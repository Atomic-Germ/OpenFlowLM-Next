//===- tokenizer.cpp ----------------------------------------*- C++ -*-===//
//
// NpuEmbeddings -- BERT WordPiece. See tokenizer.hpp.
// SPDX-License-Identifier: MIT
//
// The pipeline, in HuggingFace's order (tokenization_bert.py):
//
//   _clean_text            drop NUL, U+FFFD and control chars; spaces -> ' '
//   _tokenize_chinese_chars pad CJK ideographs with spaces so each is its own
//                          token (Hiragana, Katakana and Hangul are NOT CJK
//                          by this definition -- only the Han ranges)
//   whitespace split
//   per token: lower() then strip accents (NFD, drop Mn)   <- one table lookup
//   _run_split_on_punc     every punctuation char becomes its own token
//   WordPiece              greedy longest-match-first, '##' continuations
//
// ON THE NFC STEP HuggingFace DOES AND THIS DOES NOT
// --------------------------------------------------
// The reference normalizes to NFC before splitting. Full NFC needs canonical
// composition tables; this implementation skips it, and the reason it is safe
// is structural rather than hopeful: the very next per-token step decomposes
// (NFD) and drops the combining marks. Composing a base+mark pair only to
// decompose it again and discard the mark lands in the same place as never
// composing it. The residue is characters whose NFD keeps a non-Mn second
// element, which are vanishingly rare in practice.
//
// "Vanishingly rare" is a claim, so it is measured, not assumed:
// tools/verify_tokenizer.py compares against HuggingFace over a corpus that
// deliberately includes combining sequences, and reports exact agreement.

#include "tokenizer.hpp"

#include <algorithm>
#include <fstream>
#include <sstream>
#include <stdexcept>

#include "bert_unicode_tables.hpp"

namespace npue {
namespace {

using bert_uni::CpRange;

bool in_ranges(uint32_t cp, const CpRange *r, int n) {
  int lo = 0, hi = n - 1;
  while (lo <= hi) {
    const int mid = (lo + hi) / 2;
    if (cp < r[mid].lo) hi = mid - 1;
    else if (cp > r[mid].hi) lo = mid + 1;
    else return true;
  }
  return false;
}

bool is_punct(uint32_t cp) {
  return in_ranges(cp, bert_uni::kPunct, bert_uni::kPunct_n);
}
bool is_control(uint32_t cp) {
  return in_ranges(cp, bert_uni::kControl, bert_uni::kControl_n);
}
bool is_space(uint32_t cp) {
  return in_ranges(cp, bert_uni::kSpace, bert_uni::kSpace_n);
}

// The Han ranges HuggingFace calls "chinese chars". Deliberately excludes
// Hiragana, Katakana and Hangul -- a difference that silently changes
// tokenization for Japanese and Korean if it is got wrong.
bool is_cjk(uint32_t cp) {
  return (cp >= 0x4E00 && cp <= 0x9FFF) || (cp >= 0x3400 && cp <= 0x4DBF) ||
         (cp >= 0x20000 && cp <= 0x2A6DF) || (cp >= 0x2A700 && cp <= 0x2B73F) ||
         (cp >= 0x2B740 && cp <= 0x2B81F) || (cp >= 0x2B820 && cp <= 0x2CEAF) ||
         (cp >= 0xF900 && cp <= 0xFAFF) || (cp >= 0x2F800 && cp <= 0x2FA1F);
}

const bert_uni::FoldEntry *find_fold(uint32_t cp) {
  int lo = 0, hi = bert_uni::kFold_n - 1;
  while (lo <= hi) {
    const int mid = (lo + hi) / 2;
    if (cp < bert_uni::kFold[mid].cp) hi = mid - 1;
    else if (cp > bert_uni::kFold[mid].cp) lo = mid + 1;
    else return &bert_uni::kFold[mid];
  }
  return nullptr;
}

// --- UTF-8 ----------------------------------------------------------------
// Invalid bytes are passed through as U+FFFD, which _clean_text then drops --
// the same net effect as Python refusing to decode them, without throwing on
// user input.
std::vector<uint32_t> utf8_decode(const std::string &s) {
  std::vector<uint32_t> out;
  out.reserve(s.size());
  size_t i = 0;
  const size_t n = s.size();
  while (i < n) {
    const unsigned char c = static_cast<unsigned char>(s[i]);
    uint32_t cp = 0xFFFD;
    int len = 1;
    if (c < 0x80) { cp = c; len = 1; }
    else if ((c & 0xE0) == 0xC0 && i + 1 < n) { cp = c & 0x1F; len = 2; }
    else if ((c & 0xF0) == 0xE0 && i + 2 < n) { cp = c & 0x0F; len = 3; }
    else if ((c & 0xF8) == 0xF0 && i + 3 < n) { cp = c & 0x07; len = 4; }
    else { out.push_back(0xFFFD); ++i; continue; }
    bool ok = true;
    for (int k = 1; k < len; ++k) {
      const unsigned char cc = static_cast<unsigned char>(s[i + k]);
      if ((cc & 0xC0) != 0x80) { ok = false; break; }
      cp = (cp << 6) | (cc & 0x3F);
    }
    if (!ok) { out.push_back(0xFFFD); ++i; continue; }
    out.push_back(cp);
    i += len;
  }
  return out;
}

void utf8_append(std::string &s, uint32_t cp) {
  if (cp < 0x80) {
    s.push_back(static_cast<char>(cp));
  } else if (cp < 0x800) {
    s.push_back(static_cast<char>(0xC0 | (cp >> 6)));
    s.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
  } else if (cp < 0x10000) {
    s.push_back(static_cast<char>(0xE0 | (cp >> 12)));
    s.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
    s.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
  } else {
    s.push_back(static_cast<char>(0xF0 | (cp >> 18)));
    s.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3F)));
    s.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
    s.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
  }
}

}  // namespace

// --- vocabulary -----------------------------------------------------------

void Tokenizer::build_index(const std::string &blob) {
  std::istringstream in(blob);
  std::string line;
  int32_t id = 0;
  while (std::getline(in, line)) {
    if (!line.empty() && line.back() == '\r') line.pop_back();
    // A trailing newline at end of file must not create an empty token.
    if (line.empty() && in.eof()) break;
    tokens_.push_back(line);
    id_of_.emplace(line, id);
    ++id;
  }
  if (tokens_.empty()) throw std::runtime_error("tokenizer: empty vocabulary");
  auto get = [&](const char *t) {
    auto it = id_of_.find(t);
    if (it == id_of_.end())
      throw std::runtime_error(std::string("tokenizer: vocabulary has no ") + t);
    return it->second;
  };
  cls_id = get("[CLS]");
  sep_id = get("[SEP]");
  pad_id = get("[PAD]");
  unk_id = get("[UNK]");

  // The added-token vocabulary. The reference matches these literally in the
  // raw text and never lets basic tokenization touch them -- without this,
  // "[CLS]" in a user string tokenizes to '[', 'cl', '##s', ']'.
  for (const char *s : {"[CLS]", "[SEP]", "[PAD]", "[UNK]", "[MASK]"})
    if (id_of_.find(s) != id_of_.end()) specials_.emplace_back(s);
  std::sort(specials_.begin(), specials_.end(),
            [](const std::string &a, const std::string &b) {
              return a.size() > b.size();
            });
}

Tokenizer Tokenizer::from_vocab_file(const std::string &path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("tokenizer: cannot open " + path);
  std::stringstream ss;
  ss << f.rdbuf();
  Tokenizer t;
  t.build_index(ss.str());
  return t;
}

Tokenizer Tokenizer::from_vocab_bytes(const char *data, size_t bytes) {
  Tokenizer t;
  t.build_index(std::string(data, bytes));
  return t;
}

int32_t Tokenizer::id_of(const std::string &token) const {
  auto it = id_of_.find(token);
  return it == id_of_.end() ? unk_id : it->second;
}

const std::string &Tokenizer::token_of(int32_t id) const {
  static const std::string kEmpty;
  if (id < 0 || static_cast<size_t>(id) >= tokens_.size()) return kEmpty;
  return tokens_[static_cast<size_t>(id)];
}

// --- the pipeline ---------------------------------------------------------

std::vector<std::string> Tokenizer::tokenize(const std::string &text) const {
  // Added-token pre-pass: cut the raw string around any literal special
  // token, tokenize the gaps normally, and pass the specials through whole.
  if (!specials_.empty()) {
    size_t pos = 0;
    while (pos < text.size()) {
      size_t best = std::string::npos;
      const std::string *which = nullptr;
      for (const auto &s : specials_) {
        const size_t at = text.find(s, pos);
        if (at != std::string::npos && (best == std::string::npos || at < best ||
                                        (at == best && which &&
                                         s.size() > which->size()))) {
          best = at;
          which = &s;
        }
      }
      if (which == nullptr) break;
      std::vector<std::string> out;
      if (best > pos) {
        auto head = tokenize_plain(text.substr(pos, best - pos));
        out.insert(out.end(), head.begin(), head.end());
      }
      out.push_back(*which);
      auto tail = tokenize(text.substr(best + which->size()));
      out.insert(out.end(), tail.begin(), tail.end());
      if (pos > 0) {
        auto pre = tokenize_plain(text.substr(0, pos));
        out.insert(out.begin(), pre.begin(), pre.end());
      }
      return out;
    }
  }
  return tokenize_plain(text);
}

std::vector<std::string> Tokenizer::tokenize_plain(
    const std::string &text) const {
  const std::vector<uint32_t> cps = utf8_decode(text);

  // _clean_text + _tokenize_chinese_chars in one pass: both are per-character
  // rewrites and neither depends on the other's output.
  std::vector<uint32_t> clean;
  clean.reserve(cps.size() + 16);
  for (uint32_t cp : cps) {
    if (cp == 0 || cp == 0xFFFD || is_control(cp)) continue;
    if (is_space(cp)) { clean.push_back(' '); continue; }
    if (is_cjk(cp)) {
      clean.push_back(' ');
      clean.push_back(cp);
      clean.push_back(' ');
    } else {
      clean.push_back(cp);
    }
  }

  std::vector<std::string> out;
  std::vector<uint32_t> word;

  // One whitespace-separated word: fold, then split on punctuation, then
  // WordPiece each piece.
  auto flush_word = [&]() {
    if (word.empty()) return;

    // lower() + strip accents, PER CHARACTER.
    //
    // The first version implemented Python's context-dependent final-sigma
    // rule here ("\u03a3" at the end of a word lowercases to "\u03c2"), because
    // that is what tokenization_bert.py's token.lower() does. The differential
    // test against the reference said otherwise: this checkpoint loads the
    // FAST (Rust) tokenizer, which lowercases per character with no context,
    // so "\u039f\u0394\u03a5\u03a3\u03a3\u0395\u03a5\u03a3" ends in "##\u03c3" and not "##\u03c2".
    //
    // The documented behaviour and the shipped behaviour differ, and what
    // matters is matching the tokenizer the embeddings were produced with.
    // Only a test that compares every id could have caught this.
    std::vector<uint32_t> folded;
    folded.reserve(word.size());
    for (size_t i = 0; i < word.size(); ++i) {
      const uint32_t cp = word[i];
      if (const auto *f = find_fold(cp)) {
        for (uint32_t k = 0; k < f->len; ++k)
          folded.push_back(bert_uni::kFoldData[f->off + k]);
      } else {
        folded.push_back(cp);
      }
    }

    // _run_split_on_punc, then WordPiece on each resulting piece.
    std::vector<std::vector<uint32_t>> pieces;
    std::vector<uint32_t> cur;
    for (uint32_t cp : folded) {
      if (is_punct(cp)) {
        if (!cur.empty()) { pieces.push_back(cur); cur.clear(); }
        pieces.push_back({cp});
      } else {
        cur.push_back(cp);
      }
    }
    if (!cur.empty()) pieces.push_back(cur);

    for (const auto &piece : pieces) {
      if (piece.empty()) continue;
      // Longer than max_chars_per_word is [UNK] without even trying, which
      // is HuggingFace's behaviour and keeps a pathological input O(1).
      if (static_cast<int>(piece.size()) > max_chars_per_word_) {
        out.push_back("[UNK]");
        continue;
      }
      // Greedy longest-match-first over CHARACTER boundaries. Substrings are
      // built from codepoints, never from bytes -- slicing UTF-8 by byte
      // would cut multi-byte characters in half and miss vocabulary entries.
      std::vector<size_t> byte_at(piece.size() + 1, 0);
      std::string flat;
      for (size_t i = 0; i < piece.size(); ++i) {
        byte_at[i] = flat.size();
        utf8_append(flat, piece[i]);
      }
      byte_at[piece.size()] = flat.size();

      std::vector<std::string> sub;
      size_t start = 0;
      bool bad = false;
      while (start < piece.size()) {
        size_t end = piece.size();
        std::string found;
        while (start < end) {
          std::string s = flat.substr(byte_at[start], byte_at[end] - byte_at[start]);
          if (start > 0) s = "##" + s;
          if (id_of_.find(s) != id_of_.end()) { found = s; break; }
          --end;
        }
        if (found.empty()) { bad = true; break; }
        sub.push_back(found);
        start = end;
      }
      if (bad) out.push_back("[UNK]");
      else out.insert(out.end(), sub.begin(), sub.end());
    }
    word.clear();
  };

  for (uint32_t cp : clean) {
    if (cp == ' ') flush_word();
    else word.push_back(cp);
  }
  flush_word();
  return out;
}

Encoded Tokenizer::encode(const std::string &text, int max_len) const {
  Encoded e;
  const std::vector<std::string> toks = tokenize(text);
  const int room = max_len - 2;                     // [CLS] ... [SEP]
  const int take = std::min<int>(static_cast<int>(toks.size()),
                                 std::max(0, room));

  e.input_ids.reserve(max_len);
  e.input_ids.push_back(cls_id);
  for (int i = 0; i < take; ++i) e.input_ids.push_back(id_of(toks[i]));
  e.input_ids.push_back(sep_id);
  e.n_tokens = static_cast<int32_t>(e.input_ids.size());
  // Measured before the cap, so the caller learns the size of the input it
  // gave rather than the size of the buffer it hit.
  e.n_tokens_full = static_cast<int32_t>(toks.size()) + 2;   // + [CLS]/[SEP]
  e.truncated = take < static_cast<int>(toks.size());

  e.attention_mask.assign(e.input_ids.size(), 1);
  while (static_cast<int>(e.input_ids.size()) < max_len) {
    e.input_ids.push_back(pad_id);
    e.attention_mask.push_back(0);
  }
  e.token_type_ids.assign(e.input_ids.size(), 0);
  return e;
}

std::vector<Encoded> Tokenizer::encode_batch(
    const std::vector<std::string> &texts, int max_len) const {
  std::vector<Encoded> out;
  out.reserve(texts.size());
  for (const auto &t : texts) out.push_back(encode(t, max_len));
  return out;
}

}  // namespace npue
