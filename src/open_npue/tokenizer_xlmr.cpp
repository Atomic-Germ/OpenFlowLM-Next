//===- tokenizer_xlmr.cpp ------------------------------------*- C++ -*-===//
//
// NpuEmbeddings -- XLM-R family SentencePiece Unigram. See
// tokenizer_xlmr.hpp for the pipeline summary and design rationale.
// SPDX-License-Identifier: MIT
//
// LINE-FOR-LINE PORT of tools/xlmr_tokenizer_ref.py (the executable spec,
// tasks/0127). The section order below follows that file's -- grapheme
// classifier, White_Space set, Darts trie, then the tokenizer's stages in
// pipeline order -- so the two can be diffed side by side. Comments repeat
// only what is not obvious from the port itself; the WHY of every quirk is
// in the reference's own comments and 0127's task log.
//
// f64 END TO END: the scores are stored f64 in the table and summed as
// double in viterbi() below. 65,856 of the 250,002 log-probs do not
// round-trip through float, and HuggingFace's Viterbi sums f64 -- a float
// anywhere on this path breaks near-tie byte-exactness (0127 section 3).

#include "tokenizer_xlmr.hpp"

#include "xlmr_unicode_tables.hpp"

#include <algorithm>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace npue {
namespace {

uint16_t read_u16(const char *p) {
  uint16_t v;
  std::memcpy(&v, p, 2);
  return v;
}
uint32_t read_u32(const char *p) {
  uint32_t v;
  std::memcpy(&v, p, 4);
  return v;
}
double read_f64(const char *p) {
  double v;
  std::memcpy(&v, p, 8);
  return v;
}

// --- UTF-8, same conventions as tokenizer_gemma.cpp's -----------------
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

size_t utf8_len(uint32_t cp) {
  if (cp < 0x80) return 1;
  if (cp < 0x800) return 2;
  if (cp < 0x10000) return 3;
  return 4;
}

bool in_ranges(uint32_t cp, const xlmr_uni::CpRange *rs, int n) {
  int lo = 0, hi = n - 1;
  while (lo <= hi) {
    const int mid = (lo + hi) / 2;
    if (cp < rs[mid].lo) hi = mid - 1;
    else if (cp > rs[mid].hi) lo = mid + 1;
    else return true;
  }
  return false;
}

// --------------------------------------------------------------------------
// UAX #29 extended grapheme clusters -- the same approximation as the
// reference's graphemes()/_gcb_class(), check for check, in the same order.
// The Mn/Me, Mc and Cc/Cf/Zl/Zp category ranges come from the generated
// xlmr_unicode_tables.hpp (same unicodedata the reference reads); the
// explicit sets below are transcribed from the reference directly.
// --------------------------------------------------------------------------

enum class Gcb {
  CR, LF, ZWJ, Extend, Prepend, RI, L, V, T, LV, LVT,
  SpacingMark, Control, Other,
};

bool is_prepend(uint32_t cp) {
  return (0x0600 <= cp && cp <= 0x0605) || cp == 0x06DD || cp == 0x070F ||
         cp == 0x0890 || cp == 0x0891 || cp == 0x08E2 || cp == 0x0D4E ||
         cp == 0x110BD || cp == 0x110CD;
}

Gcb gcb_class(uint32_t cp) {
  if (cp == 0x000D) return Gcb::CR;
  if (cp == 0x000A) return Gcb::LF;
  if (cp == 0x200D) return Gcb::ZWJ;
  if (cp == 0x200C) return Gcb::Extend;  // ZWNJ
  if (is_prepend(cp)) return Gcb::Prepend;
  if (0x1F1E6 <= cp && cp <= 0x1F1FF) return Gcb::RI;
  // Hangul
  if ((0x1100 <= cp && cp <= 0x115F) || (0xA960 <= cp && cp <= 0xA97C))
    return Gcb::L;
  if ((0x1160 <= cp && cp <= 0x11A7) || (0xD7B0 <= cp && cp <= 0xD7C6))
    return Gcb::V;
  if ((0x11A8 <= cp && cp <= 0x11FF) || (0xD7CB <= cp && cp <= 0xD7FB))
    return Gcb::T;
  if (0xAC00 <= cp && cp <= 0xD7A3)
    return (cp - 0xAC00) % 28 == 0 ? Gcb::LV : Gcb::LVT;
  if (in_ranges(cp, xlmr_uni::kExtend, xlmr_uni::kExtend_n))
    return Gcb::Extend;                  // Mn, Me
  if (in_ranges(cp, xlmr_uni::kSpacingMark, xlmr_uni::kSpacingMark_n))
    return Gcb::SpacingMark;             // Mc
  if (in_ranges(cp, xlmr_uni::kControlCat, xlmr_uni::kControlCat_n))
    return Gcb::Control;                 // Cc, Zl, Zp, Cf
  return Gcb::Other;
}

bool is_ext_pict(uint32_t cp) {
  // Extended_Pictographic approximation for GB11 (ZWJ emoji sequences).
  return (0x1F000 <= cp && cp <= 0x1FFFD) || (0x2600 <= cp && cp <= 0x27BF) ||
         (0x2B00 <= cp && cp <= 0x2BFF) || cp == 0x00A9 || cp == 0x00AE ||
         cp == 0x2122 || cp == 0x203C || cp == 0x2049 || cp == 0x2139 ||
         (0x2190 <= cp && cp <= 0x21FF) || (0x2300 <= cp && cp <= 0x23FF) ||
         (0x25A0 <= cp && cp <= 0x25FF) || (0x2900 <= cp && cp <= 0x297F) ||
         (0x3030 <= cp && cp <= 0x303D) || cp == 0x3297 || cp == 0x3299 ||
         (0xFE00 <= cp && cp <= 0xFE0F);  // variation selectors (Mn anyway)
}

// Cluster START indices into `cps`, with cps.size() appended -- the same
// clusters the reference's graphemes() yields, as index pairs.
std::vector<size_t> grapheme_starts(const std::vector<uint32_t> &cps) {
  std::vector<size_t> starts;
  if (cps.empty()) return starts;
  std::vector<Gcb> cls(cps.size());
  for (size_t i = 0; i < cps.size(); ++i) cls[i] = gcb_class(cps[i]);
  starts.push_back(0);
  int ri_run = 0;
  for (size_t i = 1; i < cps.size(); ++i) {
    const Gcb left = cls[i - 1], right = cls[i];
    if (left == Gcb::RI) ++ri_run; else ri_run = 0;
    bool brk;
    if (left == Gcb::CR && right == Gcb::LF)
      brk = false;                                          // GB3
    else if (left == Gcb::CR || left == Gcb::LF || left == Gcb::Control)
      brk = true;                                           // GB4
    else if (right == Gcb::CR || right == Gcb::LF || right == Gcb::Control)
      brk = true;                                           // GB5
    else if (left == Gcb::L &&
             (right == Gcb::L || right == Gcb::V || right == Gcb::LV ||
              right == Gcb::LVT))
      brk = false;                                          // GB6
    else if ((left == Gcb::LV || left == Gcb::V) &&
             (right == Gcb::V || right == Gcb::T))
      brk = false;                                          // GB7
    else if ((left == Gcb::LVT || left == Gcb::T) && right == Gcb::T)
      brk = false;                                          // GB8
    else if (right == Gcb::Extend || right == Gcb::ZWJ)
      brk = false;                                          // GB9
    else if (right == Gcb::SpacingMark)
      brk = false;                                          // GB9a
    else if (left == Gcb::Prepend)
      brk = false;                                          // GB9b
    else if (left == Gcb::ZWJ && is_ext_pict(cps[i]))
      brk = false;                                          // GB11 (approx)
    else if (left == Gcb::RI && right == Gcb::RI && ri_run % 2 == 1)
      brk = false;                                          // GB12/13
    else
      brk = true;                                           // GB999
    if (brk) starts.push_back(i);
  }
  starts.push_back(cps.size());
  return starts;
}

// --------------------------------------------------------------------------
// Unicode White_Space, as Rust char::is_whitespace sees it -- the same
// explicit set as the reference's _WHITE_SPACE (NOT isspace(): Python's
// str.isspace() additionally claims U+001C..U+001F, C's isspace() sees only
// ASCII and is locale-dependent; both diverge from tokenizers-rs).
// --------------------------------------------------------------------------
bool is_white_space(uint32_t cp) {
  return cp == 0x09 || cp == 0x0A || cp == 0x0B || cp == 0x0C || cp == 0x0D ||
         cp == 0x20 || cp == 0x85 || cp == 0xA0 || cp == 0x1680 ||
         (0x2000 <= cp && cp <= 0x200A) || cp == 0x2028 || cp == 0x2029 ||
         cp == 0x202F || cp == 0x205F || cp == 0x3000;
}

}  // namespace

// --------------------------------------------------------------------------
// Darts double-array trie (darts_clone as embedded in sentencepiece and the
// spm_precompiled crate). Bit layout per u32 unit, as in the reference:
//   has_leaf(u) = (u >> 8) & 1
//   value(u)    = u & 0x7FFFFFFF          (on the leaf unit)
//   label(u)    = u & 0x800000FF          (compared against the key byte)
//   offset(u)   = (u >> 10) << ((u & 0x200) >> 6)
// The reference collects every prefix match and uses results[0]; only the
// FIRST (shortest) match is ever consumed, so this walk stops there.
// --------------------------------------------------------------------------
int64_t XlmrTokenizer::trie_first_match(const char *key, size_t len) const {
  const uint32_t *units = trie_units_.data();
  const size_t n_units = trie_units_.size();
  uint32_t node_pos = 0;
  uint32_t unit = units[node_pos];
  node_pos ^= (unit >> 10) << ((unit & 0x200) >> 6);
  for (size_t i = 0; i < len; ++i) {
    const uint32_t c = static_cast<unsigned char>(key[i]);
    node_pos ^= c;
    if (node_pos >= n_units)
      throw std::runtime_error("tokenizer_xlmr: charsmap trie walk out of "
                               "bounds -- corrupt table");
    unit = units[node_pos];
    if ((unit & 0x800000FF) != c) return -1;
    node_pos ^= (unit >> 10) << ((unit & 0x200) >> 6);
    if ((unit >> 8) & 1) {
      if (node_pos >= n_units)
        throw std::runtime_error("tokenizer_xlmr: charsmap trie leaf out of "
                                 "bounds -- corrupt table");
      return static_cast<int64_t>(units[node_pos] & 0x7FFFFFFF);
    }
  }
  return -1;
}

// --- table loading ----------------------------------------------------

void XlmrTokenizer::build_index(const std::string &blob) {
  const char *p = blob.data();
  const char *end = p + blob.size();
  auto need = [&](size_t n) {
    if (static_cast<size_t>(end - p) < n)
      throw std::runtime_error("tokenizer_xlmr: truncated table");
  };

  need(8);
  if (std::memcmp(p, "XLMRTOK1", 8) != 0)
    throw std::runtime_error("tokenizer_xlmr: bad magic");
  p += 8;

  need(14 * 4);
  const uint32_t version = read_u32(p); p += 4;
  if (version != 1)
    throw std::runtime_error("tokenizer_xlmr: unsupported table version " +
                             std::to_string(version));
  const uint32_t vocab_size = read_u32(p); p += 4;
  unk_id = static_cast<int32_t>(read_u32(p)); p += 4;
  bos_id = static_cast<int32_t>(read_u32(p)); p += 4;
  eos_id = static_cast<int32_t>(read_u32(p)); p += 4;
  pad_id = static_cast<int32_t>(read_u32(p)); p += 4;
  mask_id = static_cast<int32_t>(read_u32(p)); p += 4;
  add_bos_ = read_u32(p); p += 4;
  add_eos_ = read_u32(p); p += 4;
  metaspace_cp_ = read_u32(p); p += 4;
  prepend_scheme_ = read_u32(p); p += 4;
  metaspace_split_ = read_u32(p); p += 4;
  p += 4;  // max_piece_bytes: informational, the Viterbi caps on chars
  max_piece_chars_ = read_u32(p); p += 4;

  need(16);
  min_score_ = read_f64(p); p += 8;
  unk_score_ = read_f64(p); p += 8;

  need(8);
  const uint32_t trie_size = read_u32(p); p += 4;
  const uint32_t norm_size = read_u32(p); p += 4;
  if (trie_size % 4 != 0)
    throw std::runtime_error("tokenizer_xlmr: trie blob not a multiple of "
                             "4 bytes");
  need(trie_size);
  trie_units_.resize(trie_size / 4);
  std::memcpy(trie_units_.data(), p, trie_size);
  p += trie_size;
  need(norm_size);
  normalized_blob_.assign(p, norm_size);
  p += norm_size;

  need(static_cast<size_t>(vocab_size) * 8);
  scores_.resize(vocab_size);
  for (uint32_t i = 0; i < vocab_size; ++i) { scores_[i] = read_f64(p); p += 8; }

  pieces_.resize(vocab_size);
  piece_to_id_.reserve(static_cast<size_t>(vocab_size) * 2);
  for (uint32_t i = 0; i < vocab_size; ++i) {
    need(2);
    const uint16_t len = read_u16(p); p += 2;
    need(len);
    pieces_[i].assign(p, len);
    p += len;
    piece_to_id_.emplace(pieces_[i], static_cast<int32_t>(i));
  }
  if (p != end)
    throw std::runtime_error("tokenizer_xlmr: trailing bytes: read " +
                             std::to_string(p - blob.data()) + " of " +
                             std::to_string(blob.size()));

  if (pieces_.empty())
    throw std::runtime_error("tokenizer_xlmr: empty vocabulary");
  if (bos_id < 0 || eos_id < 0 || pad_id < 0 || unk_id < 0 ||
      static_cast<uint32_t>(unk_id) >= vocab_size)
    throw std::runtime_error("tokenizer_xlmr: missing a required special id");
  if (trie_units_.empty())
    throw std::runtime_error("tokenizer_xlmr: empty charsmap trie");
  if (!normalized_blob_.empty() && normalized_blob_.back() != '\0')
    throw std::runtime_error("tokenizer_xlmr: normalized-strings blob does "
                             "not end in NUL");
  if (max_piece_chars_ == 0)
    throw std::runtime_error("tokenizer_xlmr: max_piece_chars is 0");
}

XlmrTokenizer XlmrTokenizer::from_table_file(const std::string &path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("tokenizer_xlmr: cannot open " + path);
  std::stringstream ss;
  ss << f.rdbuf();
  XlmrTokenizer t;
  t.build_index(ss.str());
  return t;
}

XlmrTokenizer XlmrTokenizer::from_table_bytes(const char *data, size_t bytes) {
  XlmrTokenizer t;
  t.build_index(std::string(data, bytes));
  return t;
}

const std::string &XlmrTokenizer::piece_of(int32_t id) const {
  static const std::string kEmpty;
  if (id < 0 || static_cast<size_t>(id) >= pieces_.size()) return kEmpty;
  return pieces_[static_cast<size_t>(id)];
}

// -- 1. Precompiled normalizer ------------------------------------------

bool XlmrTokenizer::transform(const char *chunk, size_t len,
                              std::string &out) const {
  const int64_t index = trie_first_match(chunk, len);
  if (index < 0) return false;
  if (static_cast<size_t>(index) >= normalized_blob_.size())
    throw std::runtime_error("tokenizer_xlmr: charsmap value out of range");
  const size_t nul = normalized_blob_.find('\0', static_cast<size_t>(index));
  if (nul == std::string::npos)
    throw std::runtime_error("tokenizer_xlmr: unterminated charsmap "
                             "replacement string");
  out.append(normalized_blob_, static_cast<size_t>(index),
             nul - static_cast<size_t>(index));
  return true;
}

std::string XlmrTokenizer::normalize(const std::string &text) const {
  const std::vector<uint32_t> cps = utf8_decode(text);
  const std::vector<size_t> starts = grapheme_starts(cps);
  std::string out;
  out.reserve(text.size());
  std::string chunk;
  for (size_t gi = 0; gi + 1 < starts.size(); ++gi) {
    const size_t lo = starts[gi], hi = starts[gi + 1];
    size_t g_bytes = 0;
    for (size_t k = lo; k < hi; ++k) g_bytes += utf8_len(cps[k]);
    if (g_bytes < 6) {
      chunk.clear();
      for (size_t k = lo; k < hi; ++k) utf8_append(chunk, cps[k]);
      if (transform(chunk.data(), chunk.size(), out))
        continue;  // replaces the WHOLE grapheme
    }
    for (size_t k = lo; k < hi; ++k) {
      chunk.clear();
      utf8_append(chunk, cps[k]);
      if (!transform(chunk.data(), chunk.size(), out))
        out += chunk;
    }
  }
  return out;
}

// -- 2 + 3. WhitespaceSplit then Metaspace ------------------------------

std::vector<std::string> XlmrTokenizer::pre_tokenize(
    const std::string &normalized) const {
  const std::vector<uint32_t> cps = utf8_decode(normalized);
  std::vector<std::vector<uint32_t>> words;
  std::vector<uint32_t> cur;
  for (uint32_t cp : cps) {
    if (is_white_space(cp)) {
      if (!cur.empty()) { words.push_back(cur); cur.clear(); }
    } else {
      cur.push_back(cp);
    }
  }
  if (!cur.empty()) words.push_back(cur);

  const uint32_t rep = metaspace_cp_;
  std::vector<std::string> pretokens;
  for (auto &word : words) {
    // replace(' ', rep) is a no-op here: WhitespaceSplit removed them
    if (prepend_scheme_ == 2 && !(!word.empty() && word[0] == rep)) {
      word.insert(word.begin(), rep);
    } else if (prepend_scheme_ == 1 && pretokens.empty() &&
               !(!word.empty() && word[0] == rep)) {
      word.insert(word.begin(), rep);
    }
    if (metaspace_split_) {
      // split on rep, delimiter merged with what FOLLOWS it
      std::vector<uint32_t> cur_part;
      for (uint32_t cp : word) {
        if (cp == rep && !cur_part.empty()) {
          std::string s;
          for (uint32_t c : cur_part) utf8_append(s, c);
          pretokens.push_back(std::move(s));
          cur_part.assign(1, cp);
        } else {
          cur_part.push_back(cp);
        }
      }
      if (!cur_part.empty()) {
        std::string s;
        for (uint32_t c : cur_part) utf8_append(s, c);
        pretokens.push_back(std::move(s));
      }
    } else {
      std::string s;
      for (uint32_t c : word) utf8_append(s, c);
      pretokens.push_back(std::move(s));
    }
  }
  return pretokens;
}

// -- 4. Unigram Viterbi (encode_optimized mirror) -----------------------

std::vector<int32_t> XlmrTokenizer::viterbi(const std::string &pretoken) const {
  // Char boundaries: pretoken is our own normalize() output, so it is valid
  // UTF-8 and lead-byte scanning finds exactly the reference's char indices.
  std::vector<size_t> offs;
  offs.reserve(pretoken.size() + 1);
  for (size_t i = 0; i < pretoken.size(); ++i) {
    if ((static_cast<unsigned char>(pretoken[i]) & 0xC0) != 0x80)
      offs.push_back(i);
  }
  offs.push_back(pretoken.size());
  const size_t n = offs.size() - 1;
  if (n == 0) return {};

  const double NEG = -1.0e300;  // only read where best_start says unreached
  std::vector<double> best_score(n + 1, NEG);
  std::vector<int32_t> best_start(n + 1, -1);
  std::vector<int32_t> best_id(n + 1, -1);
  best_score[0] = 0.0;
  const size_t maxc = max_piece_chars_;
  std::string piece;
  for (size_t start = 0; start < n; ++start) {
    const double here = best_score[start];
    if (best_start[start] == -1 && start != 0)
      continue;  // unreachable (cannot happen: unk covers every char)
    bool has_single = false;
    const size_t top = std::min(n, start + maxc);
    for (size_t end = start + 1; end <= top; ++end) {
      piece.assign(pretoken, offs[start], offs[end] - offs[start]);
      const auto it = piece_to_id_.find(piece);
      if (it == piece_to_id_.end()) continue;
      const int32_t tid = it->second;
      const double cand = here + scores_[static_cast<size_t>(tid)];
      if (best_start[end] == -1 || cand > best_score[end]) {
        best_score[end] = cand;
        best_start[end] = static_cast<int32_t>(start);
        best_id[end] = tid;
      }
      if (end - start == 1) has_single = true;
    }
    if (!has_single) {
      const size_t end = start + 1;
      const double cand = here + unk_score_;
      if (best_start[end] == -1 || cand > best_score[end]) {
        best_score[end] = cand;
        best_start[end] = static_cast<int32_t>(start);
        best_id[end] = unk_id;
      }
    }
  }
  // backtrack
  std::vector<int32_t> ids;
  size_t pos = n;
  while (pos > 0) {
    ids.push_back(best_id[pos]);
    pos = static_cast<size_t>(best_start[pos]);
  }
  std::reverse(ids.begin(), ids.end());
  // fuse consecutive <unk> (fuse_unk = true in the HF Unigram model)
  std::vector<int32_t> fused;
  fused.reserve(ids.size());
  for (int32_t tid : ids) {
    if (tid == unk_id && !fused.empty() && fused.back() == unk_id) continue;
    fused.push_back(tid);
  }
  return fused;
}

// -- full pipeline -------------------------------------------------------

std::vector<int32_t> XlmrTokenizer::encode(const std::string &text) const {
  std::vector<int32_t> ids;
  for (const std::string &pretoken : pre_tokenize(normalize(text))) {
    const std::vector<int32_t> part = viterbi(pretoken);
    ids.insert(ids.end(), part.begin(), part.end());
  }
  if (add_bos_) ids.insert(ids.begin(), bos_id);
  if (add_eos_) ids.push_back(eos_id);
  return ids;
}

}  // namespace npue
