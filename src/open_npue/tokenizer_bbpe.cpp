//===- tokenizer_bbpe.cpp ------------------------------------*- C++ -*-===//
//
// NpuEmbeddings -- byte-level BPE. See tokenizer_bbpe.hpp for the design.
// SPDX-License-Identifier: MIT
//
// THE PIPELINE, in the order it runs. Each stage was read out of a real
// tokenizer.json and then confirmed against HuggingFace's own output before
// being written here; the fuzz numbers are in tasks/0153.
//
//   1. ADDED TOKENS. The checkpoint's added_tokens are matched literally
//      against the raw text, leftmost-longest, splitting it into "this span
//      IS id N" pieces and ordinary pieces. The OLMo family uses this for
//      runs of 2-24 spaces, so skipping the stage would silently change the
//      tokenization of every indented text. `lstrip` extends a match over the
//      whitespace before it, which is then swallowed rather than tokenized --
//      BERT's [MASK] carries it, and "a [MASK] b" gives ['a', ' [MASK]',
//      'Gb'], one token fewer than the naive reading. (`rstrip` is refused by
//      the generator: nothing here sets it, so it could only ship unverified.)
//
//   2. NFC, if the checkpoint asks for it. Composition is the standard
//      algorithm over bbpe_unicode_tables.hpp's canonical data, with Hangul
//      handled algorithmically.
//
//   3. PRE-TOKENIZE. The GPT-2 regex
//         's|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+
//      as a hand-written scanner over the four-way character class. The one
//      subtle alternative is `\s+(?!\S)`: inside a maximal whitespace run it
//      can only stop where the next character is also whitespace, so greedily
//      it takes the whole run at end-of-text and everything-but-the-last
//      space otherwise -- which is precisely how the space before a word ends
//      up attached to that word as "Gword".
//
//   4. BYTE MAP. Each word's UTF-8 bytes become printable codepoints via
//      GPT-2's bytes_to_unicode. Every byte that can occur in valid UTF-8 has
//      a vocabulary entry, so there is no fallback and no <unk> -- the
//      generator verifies that rather than trusting it.
//
//   5. BPE, per word, merges applied lowest-rank first. Same engine as
//      tokenizer_gemma.cpp: doubly-linked list plus a min-heap whose entries
//      are validated on pop, because an earlier merge may have consumed an
//      endpoint. Merges never cross a word boundary, which is the whole point
//      of stage 3.

#include "tokenizer_bbpe.hpp"

#include "bbpe_unicode_tables.hpp"

#include <algorithm>
#include <cstring>
#include <fstream>
#include <queue>
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

// --- UTF-8, the same conventions as tokenizer_gemma.cpp's ---------------
//
// An invalid sequence becomes U+FFFD, exactly as Python's decoder would with
// errors="replace". That matters for more than tidiness: re-encoding the
// replacement character produces valid UTF-8, which is what guarantees the
// byte map below never sees 0xC0, 0xC1 or 0xF5-0xFF -- the five bytes a
// byte-level vocabulary is allowed to omit.
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

// --- Unicode lookups over the generated tables --------------------------

uint8_t char_class(uint32_t cp) {
  int lo = 0, hi = bbpe_uni::kClass_n - 1;
  while (lo <= hi) {
    const int mid = lo + (hi - lo) / 2;
    if (cp < bbpe_uni::kClass[mid].lo) hi = mid - 1;
    else if (cp > bbpe_uni::kClass[mid].hi) lo = mid + 1;
    else return bbpe_uni::kClass[mid].cls;
  }
  return bbpe_uni::kOther;      // absent means Other, by construction
}

uint8_t ccc_of(uint32_t cp) {
  int lo = 0, hi = bbpe_uni::kCcc_n - 1;
  while (lo <= hi) {
    const int mid = lo + (hi - lo) / 2;
    if (cp < bbpe_uni::kCcc[mid].cp) hi = mid - 1;
    else if (cp > bbpe_uni::kCcc[mid].cp) lo = mid + 1;
    else return bbpe_uni::kCcc[mid].ccc;
  }
  return 0;
}

const bbpe_uni::Decomp *decomp_of(uint32_t cp) {
  int lo = 0, hi = bbpe_uni::kDecomp_n - 1;
  while (lo <= hi) {
    const int mid = lo + (hi - lo) / 2;
    if (cp < bbpe_uni::kDecomp[mid].cp) hi = mid - 1;
    else if (cp > bbpe_uni::kDecomp[mid].cp) lo = mid + 1;
    else return &bbpe_uni::kDecomp[mid];
  }
  return nullptr;
}

uint32_t compose_pair(uint32_t a, uint32_t b) {
  int lo = 0, hi = bbpe_uni::kCompose_n - 1;
  while (lo <= hi) {
    const int mid = lo + (hi - lo) / 2;
    const auto &e = bbpe_uni::kCompose[mid];
    if (a < e.a || (a == e.a && b < e.b)) hi = mid - 1;
    else if (a > e.a || (a == e.a && b > e.b)) lo = mid + 1;
    else return e.cp;
  }
  return 0;
}

// Canonical decomposition, plus algorithmic Hangul.
//
// NO RECURSION: the table holds HuggingFace's own NFD output, which is
// already fully decomposed and canonically ordered, so one lookup gives the
// final sequence.
void decompose_cp(uint32_t cp, std::vector<uint32_t> &out) {
  using namespace bbpe_uni;
  if (cp >= kSBase && cp < kSBase + kSCount) {
    const uint32_t s = cp - kSBase;
    out.push_back(kLBase + s / kNCount);
    out.push_back(kVBase + (s % kNCount) / kTCount);
    const uint32_t t = s % kTCount;
    if (t) out.push_back(kTBase + t);
    return;
  }
  const Decomp *d = decomp_of(cp);
  if (d == nullptr) { out.push_back(cp); return; }
  for (uint32_t i = 0; i < d->len; ++i)
    out.push_back(kDecompData[d->off + i]);
}

// NFC = canonical decomposition, canonical ordering, canonical composition.
// The standard algorithm (Unicode 15.0 sections 3.11 and 3.12); the only
// thing worth pointing at is the `last_ccc` blocking rule in the composition
// pass, which is what stops a starter from reaching across an intervening
// mark of equal or higher combining class.
std::vector<uint32_t> nfc(const std::vector<uint32_t> &cps) {
  std::vector<uint32_t> d;
  d.reserve(cps.size() + cps.size() / 4);
  for (uint32_t cp : cps) decompose_cp(cp, d);

  // Canonical ordering: a stable bubble over runs of nonzero ccc.
  for (size_t i = 1; i < d.size(); ++i) {
    const uint8_t k = ccc_of(d[i]);
    if (k == 0) continue;
    size_t j = i;
    while (j > 0) {
      const uint8_t kp = ccc_of(d[j - 1]);
      if (kp == 0 || kp <= k) break;
      std::swap(d[j - 1], d[j]);
      --j;
    }
  }

  using namespace bbpe_uni;
  std::vector<uint32_t> out;
  out.reserve(d.size());
  size_t starter = static_cast<size_t>(-1);
  // `last_ccc` is the combining class of the last character emitted SINCE the
  // current starter, and it is -1 -- not 0 -- immediately after a starter.
  // The blocking rule is "C is blocked from the starter if something between
  // them has ccc 0 or ccc >= ccc(C)", so a character DIRECTLY after the
  // starter is never blocked, whatever its class. Initialising to 0 instead
  // makes `last_ccc < ccc(C)` false for every C with ccc 0 -- which is every
  // Hangul jamo, so "한글" decomposed to jamo and never came back, and
  // tokenized as 18 raw byte pieces against HuggingFace's 3 (tasks/0153).
  int last_ccc = -1;
  for (size_t i = 0; i < d.size(); ++i) {
    const uint32_t c = d[i];
    const uint8_t k = ccc_of(c);
    if (starter != static_cast<size_t>(-1) && last_ccc < static_cast<int>(k)) {
      const uint32_t s = out[starter];
      uint32_t composed = 0;
      // Hangul, algorithmically: L+V and LV+T.
      if (s >= kLBase && s < kLBase + kLCount && c >= kVBase &&
          c < kVBase + kVCount) {
        composed = kSBase + ((s - kLBase) * kVCount + (c - kVBase)) * kTCount;
      } else if (s >= kSBase && s < kSBase + kSCount &&
                 (s - kSBase) % kTCount == 0 && c > kTBase &&
                 c < kTBase + kTCount) {
        composed = s + (c - kTBase);
      } else {
        composed = compose_pair(s, c);
      }
      if (composed) {
        out[starter] = composed;
        continue;                      // c consumed; last_ccc unchanged
      }
    }
    if (k == 0) {
      starter = out.size();
      last_ccc = -1;                 // nothing between the starter and what
    } else {                         // comes next, so nothing can block it
      last_ccc = static_cast<int>(k);
    }
    out.push_back(c);
  }
  return out;
}

// --- GPT-2's byte <-> printable-codepoint map ---------------------------
//
// Built once, the same construction the generator uses. Byte -> codepoint is
// a 256-entry array; codepoint -> byte is only needed by the table generator,
// not here.
struct ByteMap {
  uint32_t to_cp[256];
  ByteMap() {
    bool used[256] = {false};
    int n = 0;
    auto take = [&](int lo, int hi) {
      for (int b = lo; b <= hi; ++b) { to_cp[b] = static_cast<uint32_t>(b); used[b] = true; }
    };
    take('!', '~');
    take(0xA1, 0xAC);
    take(0xAE, 0xFF);
    for (int b = 0; b < 256; ++b)
      if (!used[b]) to_cp[b] = static_cast<uint32_t>(256 + n++);
  }
};
const ByteMap &byte_map() {
  static const ByteMap m;
  return m;
}

uint64_t pair_key(int32_t a, int32_t b) {
  return (static_cast<uint64_t>(static_cast<uint32_t>(a)) << 32) |
         static_cast<uint32_t>(b);
}

const char *kContractions[] = {"'s", "'t", "'re", "'ve", "'m", "'ll", "'d"};

}  // namespace

// --- table loading ------------------------------------------------------

void BbpeTokenizer::build_index(const std::string &blob) {
  const char *p = blob.data();
  const char *end = p + blob.size();
  auto need = [&](size_t n) {
    if (static_cast<size_t>(end - p) < n)
      throw std::runtime_error("tokenizer_bbpe: truncated table");
  };

  need(8);
  if (std::memcmp(p, "BBPETOK1", 8) != 0)
    throw std::runtime_error("tokenizer_bbpe: bad magic");
  p += 8;

  need(4); const uint32_t version = read_u32(p); p += 4;
  if (version != 1)
    throw std::runtime_error("tokenizer_bbpe: unsupported table version " +
                             std::to_string(version));

  need(4); normalizer = read_u32(p); p += 4;
  if (normalizer > 1)
    throw std::runtime_error("tokenizer_bbpe: unknown normalizer id " +
                             std::to_string(normalizer));
  need(4); add_prefix_space = read_u32(p) != 0; p += 4;

  need(12);
  const uint32_t vocab_size = read_u32(p); p += 4;
  const uint32_t num_merges = read_u32(p); p += 4;
  const uint32_t num_added = read_u32(p); p += 4;

  need(20);
  cls_id = static_cast<int32_t>(read_u32(p)); p += 4;
  sep_id = static_cast<int32_t>(read_u32(p)); p += 4;
  pad_id = static_cast<int32_t>(read_u32(p)); p += 4;
  unk_id = static_cast<int32_t>(read_u32(p)); p += 4;
  mask_id = static_cast<int32_t>(read_u32(p)); p += 4;

  need(4); const uint32_t n_prefix = read_u32(p); p += 4;
  prefix_ids_.resize(n_prefix);
  for (uint32_t i = 0; i < n_prefix; ++i) {
    need(4); prefix_ids_[i] = static_cast<int32_t>(read_u32(p)); p += 4;
  }
  need(4); const uint32_t n_suffix = read_u32(p); p += 4;
  suffix_ids_.resize(n_suffix);
  for (uint32_t i = 0; i < n_suffix; ++i) {
    need(4); suffix_ids_[i] = static_cast<int32_t>(read_u32(p)); p += 4;
  }

  id_to_token_.resize(vocab_size);
  token_to_id_.reserve(vocab_size * 2);
  for (uint32_t i = 0; i < vocab_size; ++i) {
    need(2);
    const uint16_t len = read_u16(p); p += 2;
    need(len);
    id_to_token_[i].assign(p, len);
    p += len;
    token_to_id_.emplace(id_to_token_[i], static_cast<int32_t>(i));
  }

  merge_of_.reserve(static_cast<size_t>(num_merges) * 2);
  for (uint32_t rank = 0; rank < num_merges; ++rank) {
    need(12);
    const uint32_t a = read_u32(p); p += 4;
    const uint32_t b = read_u32(p); p += 4;
    const uint32_t merged = read_u32(p); p += 4;
    merge_of_.emplace(pair_key(static_cast<int32_t>(a), static_cast<int32_t>(b)),
                      MergeInfo{rank, static_cast<int32_t>(merged)});
  }

  added_.resize(num_added);
  for (uint32_t i = 0; i < num_added; ++i) {
    need(2);
    const uint16_t len = read_u16(p); p += 2;
    need(len + 8u);
    added_[i].content.assign(p, len); p += len;
    added_[i].id = static_cast<int32_t>(read_u32(p)); p += 4;
    const uint32_t flags = read_u32(p); p += 4;
    added_[i].lstrip = (flags & 1u) != 0;
    if (flags & ~1u)
      throw std::runtime_error(
          "tokenizer_bbpe: added-token flag bit set that this build does not "
          "implement (only lstrip = bit 0). The table was written by a newer "
          "generator; rebuild the runtime rather than ignoring the bit.");
    added_max_bytes_ = std::max(added_max_bytes_, added_[i].content.size());
  }
  // Longest first, so a linear scan at each position is leftmost-LONGEST
  // without a second pass. HuggingFace's matcher is an Aho-Corasick; with
  // ~100 patterns the difference is not measurable and the ordering is the
  // part that has to be right.
  std::sort(added_.begin(), added_.end(),
            [](const Added &a, const Added &b) {
              if (a.content.size() != b.content.size())
                return a.content.size() > b.content.size();
              return a.content < b.content;
            });

  if (id_to_token_.empty())
    throw std::runtime_error("tokenizer_bbpe: empty vocabulary");
}

BbpeTokenizer BbpeTokenizer::from_table_file(const std::string &path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("tokenizer_bbpe: cannot open " + path);
  std::stringstream ss;
  ss << f.rdbuf();
  BbpeTokenizer t;
  t.build_index(ss.str());
  return t;
}

BbpeTokenizer BbpeTokenizer::from_table_bytes(const char *data, size_t bytes) {
  BbpeTokenizer t;
  t.build_index(std::string(data, bytes));
  return t;
}

int32_t BbpeTokenizer::id_of(const std::string &token) const {
  auto it = token_to_id_.find(token);
  return it == token_to_id_.end() ? unk_id : it->second;
}

const std::string &BbpeTokenizer::token_of(int32_t id) const {
  static const std::string kEmpty;
  if (id < 0 || static_cast<size_t>(id) >= id_to_token_.size()) return kEmpty;
  return id_to_token_[static_cast<size_t>(id)];
}

// --- BPE over one pre-tokenized, byte-mapped word ------------------------

void BbpeTokenizer::bpe_word(const std::string &mapped,
                             std::vector<int32_t> &out) const {
  const std::vector<uint32_t> cps = utf8_decode(mapped);
  std::vector<int32_t> symbols;
  symbols.reserve(cps.size());
  for (uint32_t cp : cps) {
    std::string ch;
    utf8_append(ch, cp);
    auto it = token_to_id_.find(ch);
    if (it == token_to_id_.end()) {
      // Unreachable for valid UTF-8 input: the generator verifies that every
      // byte which can occur in UTF-8 has a vocabulary entry, and the decoder
      // above guarantees the input is valid. Dropped rather than guessed --
      // there is no <unk> in this family to substitute.
      continue;
    }
    symbols.push_back(it->second);
  }
  if (symbols.empty()) return;

  struct Node { int32_t id; int prev, next; bool alive; };
  std::vector<Node> nodes(symbols.size());
  for (size_t i = 0; i < symbols.size(); ++i) {
    nodes[i].id = symbols[i];
    nodes[i].prev = static_cast<int>(i) - 1;
    nodes[i].next = (i + 1 < symbols.size()) ? static_cast<int>(i + 1) : -1;
    nodes[i].alive = true;
  }

  struct Candidate {
    uint32_t rank;
    int left, right;
    int32_t left_id, right_id;
    int32_t merged_id;
  };
  struct Cmp {
    bool operator()(const Candidate &a, const Candidate &b) const {
      if (a.rank != b.rank) return a.rank > b.rank;
      return a.left > b.left;
    }
  };
  std::priority_queue<Candidate, std::vector<Candidate>, Cmp> heap;

  auto try_queue = [&](int left, int right) {
    if (left < 0 || right < 0) return;
    auto it = merge_of_.find(pair_key(nodes[left].id, nodes[right].id));
    if (it == merge_of_.end()) return;
    heap.push(Candidate{it->second.rank, left, right, nodes[left].id,
                        nodes[right].id, it->second.merged_id});
  };

  for (size_t i = 0; i + 1 < nodes.size(); ++i)
    try_queue(static_cast<int>(i), static_cast<int>(i) + 1);

  while (!heap.empty()) {
    Candidate c = heap.top();
    heap.pop();
    if (!nodes[c.left].alive || !nodes[c.right].alive) continue;
    if (nodes[c.left].id != c.left_id || nodes[c.right].id != c.right_id) continue;
    if (nodes[c.left].next != c.right) continue;

    nodes[c.left].id = c.merged_id;
    nodes[c.right].alive = false;
    const int after = nodes[c.right].next;
    nodes[c.left].next = after;
    if (after >= 0) nodes[after].prev = c.left;

    try_queue(nodes[c.left].prev, c.left);
    try_queue(c.left, nodes[c.left].next);
  }

  for (int i = 0; i >= 0 && static_cast<size_t>(i) < nodes.size();
       i = nodes[i].next)
    if (nodes[i].alive) out.push_back(nodes[i].id);
}

// --- the whole pipeline --------------------------------------------------

std::vector<int32_t> BbpeTokenizer::tokenize(const std::string &text) const {
  std::vector<int32_t> out;

  // Stage 1: split on added tokens. Everything between matches is an
  // ordinary piece and goes through stages 2-5; a match contributes its id
  // and nothing else.
  auto run_ordinary = [&](const std::string &piece) {
    if (piece.empty()) return;

    // Stage 2: normalize.
    std::vector<uint32_t> cps = utf8_decode(piece);
    if (normalizer == 1) cps = nfc(cps);
    if (add_prefix_space && !cps.empty() && cps[0] != 0x20)
      cps.insert(cps.begin(), 0x20);

    // Stage 3: pre-tokenize with the GPT-2 scanner.
    const size_t n = cps.size();
    size_t i = 0;
    const ByteMap &bm = byte_map();
    std::string word, mapped;
    while (i < n) {
      size_t j = i;
      bool matched_contraction = false;
      for (const char *c : kContractions) {
        const size_t len = std::strlen(c);
        if (i + len > n) continue;
        bool eq = true;
        for (size_t k = 0; k < len; ++k)
          if (cps[i + k] != static_cast<uint32_t>(static_cast<unsigned char>(c[k]))) {
            eq = false;
            break;
          }
        if (eq) { j = i + len; matched_contraction = true; break; }
      }
      if (!matched_contraction) {
        j = i;
        // The regex's ` ?` is a literal U+0020, not `\s?`.
        if (cps[j] == 0x20 && j + 1 < n && char_class(cps[j + 1]) != bbpe_uni::kSpace)
          ++j;
        const uint8_t k = char_class(cps[j]);
        if (k != bbpe_uni::kSpace) {
          while (j < n && char_class(cps[j]) == k) ++j;
        } else {
          // `\s+(?!\S)` then `\s+`: hold the last space back for the word
          // that follows it, unless the run ends the text.
          while (j < n && char_class(cps[j]) == bbpe_uni::kSpace) ++j;
          if (j < n && j - 1 > i) --j;
        }
      }

      // Stage 4: this word's UTF-8 bytes -> printable codepoints.
      word.clear();
      for (size_t k = i; k < j; ++k) utf8_append(word, cps[k]);
      mapped.clear();
      for (char ch : word)
        utf8_append(mapped, bm.to_cp[static_cast<unsigned char>(ch)]);

      // Stage 5.
      bpe_word(mapped, out);
      i = j;
    }
  };

  // The added-token spans, left to right, leftmost-longest -- then LSTRIP,
  // which is why this is two phases and not one.
  //
  // An lstrip match extends its start left over whitespace, and that
  // extension OVERRIDES a match already accepted inside it. Measured:
  // "a  [MASK]  [MASK]  b" gives ['a', '  [MASK]', '  [MASK]', '  ', 'b'] --
  // the two-space runs before each [MASK] are swallowed while the one before
  // "b" survives as its own added token, and "a  |||IP_ADDRESS|||  b" (same
  // shape, no lstrip) keeps both. A single-pass matcher accepts the
  // whitespace token first and can never take it back, which is exactly the
  // extra token the first version emitted.
  struct Span { size_t start, stop; int32_t id; };
  std::vector<Span> spans;
  size_t pos = 0;
  while (pos < text.size()) {
    const Added *hit = nullptr;
    for (const Added &a : added_) {
      if (a.content.size() > text.size() - pos) continue;
      if (std::memcmp(text.data() + pos, a.content.data(), a.content.size()) == 0) {
        hit = &a;
        break;                       // added_ is sorted longest-first
      }
    }
    if (hit == nullptr) { ++pos; continue; }
    size_t start = pos;
    if (hit->lstrip) {
      // Walk back over whole codepoints while they are whitespace.
      while (start > 0) {
        size_t k = start;
        while (k > 0 && (static_cast<unsigned char>(text[k - 1]) & 0xC0) == 0x80) --k;
        if (k == 0) break;
        --k;
        const std::vector<uint32_t> prev = utf8_decode(text.substr(k, start - k));
        if (prev.size() != 1 || char_class(prev[0]) != bbpe_uni::kSpace) break;
        start = k;
      }
      while (!spans.empty() && spans.back().stop > start) spans.pop_back();
    }
    spans.push_back(Span{start, pos + hit->content.size(), hit->id});
    pos += hit->content.size();
  }

  size_t cursor = 0;
  for (const Span &sp : spans) {
    if (sp.start > cursor) run_ordinary(text.substr(cursor, sp.start - cursor));
    out.push_back(sp.id);
    cursor = sp.stop;
  }
  if (cursor < text.size()) run_ordinary(text.substr(cursor));
  return out;
}

BbpeEncoded BbpeTokenizer::encode(const std::string &text, int max_len) const {
  BbpeEncoded e;
  const std::vector<int32_t> body = tokenize(text);
  const int wrap = n_special();
  const int room = max_len - wrap;
  const int take = std::min<int>(static_cast<int>(body.size()),
                                 std::max(0, room));

  e.input_ids.reserve(max_len);
  for (int32_t id : prefix_ids_) e.input_ids.push_back(id);
  for (int i = 0; i < take; ++i) e.input_ids.push_back(body[i]);
  for (int32_t id : suffix_ids_) e.input_ids.push_back(id);
  e.n_tokens = static_cast<int32_t>(e.input_ids.size());
  e.n_tokens_full = static_cast<int32_t>(body.size()) + wrap;
  e.truncated = take < static_cast<int>(body.size());

  e.attention_mask.assign(e.input_ids.size(), 1);
  if (static_cast<int>(e.input_ids.size()) < max_len && pad_id < 0)
    throw std::runtime_error(
        "tokenizer_bbpe: this checkpoint's table records no padding token, "
        "so a sequence shorter than max_len cannot be padded. Refusing "
        "rather than inventing an id -- padded positions are masked out, so "
        "a wrong one stays invisible until something reads them.");
  while (static_cast<int>(e.input_ids.size()) < max_len) {
    e.input_ids.push_back(pad_id);
    e.attention_mask.push_back(0);
  }
  return e;
}

std::vector<BbpeEncoded> BbpeTokenizer::encode_batch(
    const std::vector<std::string> &texts, int max_len) const {
  std::vector<BbpeEncoded> out;
  out.reserve(texts.size());
  for (const auto &t : texts) out.push_back(encode(t, max_len));
  return out;
}

}  // namespace npue
