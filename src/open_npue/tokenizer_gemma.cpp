//===- tokenizer_gemma.cpp -----------------------------------*- C++ -*-===//
//
// NpuEmbeddings -- EmbeddingGemma-300M SentencePiece BPE. See
// tokenizer_gemma.hpp for the pipeline summary and the design rationale.
// SPDX-License-Identifier: MIT
//
// THE MERGE ALGORITHM
// --------------------
// Standard BPE: a word (here, the whole normalized+prefixed text -- this
// checkpoint's pre_tokenizer does not split on word boundaries, confirmed in
// tools/gen_gemma_tokenizer_table.py's docstring) starts as one symbol per
// Unicode codepoint. Repeatedly find the adjacent pair with the LOWEST merge
// rank (earliest-trained merge = highest priority) and replace it with its
// merged symbol, until no adjacent pair has a merge rule. This is the
// classic "doubly-linked-list + lazy-invalidation min-heap" implementation:
// a candidate merge is pushed with the (id, generation) of both symbols it
// was computed against, and validated again when popped, because earlier
// merges may have consumed one of its endpoints in the meantime.
//
// BYTE FALLBACK
// -------------
// A codepoint whose own UTF-8 string is not itself a vocabulary entry is
// decomposed into its raw UTF-8 bytes, each becoming its own "<0xXX>"
// symbol (uppercase hex, e.g. "<0xF0>") -- and then those byte symbols enter
// the SAME merge loop as everything else, using the SAME merge table. In
// practice they almost never re-merge (no training data pairs a rare
// codepoint's raw bytes together often enough to earn a merge rule), but
// nothing in this implementation assumes that -- it is not special-cased.

#include "tokenizer_gemma.hpp"

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

// --- UTF-8, same conventions as tokenizer.cpp's -----------------------
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

// Metaspace: replace every literal ' ' (0x20) byte with the 3-byte '▁'
// (U+2581 = E2 96 81). Byte-level is safe: 0x20 never occurs as a UTF-8
// continuation byte (those are 0x80-0xBF), so this is exactly equivalent to
// decode -> per-codepoint replace -> re-encode, and much cheaper. This is
// the checkpoint's ENTIRE normalizer -- confirmed empirically (no NFKC, no
// case folding) in tools/gen_gemma_tokenizer_table.py's docstring.
std::string metaspace(const std::string &text) {
  std::string out;
  out.reserve(text.size() + text.size() / 4);
  for (char c : text) {
    if (c == ' ') { out.push_back('\xE2'); out.push_back('\x96'); out.push_back('\x81'); }
    else out.push_back(c);
  }
  return out;
}

uint64_t pair_key(int32_t a, int32_t b) {
  return (static_cast<uint64_t>(static_cast<uint32_t>(a)) << 32) |
         static_cast<uint32_t>(b);
}

}  // namespace

// --- table loading ----------------------------------------------------

void GemmaTokenizer::build_index(const std::string &blob) {
  const char *p = blob.data();
  const char *end = p + blob.size();
  auto need = [&](size_t n) {
    if (static_cast<size_t>(end - p) < n)
      throw std::runtime_error("tokenizer_gemma: truncated table");
  };

  need(8);
  if (std::memcmp(p, "GEMATOK1", 8) != 0)
    throw std::runtime_error("tokenizer_gemma: bad magic");
  p += 8;

  need(4); const uint32_t version = read_u32(p); p += 4;
  if (version != 1)
    throw std::runtime_error("tokenizer_gemma: unsupported table version " +
                             std::to_string(version));

  need(4); const uint32_t vocab_size = read_u32(p); p += 4;
  need(4); const uint32_t num_merges = read_u32(p); p += 4;

  need(20);
  pad_id = static_cast<int32_t>(read_u32(p)); p += 4;
  eos_id = static_cast<int32_t>(read_u32(p)); p += 4;
  bos_id = static_cast<int32_t>(read_u32(p)); p += 4;
  unk_id = static_cast<int32_t>(read_u32(p)); p += 4;
  mask_id = static_cast<int32_t>(read_u32(p)); p += 4;

  need(8);
  const uint32_t add_bos = read_u32(p); p += 4;
  const uint32_t add_eos = read_u32(p); p += 4;
  if (!add_bos || !add_eos)
    throw std::runtime_error("tokenizer_gemma: table built without bos/eos "
                             "wrapping, this class assumes both");

  need(8);
  const uint32_t num_prefixes = read_u32(p); p += 4;
  default_prefix_index_ = static_cast<int32_t>(read_u32(p)); p += 4;

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

  prefixes_.resize(num_prefixes);
  for (uint32_t i = 0; i < num_prefixes; ++i) {
    need(2);
    uint16_t nlen = read_u16(p); p += 2;
    need(nlen);
    prefixes_[i].name.assign(p, nlen); p += nlen;
    need(2);
    uint16_t plen = read_u16(p); p += 2;
    need(plen);
    prefixes_[i].text.assign(p, plen); p += plen;
    prefix_index_.emplace(prefixes_[i].name, i);
  }

  if (id_to_token_.empty())
    throw std::runtime_error("tokenizer_gemma: empty vocabulary");
  if (bos_id < 0 || eos_id < 0 || pad_id < 0 || unk_id < 0)
    throw std::runtime_error("tokenizer_gemma: missing a required special id");
  if (default_prefix_index_ < 0 ||
      static_cast<size_t>(default_prefix_index_) >= prefixes_.size())
    throw std::runtime_error("tokenizer_gemma: bad default prefix index");
}

GemmaTokenizer GemmaTokenizer::from_table_file(const std::string &path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("tokenizer_gemma: cannot open " + path);
  std::stringstream ss;
  ss << f.rdbuf();
  GemmaTokenizer t;
  t.build_index(ss.str());
  return t;
}

GemmaTokenizer GemmaTokenizer::from_table_bytes(const char *data, size_t bytes) {
  GemmaTokenizer t;
  t.build_index(std::string(data, bytes));
  return t;
}

int32_t GemmaTokenizer::id_of(const std::string &token) const {
  auto it = token_to_id_.find(token);
  return it == token_to_id_.end() ? unk_id : it->second;
}

const std::string &GemmaTokenizer::token_of(int32_t id) const {
  static const std::string kEmpty;
  if (id < 0 || static_cast<size_t>(id) >= id_to_token_.size()) return kEmpty;
  return id_to_token_[static_cast<size_t>(id)];
}

const std::string &GemmaTokenizer::prefix_text(const std::string &name) const {
  auto it = prefix_index_.find(name);
  if (it == prefix_index_.end())
    throw std::runtime_error("tokenizer_gemma: no task prefix named " + name);
  return prefixes_[it->second].text;
}

const std::string &GemmaTokenizer::default_prefix() const {
  return prefixes_[static_cast<size_t>(default_prefix_index_)].text;
}

std::vector<std::string> GemmaTokenizer::prefix_names() const {
  std::vector<std::string> out;
  out.reserve(prefixes_.size());
  for (const auto &pr : prefixes_) out.push_back(pr.name);
  return out;
}

int32_t GemmaTokenizer::byte_fallback_id(uint8_t byte) const {
  static const char kHex[] = "0123456789ABCDEF";
  char buf[7] = {'<', '0', 'x', kHex[byte >> 4], kHex[byte & 0xF], '>', '\0'};
  auto it = token_to_id_.find(std::string(buf, 6));
  return it == token_to_id_.end() ? -1 : it->second;
}

// --- BPE ----------------------------------------------------------------

std::vector<int32_t> GemmaTokenizer::tokenize(const std::string &text) const {
  const std::string normalized = metaspace(text);
  const std::vector<uint32_t> cps = utf8_decode(normalized);

  // Initial symbolization: one symbol per codepoint, falling back to raw
  // UTF-8 bytes for any codepoint whose own string is not a vocab entry.
  std::vector<int32_t> symbols;
  symbols.reserve(cps.size());
  for (uint32_t cp : cps) {
    std::string ch;
    utf8_append(ch, cp);
    auto it = token_to_id_.find(ch);
    if (it != token_to_id_.end()) {
      symbols.push_back(it->second);
      continue;
    }
    // byte_fallback: decompose this ONE codepoint's UTF-8 bytes.
    bool any_fallback = false;
    for (unsigned char b : ch) {
      const int32_t bid = byte_fallback_id(b);
      if (bid < 0) {
        // Should be unreachable -- all 256 <0xXX> entries exist in this
        // checkpoint's vocabulary (verified by the generator). fuse_unk:
        // collapse into the previous symbol if it too is <unk>.
        if (!symbols.empty() && symbols.back() == unk_id) continue;
        symbols.push_back(unk_id);
        any_fallback = true;
        continue;
      }
      symbols.push_back(bid);
      any_fallback = true;
    }
    (void)any_fallback;
  }
  if (symbols.empty()) return symbols;

  // Doubly linked list over `symbols`, mutated in place by merges.
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
    int32_t left_id, right_id;   // ids at the time this candidate was queued
    int32_t merged_id;
  };
  struct Cmp {
    bool operator()(const Candidate &a, const Candidate &b) const {
      // Min-heap on rank: lowest rank (earliest-trained merge) pops first.
      // Break ties on position for determinism (matches left-to-right scan
      // order, though the trained merge table should not actually collide).
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
    if (nodes[c.left].next != c.right) continue;  // no longer adjacent

    nodes[c.left].id = c.merged_id;
    nodes[c.right].alive = false;
    const int after = nodes[c.right].next;
    nodes[c.left].next = after;
    if (after >= 0) nodes[after].prev = c.left;

    try_queue(nodes[c.left].prev, c.left);
    try_queue(c.left, nodes[c.left].next);
  }

  std::vector<int32_t> out;
  out.reserve(nodes.size());
  for (int i = 0; i >= 0 && static_cast<size_t>(i) < nodes.size();
       i = nodes[i].next) {
    if (nodes[i].alive) out.push_back(nodes[i].id);
  }
  return out;
}

GemmaEncoded GemmaTokenizer::encode(const std::string &text, int max_len,
                                   const std::string &prefix_name) const {
  GemmaEncoded e;
  std::string full;
  if (!prefix_name.empty()) full = prefix_text(prefix_name);
  full += text;

  const std::vector<int32_t> body = tokenize(full);
  const int room = max_len - 2;                       // <bos> ... <eos>
  const int take = std::min<int>(static_cast<int>(body.size()),
                                 std::max(0, room));

  e.input_ids.reserve(max_len);
  e.input_ids.push_back(bos_id);
  for (int i = 0; i < take; ++i) e.input_ids.push_back(body[i]);
  e.input_ids.push_back(eos_id);
  e.n_tokens = static_cast<int32_t>(e.input_ids.size());
  // `body` is the prefix AND the text, so this counts what actually competed
  // for the sequence budget.
  e.n_tokens_full = static_cast<int32_t>(body.size()) + 2;   // + <bos>/<eos>
  e.truncated = take < static_cast<int>(body.size());

  e.attention_mask.assign(e.input_ids.size(), 1);
  while (static_cast<int>(e.input_ids.size()) < max_len) {
    e.input_ids.push_back(pad_id);
    e.attention_mask.push_back(0);
  }
  return e;
}

std::vector<GemmaEncoded> GemmaTokenizer::encode_batch(
    const std::vector<std::string> &texts, int max_len,
    const std::string &prefix_name) const {
  std::vector<GemmaEncoded> out;
  out.reserve(texts.size());
  for (const auto &t : texts) out.push_back(encode(t, max_len, prefix_name));
  return out;
}

}  // namespace npue
