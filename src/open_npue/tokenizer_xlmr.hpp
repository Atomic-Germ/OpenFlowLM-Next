//===- tokenizer_xlmr.hpp ------------------------------------*- C++ -*-===//
//
// NpuEmbeddings -- XLM-R family SentencePiece **Unigram** tokenizer
// (gte-multilingual-base and, with it, the XLM-R / multilingual-E5 / mGTE
// family).
// SPDX-License-Identifier: MIT
//
// NOT WordPiece (tokenizer.hpp) and NOT SentencePiece BPE
// (tokenizer_gemma.hpp): this checkpoint's tokenizer.json declares
// model.type == "Unigram" (confirmed by reading it, tasks/0127, per T29's
// read-before-writing-code lesson) -- a Viterbi search over per-piece f64
// log-probs, behind a Precompiled-charsmap normalizer (a Darts double-array
// trie), WhitespaceSplit + Metaspace pre-tokenization, and <s>...</s>
// wrapping. The third tokenizer family in this repo (T52).
//
// THE EXECUTABLE SPEC IS tools/xlmr_tokenizer_ref.py. This class is its
// line-for-line C++ port -- same method boundaries (normalize /
// pre_tokenize / viterbi / encode), same quirks, so the two can be diffed
// function by function (the gemma two-implementation discipline). Where
// HuggingFace deviates from upstream sentencepiece, HuggingFace wins,
// because that is what the golden embeddings were produced with. The
// pipeline facts, all probed live in tasks/0127, none inferred from docs:
//
//   1. Precompiled normalizer: iterate UAX #29 extended grapheme clusters;
//      a cluster < 6 UTF-8 bytes whose PREFIX hits the charsmap trie is
//      replaced WHOLE by the first (shortest) match -- HF's own quirk
//      (precompiled.rs: "Yes, this seems broken"), mirrored deliberately.
//      Otherwise each char is transformed independently.
//   2. WhitespaceSplit on the Rust White_Space set -- NOT isspace(), and
//      NOT Python str.isspace() (which also claims U+001C..U+001F).
//   3. Metaspace(U+2581, prepend_scheme from the table, split=true).
//   4. Unigram Viterbi, f64 sums end to end (65,856 of 250,002 scores do
//      not survive f32 -- storing or accumulating float breaks near-tie
//      byte-exactness), unk node = min_score - 10.0 covering exactly one
//      char where no single-char piece matches, consecutive unks fused.
//   5. Post-processor: <s> ids </s>.
//
// KNOWN LIMITATION (deliberate, shared with the reference, recorded in
// 0127): no added-tokens splitter -- a literal special-token string in the
// input goes through the Unigram model instead of being extracted
// verbatim. Embedding inputs are plain text, never templated.
//
// The table is GENERATED offline by tools/gen_xlmr_tokenizer_table.py from
// the checkpoint's tokenizer.json (17 MB) into a flat XLMRTOK1 binary
// (5.3 MB: charsmap trie + normalized-strings blob stored VERBATIM and
// walked in place, f64 scores, length-prefixed pieces) -- no JSON at
// runtime, CLAUDE.md rule 5. The C++ generator port is
// xlmr_tokenizer_gen.hpp (fresh clones pack without Python).
//
// STANDALONE ON PURPOSE: no XRT dependency, not wired into main.cpp's
// Encoder or hub.cpp's catalogue. The arch-3 integration is a later task.
//
// Correctness is measured, not asserted: tools/verify_tokenizer_xlmr.py
// holds this implementation (via tokenizer_xlmr_cli.exe), the Python
// reference and HuggingFace to byte-exact agreement over a 343-sequence
// multilingual adversarial corpus.

#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace npue {

class XlmrTokenizer {
public:
  // The binary table tools/gen_xlmr_tokenizer_table.py writes (default
  // path models/gte-multilingual-base/xlmr_tokenizer.bin). from_table_bytes
  // exists so the table can later come out of a .npue container tensor.
  static XlmrTokenizer from_table_file(const std::string &path);
  static XlmrTokenizer from_table_bytes(const char *data, size_t bytes);

  // Pipeline stages, public so tools and tests can diff them against the
  // reference stage by stage. All strings are UTF-8.
  std::string normalize(const std::string &text) const;
  std::vector<std::string> pre_tokenize(const std::string &normalized) const;
  std::vector<int32_t> viterbi(const std::string &pretoken) const;

  // Full pipeline: normalize -> pre-tokenize -> Viterbi per pre-token ->
  // <s> ids </s>. No padding, no truncation (0110: an input that does not
  // fit is the CALLER's error to raise, not this class's to hide).
  std::vector<int32_t> encode(const std::string &text) const;

  size_t vocab_size() const { return pieces_.size(); }
  const std::string &piece_of(int32_t id) const;

  int32_t bos_id = -1, eos_id = -1, pad_id = -1, unk_id = -1, mask_id = -1;

private:
  void build_index(const std::string &blob);

  // Darts double-array common-prefix search over the charsmap trie, walked
  // in place exactly as the reference walks it (darts_clone bit layout).
  // Returns the FIRST match's value or -1 -- the normalizer only ever uses
  // results[0], so the shortest match is the only one collected.
  int64_t trie_first_match(const char *key, size_t len) const;

  // The Precompiled transform for one chunk (a whole grapheme or one
  // char): nullptr-equivalent is signalled by `found = false`.
  bool transform(const char *chunk, size_t len, std::string &out) const;

  std::vector<uint32_t> trie_units_;   // charsmap trie, u32 LE units
  std::string normalized_blob_;        // NUL-terminated replacement strings

  std::vector<double> scores_;         // f64, id order
  std::vector<std::string> pieces_;    // UTF-8, id order
  std::unordered_map<std::string, int32_t> piece_to_id_;

  double min_score_ = 0.0, unk_score_ = 0.0;
  uint32_t metaspace_cp_ = 0;          // U+2581
  uint32_t prepend_scheme_ = 0;        // 0=never 1=first 2=always
  uint32_t metaspace_split_ = 0;
  uint32_t max_piece_chars_ = 0;
  uint32_t add_bos_ = 0, add_eos_ = 0;
};

}  // namespace npue
