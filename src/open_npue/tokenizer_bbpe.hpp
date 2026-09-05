//===- tokenizer_bbpe.hpp ------------------------------------*- C++ -*-===//
//
// NpuEmbeddings -- byte-level BPE (GPT-2 / OLMo / Qwen / tekken family).
// SPDX-License-Identifier: MIT
//
// The fourth tokenizer in this tree, and the one T43 named as the gate on
// every modern encoder:
//
//   WordPiece             tokenizer.hpp        arch 0, arch 2
//   SentencePiece BPE     tokenizer_gemma.hpp  arch 1
//   SentencePiece Unigram tokenizer_xlmr.hpp   arch 3
//   byte-level BPE        this                 -- ModernBERT, Qwen3, Ministral
//
// WHAT IS ACTUALLY NEW. The merge engine is the same one tokenizer_gemma.cpp
// runs -- linked list plus a lazily-invalidated min-heap on merge rank -- and
// it is deliberately the same code shape so the two can be read against each
// other. What is new is everything around it:
//
//   * an ADDED-TOKEN matcher (leftmost-longest, with lstrip), which the
//     SentencePiece tokenizers do not need,
//   * an NFC normalizer,
//   * the GPT-2 PRE-TOKENIZER -- a regex split into words, each of which is
//     BPE'd independently so no merge ever crosses a word boundary, and
//   * the BYTE-TO-UNICODE map, which replaces byte-fallback: every byte is
//     already a vocabulary entry, so there is no unknown piece and no <unk>.
//
// The pre-tokenizer's character classes come from bbpe_unicode_tables.hpp,
// which is MEASURED against HuggingFace's own splitter rather than computed
// from Python's unicodedata -- the two disagree on 4,386 codepoints, and the
// generator's docstring says exactly which. That is the difference between
// this file agreeing with HuggingFace and merely being written carefully.
//
// The vocabulary, merges, added tokens and post-processor wrapping are
// GENERATED offline by tools/gen_bbpe_tokenizer_table.py (and by its C++ port
// in bbpe_tokenizer_gen.hpp, so a fresh clone can pack without Python) into a
// flat `BBPETOK1` binary this class reads with ifstream + memcpy -- no JSON at
// runtime, per CLAUDE.md rule 5.
//
// Correctness is measured, not asserted: tools/verify_tokenizer_bbpe.py runs
// this implementation and HuggingFace's over the same corpus and requires
// exact agreement on every id.

#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace npue {

struct BbpeEncoded {
  std::vector<int32_t> input_ids;
  std::vector<int32_t> attention_mask;   // 1 for real tokens, 0 for padding
  int32_t n_tokens = 0;                  // before padding, incl. specials

  // The untruncated count, and whether anything was dropped -- the same pair
  // as npue::Encoded and GemmaEncoded carry, for the same reason: since
  // tasks/0110 an input that does not fit is an ERROR, and the caller cannot
  // raise it without knowing the real length.
  int32_t n_tokens_full = 0;
  bool truncated = false;
};

class BbpeTokenizer {
public:
  static BbpeTokenizer from_table_file(const std::string &path);
  static BbpeTokenizer from_table_bytes(const char *data, size_t bytes);

  // Ids for one text WITHOUT the post-processor's wrapping -- the added-token
  // split, normalization, pre-tokenization and BPE only. Exposed for
  // debugging and for tools/verify_tokenizer_bbpe.py, which compares against
  // HuggingFace's `add_special_tokens=False`.
  std::vector<int32_t> tokenize(const std::string &text) const;

  // The template's prefix ids + tokenize(text) + its suffix ids, padded or
  // truncated to `max_len`. Throws if the table records no padding token and
  // padding is actually needed -- a decoder checkpoint has none, and
  // inventing one is invisible until something reads the padded positions.
  BbpeEncoded encode(const std::string &text, int max_len) const;
  std::vector<BbpeEncoded> encode_batch(const std::vector<std::string> &texts,
                                        int max_len) const;

  size_t vocab_size() const { return id_to_token_.size(); }
  int32_t id_of(const std::string &token) const;
  const std::string &token_of(int32_t id) const;

  // How many ids the wrapping costs, so a caller can budget `max_len`.
  int32_t n_special() const {
    return static_cast<int32_t>(prefix_ids_.size() + suffix_ids_.size());
  }

  int32_t cls_id = -1, sep_id = -1, pad_id = -1, unk_id = -1, mask_id = -1;
  // 0 = none, 1 = NFC. Read from the table, never assumed -- both occur in
  // this family and the difference is silent.
  uint32_t normalizer = 0;
  bool add_prefix_space = false;

private:
  void build_index(const std::string &blob);

  // One pre-tokenized, byte-mapped word -> its ids, appended to `out`.
  void bpe_word(const std::string &mapped, std::vector<int32_t> &out) const;

  std::vector<std::string> id_to_token_;
  std::unordered_map<std::string, int32_t> token_to_id_;

  struct MergeInfo { uint32_t rank; int32_t merged_id; };
  std::unordered_map<uint64_t, MergeInfo> merge_of_;

  struct Added {
    std::string content;
    int32_t id;
    bool lstrip;
  };
  std::vector<Added> added_;
  size_t added_max_bytes_ = 0;

  std::vector<int32_t> prefix_ids_, suffix_ids_;
};

}  // namespace npue
