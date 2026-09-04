//===- tokenizer_gemma.hpp -----------------------------------*- C++ -*-===//
//
// NpuEmbeddings -- EmbeddingGemma-300M SentencePiece BPE tokenizer.
// SPDX-License-Identifier: MIT
//
// NOT WordPiece (see tokenizer.hpp for that). This checkpoint's own
// tokenizer.json declares model.type == "BPE" (confirmed by reading it, not
// assumed from the model family) -- SentencePiece-style BPE with a metaspace
// normalizer and byte-fallback, 262,144-entry vocabulary, <bos>/<eos>
// wrapping instead of [CLS]/[SEP]. See tools/gen_gemma_tokenizer_table.py's
// module docstring for the full pipeline, confirmed empirically against the
// live `transformers` tokenizer.
//
// The vocabulary + merge-rank table is GENERATED offline by
// tools/gen_gemma_tokenizer_table.py from the checkpoint's tokenizer.json
// (33 MB of JSON, 262k vocab entries, 515k merges) into a flat binary this
// class reads with ifstream + memcpy -- no JSON, no protobuf, at runtime,
// per CLAUDE.md rule 5.
//
// STANDALONE ON PURPOSE: this class has no XRT dependency and is not wired
// into main.cpp's Encoder::run() or hub.cpp's model catalogue. That
// integration (arch=1, RMSNorm/RoPE/GeGLU, packer changes) is separate,
// larger, and out of scope for the tokenizer alone -- see
// tasks/00XX-gemma-tokenizer/TASK.md.
//
// Correctness is measured, not asserted: tools/verify_tokenizer_gemma.py
// runs this implementation and HuggingFace's over the same corpus (with the
// same <bos>/task-prefix/text/<eos> wrapping) and reports exact agreement.

#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace npue {

struct GemmaEncoded {
  std::vector<int32_t> input_ids;
  std::vector<int32_t> attention_mask;   // 1 for real tokens, 0 for padding
  int32_t n_tokens = 0;                  // before padding, incl. <bos>/<eos>

  // The untruncated count, and whether anything was dropped. Same reasoning
  // as npue::Encoded's pair -- see tokenizer.hpp. Note that for this
  // tokenizer the count INCLUDES the task prefix, because the prefix is
  // prepended before tokenization and therefore spends real sequence budget:
  // a caller sizing its inputs against `max_len` without accounting for the
  // prefix would be over by however many tokens "search_document: " costs.
  int32_t n_tokens_full = 0;
  bool truncated = false;
};

class GemmaTokenizer {
public:
  // The binary table tools/gen_gemma_tokenizer_table.py writes (default path
  // models/embeddinggemma-300m/gemma_tokenizer.bin).
  static GemmaTokenizer from_table_file(const std::string &path);
  static GemmaTokenizer from_table_bytes(const char *data, size_t bytes);

  // Token ids for one piece of already-normalized+prefixed text, WITHOUT
  // <bos>/<eos> -- the BPE segmentation only. Exposed for debugging (an id
  // list alone does not tell you what went wrong) and for tools/verify_*.
  std::vector<int32_t> tokenize(const std::string &text) const;

  // <bos> + prefix(prefix_name) + text + <eos>, padded/truncated to
  // max_len. `prefix_name` selects a row of the task-prefix table baked into
  // the binary (e.g. "document", "query", "STS", ...); pass "" for no
  // prefix at all (raw <bos> + text + <eos>, sentence-transformers' own
  // behaviour when no prompt is named).
  GemmaEncoded encode(const std::string &text, int max_len,
                     const std::string &prefix_name = default_prefix_name()) const;

  std::vector<GemmaEncoded> encode_batch(
      const std::vector<std::string> &texts, int max_len,
      const std::string &prefix_name = default_prefix_name()) const;

  // The prefix text for a named task, e.g. prefix_text("document") ->
  // "title: none | text: ". Throws if the name is not in the table.
  const std::string &prefix_text(const std::string &name) const;
  // This project's own default -- baked into the table by the generator,
  // NOT the checkpoint's own `default_prompt_name` (which is null, i.e.
  // sentence-transformers applies no prefix unless one is named). See
  // tools/gen_gemma_tokenizer_table.py's docstring for the reasoning.
  const std::string &default_prefix() const;
  static const std::string &default_prefix_name() {
    static const std::string s = "document";
    return s;
  }
  std::vector<std::string> prefix_names() const;

  size_t vocab_size() const { return id_to_token_.size(); }
  int32_t id_of(const std::string &token) const;
  const std::string &token_of(int32_t id) const;

  int32_t bos_id = -1, eos_id = -1, pad_id = -1, unk_id = -1, mask_id = -1;

private:
  void build_index(const std::string &blob);

  std::vector<std::string> id_to_token_;
  std::unordered_map<std::string, int32_t> token_to_id_;

  // merge lookup: key = (uint64_t)id_a << 32 | id_b, value = (rank, merged_id)
  struct MergeInfo { uint32_t rank; int32_t merged_id; };
  std::unordered_map<uint64_t, MergeInfo> merge_of_;

  struct Prefix { std::string name, text; };
  std::vector<Prefix> prefixes_;
  int32_t default_prefix_index_ = -1;
  std::unordered_map<std::string, size_t> prefix_index_;

  int32_t byte_fallback_id(uint8_t byte) const;
};

}  // namespace npue
