//===- tokenizer.hpp ----------------------------------------*- C++ -*-===//
//
// NpuEmbeddings -- BERT WordPiece tokenization, no Python, no ICU.
// SPDX-License-Identifier: MIT
//
// The last piece between this project and "text in, vector out" in one
// process. It reproduces HuggingFace's `BertTokenizer` for the uncased
// configuration this checkpoint ships:
//
//   do_lower_case true, strip_accents null (inherits true),
//   tokenize_chinese_chars true, model_max_length 512
//
// The Unicode facts it needs -- lowercase, NFD-with-Mn-dropped, and the
// punctuation / control / space categories -- are GENERATED from Python's
// unicodedata by tools/gen_tokenizer_tables.py, so they agree with the
// reference by construction rather than by careful reading.
//
// Correctness is not asserted, it is MEASURED: tools/verify_tokenizer.py runs
// this implementation and HuggingFace's over the same corpus and reports the
// exact per-sequence agreement rate.

#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace npue {

struct Encoded {
  std::vector<int32_t> input_ids;
  std::vector<int32_t> attention_mask;   // 1 for real tokens, 0 for padding
  std::vector<int32_t> token_type_ids;   // all zero for single sequences
  int32_t n_tokens = 0;                  // before padding, including [CLS]/[SEP]

  // WHAT THE TEXT ACTUALLY WAS, as opposed to what survived.
  //
  // `n_tokens` is capped at `max_len` by construction, so on a truncated input
  // it reads EXACTLY as an input that happened to fit -- which is why nothing
  // downstream could ever tell the two apart, and why `usage.prompt_tokens`
  // could only ever report the cut count. These two fields are the signal that
  // was missing: `n_tokens_full` is what `n_tokens` would have been with
  // unlimited room, and it is the number to put in an error message, because
  // it tells the caller how much they need rather than how much they lost.
  int32_t n_tokens_full = 0;
  bool truncated = false;
};

class Tokenizer {
public:
  // `vocab.txt`: one token per line, line number is the id -- the same file
  // HuggingFace reads.
  static Tokenizer from_vocab_file(const std::string &path);

  // The vocabulary as raw bytes (the contents of a vocab.txt), so it can come
  // from an .npue tensor instead of a loose file.
  static Tokenizer from_vocab_bytes(const char *data, size_t bytes);

  // One sequence, padded to `max_len`, truncated to fit [CLS] ... [SEP].
  Encoded encode(const std::string &text, int max_len) const;

  // A batch, each row padded to the same `max_len`. Rows are laid out
  // contiguously, which is exactly what the runtime's embedding gather wants.
  std::vector<Encoded> encode_batch(const std::vector<std::string> &texts,
                                    int max_len) const;

  // Token strings, before ids are looked up. Exposed because it is what makes
  // a tokenizer mismatch debuggable -- an id list tells you nothing.
  std::vector<std::string> tokenize(const std::string &text) const;

  size_t vocab_size() const { return id_of_.size(); }
  int32_t id_of(const std::string &token) const;
  const std::string &token_of(int32_t id) const;

  int32_t cls_id = -1, sep_id = -1, pad_id = -1, unk_id = -1;

private:
  void build_index(const std::string &blob);
  // The pipeline proper, on text already known to contain no special tokens.
  std::vector<std::string> tokenize_plain(const std::string &text) const;

  std::unordered_map<std::string, int32_t> id_of_;
  std::vector<std::string> tokens_;
  // Special tokens are matched LITERALLY in the raw text and never split.
  // Longest first, so "[MASK]" wins over a hypothetical "[MA".
  std::vector<std::string> specials_;
  int max_chars_per_word_ = 100;   // HuggingFace's default; longer -> [UNK]
};

}  // namespace npue
