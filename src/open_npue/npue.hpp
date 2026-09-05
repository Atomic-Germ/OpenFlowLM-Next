//===- npue.hpp ---------------------------------------------*- C++ -*-===//
//
// NpuEmbeddings -- reader for the .npue runtime weight container.
// SPDX-License-Identifier: MIT
//
// Spec: docs/04-model/npue-format.md. The Python implementation in
// tools/npue.py is the reference this must agree with, and the round-trip is
// checked by tools/verify_npue.py on the writing side.
//
// The whole point of the format is that loading is mmap plus pointer
// arithmetic: no parsing of weight data, no dtype conversion, no copying. A
// tensor's bytes are handed to a DMA descriptor exactly as they sit on disk.
//
// Windows note: this uses CreateFileMappingW / MapViewOfFile rather than mmap,
// which is what docs/04-model specified.

#pragma once

#include <cstddef>
#include <cstdint>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace npue {

// An input that does not fit the sequence length this runtime is running at.
//
// A DISTINCT type rather than a plain runtime_error, because it is the
// CALLER's fault rather than the runtime's: the serve path maps it to HTTP
// 400, and a 500 here would tell an operator to go looking at the NPU.
//
// It exists at all because silent truncation is this project's most dangerous
// fail-open, and the only one whose blast radius is the ANSWER rather than the
// run. A truncated text still returns a correctly shaped, correctly normed,
// perfectly deterministic vector, so every downstream guard passes while the
// measurement is wrong -- the same shape as traps 6b/6c/7c/7d in CLAUDE.md,
// one layer further out. Documents sharing an administrative preamble are the
// worst case: cut them all at the same point and their vectors become
// byte-identical, so retrieval degrades to a coin flip wearing a similarity
// score rather than to visibly bad results.
//
// Report the value you read, never the intention: `n_tokens` is what the text
// ACTUALLY tokenized to, not what was kept.
struct InputTooLong : std::runtime_error {
  InputTooLong(size_t index, int64_t n_tokens, int64_t limit)
      : std::runtime_error(
            "input " + std::to_string(index) + " is " +
            std::to_string(n_tokens) + " tokens, but this runtime is running "
            "at sequence length " + std::to_string(limit) + ". Split the text "
            "into shorter pieces, or run a design exported for a longer "
            "sequence (tools/export_gemm_rtp.py --seq N). Passing "
            "--allow-truncation cuts it at " + std::to_string(limit) +
            " tokens instead and DISCARDS the rest."),
        index(index), n_tokens(n_tokens), limit(limit) {}
  size_t index;      // which input in the request/batch
  int64_t n_tokens;  // the untruncated token count, including [CLS]/[SEP]
  int64_t limit;     // the sequence length the design was compiled for
};

struct TensorInfo {
  std::string name;
  std::string role;      // gemm_b | bias | layernorm | embedding
  std::string dtype;     // BF16 | F32 | ...
  std::vector<int64_t> logical_shape;
  std::vector<int64_t> padded_shape;
  uint64_t offset = 0;   // relative to data_offset
  uint64_t nbytes = 0;
  std::string layout_hash;   // empty when the tensor is not tiled
};

// A view into the mapped file. Never owns, never copies.
struct Span {
  const void *data = nullptr;
  size_t bytes = 0;
  template <typename T> const T *as() const {
    return static_cast<const T *>(data);
  }
};

class File {
public:
  explicit File(const std::string &path);
  ~File();
  File(const File &) = delete;
  File &operator=(const File &) = delete;

  // Raw bytes exactly as stored: pre-tiled, bf16 as uint16. What DMA sees.
  Span raw(const std::string &name) const;
  const TensorInfo &info(const std::string &name) const;
  bool has(const std::string &name) const {
    return tensors_.find(name) != tensors_.end();
  }

  // Config values from the JSON directory. The runtime asserts on these rather
  // than assuming a build matches the file it was built against.
  int64_t config_int(const std::string &key) const;
  double config_double(const std::string &key) const;
  std::string config_string(const std::string &key) const;

  uint32_t version() const { return version_; }
  uint64_t data_length() const { return data_length_; }
  size_t tensor_count() const { return tensors_.size(); }

private:
  void *handle_file_ = nullptr;
  void *handle_map_ = nullptr;
  const uint8_t *base_ = nullptr;
  size_t size_ = 0;

  uint32_t version_ = 0;
  uint64_t data_offset_ = 0;
  uint64_t data_length_ = 0;

  std::map<std::string, TensorInfo> tensors_;
  std::map<std::string, std::string> config_;   // key -> raw JSON scalar text
};

}  // namespace npue
