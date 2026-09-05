//===- gemma_encode.hpp ---------------------------------------*- C++ -*-===//
//
// NpuEmbeddings -- EmbeddingGemma-300M (arch=1), CPU-only forward pass.
// SPDX-License-Identifier: MIT
//
// WHY THIS EXISTS AND WHY IT HAS NO NPU CODE AT ALL. There is no
// Gemma-specific NPU kernel or design yet -- see
// tasks/0064-m12-embeddinggemma-arch1-integration/TASK.md. Per this
// project's own established precedent for the BERT models ("eltwise lives
// on the host", CLAUDE.md's production architecture section), every op in
// this class runs on the HOST, including every GEMM (Q/K/V, attn-out,
// GeGLU's gate/up/down, the two post-pool Dense heads) -- there is simply
// nothing to offload to yet. This is deliberately a plain, reference-quality
// implementation (double-precision GEMM accumulation, no AVX2, no
// batching -- one sequence at a time): correctness against
// reference/encoder_gemma.py is the point of this file, not speed. A future
// session can profile this and decide whether host-side AVX2/threading (the
// BERT runtime's own precedent) is worth it before ever touching the NPU.
//
// Built from three already-verified, standalone pieces, wired together for
// the first time here:
//   - tokenizer_gemma.hpp/.cpp (tasks/0061) -- SentencePiece BPE, 1,925/1,925
//     byte-identical to HuggingFace.
//   - gemma_kernels.hpp/.cpp (tasks/0063) -- RMSNorm, RoPE, GeGLU, 36/36
//     records PASS against real tapped intermediates.
//   - npue.hpp/.cpp (M7) -- the .npue reader, unmodified; this class only
//     reads tensors by name via File::raw()/config_*(), the same generic
//     interface the BERT path uses.
// The architecture itself (4-RMSNorm sandwich, MQA + q_norm/k_norm + RoPE,
// GeGLU, mean-pool + 2x Dense + L2-normalize) is read from
// reference/encoder_gemma.py (tasks/0055, 1-cos 1.065e-07 against real
// HuggingFace) -- see that file's header for the primary-source citations.

#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "npue.hpp"
#include "tokenizer_gemma.hpp"

namespace npue {

class GemmaEncoder {
public:
  explicit GemmaEncoder(File &model);

  // Tokenize -> embed (xsqrt(hidden)) -> 24x[4 RMSNorm + MQA/RoPE attn +
  // GeGLU] -> final RMSNorm -> masked mean pool (include_prompt=true) ->
  // Dense(hidden->dense_hidden) -> Dense(dense_hidden->hidden) ->
  // L2-normalize. Returns `hidden` floats. One text at a time: attention
  // never mixes across sequences anyway (each row's mask only touches its
  // own padding), so batch=1 gives bit-for-bit the same per-text result a
  // batched implementation would, at the cost of not sharing GEMM calls
  // across texts -- an acceptable trade for a correctness-first first cut.
  //
  // `index` and `allow_truncation` exist only for the error this throws when
  // the text does not fit `max_len` (npue::InputTooLong). The policy is an
  // ARGUMENT rather than a global because this class is a library type that
  // the CLI, the verify tools and main() all construct independently; a
  // global read from here would bind it to one of those three. Default is to
  // refuse, so a caller that has not thought about it gets the safe answer.
  std::vector<float> encode_one(
      const std::string &text, int max_len,
      const std::string &prefix_name = GemmaTokenizer::default_prefix_name(),
      size_t index = 0, bool allow_truncation = false) const;

  GemmaTokenizer tok;

  int64_t hidden() const { return hidden_; }

private:
  File &model_;
  int64_t hidden_ = 0, heads_ = 0, kv_heads_ = 0, head_dim_ = 0, inter_ = 0,
          layers_ = 0, dense_hidden_ = 0, sliding_window_pattern_ = 6;
  double eps_ = 1e-6, rope_theta_ = 1e6, rope_theta_local_ = 1e4,
         query_pre_attn_scalar_ = 256.0;

  const float *w_embed_ = nullptr;
  const float *w_norm_ = nullptr;
  const float *w_dense2_ = nullptr;   // [hidden, dense_hidden]
  const float *w_dense3_ = nullptr;   // [dense_hidden, hidden]

  struct LayerPtrs {
    const float *q_proj, *k_proj, *v_proj, *o_proj;
    const float *q_norm, *k_norm;
    const float *ln_in, *ln_post_attn, *ln_pre_ffn, *ln_post_ffn;
    const float *gate_proj, *up_proj, *down_proj;
  };
  std::vector<LayerPtrs> lp_;
};

}  // namespace npue
