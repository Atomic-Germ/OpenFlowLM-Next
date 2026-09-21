//===- decoder.hpp -------------------------------------------*- C++ -*-===//
//
// open_whisper -- the Whisper-large-v3-turbo DECODER, entirely on the host in
// fp32 (phase 3, issue #72). 4 layers, d_model 1280, 20 heads x 64, FFN 5120,
// max 448 positions, vocab 51866 (tied to embed_tokens -- there is no lm_head
// tensor). Cross-attention reads the encoder's fixed K/V (Encoder::xkv()) and
// never recomputes or appends to it; self-attention keeps its own growing KV
// cache, which clear_context() resets between generations over one window.
//
// Weights are read straight off the container's own [out, in] bf16 layout
// (utilities/q4nx-build/q4nx/open_whisper.py's whisper_tensors(), the
// "Decoder: transformers' names..." branch) and kept bf16 in memory, widened
// to fp32 on the fly per dot product -- see decoder.cpp's dot_bf16(), ported
// with attribution from open_qwen36/vision/vit.cpp's linear()/widen_avx2. The
// alternative (widen everything once at load) would double the resident
// weight size for ~158M parameters (~316 MB bf16 vs ~632 MB fp32), and the
// tied embed_tokens matrix -- the single largest sweep in a decode step,
// 51866 x 1280 dot products against the tied lm_head -- is read in full
// exactly once per token either way, so keeping it bf16 halves the bytes
// actually moved from RAM for the one operation that touches the whole
// vocabulary every step.
// SPDX-License-Identifier: MIT
//
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace ow {

// y = x . W^T + b, W kept as bf16 [out, in] (the container's own layout, not
// tiled -- the decoder never touches the NPU, so there is no kernel-set tile
// tuple to tile for). An absent bias (self-attention's k_proj, and the tied
// lm_head used as a projection) is stored as an explicit all-zero vector, so
// linear() in decoder.cpp never needs a separate no-bias code path.
struct Linear {
  std::vector<uint16_t> w;   // [out, in] bf16
  std::vector<float> b;      // [out]
  int64_t out = 0, in = 0;
};

struct DecoderLayerWeights {
  Linear self_q, self_k, self_v, self_out;              // self_k has no bias
  std::vector<float> ln_self_w, ln_self_b;
  Linear cross_q, cross_out;                            // cross K/V are NOT here -- see xkv()
  std::vector<float> ln_cross_w, ln_cross_b;
  Linear fc1, fc2;
  std::vector<float> ln_final_w, ln_final_b;
};

// The only geometry this build implements -- whisper-large-v3-turbo's
// decoder. Checked against config.json in Decoder's constructor before a
// single tensor is read, the same discipline weights.hpp's Geometry uses for
// the encoder: a wrong geometry loaded anyway would index into the wrong
// widths and return plausible, wrong logits (this project's "fails open"
// class), not an error.
struct DecoderGeometry {
  static constexpr int64_t d_model = 1280;
  static constexpr int64_t n_layers = 4;
  static constexpr int64_t n_heads = 20;
  static constexpr int64_t head_dim = 64;      // d_model / n_heads
  static constexpr int64_t ffn = 5120;
  static constexpr int64_t vocab = 51866;
  // The host sampler's Whisper_Config width (oflm's own struct, not a choice
  // made here) -- Decoder::step() always writes this many logits, with the
  // tail set to -inf so the pad can never win an argmax or a sample.
  static constexpr int64_t vocab_padded = 51872;
  static constexpr int64_t max_target_positions = 448;
};

// Host wall-clock only (the same rule as encoder.hpp's Timers -- never an NPU
// performance claim; the decoder never dispatches to the NPU at all, so there
// is no npu_* split here).
struct DecoderTimers {
  double embed = 0, layer_norm = 0, linear = 0, attention = 0, gelu = 0;
  double total = 0;
  int64_t steps = 0;
};

class Decoder {
public:
  // Throws on a weights_manifest.json / config.json this engine does not
  // recognise, or any decoder tensor missing / of the wrong shape.
  explicit Decoder(const std::string &model_dir);

  // Resets the self-attention KV cache and the position counter ONLY.
  // Cross-attention K/V (set_encoder_output()) is NOT touched: it is fixed
  // for the whole 30 s window and outlives any number of generations over it
  // (the host calls this once per generation, after encode() and before the
  // first token).
  void clear_context();

  // Encoder::xkv()'s fused [1500, 10240] cross-attention K/V (bias already
  // applied). Decoder does not own or copy this buffer -- the caller keeps
  // the Encoder (or an equivalent buffer) alive for as long as it calls
  // step().
  void set_encoder_output(const float *xkv_1500x10240);

  // One decode step: embeds `token_id` at the current position, appends its
  // self-attention K/V to the cache, and writes DecoderGeometry::vocab_padded
  // logits to `logits_out` (indices [vocab, vocab_padded) are -inf). Advances
  // the position counter by one. Throws if set_encoder_output() was never
  // called or the position counter has reached max_target_positions.
  void step(int32_t token_id, float *logits_out /* [vocab_padded] */);

  int64_t position() const { return pos_; }

  DecoderTimers timers;

private:
  Linear embed_tokens_;                              // [vocab, d_model] bf16, tied to the head
  std::vector<float> embed_positions_;                // [max_target_positions, d_model] f32
  std::vector<float> ln_w_, ln_b_;                    // decoder.layer_norm
  std::vector<DecoderLayerWeights> layers_;

  const float *xkv_ = nullptr;                        // [1500, 10240], NOT owned
  int64_t pos_ = 0;

  // Self-attention KV cache: one [max_target_positions, d_model] block per
  // layer, per side. clear_context() only resets pos_ -- rows at or beyond it
  // are never read, so there is nothing to zero.
  std::vector<std::vector<float>> self_k_cache_, self_v_cache_;

  // Scratch reused across steps: a generation is hundreds of one-row calls,
  // so a fresh set of std::vector allocations every step would be measurable
  // next to the arithmetic itself (each layer touches ~14 vectors of up to
  // 5120 floats). Sized once in the constructor.
  std::vector<float> x_, h_, q_, k_, v_, attn_, tmp_, ff_;
};

}  // namespace ow
