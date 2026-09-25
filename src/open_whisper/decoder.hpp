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

#include "decoder_quant.hpp"

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
  // First id of the "special" region: eos/<|endoftext|>, then every language,
  // task, no-timestamps and timestamp token through vocab-1 (~1609 rows for
  // this geometry). Checked against config.json's OWN "eos_token_id" in the
  // constructor, the same way every other DG field is -- not a magic number,
  // a value this decoder is verified to have been built for (PR #111 review,
  // finding E). OW_DEC_HEAD=int8x recomputes every row in [eos_token_id,
  // vocab) exactly, unconditionally, because the hf decode protocol's own
  // logits processing (generation_hf.cpp: suppress_tokens, the timestamp
  // log-sum-exp, detect_language) reads almost exclusively from this region,
  // and it is far too small (~3%) of the vocabulary to reliably land inside
  // an int8-ranked top-64 on its own.
  static constexpr int64_t eos_token_id = 50257;
};

// Host wall-clock only (the same rule as encoder.hpp's Timers -- never an NPU
// performance claim; the decoder never dispatches to the NPU at all, so there
// is no npu_* split here).
struct DecoderTimers {
  double embed = 0, layer_norm = 0, linear = 0, attention = 0, gelu = 0;
  double total = 0;
  int64_t steps = 0;

  // Finer split of `linear` and `attention` above -- timing only, added for
  // task b1's baseline (the two coarse fields are still accumulated exactly
  // as before, so any comparison against a pre-existing number still works;
  // each fine-grained field's sum equals its coarse parent to within fp
  // accumulation order, which is asserted nowhere but true by construction:
  // both are summed from the same now_s() intervals).
  double linear_self_qkv = 0;   // self_q + self_k + self_v (three linear() calls, one bucket)
  double linear_self_out = 0;
  double linear_cross_q = 0;
  double linear_cross_out = 0;
  double linear_fc1 = 0;
  double linear_fc2 = 0;
  double linear_head = 0;       // the tied 51866 x 1280 sweep
  double attention_self = 0;
  double attention_cross = 0;

  // set_encoder_output()'s head-contiguous K/V gather (task b2 step 1) --
  // once per 30 s window, not per step, so it is its own bucket rather than
  // folded into `attention`.
  double xkv_gather = 0;
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

  // The three OW_DEC_* precision knobs' resolved values, for the startup
  // summary (engine_adapter.cpp's config_summary()).
  XkvPrecision xkv_precision() const { return xkv_precision_; }
  WeightPrecision weight_precision() const { return weight_precision_; }
  HeadPrecision head_precision() const { return head_precision_; }

  DecoderTimers timers;

private:
  Linear embed_tokens_;                              // [vocab, d_model] bf16, tied to the head
  std::vector<float> embed_positions_;                // [max_target_positions, d_model] f32
  std::vector<float> ln_w_, ln_b_;                    // decoder.layer_norm
  std::vector<DecoderLayerWeights> layers_;

  const float *xkv_ = nullptr;                        // [1500, 10240], NOT owned
  int64_t pos_ = 0;

  // Env-selected precision variants (see decoder_quant.hpp), read ONCE in
  // the constructor. Unset, they are bf16 / int8 / int8x (the speed defaults,
  // WER-neutral over 1200 utterances); OW_DEC_XKV=fp32, OW_DEC_W=bf16 and
  // OW_DEC_HEAD=bf16 restore the exact path. The initialisers below are only
  // placeholders -- the constructor always assigns all three.
  XkvPrecision xkv_precision_ = XkvPrecision::FP32;
  WeightPrecision weight_precision_ = WeightPrecision::BF16;
  HeadPrecision head_precision_ = HeadPrecision::BF16;

  // Cross-attention K/V, gathered head-contiguous once per window by
  // set_encoder_output() -- xkv_'s own fused-row layout puts a 64-float head
  // slice 10240 floats (40 KB) apart from the next row's same head, so
  // attend_one's cross-attention pass walked 20 strided passes over a 61 MB
  // buffer. Gathered layout: [layer][k=0|v=1][head][t in 0..1500)][head_dim],
  // contiguous per (layer, k/v, head) -- same VALUES (memcpy, no arithmetic),
  // only the addresses attend_one reads change. Sized on first use because
  // DecoderGeometry has no `1500` of its own (that is the encoder's frame
  // count, not decoder geometry) -- see set_encoder_output().
  //
  // Exactly ONE of the next two is ever populated, selected once at
  // construction by xkv_precision_: the fp32 buffer (default, UNCHANGED from
  // before this task) or the bf16 buffer (OW_DEC_XKV=bf16, half the bytes).
  std::vector<float> xkv_gathered_;
  std::vector<uint16_t> xkv_gathered_bf16_;

  // Per-layer int8 linears (OW_DEC_W=int8) -- parallel to layers_'s own bf16
  // Linear fields, quantized once at load from the SAME bf16 bits layers_
  // already holds. Empty unless weight_precision_ == INT8; layers_ itself is
  // always fully loaded regardless (it is also what quantize_int8_rows_bf16
  // reads from), so OW_DEC_W never changes which tensors are read off disk,
  // only which representation step() dispatches through.
  struct LayerInt8 {
    QLinear self_q, self_k, self_v, self_out;
    QLinear cross_q, cross_out;
    QLinear fc1, fc2;
  };
  std::vector<LayerInt8> layers_int8_;

  // Tied head, int8 (OW_DEC_HEAD=int8 or int8x) -- quantized once from
  // embed_tokens_.w. The embedding LOOKUP (step()'s very first read, token
  // id -> row) always reads embed_tokens_ itself (bf16), never this: only
  // the 51866-row OUTPUT sweep is affected by OW_DEC_HEAD.
  QLinear head_int8_;

  // attend_one's scores scratch, one row per HEAD ([n_heads][1500]) --
  // preallocated once here so parallelising attend_one across heads needs no
  // allocation on the hot path, and indexed by head rather than thread id so
  // a larger OpenMP team than at construction cannot overflow it. 1500 covers
  // both self-attention's cache (<= max_target_positions = 448) and
  // cross-attention (1500 encoder frames).
  std::vector<float> attn_scores_scratch_;
  int64_t attn_scratch_stride_ = 1500;

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
