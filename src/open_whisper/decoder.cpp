//===- decoder.cpp -------------------------------------------*- C++ -*-===//
// open_whisper -- see decoder.hpp. SPDX-License-Identifier: MIT
#include "decoder.hpp"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>

#if defined(_OPENMP)
#include <omp.h>
#endif

#if defined(__AVX2__)
#include <immintrin.h>
#endif

#include "host_ops.hpp"
#include "kernels.hpp"
#include "nlohmann/json.hpp"
#include "open_qwen36/q4nx_file.hpp"

namespace ow {
namespace {

using DG = DecoderGeometry;

double now_s() {
  return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

std::string read_file(const std::string &path) {
  std::ifstream fs(path, std::ios::binary);
  if (!fs) throw std::runtime_error("cannot open " + path);
  std::stringstream ss;
  ss << fs.rdbuf();
  return ss.str();
}

void require_shape(const open_qwen36::Q4nxFile &f, const std::string &name,
                   const std::vector<size_t> &want) {
  if (!f.has(name))
    throw std::runtime_error("model.open.safetensors: missing tensor '" + name + "'");
  const auto &m = f.meta(name);
  if (m.shape != want) {
    std::string got, exp;
    for (size_t i = 0; i < m.shape.size(); ++i) got += (i ? "," : "") + std::to_string(m.shape[i]);
    for (size_t i = 0; i < want.size(); ++i) exp += (i ? "," : "") + std::to_string(want[i]);
    throw std::runtime_error("model.open.safetensors: '" + name + "' has shape [" + got +
                             "], expected [" + exp + "]");
  }
}

std::vector<float> load_f32(const open_qwen36::Q4nxFile &f, const std::string &name,
                            const std::vector<size_t> &shape) {
  require_shape(f, name, shape);
  return f.f32(name);
}

// Raw bf16 bits, [out, in], NOT un-tiled: the decoder's tensors are stored at
// their natural shape (utilities/q4nx-build/q4nx/open_whisper.py's
// whisper_tensors(), "Decoder: transformers' names..." branch), unlike the
// encoder's GEMM operands, which weights.cpp tiles for the NPU at load.
std::vector<uint16_t> load_bf16_raw(const open_qwen36::Q4nxFile &f, const std::string &name,
                                    int64_t out, int64_t in) {
  require_shape(f, name, {static_cast<size_t>(out), static_cast<size_t>(in)});
  require_bf16(f, name);   // the dtype, not just the byte count -- see kernels.hpp
  size_t nbytes = 0;
  const uint8_t *raw = f.raw(name, &nbytes);
  const size_t want = static_cast<size_t>(out) * static_cast<size_t>(in);
  if (nbytes != want * 2)
    throw std::runtime_error("model.open.safetensors: '" + name + "' is " + std::to_string(nbytes) +
                             " bytes, expected " + std::to_string(want * 2) + " (bf16)");
  std::vector<uint16_t> v(want);
  std::memcpy(v.data(), raw, want * 2);
  return v;
}

Linear load_linear(const open_qwen36::Q4nxFile &f, const std::string &wname, const std::string &bname,
                   int64_t out, int64_t in) {
  Linear L;
  L.out = out;
  L.in = in;
  L.w = load_bf16_raw(f, wname, out, in);
  L.b = load_f32(f, bname, {static_cast<size_t>(out)});
  return L;
}

// A Linear whose bias tensor does not exist in the container (self-attention
// k_proj, and embed_tokens used as the tied output projection) -- an
// explicit zero vector, so linear() below never needs a no-bias branch.
Linear load_linear_nobias(const open_qwen36::Q4nxFile &f, const std::string &wname, int64_t out,
                          int64_t in) {
  Linear L;
  L.out = out;
  L.in = in;
  L.w = load_bf16_raw(f, wname, out, in);
  L.b.assign(static_cast<size_t>(out), 0.f);
  return L;
}

#if defined(__AVX2__)
// Identical to host_ops.cpp's dot8/axpy8 (also anonymous-namespace there, so
// duplicated rather than shared -- the same choice open_qwen36/vision/vit.cpp
// makes for its own copies of the same pattern).
inline float dot8(const float *a, const float *b, int64_t n) {
  __m256 acc = _mm256_setzero_ps();
  int64_t i = 0;
  for (; i + 8 <= n; i += 8)
    acc = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), acc);
  __m128 lo = _mm_add_ps(_mm256_castps256_ps128(acc), _mm256_extractf128_ps(acc, 1));
  lo = _mm_hadd_ps(lo, lo);
  lo = _mm_hadd_ps(lo, lo);
  float s = _mm_cvtss_f32(lo);
  for (; i < n; ++i) s += a[i] * b[i];
  return s;
}
inline void axpy8(float *y, const float *x, float alpha, int64_t n) {
  const __m256 av = _mm256_set1_ps(alpha);
  int64_t i = 0;
  for (; i + 8 <= n; i += 8)
    _mm256_storeu_ps(y + i, _mm256_fmadd_ps(av, _mm256_loadu_ps(x + i), _mm256_loadu_ps(y + i)));
  for (; i < n; ++i) y[i] += alpha * x[i];
}
// x (fp32) . w (bf16), widening 8 lanes of w to fp32 at a time -- ported with
// attribution from open_qwen36/vision/vit.cpp's widen_avx2/dot8_avx2 (this is
// the n=1-row specialisation: one output column, not eight).
inline float dot_bf16(const float *x, const uint16_t *w, int64_t n) {
  __m256 acc = _mm256_setzero_ps();
  int64_t i = 0;
  for (; i + 8 <= n; i += 8) {
    const __m128i h = _mm_loadu_si128(reinterpret_cast<const __m128i *>(w + i));
    const __m256 wf = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(h), 16));
    acc = _mm256_fmadd_ps(_mm256_loadu_ps(x + i), wf, acc);
  }
  __m128 lo = _mm_add_ps(_mm256_castps256_ps128(acc), _mm256_extractf128_ps(acc, 1));
  lo = _mm_hadd_ps(lo, lo);
  lo = _mm_hadd_ps(lo, lo);
  float s = _mm_cvtss_f32(lo);
  for (; i < n; ++i) s += x[i] * from_bf16(w[i]);
  return s;
}
#else
inline float dot8(const float *a, const float *b, int64_t n) {
  float s = 0.f;
  for (int64_t i = 0; i < n; ++i) s += a[i] * b[i];
  return s;
}
inline void axpy8(float *y, const float *x, float alpha, int64_t n) {
  for (int64_t i = 0; i < n; ++i) y[i] += alpha * x[i];
}
inline float dot_bf16(const float *x, const uint16_t *w, int64_t n) {
  float s = 0.f;
  for (int64_t i = 0; i < n; ++i) s += x[i] * from_bf16(w[i]);
  return s;
}
#endif

// y[out] = x[in] . W^T + b. Row-parallel over `out` -- pays off at the tied
// lm_head's 51866 outputs, the single largest sweep in a decode step; the
// per-layer projections (out <= 5120) are small enough that at n=1 row this
// is mostly bookkeeping either way, so one function serves both rather than
// adding a size-dependent branch.
void linear(const float *x, const Linear &W, float *y) {
#pragma omp parallel for schedule(static) num_threads(::ow::omp_threads())
  for (int64_t o = 0; o < W.out; ++o) y[static_cast<size_t>(o)] = dot_bf16(x, W.w.data() + o * W.in, W.in) + W.b[static_cast<size_t>(o)];
}

// Self-attention's q_proj/k_proj/v_proj against the SAME input `x`, as one
// omp parallel region instead of three: three fork/joins per layer became
// one. Each output element is exactly linear()'s own computation for that
// (W, o) pair -- q, k and v never read or write each other's memory, so
// which of the three sub-ranges a given iteration of a single flattened
// [0, 3*out) loop lands in changes nothing about its value, only how many
// times the thread team is spun up around the group.
void linear_qkv(const float *x, const Linear &Wq, const Linear &Wk, const Linear &Wv, float *q,
                float *k, float *v) {
  const int64_t out = Wq.out;   // == Wk.out == Wv.out == d_model, asserted by the caller's geometry
#pragma omp parallel for schedule(static) num_threads(::ow::omp_threads())
  for (int64_t o = 0; o < 3 * out; ++o) {
    if (o < out) {
      q[static_cast<size_t>(o)] = dot_bf16(x, Wq.w.data() + o * Wq.in, Wq.in) + Wq.b[static_cast<size_t>(o)];
    } else if (o < 2 * out) {
      const int64_t oo = o - out;
      k[static_cast<size_t>(oo)] = dot_bf16(x, Wk.w.data() + oo * Wk.in, Wk.in) + Wk.b[static_cast<size_t>(oo)];
    } else {
      const int64_t oo = o - 2 * out;
      v[static_cast<size_t>(oo)] = dot_bf16(x, Wv.w.data() + oo * Wv.in, Wv.in) + Wv.b[static_cast<size_t>(oo)];
    }
  }
}

// One query row against `len` K/V rows, per head, softmax with max
// subtraction. Generalised over TWO strides per side rather than one:
// `k_row_stride` is the float distance from row t's K to row t+1's (same
// head), `k_head_stride` is the float distance from head h's K block to head
// h+1's (same row) -- and likewise for V. This one function serves three
// layouts unchanged: self-attention's cache (row_stride = d_model,
// head_stride = head_dim -- consecutive heads sit inside one row), and
// cross-attention's GATHERED K/V (row_stride = head_dim, head_stride =
// len*head_dim -- consecutive head_dim-wide rows sit inside one head's
// block). Each head's whole computation (dot products, softmax, weighted
// sum) is independent of every other head and writes only its own
// `head_dim`-wide slice of `out`, so parallelising over `h` changes no
// floating-point accumulation order -- see host_ops.cpp's attention() for
// the same argument made about the encoder's own multi-head loop.
// `scores_scratch` is [heads][scores_stride], preallocated by the
// caller; `scores_stride` must be >= len.
void attend_one(const float *q, const float *k_base, int64_t k_row_stride, int64_t k_head_stride,
                const float *v_base, int64_t v_row_stride, int64_t v_head_stride, int64_t len,
                int64_t heads, int64_t head_dim, float scale, float *out, float *scores_scratch,
                int64_t scores_stride) {
#pragma omp parallel for schedule(static) num_threads(::ow::omp_threads())
  for (int64_t h = 0; h < heads; ++h) {
    // One scratch row per HEAD, not per thread: indexing by
    // omp_get_thread_num() would overflow if the team grew past the count
    // sized at construction (another component in the same process may call
    // omp_set_num_threads), and heads are few enough (20 x 1500 floats) that
    // the per-thread saving is not worth that failure mode.
    float *scores = scores_scratch + h * scores_stride;
    const float *qh = q + h * head_dim;
    const float *kh = k_base + h * k_head_stride;
    const float *vh = v_base + h * v_head_stride;
    float mx = -std::numeric_limits<float>::infinity();
    for (int64_t t = 0; t < len; ++t) {
      const float *krow = kh + t * k_row_stride;
      const float s = dot8(qh, krow, head_dim) * scale;
      scores[t] = s;
      if (s > mx) mx = s;
    }
    float sum = 0.f;
    for (int64_t t = 0; t < len; ++t) {
      const float e = std::exp(scores[t] - mx);
      scores[t] = e;
      sum += e;
    }
    const float inv = 1.0f / sum;
    float *oh = out + h * head_dim;
    std::memset(oh, 0, static_cast<size_t>(head_dim) * sizeof(float));
    for (int64_t t = 0; t < len; ++t)
      axpy8(oh, vh + t * v_row_stride, scores[t] * inv, head_dim);
  }
}

}  // namespace

Decoder::Decoder(const std::string &model_dir) {
  // 1. weights_manifest.json format -- refused before opening the safetensors
  //    at all, same discipline as weights.cpp for the encoder.
  const nlohmann::json manifest = nlohmann::json::parse(read_file(model_dir + "/weights_manifest.json"));
  const std::string format = manifest.value("format", std::string());
  if (format != "oflm-open-whisper-v1")
    throw std::runtime_error("weights_manifest.json: format is '" + format +
                             "', expected 'oflm-open-whisper-v1'");

  // 2. config.json geometry -- the decoder-specific fields weights.cpp does
  //    not check (it only validates the encoder's).
  const nlohmann::json cfg = nlohmann::json::parse(read_file(model_dir + "/config.json"));
  auto want_int = [&](const char *key, int64_t want) {
    if (!cfg.contains(key)) throw std::runtime_error(std::string("config.json: missing '") + key + "'");
    const int64_t got = cfg.at(key).get<int64_t>();
    if (got != want)
      throw std::runtime_error(std::string("config.json: '") + key + "' is " + std::to_string(got) +
                               ", this decoder only implements " + std::to_string(want));
  };
  if (cfg.value("model_type", std::string()) != "whisper")
    throw std::runtime_error("config.json: model_type is not 'whisper'");
  want_int("d_model", DG::d_model);
  want_int("decoder_layers", DG::n_layers);
  want_int("decoder_attention_heads", DG::n_heads);
  want_int("decoder_ffn_dim", DG::ffn);
  want_int("max_target_positions", DG::max_target_positions);
  want_int("vocab_size", DG::vocab);
  // scale_embedding is false on whisper-large-v3-turbo: step() adds the raw
  // embedding with no sqrt(d_model) scale. A checkpoint that sets it true
  // would silently need a different embed step, so this is refused rather
  // than guessed.
  if (cfg.value("scale_embedding", false))
    throw std::runtime_error(
        "config.json: scale_embedding is true -- this decoder does not scale the token "
        "embedding, matching whisper-large-v3-turbo's own false");

  // 3. The tensors themselves.
  open_qwen36::Q4nxFile f(model_dir + "/model.open.safetensors");
  const int64_t D = DG::d_model, FFN = DG::ffn, V = DG::vocab;

  embed_tokens_ = load_linear_nobias(f, "decoder.embed_tokens.weight", V, D);
  embed_positions_ = load_f32(f, "decoder.embed_positions.weight",
                              {static_cast<size_t>(DG::max_target_positions), static_cast<size_t>(D)});
  ln_w_ = load_f32(f, "decoder.layer_norm.weight", {static_cast<size_t>(D)});
  ln_b_ = load_f32(f, "decoder.layer_norm.bias", {static_cast<size_t>(D)});

  layers_.resize(static_cast<size_t>(DG::n_layers));
  for (int64_t l = 0; l < DG::n_layers; ++l) {
    auto &L = layers_[static_cast<size_t>(l)];
    const std::string p = "decoder.layers." + std::to_string(l) + ".";
    L.self_q = load_linear(f, p + "self_attn.q_proj.weight", p + "self_attn.q_proj.bias", D, D);
    L.self_k = load_linear_nobias(f, p + "self_attn.k_proj.weight", D, D);
    L.self_v = load_linear(f, p + "self_attn.v_proj.weight", p + "self_attn.v_proj.bias", D, D);
    L.self_out = load_linear(f, p + "self_attn.out_proj.weight", p + "self_attn.out_proj.bias", D, D);
    L.ln_self_w = load_f32(f, p + "self_attn_layer_norm.weight", {static_cast<size_t>(D)});
    L.ln_self_b = load_f32(f, p + "self_attn_layer_norm.bias", {static_cast<size_t>(D)});

    // k_proj/v_proj are NOT read here: dec.xkv (Encoder::xkv()) already
    // carries them, computed once for the whole window.
    L.cross_q = load_linear(f, p + "encoder_attn.q_proj.weight", p + "encoder_attn.q_proj.bias", D, D);
    L.cross_out =
        load_linear(f, p + "encoder_attn.out_proj.weight", p + "encoder_attn.out_proj.bias", D, D);
    L.ln_cross_w = load_f32(f, p + "encoder_attn_layer_norm.weight", {static_cast<size_t>(D)});
    L.ln_cross_b = load_f32(f, p + "encoder_attn_layer_norm.bias", {static_cast<size_t>(D)});

    L.fc1 = load_linear(f, p + "fc1.weight", p + "fc1.bias", FFN, D);
    L.fc2 = load_linear(f, p + "fc2.weight", p + "fc2.bias", D, FFN);
    L.ln_final_w = load_f32(f, p + "final_layer_norm.weight", {static_cast<size_t>(D)});
    L.ln_final_b = load_f32(f, p + "final_layer_norm.bias", {static_cast<size_t>(D)});
  }

  // OPTIONAL precision variants (task 0180 Part 8): read once, strict --
  // print the VALUE PARSED, not the intention, matching encoder.cpp's own
  // OW_ATTN startup line.
  xkv_precision_ = parse_xkv_precision();
  weight_precision_ = parse_weight_precision();
  head_precision_ = parse_head_precision();
  std::printf("  decoder    xkv=%s weights=%s head=%s\n", ow::to_string(xkv_precision_),
             ow::to_string(weight_precision_), ow::to_string(head_precision_));

  if (weight_precision_ == WeightPrecision::INT8) {
    layers_int8_.resize(static_cast<size_t>(DG::n_layers));
    for (int64_t l = 0; l < DG::n_layers; ++l) {
      const auto &W = layers_[static_cast<size_t>(l)];
      auto &Q = layers_int8_[static_cast<size_t>(l)];
      Q.self_q = quantize_int8_rows_bf16(W.self_q.w.data(), W.self_q.b, W.self_q.out, W.self_q.in);
      Q.self_k = quantize_int8_rows_bf16(W.self_k.w.data(), W.self_k.b, W.self_k.out, W.self_k.in);
      Q.self_v = quantize_int8_rows_bf16(W.self_v.w.data(), W.self_v.b, W.self_v.out, W.self_v.in);
      Q.self_out =
          quantize_int8_rows_bf16(W.self_out.w.data(), W.self_out.b, W.self_out.out, W.self_out.in);
      Q.cross_q = quantize_int8_rows_bf16(W.cross_q.w.data(), W.cross_q.b, W.cross_q.out, W.cross_q.in);
      Q.cross_out =
          quantize_int8_rows_bf16(W.cross_out.w.data(), W.cross_out.b, W.cross_out.out, W.cross_out.in);
      Q.fc1 = quantize_int8_rows_bf16(W.fc1.w.data(), W.fc1.b, W.fc1.out, W.fc1.in);
      Q.fc2 = quantize_int8_rows_bf16(W.fc2.w.data(), W.fc2.b, W.fc2.out, W.fc2.in);
    }
  }
  if (head_precision_ == HeadPrecision::INT8 || head_precision_ == HeadPrecision::INT8X) {
    head_int8_ = quantize_int8_rows_bf16(embed_tokens_.w.data(), embed_tokens_.b, embed_tokens_.out,
                                         embed_tokens_.in);
  }

  self_k_cache_.assign(static_cast<size_t>(DG::n_layers),
                       std::vector<float>(static_cast<size_t>(DG::max_target_positions) *
                                         static_cast<size_t>(D)));
  self_v_cache_ = self_k_cache_;

  x_.resize(static_cast<size_t>(D));
  h_.resize(static_cast<size_t>(D));
  q_.resize(static_cast<size_t>(D));
  k_.resize(static_cast<size_t>(D));
  v_.resize(static_cast<size_t>(D));
  attn_.resize(static_cast<size_t>(D));
  tmp_.resize(static_cast<size_t>(D));
  ff_.resize(static_cast<size_t>(FFN));

  attn_scratch_stride_ = 1500;   // covers both self's <=448 and cross's 1500
  attn_scores_scratch_.assign(static_cast<size_t>(DG::n_heads) * static_cast<size_t>(attn_scratch_stride_),
                              0.f);

  clear_context();
}

void Decoder::clear_context() { pos_ = 0; }   // cache rows at/beyond pos_ are never read

// Copies xkv_1500x10240's K and V into head-contiguous scratch (see
// decoder.hpp's xkv_gathered_ comment): same VALUES via memcpy, so this
// changes memory ADDRESSES ONLY -- attend_one's arithmetic is untouched.
// Runs once per 30 s window (whenever the caller re-encodes), not per step --
// engine_adapter.cpp's encode_audio() and cli.cpp's run_decode_gate() both
// call this exactly once after encode(), for exactly the window's worth of
// decode steps that follow, so xkv_1500x10240 and this gather are always in
// sync with each other.
void Decoder::set_encoder_output(const float *xkv_1500x10240) {
  xkv_ = xkv_1500x10240;
  const int64_t D = DG::d_model, H = DG::n_heads, HD = DG::head_dim, L = DG::n_layers;
  const int64_t XKV_STRIDE = 2 * L * D;
  const int64_t LEN = 1500;
  const int64_t blocks = L * 2 * H;   // one block per (layer, k-or-v, head)
  const double t0 = now_s();

  if (xkv_precision_ == XkvPrecision::BF16) {
    // OW_DEC_XKV=bf16: same gather, same addresses, but each 64-float row is
    // converted to bf16 (RNE, host_ops.hpp's bf16_fill) on the way in rather
    // than memcpy'd -- halves xkv_gathered_'s 61.4 MB. This branch touches
    // NOTHING the fp32 branch below reads or writes.
    if (xkv_gathered_bf16_.empty())
      xkv_gathered_bf16_.assign(static_cast<size_t>(L) * 2 * static_cast<size_t>(H) *
                                    static_cast<size_t>(LEN) * static_cast<size_t>(HD),
                                0);
#pragma omp parallel for schedule(static) num_threads(::ow::omp_threads())
    for (int64_t idx = 0; idx < blocks; ++idx) {
      const int64_t l = idx / (2 * H);
      const int64_t kv = (idx / H) % 2;   // 0 = K, 1 = V
      const int64_t h = idx % H;
      const float *src = xkv_ + (2 * l + kv) * D + h * HD;
      uint16_t *dst = xkv_gathered_bf16_.data() +
                     (static_cast<size_t>(l) * 2 + static_cast<size_t>(kv)) * static_cast<size_t>(H) *
                         static_cast<size_t>(LEN) * static_cast<size_t>(HD) +
                     static_cast<size_t>(h) * static_cast<size_t>(LEN) * static_cast<size_t>(HD);
      for (int64_t t = 0; t < LEN; ++t)
        bf16_fill(dst + t * HD, src + t * XKV_STRIDE, static_cast<size_t>(HD));
    }
    timers.xkv_gather += now_s() - t0;
    return;
  }

  // Default (OW_DEC_XKV unset or "fp32"): UNCHANGED from before this task.
  if (xkv_gathered_.empty())
    xkv_gathered_.assign(static_cast<size_t>(L) * 2 * static_cast<size_t>(H) *
                             static_cast<size_t>(LEN) * static_cast<size_t>(HD),
                         0.f);

#pragma omp parallel for schedule(static) num_threads(::ow::omp_threads())
  for (int64_t idx = 0; idx < blocks; ++idx) {
    const int64_t l = idx / (2 * H);
    const int64_t kv = (idx / H) % 2;   // 0 = K, 1 = V
    const int64_t h = idx % H;
    const float *src = xkv_ + (2 * l + kv) * D + h * HD;
    float *dst = xkv_gathered_.data() +
                (static_cast<size_t>(l) * 2 + static_cast<size_t>(kv)) * static_cast<size_t>(H) *
                    static_cast<size_t>(LEN) * static_cast<size_t>(HD) +
                static_cast<size_t>(h) * static_cast<size_t>(LEN) * static_cast<size_t>(HD);
    for (int64_t t = 0; t < LEN; ++t)
      std::memcpy(dst + t * HD, src + t * XKV_STRIDE, static_cast<size_t>(HD) * sizeof(float));
  }
  timers.xkv_gather += now_s() - t0;
}

void Decoder::step(int32_t token_id, float *logits_out) {
  if (pos_ >= DG::max_target_positions)
    throw std::runtime_error("Decoder::step: position " + std::to_string(pos_) +
                             " has reached max_target_positions (" +
                             std::to_string(DG::max_target_positions) + ")");
  if (!xkv_) throw std::runtime_error("Decoder::step: set_encoder_output() was never called");

  const int64_t D = DG::d_model, H = DG::n_heads, HD = DG::head_dim, FFN = DG::ffn;
  const float scale = 1.0f / std::sqrt(static_cast<float>(HD));
  const double t_step0 = now_s();
  double t0;

  // Embed: token lookup (bf16, widened) + learned absolute position.
  t0 = now_s();
  bf16_read(x_.data(), embed_tokens_.w.data() + static_cast<size_t>(token_id) * static_cast<size_t>(D), D);
  const float *posr = embed_positions_.data() + pos_ * D;
  for (int64_t c = 0; c < D; ++c) x_[static_cast<size_t>(c)] += posr[c];
  timers.embed += now_s() - t0;

  for (int64_t l = 0; l < DG::n_layers; ++l) {
    const auto &W = layers_[static_cast<size_t>(l)];

    // h = self_attn_layer_norm(x); q,k,v = proj(h); append k,v to this
    // layer's cache at pos_; attend over cache[0, pos_] (causal by
    // construction -- nothing past pos_ is in the cache yet).
    t0 = now_s();
    layer_norm(x_.data(), W.ln_self_w.data(), W.ln_self_b.data(), 1, D, h_.data());
    timers.layer_norm += now_s() - t0;

    t0 = now_s();
    if (weight_precision_ == WeightPrecision::INT8) {
      const auto &Qi = layers_int8_[static_cast<size_t>(l)];
      linear_int8(h_.data(), Qi.self_q, q_.data());
      linear_int8(h_.data(), Qi.self_k, k_.data());
      linear_int8(h_.data(), Qi.self_v, v_.data());
    } else {
      linear_qkv(h_.data(), W.self_q, W.self_k, W.self_v, q_.data(), k_.data(), v_.data());
    }
    {
      const double d = now_s() - t0;
      timers.linear += d;
      timers.linear_self_qkv += d;
    }

    float *kc = self_k_cache_[static_cast<size_t>(l)].data() + pos_ * D;
    float *vc = self_v_cache_[static_cast<size_t>(l)].data() + pos_ * D;
    std::memcpy(kc, k_.data(), static_cast<size_t>(D) * sizeof(float));
    std::memcpy(vc, v_.data(), static_cast<size_t>(D) * sizeof(float));

    t0 = now_s();
    // Self cache layout unchanged: row_stride = D (consecutive positions),
    // head_stride = HD (heads packed inside one row).
    attend_one(q_.data(), self_k_cache_[static_cast<size_t>(l)].data(), D, HD,
              self_v_cache_[static_cast<size_t>(l)].data(), D, HD, pos_ + 1, H, HD, scale,
              attn_.data(), attn_scores_scratch_.data(), attn_scratch_stride_);
    {
      const double d = now_s() - t0;
      timers.attention += d;
      timers.attention_self += d;
    }

    t0 = now_s();
    if (weight_precision_ == WeightPrecision::INT8)
      linear_int8(attn_.data(), layers_int8_[static_cast<size_t>(l)].self_out, tmp_.data());
    else
      linear(attn_.data(), W.self_out, tmp_.data());
    {
      const double d = now_s() - t0;
      timers.linear += d;
      timers.linear_self_out += d;
    }
    for (int64_t c = 0; c < D; ++c) x_[static_cast<size_t>(c)] += tmp_[static_cast<size_t>(c)];

    // h = encoder_attn_layer_norm(x); q = proj(h); attend over the encoder's
    // fixed [1500, D] K/V for this layer -- no mask, no cache append.
    t0 = now_s();
    layer_norm(x_.data(), W.ln_cross_w.data(), W.ln_cross_b.data(), 1, D, h_.data());
    timers.layer_norm += now_s() - t0;

    t0 = now_s();
    if (weight_precision_ == WeightPrecision::INT8)
      linear_int8(h_.data(), layers_int8_[static_cast<size_t>(l)].cross_q, q_.data());
    else
      linear(h_.data(), W.cross_q, q_.data());
    {
      const double d = now_s() - t0;
      timers.linear += d;
      timers.linear_cross_q += d;
    }

    t0 = now_s();
    // Gathered cross K/V: row_stride = HD (one head's rows are contiguous),
    // head_stride = 1500*HD (distance between head blocks). Same values as
    // xkv_ + 2*l*D / xkv_ + (2*l+1)*D at stride XKV_STRIDE would have given --
    // see set_encoder_output()'s gather comment.
    if (xkv_precision_ == XkvPrecision::BF16) {
      const int64_t HB = 1500 * HD;
      const uint16_t *Kg = xkv_gathered_bf16_.data() +
                          (static_cast<size_t>(l) * 2 + 0) * static_cast<size_t>(H) * static_cast<size_t>(HB);
      const uint16_t *Vg = xkv_gathered_bf16_.data() +
                          (static_cast<size_t>(l) * 2 + 1) * static_cast<size_t>(H) * static_cast<size_t>(HB);
      attend_one_xkv_bf16(q_.data(), Kg, HD, HB, Vg, HD, HB, 1500, H, HD, scale, attn_.data(),
                          attn_scores_scratch_.data(), attn_scratch_stride_);
    } else {
      const int64_t HB = 1500 * HD;
      const float *Kg = xkv_gathered_.data() + (static_cast<size_t>(l) * 2 + 0) *
                                                   static_cast<size_t>(H) * static_cast<size_t>(HB);
      const float *Vg = xkv_gathered_.data() + (static_cast<size_t>(l) * 2 + 1) *
                                                   static_cast<size_t>(H) * static_cast<size_t>(HB);
      attend_one(q_.data(), Kg, HD, HB, Vg, HD, HB, 1500, H, HD, scale, attn_.data(),
                attn_scores_scratch_.data(), attn_scratch_stride_);
    }
    {
      const double d = now_s() - t0;
      timers.attention += d;
      timers.attention_cross += d;
    }

    t0 = now_s();
    if (weight_precision_ == WeightPrecision::INT8)
      linear_int8(attn_.data(), layers_int8_[static_cast<size_t>(l)].cross_out, tmp_.data());
    else
      linear(attn_.data(), W.cross_out, tmp_.data());
    {
      const double d = now_s() - t0;
      timers.linear += d;
      timers.linear_cross_out += d;
    }
    for (int64_t c = 0; c < D; ++c) x_[static_cast<size_t>(c)] += tmp_[static_cast<size_t>(c)];

    // h = final_layer_norm(x); x += fc2(GELU(fc1(h)))
    t0 = now_s();
    layer_norm(x_.data(), W.ln_final_w.data(), W.ln_final_b.data(), 1, D, h_.data());
    timers.layer_norm += now_s() - t0;

    t0 = now_s();
    if (weight_precision_ == WeightPrecision::INT8)
      linear_int8(h_.data(), layers_int8_[static_cast<size_t>(l)].fc1, ff_.data());
    else
      linear(h_.data(), W.fc1, ff_.data());
    {
      const double d = now_s() - t0;
      timers.linear += d;
      timers.linear_fc1 += d;
    }
    t0 = now_s();
    gelu(ff_.data(), 1, FFN, ff_.data());
    timers.gelu += now_s() - t0;
    t0 = now_s();
    if (weight_precision_ == WeightPrecision::INT8)
      linear_int8(ff_.data(), layers_int8_[static_cast<size_t>(l)].fc2, tmp_.data());
    else
      linear(ff_.data(), W.fc2, tmp_.data());
    {
      const double d = now_s() - t0;
      timers.linear += d;
      timers.linear_fc2 += d;
    }
    for (int64_t c = 0; c < D; ++c) x_[static_cast<size_t>(c)] += tmp_[static_cast<size_t>(c)];
  }

  t0 = now_s();
  layer_norm(x_.data(), ln_w_.data(), ln_b_.data(), 1, D, h_.data());
  timers.layer_norm += now_s() - t0;

  // Tied head: logits = h . embed_tokens^T (no bias). Pad to vocab_padded
  // with -inf so a caller sampling over the padded width can never pick one
  // of the six unused ids.
  t0 = now_s();
  if (head_precision_ == HeadPrecision::BF16) {
    linear(h_.data(), embed_tokens_, logits_out);
  } else {
    // int8 / int8x: every logit from the int8 approximation first...
    linear_int8(h_.data(), head_int8_, logits_out);
    if (head_precision_ == HeadPrecision::INT8X) {
      // ...then the K=64 rows the int8 pass ranked highest are recomputed
      // EXACTLY against embed_tokens_'s own bf16 bits (the same tensor and
      // the same dot_bf16-shaped kernel the HeadPrecision::BF16 branch
      // above uses) and overwritten in place -- see
      // recompute_top_k_exact()'s header comment for why this keeps greedy
      // argmax exact unless the true maximum falls outside the int8 top-64.
      // embed_tokens_ has no bias tensor (load_linear_nobias), so `bias` is
      // null here, matching linear()'s own zero-bias vector for this Linear.
      recompute_top_k_exact(h_.data(), DG::vocab, D, 64, embed_tokens_.w.data(), nullptr, logits_out);
    }
  }
  {
    const double d = now_s() - t0;
    timers.linear += d;
    timers.linear_head += d;
  }
  for (int64_t i = DG::vocab; i < DG::vocab_padded; ++i)
    logits_out[i] = -std::numeric_limits<float>::infinity();

  ++pos_;
  ++timers.steps;
  timers.total += now_s() - t_step0;
}

}  // namespace ow
