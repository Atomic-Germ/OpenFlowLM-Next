//===- gemma_encode.cpp ---------------------------------------*- C++ -*-===//
//
// NpuEmbeddings -- EmbeddingGemma-300M (arch=1), CPU-only forward pass.
// SPDX-License-Identifier: MIT
//
// See gemma_encode.hpp for why this has no NPU code. Every numeric decision
// below (which precision each step runs in) is copied from
// reference/encoder_gemma.py, not re-derived -- see that file's docstrings
// for the primary-source citations. Where this file's rounding choices
// differ in kind from the reference's (see gemm_f32 below), that is called
// out explicitly rather than silently assumed equivalent.

#include "gemma_encode.hpp"

#include "gemma_kernels.hpp"

#include <cmath>
#include <cstring>
#include <stdexcept>

namespace npue {

namespace {

// A plain (M,K)x(K,N)->(M,N) GEMM, row-major, double-precision accumulation
// rounded to float32 on the way out.
//
// This is NOT bit-identical to reference/encoder_gemma.py's fp32_gemm()
// (which is `(a.astype(f32) @ b.astype(f32)).astype(f32)` -- numpy's own
// BLAS path, whose accumulation order/precision is opaque and unspecified).
// Accumulating in double is a DELIBERATE choice, not an oversight: it can
// only be at least as accurate as a native fp32 accumulation for any given
// reduction length (256-1152 here), never less, so it cannot be the source
// of the encode's error against the goldens -- any measured 1-cos gap is
// upper-bounded by this choice, not caused by it. Matches this project's
// own convention elsewhere (RMSNorm's reduction, encoder.py's LayerNorm) of
// upcasting reductions as "cheap insurance", not a precision trade.
void gemm_f32(const float *a, int64_t M, int64_t K, const float *b, int64_t N,
              float *c) {
  for (int64_t i = 0; i < M; ++i) {
    const float *arow = a + i * K;
    float *crow = c + i * N;
    for (int64_t j = 0; j < N; ++j) {
      double acc = 0.0;
      for (int64_t k = 0; k < K; ++k)
        acc += static_cast<double>(arow[k]) * static_cast<double>(b[k * N + j]);
      crow[j] = static_cast<float>(acc);
    }
  }
}

}  // namespace

GemmaEncoder::GemmaEncoder(File &model) : model_(model) {
  if (model.config_string("arch") != "gemma3_mqa_rope_geglu")
    throw std::runtime_error(
        "GemmaEncoder given a .npue whose config arch is '" +
        model.config_string("arch") + "', not 'gemma3_mqa_rope_geglu'");

  hidden_ = model.config_int("hidden");
  heads_ = model.config_int("num_heads");
  kv_heads_ = model.config_int("num_key_value_heads");
  head_dim_ = model.config_int("head_dim");
  inter_ = model.config_int("intermediate");
  layers_ = model.config_int("num_layers");
  dense_hidden_ = model.config_int("dense_hidden");
  eps_ = model.config_double("rms_norm_eps");
  rope_theta_ = model.config_double("rope_theta");
  rope_theta_local_ = model.config_double("rope_local_base_freq");
  sliding_window_pattern_ = model.config_int("sliding_window_pattern");
  query_pre_attn_scalar_ = model.config_double("query_pre_attn_scalar");

  if (hidden_ <= 0 || heads_ <= 0 || kv_heads_ <= 0 || head_dim_ <= 0 ||
      layers_ <= 0 || dense_hidden_ <= 0)
    throw std::runtime_error("the .npue reports a non-positive Gemma shape");
  if (head_dim_ * heads_ != hidden_)
    throw std::runtime_error("head_dim * num_heads != hidden in the .npue");
  if (heads_ % kv_heads_)
    throw std::runtime_error("num_heads is not a multiple of "
                             "num_key_value_heads (MQA/GQA repeat factor)");

  w_embed_ = model.raw("embed_tokens.weight").as<float>();
  w_norm_ = model.raw("norm.weight").as<float>();
  w_dense2_ = model.raw("dense2.weight").as<float>();
  w_dense3_ = model.raw("dense3.weight").as<float>();

  lp_.resize(static_cast<size_t>(layers_));
  for (int64_t i = 0; i < layers_; ++i) {
    const std::string p = "layer." + std::to_string(i) + ".";
    LayerPtrs &l = lp_[static_cast<size_t>(i)];
    l.q_proj = model.raw(p + "q_proj").as<float>();
    l.k_proj = model.raw(p + "k_proj").as<float>();
    l.v_proj = model.raw(p + "v_proj").as<float>();
    l.o_proj = model.raw(p + "o_proj").as<float>();
    l.q_norm = model.raw(p + "q_norm.weight").as<float>();
    l.k_norm = model.raw(p + "k_norm.weight").as<float>();
    l.ln_in = model.raw(p + "input_layernorm.weight").as<float>();
    l.ln_post_attn = model.raw(p + "post_attention_layernorm.weight").as<float>();
    l.ln_pre_ffn = model.raw(p + "pre_feedforward_layernorm.weight").as<float>();
    l.ln_post_ffn = model.raw(p + "post_feedforward_layernorm.weight").as<float>();
    l.gate_proj = model.raw(p + "gate_proj").as<float>();
    l.up_proj = model.raw(p + "up_proj").as<float>();
    l.down_proj = model.raw(p + "down_proj").as<float>();
  }

  auto tv = model.raw("tokenizer.gemma_table");
  tok = GemmaTokenizer::from_table_bytes(
      reinterpret_cast<const char *>(tv.data), tv.bytes);
}

std::vector<float> GemmaEncoder::encode_one(const std::string &text,
                                            int max_len,
                                            const std::string &prefix_name,
                                            size_t index,
                                            bool allow_truncation) const {
  const GemmaEncoded en = tok.encode(text, max_len, prefix_name);
  // Note this counts the task prefix too -- it is prepended before
  // tokenization and spends the same budget, so a caller that sized its text
  // against max_len alone can legitimately land here.
  if (en.truncated && !allow_truncation)
    throw InputTooLong(index, en.n_tokens_full, max_len);
  const int64_t S = max_len;
  const int64_t H = heads_, KVH = kv_heads_, hd = head_dim_;
  const bool kv_heads_is_one = (KVH == 1);
  if (!kv_heads_is_one)
    throw std::runtime_error(
        "GemmaEncoder::encode_one's attention loop assumes num_key_value_heads "
        "== 1 (true for EmbeddingGemma-300M); a checkpoint with KVH > 1 needs "
        "the repeat_kv loop generalised before this can run it");

  // -- embed: gather + x*sqrt(hidden), matching reference/encoder_gemma.py's
  // embed() exactly (upcast to double for the multiply, round once). ------
  const double embed_scale = std::sqrt(static_cast<double>(hidden_));
  std::vector<float> x(static_cast<size_t>(S * hidden_));
  for (int64_t s = 0; s < S; ++s) {
    const int32_t id = en.input_ids[static_cast<size_t>(s)];
    const float *wv = w_embed_ + static_cast<size_t>(id) * hidden_;
    float *dst = x.data() + s * hidden_;
    for (int64_t c = 0; c < hidden_; ++c)
      dst[c] = static_cast<float>(static_cast<double>(wv[c]) * embed_scale);
  }

  // Additive padding mask, shared by every layer (exact at seq_len <= 512,
  // see reference/encoder_gemma.py's file header for why sliding-window
  // masking collapses to this at the lengths this project's goldens use).
  const float MASK_FILL = -3.4028235e38f;  // np.finfo(float32).min
  std::vector<float> add_mask(static_cast<size_t>(S));
  for (int64_t s = 0; s < S; ++s)
    add_mask[static_cast<size_t>(s)] =
        en.attention_mask[static_cast<size_t>(s)] ? 0.0f : MASK_FILL;

  // RoPE tables, one per base frequency actually used this run.
  std::vector<float> cos_g(static_cast<size_t>(S * hd)), sin_g(static_cast<size_t>(S * hd));
  std::vector<float> cos_l(static_cast<size_t>(S * hd)), sin_l(static_cast<size_t>(S * hd));
  gemma_rope_tables(S, hd, rope_theta_, cos_g.data(), sin_g.data());
  gemma_rope_tables(S, hd, rope_theta_local_, cos_l.data(), sin_l.data());

  const double attn_scale = std::pow(query_pre_attn_scalar_, -0.5);

  std::vector<float> h(static_cast<size_t>(S * hidden_));
  std::vector<float> q(static_cast<size_t>(S * H * hd));
  std::vector<float> k(static_cast<size_t>(S * hd));
  std::vector<float> v(static_cast<size_t>(S * hd));
  std::vector<float> rope_scratch(static_cast<size_t>(S * hd));
  std::vector<float> scores(static_cast<size_t>(S * S));
  std::vector<float> ctx(static_cast<size_t>(S * hidden_));
  std::vector<float> proj(static_cast<size_t>(S * hidden_));
  std::vector<float> gate(static_cast<size_t>(S * inter_));
  std::vector<float> up(static_cast<size_t>(S * inter_));
  std::vector<float> geglu(static_cast<size_t>(S * inter_));
  std::vector<float> down(static_cast<size_t>(S * hidden_));

  for (int64_t L = 0; L < layers_; ++L) {
    const LayerPtrs &lp = lp_[static_cast<size_t>(L)];
    const bool full_attn = gemma_is_full_attention_layer(L, sliding_window_pattern_);
    const float *cos_t = full_attn ? cos_g.data() : cos_l.data();
    const float *sin_t = full_attn ? sin_g.data() : sin_l.data();

    // residual = x; h = input_layernorm(x)
    rms_norm_cpu(x.data(), lp.ln_in, h.data(), S, hidden_, static_cast<float>(eps_));

    // -- attention --------------------------------------------------------
    gemm_f32(h.data(), S, hidden_, lp.q_proj, H * hd, q.data());
    gemm_f32(h.data(), S, hidden_, lp.k_proj, hd, k.data());
    gemm_f32(h.data(), S, hidden_, lp.v_proj, hd, v.data());

    // q_norm / k_norm: RMSNorm over head_dim, PER HEAD, before RoPE. q's
    // natural GEMM output layout is [S, H, hd] (s-major, h-minor), and each
    // (s,h) slice of hd floats is already contiguous, so rows=S*H works
    // directly with no repacking. k has KVH=1, so its [S, hd] layout is
    // already exactly "one row per s".
    rms_norm_cpu(q.data(), lp.q_norm, q.data(), S * H, hd, static_cast<float>(eps_));
    rms_norm_cpu(k.data(), lp.k_norm, k.data(), S, hd, static_cast<float>(eps_));

    // RoPE needs "one row per (head, sequence-position)" to recover position
    // as row%seq_len -- q's natural layout is the other order (s-major,
    // h-minor), so each head's S rows are extracted into a contiguous
    // scratch buffer, rotated, and written back (gemma_kernels.hpp's own
    // documented caller contract for a non-matching row order).
    for (int64_t hh = 0; hh < H; ++hh) {
      for (int64_t s = 0; s < S; ++s)
        std::memcpy(rope_scratch.data() + s * hd,
                    q.data() + (s * H + hh) * hd, sizeof(float) * static_cast<size_t>(hd));
      apply_rope_cpu(rope_scratch.data(), cos_t, sin_t, rope_scratch.data(), S, S, hd);
      for (int64_t s = 0; s < S; ++s)
        std::memcpy(q.data() + (s * H + hh) * hd,
                    rope_scratch.data() + s * hd, sizeof(float) * static_cast<size_t>(hd));
    }
    apply_rope_cpu(k.data(), cos_t, sin_t, k.data(), S, S, hd);

    // MQA: KVH=1, so every one of the H query heads attends to the SAME k/v
    // -- mathematically identical to repeat_kv(k,H) but without materialising
    // the repeat.
    for (int64_t hh = 0; hh < H; ++hh) {
      // scores[i,j] = dot(q[i,hh,:], k[j,:]) * scale + mask[j], matching the
      // reference's own order: fp32 GEMM, THEN scale in double or later,
      // then mask, then softmax in double.
      for (int64_t i = 0; i < S; ++i) {
        const float *qi = q.data() + (i * H + hh) * hd;
        float *srow = scores.data() + i * S;
        for (int64_t j = 0; j < S; ++j) {
          const float *kj = k.data() + j * hd;
          double acc = 0.0;
          for (int64_t d = 0; d < hd; ++d)
            acc += static_cast<double>(qi[d]) * static_cast<double>(kj[d]);
          const double scaled = acc * attn_scale;
          srow[j] = static_cast<float>(scaled) + add_mask[static_cast<size_t>(j)];
        }
      }
      // softmax, double precision (reference's softmax() upcasts fully).
      for (int64_t i = 0; i < S; ++i) {
        float *srow = scores.data() + i * S;
        double mx = srow[0];
        for (int64_t j = 1; j < S; ++j) mx = std::max(mx, static_cast<double>(srow[j]));
        double sum = 0.0;
        std::vector<double> e(static_cast<size_t>(S));
        for (int64_t j = 0; j < S; ++j) {
          e[static_cast<size_t>(j)] = std::exp(static_cast<double>(srow[j]) - mx);
          sum += e[static_cast<size_t>(j)];
        }
        for (int64_t j = 0; j < S; ++j)
          srow[j] = static_cast<float>(e[static_cast<size_t>(j)] / sum);
      }
      // ctx[i,hh,:] = sum_j probs[i,j] * v[j,:] -- exactly gemm_f32(probs,
      // S,S, v, hd, ...) written into this head's slice of ctx.
      for (int64_t i = 0; i < S; ++i) {
        const float *prow = scores.data() + i * S;
        float *co = ctx.data() + (i * H + hh) * hd;
        for (int64_t d = 0; d < hd; ++d) {
          double acc = 0.0;
          for (int64_t j = 0; j < S; ++j)
            acc += static_cast<double>(prow[j]) * static_cast<double>(v[j * hd + d]);
          co[d] = static_cast<float>(acc);
        }
      }
    }

    gemm_f32(ctx.data(), S, hidden_, lp.o_proj, hidden_, proj.data());

    // h = post_attention_layernorm(attn_out); x = residual + h
    rms_norm_cpu(proj.data(), lp.ln_post_attn, proj.data(), S, hidden_, static_cast<float>(eps_));
    for (size_t i = 0; i < x.size(); ++i) x[i] += proj[i];

    // -- GeGLU FFN ----------------------------------------------------------
    rms_norm_cpu(x.data(), lp.ln_pre_ffn, h.data(), S, hidden_, static_cast<float>(eps_));
    gemm_f32(h.data(), S, hidden_, lp.gate_proj, inter_, gate.data());
    gemm_f32(h.data(), S, hidden_, lp.up_proj, inter_, up.data());
    geglu_cpu(gate.data(), up.data(), geglu.data(),
              static_cast<size_t>(S * inter_));
    gemm_f32(geglu.data(), S, inter_, lp.down_proj, hidden_, down.data());

    // h = post_feedforward_layernorm(mlp_out); x = residual + h
    rms_norm_cpu(down.data(), lp.ln_post_ffn, down.data(), S, hidden_, static_cast<float>(eps_));
    for (size_t i = 0; i < x.size(); ++i) x[i] += down[i];
  }

  // final RMSNorm
  rms_norm_cpu(x.data(), w_norm_, x.data(), S, hidden_, static_cast<float>(eps_));

  // masked mean pool, include_prompt=true (no special-casing of the
  // task-prefix tokens) -- same 1e-9 denominator clamp as the reference and
  // this project's BERT pool_rows().
  std::vector<double> pooled(static_cast<size_t>(hidden_), 0.0);
  double denom = 0.0;
  for (int64_t s = 0; s < S; ++s) {
    const float m = static_cast<float>(en.attention_mask[static_cast<size_t>(s)]);
    if (m == 0.f) continue;
    denom += m;
    const float *row = x.data() + s * hidden_;
    for (int64_t c = 0; c < hidden_; ++c) pooled[static_cast<size_t>(c)] += row[c] * m;
  }
  denom = std::max(denom, 1e-9);
  std::vector<float> pooled_f(static_cast<size_t>(hidden_));
  for (int64_t c = 0; c < hidden_; ++c)
    pooled_f[static_cast<size_t>(c)] = static_cast<float>(pooled[static_cast<size_t>(c)] / denom);

  // Dense(hidden->dense_hidden) -> Dense(dense_hidden->hidden), bias=false.
  std::vector<float> d2(static_cast<size_t>(dense_hidden_));
  gemm_f32(pooled_f.data(), 1, hidden_, w_dense2_, dense_hidden_, d2.data());
  std::vector<float> d3(static_cast<size_t>(hidden_));
  gemm_f32(d2.data(), 1, dense_hidden_, w_dense3_, hidden_, d3.data());

  double nrm = 0.0;
  for (int64_t c = 0; c < hidden_; ++c)
    nrm += static_cast<double>(d3[static_cast<size_t>(c)]) * d3[static_cast<size_t>(c)];
  nrm = std::max(std::sqrt(nrm), 1e-12);
  std::vector<float> out(static_cast<size_t>(hidden_));
  for (int64_t c = 0; c < hidden_; ++c)
    out[static_cast<size_t>(c)] = static_cast<float>(d3[static_cast<size_t>(c)] / nrm);
  return out;
}

}  // namespace npue
