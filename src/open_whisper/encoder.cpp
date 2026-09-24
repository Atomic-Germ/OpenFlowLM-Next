//===- encoder.cpp -------------------------------------------*- C++ -*-===//
// open_whisper -- see encoder.hpp. SPDX-License-Identifier: MIT
#include "encoder.hpp"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

#include "host_ops.hpp"

namespace ow {
namespace {

// OW_FA_DIR overrides where the FlashAttention kernel is looked for; unset,
// it is <kernels_dir>/fa -- next to the GEMM kernel set, the same way
// whisper_kernels.json lives beside design.json.
std::string fa_kernel_dir(const std::string &kernels_dir) {
  const char *e = std::getenv("OW_FA_DIR");
  if (e && *e) return std::string(e);
  return kernels_dir + "/fa";
}
double now_s() {
  return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}
constexpr int64_t D = Geometry::d_model;
constexpr int64_t FFN = Geometry::ffn;
constexpr int64_t H = Geometry::n_heads;
constexpr int64_t HD = Geometry::head_dim;
constexpr int64_t T = Geometry::max_src_pos;   // 1500 real rows
constexpr int64_t M = 1536;                    // padded M every layer GEMM shares
constexpr int64_t NDEC = Geometry::n_dec;
}  // namespace

Encoder::Encoder(const std::string &model_dir, const std::string &kernels_dir_hint) {
  const std::string kdir = KernelSet::resolve_dir(kernels_dir_hint, model_dir);
  const KernelSet::BLayout bl = KernelSet::read_b_layout(kdir);
  std::printf("  b_layout   tile_k=%lld tile_n=%lld mac=%lld/%lld\n",
             (long long)bl.tile_k, (long long)bl.tile_n, (long long)bl.mac_s, (long long)bl.mac_t);

  weights_ = std::make_unique<Weights>(model_dir, bl.tile_k, bl.tile_n, bl.mac_s, bl.mac_t);
  device_ = std::make_unique<npue::npu::Device>();
  kernels_ = std::make_unique<KernelSet>(*device_, kdir, model_dir, bl.tile_k, bl.tile_n,
                                        bl.mac_s, bl.mac_t);

  auto stage = [&](const std::vector<uint16_t> &tiled, int64_t K, int64_t N) {
    return kernels_->stage_b(tiled.data(), static_cast<size_t>(K) * static_cast<size_t>(N));
  };
  conv1_slot_ = stage(weights_->conv1_B, 3 * Geometry::n_mel, D);
  conv2_slot_ = stage(weights_->conv2_B, 3 * D, D);
  xkv_slot_ = stage(weights_->xkv_B, D, 2 * NDEC * D);
  layer_slots_.resize(static_cast<size_t>(Geometry::n_enc));
  for (int64_t i = 0; i < Geometry::n_enc; ++i) {
    auto &L = weights_->layers[static_cast<size_t>(i)];
    auto &S = layer_slots_[static_cast<size_t>(i)];
    S.qkv = stage(L.qkv_B, D, 3 * D);
    S.o = stage(L.o_B, D, D);
    S.fc1 = stage(L.fc1_B, D, FFN);
    S.fc2 = stage(L.fc2_B, FFN, D);
  }
  std::printf("  staged     %zu weight buffers on the device\n",
             3 + layer_slots_.size() * 4);

  // OW_ATTN: "auto" (default, unset) uses the NPU FlashAttention kernel when
  // one is found next to this kernel set AND fa.json parses and names this
  // engine's own geometry; "npu" requires all of that too, refusing rather
  // than falling back if it does not hold; "host" never uses it. Never silent
  // either way -- the chosen mode and the reason are always printed
  // (CLAUDE.md rule 8's class: a choice nothing prints is as good as unmade).
  //
  // fa_kernel_usable() (not the bare file-existence fa_kernel_present()) is
  // what decides this: a stale or mismatched fa/ -- malformed fa.json, or one
  // built for a different geometry -- must make auto fall back to host, not
  // throw here when FaAttention's own constructor re-reads and re-checks the
  // same file (PR #111 review).
  const AttnMode attn_mode = parse_attn_mode(std::getenv("OW_ATTN"));
  const std::string fa_dir = fa_kernel_dir(kdir);
  std::string fa_reason;
  const bool fa_ok = fa_kernel_usable(fa_dir, fa_reason);
  use_fa_attn_ = resolve_use_npu_attn(attn_mode, fa_ok, fa_dir, fa_reason);
  const char *env = std::getenv("OW_ATTN");
  const std::string source = (env && *env) ? (std::string("OW_ATTN=") + env) : "default (auto)";
  if (use_fa_attn_) {
    attn_summary_ = "npu, " + fa_dir + " (" + source + ")";
    std::printf("  attention  NPU FlashAttention, %s (%s)\n", fa_dir.c_str(), source.c_str());
    fa_attn_ = std::make_unique<FaAttention>(*device_, fa_dir);
  } else {
    // attn_mode == Npu with !fa_ok already threw inside resolve_use_npu_attn(),
    // so reaching here means either "host" was requested, or "auto" found no
    // usable kernel at fa_dir -- fa_reason names why (missing files, bad JSON,
    // or the specific geometry mismatch).
    const std::string why = attn_mode == AttnMode::Auto ? fa_reason : source;
    attn_summary_ = "host (" + why + ")";
    std::printf("  attention  host (%s)\n", why.c_str());
  }

  // OW_HOST_FAST (host_ops.cpp): validated here too, at construction, so a
  // bad value is caught before the first layer runs -- run_layer()'s own
  // function-local static re-reads it (cheap, deterministic, side-effect-
  // free besides the same throw) and is left untouched.
  host_fast_ = host_fast_enabled();
  {
    const char *hf_env = std::getenv("OW_HOST_FAST");
    const std::string hf_source =
        (hf_env && *hf_env) ? (std::string("OW_HOST_FAST=") + hf_env) : "default";
    host_fast_summary_ = std::string(host_fast_ ? "fast" : "exact") + " (" + hf_source + ")";
    std::printf("  host ops   %s\n", host_fast_summary_.c_str());
  }
}

void Encoder::run_layer(int64_t layer, float *x, int64_t real_rows, int64_t m_padded) {
  const auto &W = weights_->layers[static_cast<size_t>(layer)];
  const auto &S = layer_slots_[static_cast<size_t>(layer)];

  const size_t n_d = static_cast<size_t>(m_padded) * static_cast<size_t>(D);
  s_h_.resize(n_d);
  s_qkv_.resize(n_d * 3);
  s_attn_.resize(n_d);
  s_o_out_.resize(n_d);
  s_fc2_out_.resize(n_d);
  s_a_bf_.resize(n_d);
  s_fc1_h_.resize(static_cast<size_t>(m_padded) * static_cast<size_t>(FFN));
  s_a_bf2_.resize(static_cast<size_t>(m_padded) * static_cast<size_t>(FFN));
  s_attn_scratch_.resize(static_cast<size_t>(3) * static_cast<size_t>(real_rows) *
                        static_cast<size_t>(D));
  std::vector<float> &h = s_h_;
  std::vector<uint16_t> &a_bf = s_a_bf_;
  std::vector<float> &qkv = s_qkv_;
  std::vector<float> &attn = s_attn_;
  std::vector<float> &o_out = s_o_out_;
  std::vector<float> &fc1_h = s_fc1_h_;
  std::vector<uint16_t> &a_bf2 = s_a_bf2_;
  std::vector<float> &fc2_out = s_fc2_out_;

  // task 0180: OW_HOST_FAST=1 fuses several of the passes below (strict
  // parsing, see host_ops.cpp) -- the exact path above is left untouched.
  // The MEMBER, set once at construction (host_fast_enabled() validated
  // there -- see the constructor comment above), not a second
  // function-local re-read: a function-local static would give run_layer()
  // its own private copy of "was OW_HOST_FAST valid", a second source of
  // truth that could only ever agree with the constructor's by accident
  // (PR #111 review, finding G).
  const bool fast = host_fast_;

  double t0 = now_s();
  if (fast) {
    // LN1 fused with the bf16 round Qkv's A needs -- no fp32 `h` write.
    layer_norm_bf16_fast(x, W.ln1_w.data(), W.ln1_b.data(), m_padded, D, a_bf.data());
  } else {
    layer_norm(x, W.ln1_w.data(), W.ln1_b.data(), m_padded, D, h.data());
  }
  timers.layer_norm += now_s() - t0;

  // qkv = gemm(h, qkv.B) + qkv.bias
  if (!fast) {
    t0 = now_s();
    bf16_fill(a_bf.data(), h.data(), a_bf.size());
    timers.bf16 += now_s() - t0;
  }
  const float *qkv_c = kernels_->run(Op::Qkv, a_bf.data(), S.qkv, &timers.npu_in,
                                     &timers.npu_disp_op[static_cast<size_t>(Op::Qkv)], &timers.npu_out);
  if (!fast) {
    std::memcpy(qkv.data(), qkv_c, qkv.size() * sizeof(float));
    t0 = now_s();
    add_bias(qkv.data(), W.qkv_bias.data(), m_padded, 3 * D);
    timers.bias += now_s() - t0;
  }

  // attention, then x += gemm(attn, o.B) + o.bias
  t0 = now_s();
  static const bool phase_split = [] {
    const char *e = std::getenv("OW_ATTN_PHASES");
    return e && *e && *e != '0';
  }();
  if (use_fa_attn_) {
    // NPU FA path. OW_HOST_FAST=1 now reaches this path too: run_fast() fuses
    // the qkv bias add into FaAttention's own repack, reading the GEMM C
    // buffer once instead of the standalone memcpy + add_bias() pass -- no
    // fp32 `qkv` copy is made for this branch at all. The exact (!fast) path
    // is unchanged: the top-of-function `!fast` block above already built the
    // biased fp32 `qkv`, so run() here just uses it -- adding the bias again
    // would double it.
    if (fast) {
      fa_attn_->run_fast(qkv_c, m_padded, real_rows, D, H, HD, W.qkv_bias.data(), attn.data(),
                        &timers.fa_phases);
    } else {
      fa_attn_->run(qkv.data(), m_padded, real_rows, D, H, HD, attn.data(),
                   &timers.fa_phases);
    }
  } else if (fast) {
    // Fused bias-add + head gather, straight off the device C buffer --
    // skips the standalone `qkv` copy and add_bias() pass entirely.
    zero_pad_rows(attn.data(), real_rows, m_padded, D);
    attention_gather_bias_fast(qkv_c, m_padded, real_rows, D, H, HD, W.qkv_bias.data(),
                              s_attn_scratch_.data());
    attention_core(real_rows, D, H, HD, attn.data(), s_attn_scratch_.data(),
                  phase_split ? &timers.attn_phases : nullptr);
  } else {
    attention(qkv.data(), m_padded, real_rows, D, H, HD, attn.data(),
             s_attn_scratch_.data(), phase_split ? &timers.attn_phases : nullptr);
  }
  timers.attention += now_s() - t0;

  if (fast) {
    // bf16 round for O's A, fused into the attention output above would need
    // a second per-head write; done here as bf16_fill_parallel instead
    // (bit-identical to bf16_fill, just threaded).
    t0 = now_s();
    bf16_fill_parallel(a_bf.data(), attn.data(), a_bf.size());
    timers.bf16 += now_s() - t0;
  } else {
    t0 = now_s();
    bf16_fill(a_bf.data(), attn.data(), a_bf.size());
    timers.bf16 += now_s() - t0;
  }
  const float *o_c = kernels_->run(Op::O, a_bf.data(), S.o, &timers.npu_in,
                                   &timers.npu_disp_op[static_cast<size_t>(Op::O)], &timers.npu_out);
  if (fast) {
    // Fused memcpy + bias + residual: reads the device C buffer ONCE
    // (read-only, trap 27) and writes x once, in the same op order as the
    // three-pass exact path (c+bias, then +x) -- bit-identical, just fewer
    // memory round trips. zero_pad_rows unchanged: padded rows still get
    // discarded after, exactly as the exact path leaves them.
    t0 = now_s();
    add_bias_residual_fast(o_c, W.o_bias.data(), m_padded, D, x);
    zero_pad_rows(x, real_rows, m_padded, D);
    timers.residual += now_s() - t0;
  } else {
    t0 = now_s();
    std::memcpy(o_out.data(), o_c, o_out.size() * sizeof(float));
    add_bias(o_out.data(), W.o_bias.data(), m_padded, D);
    timers.bias += now_s() - t0;
    t0 = now_s();
    add_rows(x, o_out.data(), m_padded, D, x);
    zero_pad_rows(x, real_rows, m_padded, D);
    timers.residual += now_s() - t0;
  }

  // h = LN2(x); fc1 -> GELU -> fc2; x += fc2_out + bias
  t0 = now_s();
  if (fast) {
    layer_norm_bf16_fast(x, W.ln2_w.data(), W.ln2_b.data(), m_padded, D, a_bf.data());
  } else {
    layer_norm(x, W.ln2_w.data(), W.ln2_b.data(), m_padded, D, h.data());
  }
  timers.layer_norm += now_s() - t0;
  if (!fast) {
    t0 = now_s();
    bf16_fill(a_bf.data(), h.data(), a_bf.size());
    timers.bf16 += now_s() - t0;
  }
  const float *fc1_c = kernels_->run(Op::Fc1, a_bf.data(), S.fc1, &timers.npu_in,
                                     &timers.npu_disp_op[static_cast<size_t>(Op::Fc1)], &timers.npu_out);
  if (fast) {
    // GELU + bias + bf16-round in ONE pass: the fp32 fc1_h intermediate
    // (31.5 MB) is never written or re-read. The C buffer is READ-ONLY here
    // on purpose -- see gelu_bias_bf16_fast() in host_ops.hpp.
    t0 = now_s();
    gelu_bias_bf16_fast(fc1_c, m_padded, FFN, W.fc1_bias.data(), a_bf2.data());
    timers.gelu += now_s() - t0;
  } else {
    t0 = now_s();
    // The C buffer is READ-ONLY here on purpose -- see gelu_bias() in host_ops.hpp.
    gelu_bias(fc1_c, m_padded, FFN, W.fc1_bias.data(), fc1_h.data());
    timers.gelu += now_s() - t0;
    t0 = now_s();
    bf16_fill(a_bf2.data(), fc1_h.data(), a_bf2.size());
    timers.bf16 += now_s() - t0;
  }
  const float *fc2_c = kernels_->run(Op::Fc2, a_bf2.data(), S.fc2, &timers.npu_in,
                                     &timers.npu_disp_op[static_cast<size_t>(Op::Fc2)], &timers.npu_out);
  if (fast) {
    t0 = now_s();
    add_bias_residual_fast(fc2_c, W.fc2_bias.data(), m_padded, D, x);
    zero_pad_rows(x, real_rows, m_padded, D);
    timers.residual += now_s() - t0;
  } else {
    std::memcpy(fc2_out.data(), fc2_c, fc2_out.size() * sizeof(float));
    t0 = now_s();
    add_bias(fc2_out.data(), W.fc2_bias.data(), m_padded, D);
    timers.bias += now_s() - t0;
    t0 = now_s();
    add_rows(x, fc2_out.data(), m_padded, D, x);
    zero_pad_rows(x, real_rows, m_padded, D);
    timers.residual += now_s() - t0;
  }
}

void Encoder::encode(const float *mel, const StageHook &hook) {
  const double t_start = now_s();
  // Every hook call is charged to timers.hook, never to the stage it follows.
  auto H = [&](const std::string &name, const float *d, int64_t r, int64_t c) {
    if (!hook) return;
    const double th = now_s();
    hook(name, d, r, c);
    timers.hook += now_s() - th;
  };
  const int64_t n_mel = Geometry::n_mel;

  // Stem: conv1 (M=3072, K=384) -> keep 3000 rows -> conv2 (M=1536, K=3840).
  std::vector<float> mel_tm(static_cast<size_t>(3000) * static_cast<size_t>(n_mel));
  double t0 = now_s();
#pragma omp parallel for schedule(static) num_threads(::ow::omp_threads())
  for (int64_t t = 0; t < 3000; ++t)
    for (int64_t c = 0; c < n_mel; ++c) mel_tm[static_cast<size_t>(t * n_mel + c)] = mel[c * 3000 + t];

  std::vector<float> a1(static_cast<size_t>(3072) * static_cast<size_t>(3 * n_mel));
  im2col(mel_tm.data(), 3000, n_mel, 1, 3072, a1.data());
  timers.im2col += now_s() - t0;

  std::vector<uint16_t> a1_bf(a1.size());
  t0 = now_s();
  bf16_fill(a1_bf.data(), a1.data(), a1.size());
  timers.bf16 += now_s() - t0;
  const float *c1 = kernels_->run(Op::Conv1, a1_bf.data(), conv1_slot_, &timers.npu_in,
                                  &timers.npu_disp_op[static_cast<size_t>(Op::Conv1)], &timers.npu_out);
  std::vector<float> h1(static_cast<size_t>(3000) * static_cast<size_t>(D));
  t0 = now_s();
  // Only the first 3000 rows are ever used downstream, and the C buffer is
  // read-only here -- see gelu_bias() in host_ops.hpp.
  gelu_bias(c1, 3000, D, weights_->conv1_bias.data(), h1.data());
  timers.gelu += now_s() - t0;
  H("conv1", h1.data(), 3000, D);

  std::vector<float> a2(static_cast<size_t>(M) * static_cast<size_t>(3 * D));
  t0 = now_s();
  im2col(h1.data(), 3000, D, 2, M, a2.data());
  timers.im2col += now_s() - t0;
  std::vector<uint16_t> a2_bf(a2.size());
  t0 = now_s();
  bf16_fill(a2_bf.data(), a2.data(), a2.size());
  timers.bf16 += now_s() - t0;
  const float *c2 = kernels_->run(Op::Conv2, a2_bf.data(), conv2_slot_, &timers.npu_in,
                                  &timers.npu_disp_op[static_cast<size_t>(Op::Conv2)], &timers.npu_out);
  std::vector<float> x(static_cast<size_t>(M) * static_cast<size_t>(D));
  t0 = now_s();
  gelu_bias(c2, M, D, weights_->conv2_bias.data(), x.data());
  // + positional embedding, rows [0, T) only -- there is no pos[T:].
  for (int64_t r = 0; r < T; ++r) {
    float *xr = x.data() + r * D;
    const float *pr = weights_->pos.data() + r * D;
    for (int64_t c = 0; c < D; ++c) xr[c] += pr[c];
  }
  zero_pad_rows(x.data(), T, M, D);
  timers.gelu += now_s() - t0;
  H("conv2", x.data(), T, D);   // == enc.hidden.0, the input to layer 0

  for (int64_t i = 0; i < Geometry::n_enc; ++i) {
    run_layer(i, x.data(), T, M);
    H("enc.hidden." + std::to_string(i + 1), x.data(), T, D);
  }

  enc_out_.assign(static_cast<size_t>(T) * static_cast<size_t>(D), 0.f);
  std::vector<float> out_full(static_cast<size_t>(M) * static_cast<size_t>(D));
  t0 = now_s();
  layer_norm(x.data(), weights_->ln_w.data(), weights_->ln_b.data(), M, D, out_full.data());
  timers.layer_norm += now_s() - t0;
  for (int64_t r = 0; r < T; ++r)
    std::memcpy(&enc_out_[static_cast<size_t>(r) * D], out_full.data() + r * D, D * sizeof(float));
  H("enc.out", enc_out_.data(), T, D);

  std::vector<uint16_t> out_bf(out_full.size());
  t0 = now_s();
  bf16_fill(out_bf.data(), out_full.data(), out_bf.size());
  timers.bf16 += now_s() - t0;
  const float *xkv_c = kernels_->run(Op::Xkv, out_bf.data(), xkv_slot_, &timers.npu_in,
                                     &timers.npu_disp_op[static_cast<size_t>(Op::Xkv)], &timers.npu_out);
  const int64_t xkv_n = 2 * NDEC * D;
  xkv_.assign(static_cast<size_t>(T) * static_cast<size_t>(xkv_n), 0.f);
  t0 = now_s();
  for (int64_t r = 0; r < T; ++r)
    std::memcpy(&xkv_[static_cast<size_t>(r) * xkv_n], xkv_c + r * xkv_n, xkv_n * sizeof(float));
  add_bias(xkv_.data(), weights_->xkv_bias.data(), T, xkv_n);
  timers.bias += now_s() - t0;

  if (hook) {
    const double th = now_s();
    for (int64_t l = 0; l < NDEC; ++l) {
      std::vector<float> k(static_cast<size_t>(T) * static_cast<size_t>(D));
      std::vector<float> v(static_cast<size_t>(T) * static_cast<size_t>(D));
      for (int64_t r = 0; r < T; ++r) {
        std::memcpy(&k[static_cast<size_t>(r) * D], &xkv_[static_cast<size_t>(r) * xkv_n + 2 * l * D],
                   D * sizeof(float));
        std::memcpy(&v[static_cast<size_t>(r) * D],
                   &xkv_[static_cast<size_t>(r) * xkv_n + (2 * l + 1) * D], D * sizeof(float));
      }
      // Golden's own names (whisper-goldens' dec.<l>.xk / dec.<l>.xv), so the
      // CLI's hook can look them up with no translation table.
      hook("dec." + std::to_string(l) + ".xk", k.data(), T, D);
      hook("dec." + std::to_string(l) + ".xv", v.data(), T, D);
    }
    timers.hook += now_s() - th;
  }

  timers.total += now_s() - t_start;
}

int Encoder::stress_qkv(int64_t reps, int64_t n_layers) {
  // Four ops in the order a layer runs them, so the loop also switches
  // instruction streams and A/C transfer sizes between dispatches -- which is
  // what a real encode() does and the 320-dispatch qkv-only version did not.
  const Op ops[4] = {Op::Qkv, Op::O, Op::Fc1, Op::Fc2};
  const int64_t a_k[4] = {D, D, D, FFN};
  const int64_t c_n[4] = {3 * D, D, FFN, D};
  std::vector<std::vector<uint16_t>> a_of(4);
  uint32_t s = 12345;
  for (int j = 0; j < 4; ++j) {
    a_of[j].resize(static_cast<size_t>(M) * static_cast<size_t>(a_k[j]));
    for (size_t i = 0; i < a_of[j].size(); ++i) {   // fixed pseudo-random bf16 pattern
      s = s * 1664525u + 1013904223u;
      a_of[j][i] = static_cast<uint16_t>(0x3B00 | ((s >> 17) & 0xFF));
    }
  }
  std::vector<std::vector<float>> first(static_cast<size_t>(n_layers) * 4);
  int mismatches = 0;
  for (int64_t r = 0; r < reps; ++r) {
    const int64_t layer = (r / 4) % n_layers;
    const int j = static_cast<int>(r % 4);
    const auto &S = layer_slots_[static_cast<size_t>(layer)];
    const size_t slot = j == 0 ? S.qkv : j == 1 ? S.o : j == 2 ? S.fc1 : S.fc2;
    const float *c = kernels_->run(ops[j], a_of[j].data(), slot, nullptr, nullptr, nullptr);
    const size_t n = static_cast<size_t>(M) * static_cast<size_t>(c_n[j]);
    auto &ref = first[static_cast<size_t>(layer) * 4 + j];
    if (ref.empty()) {
      ref.assign(c, c + n);
      continue;
    }
    for (size_t i = 0; i < n; ++i) {
      if (ref[i] != c[i]) {
        ++mismatches;
        std::printf("  rep %lld layer %lld: C differs at element %zu (row %zu col %zu): "
                    "%.6f vs %.6f\n", (long long)r, (long long)layer, i, i / (size_t)c_n[j],
                    i % (size_t)c_n[j], (double)ref[i], (double)c[i]);
        break;
      }
    }
  }
  std::printf("  stress: %lld dispatches over %lld B slots, %d differing from the "
              "layer's first result\n", (long long)reps, (long long)n_layers, mismatches);
  return mismatches;
}

void Encoder::run_layer_from(int64_t layer, const float *input_1500x1280, float *output) {
  std::vector<float> x(static_cast<size_t>(M) * static_cast<size_t>(D), 0.f);
  for (int64_t r = 0; r < T; ++r)
    std::memcpy(&x[static_cast<size_t>(r) * D], input_1500x1280 + r * D, D * sizeof(float));
  run_layer(layer, x.data(), T, M);
  for (int64_t r = 0; r < T; ++r)
    std::memcpy(output + r * D, &x[static_cast<size_t>(r) * D], D * sizeof(float));
}

}  // namespace ow
