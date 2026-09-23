//===- fa_attention.cpp --------------------------------------*- C++ -*-===//
// open_whisper -- see fa_attention.hpp. SPDX-License-Identifier: MIT
#include "fa_attention.hpp"

#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>

#if defined(_OPENMP)
#include <omp.h>
#endif

#if defined(__AVX2__)
#include <immintrin.h>
#endif

#include "host_ops.hpp"

namespace ow {
namespace {

double now_s() {
  return std::chrono::duration<double>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

bool file_exists(const std::string &path) {
  std::ifstream f(path, std::ios::binary);
  return static_cast<bool>(f);
}

size_t file_size(const std::string &path) {
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  if (!f) throw std::runtime_error("cannot open " + path);
  return static_cast<size_t>(f.tellg());
}

// FNV-1a 64-bit -- see fa_attention.hpp's xclbin_fnv1a() comment: not
// cryptographic, just enough to name the exact bytes that were loaded.
std::string fnv1a_hex(const std::string &path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("cannot open " + path);
  uint64_t h = 1469598103934665603ull;
  char buf[65536];
  while (f.read(buf, sizeof buf) || f.gcount() > 0) {
    const std::streamsize n = f.gcount();
    for (std::streamsize i = 0; i < n; ++i) {
      h ^= static_cast<uint8_t>(buf[i]);
      h *= 1099511628211ull;
    }
    if (!f) break;
  }
  char hex[17];
  std::snprintf(hex, sizeof hex, "%016llx", static_cast<unsigned long long>(h));
  return std::string(hex);
}

// Repack open_whisper's [m_padded, 3*d] fp32 qkv (Q|K|V blocks of `d`
// columns, head h at columns h*hd..h*hd+hd within a block -- see
// host_ops.cpp's attention(), whose Q()/K()/V() lambdas read the identical
// offsets) into the FA kernel's head-first [heads][seq_pad][hd] bf16 layout,
// bf16-rounded with the SAME round-to-nearest-even as the rest of this
// engine (ow::bf16_fill). Rows [t, seq_pad) of all three are zeroed: Q's pad
// rows produce garbage output rows that are simply never scattered back: K's
// are masked off by the kernel's own apply_length_mask (valid_len=1500,
// compiled in); V's must be zero so a masked pad key contributes nothing
// even if softmax mass ever leaked onto it. Parallel over (head, row), like
// attention()'s own gather.
void repack_qkv(const float *qkv, int64_t m_padded, int64_t seq_pad, int64_t t,
                int64_t d, int64_t heads, int64_t hd, uint16_t *q_out,
                uint16_t *k_out, uint16_t *v_out) {
  const int64_t stride = 3 * d;
  const size_t per_head = static_cast<size_t>(seq_pad) * static_cast<size_t>(hd);
  std::memset(q_out, 0, static_cast<size_t>(heads) * per_head * sizeof(uint16_t));
  std::memset(k_out, 0, static_cast<size_t>(heads) * per_head * sizeof(uint16_t));
  std::memset(v_out, 0, static_cast<size_t>(heads) * per_head * sizeof(uint16_t));
  (void)m_padded;  // == seq_pad, asserted by the caller

  const int64_t work = heads * t;
#pragma omp parallel for schedule(static) num_threads(::ow::omp_threads())
  for (int64_t idx = 0; idx < work; ++idx) {
    const int64_t h = idx / t, t1 = idx % t;
    const float *row = qkv + t1 * stride + h * hd;
    const size_t off = static_cast<size_t>(h) * per_head + static_cast<size_t>(t1) * hd;
    bf16_fill(q_out + off, row, static_cast<size_t>(hd));
    bf16_fill(k_out + off, row + d, static_cast<size_t>(hd));
    bf16_fill(v_out + off, row + 2 * d, static_cast<size_t>(hd));
  }
}

// task 0180 Part A: fused bias-add + repack for OW_HOST_FAST=1. Same layout
// and same zero-padding contract as repack_qkv above, except `qkv` is now the
// GEMM's device C buffer -- READ ONLY (trap 27) -- and does NOT have the qkv
// bias added yet. The per-element operation (row value + bias[col]) is
// computed in fp32 with the SAME two operands in the SAME order add_bias()
// would use (see host_ops.cpp's attention_gather_bias_fast, whose header
// makes the identical claim for the host-attention gather), immediately
// rounded to bf16 with the same RNE bf16_fill used everywhere else in this
// engine -- so this is expected to be, and is tested for, BIT-IDENTICAL to
// add_bias(qkv) followed by repack_qkv() above. It replaces THREE passes over
// the 1536*3840 fp32 qkv tensor (memcpy C->scratch, scratch+=bias, then
// repack_qkv's own read) with one pass that reads `qkv` once and never
// materialises the fp32 [m_padded, 3*d] buffer at all.
void repack_qkv_bias_fast(const float *qkv, int64_t m_padded, int64_t seq_pad, int64_t t,
                          int64_t d, int64_t heads, int64_t hd, const float *bias,
                          uint16_t *q_out, uint16_t *k_out, uint16_t *v_out) {
  const int64_t stride = 3 * d;
  const size_t per_head = static_cast<size_t>(seq_pad) * static_cast<size_t>(hd);
  std::memset(q_out, 0, static_cast<size_t>(heads) * per_head * sizeof(uint16_t));
  std::memset(k_out, 0, static_cast<size_t>(heads) * per_head * sizeof(uint16_t));
  std::memset(v_out, 0, static_cast<size_t>(heads) * per_head * sizeof(uint16_t));
  (void)m_padded;  // == seq_pad, asserted by the caller

  const int64_t work = heads * t;
#pragma omp parallel for schedule(static) num_threads(::ow::omp_threads())
  for (int64_t idx = 0; idx < work; ++idx) {
    const int64_t h = idx / t, t1 = idx % t;
    const float *row = qkv + t1 * stride + h * hd;
    const float *bq = bias + h * hd, *bk = bias + d + h * hd, *bv = bias + 2 * d + h * hd;
    float qb[256], kb[256], vb[256];  // hd is <= 256 for every geometry this engine ships
    int64_t c = 0;
#if defined(__AVX2__)
    for (; c + 8 <= hd; c += 8) {
      _mm256_storeu_ps(qb + c, _mm256_add_ps(_mm256_loadu_ps(row + c), _mm256_loadu_ps(bq + c)));
      _mm256_storeu_ps(kb + c, _mm256_add_ps(_mm256_loadu_ps(row + d + c), _mm256_loadu_ps(bk + c)));
      _mm256_storeu_ps(vb + c, _mm256_add_ps(_mm256_loadu_ps(row + 2 * d + c), _mm256_loadu_ps(bv + c)));
    }
#endif
    for (; c < hd; ++c) {
      qb[c] = row[c] + bq[c];
      kb[c] = row[d + c] + bk[c];
      vb[c] = row[2 * d + c] + bv[c];
    }
    const size_t off = static_cast<size_t>(h) * per_head + static_cast<size_t>(t1) * hd;
    bf16_fill(q_out + off, qb, static_cast<size_t>(hd));
    bf16_fill(k_out + off, kb, static_cast<size_t>(hd));
    bf16_fill(v_out + off, vb, static_cast<size_t>(hd));
  }
}

// The inverse of repack_qkv for the kernel's O buffer: head-first
// [heads][seq_pad][hd] bf16 -> open_whisper's [m_padded, d] fp32, exactly the
// layout host_ops.cpp's attention() writes
// (`out + (q0+qi)*d + h*hd`). Only rows [0, t) are read from the kernel
// output -- the caller zero-pads [t, m_padded) afterward, matching
// attention()'s own contract.
void scatter_output(const uint16_t *o_bf, int64_t seq_pad, int64_t t, int64_t d,
                    int64_t heads, int64_t hd, float *out) {
  const size_t per_head = static_cast<size_t>(seq_pad) * static_cast<size_t>(hd);
  const int64_t work = heads * t;
#pragma omp parallel for schedule(static) num_threads(::ow::omp_threads())
  for (int64_t idx = 0; idx < work; ++idx) {
    const int64_t h = idx / t, t1 = idx % t;
    const size_t off = static_cast<size_t>(h) * per_head + static_cast<size_t>(t1) * hd;
    bf16_read(out + t1 * d + h * hd, o_bf + off, static_cast<size_t>(hd));
  }
}

}  // namespace

FaAttention::FaAttention(npue::npu::Device &dev, const std::string &fa_dir) {
  xclbin_path_ = fa_dir + "/air.xclbin";
  const std::string insts_path = fa_dir + "/air.insts.bin";
  if (!file_exists(xclbin_path_))
    throw std::runtime_error("OW_FA_DIR: missing " + xclbin_path_);
  if (!file_exists(insts_path))
    throw std::runtime_error("OW_FA_DIR: missing " + insts_path);
  xclbin_bytes_ = file_size(xclbin_path_);
  xclbin_hash_ = fnv1a_hex(xclbin_path_);

  // Fixed shape: H=20, dk=dv=64, lq=lk=1536 -- Geometry::n_heads/head_dim and
  // the M every encoder layer GEMM already shares. All four buffers (Q, K, V,
  // O) are the same size at this shape: heads * seq_pad * head_dim bf16.
  constexpr int64_t kHeads = 20, kHeadDim = 64, kSeqPad = 1536;
  const size_t buf_bytes =
      static_cast<size_t>(kHeads) * static_cast<size_t>(kSeqPad) *
      static_cast<size_t>(kHeadDim) * sizeof(uint16_t);
  const std::vector<size_t> buffers = {buf_bytes, buf_bytes, buf_bytes, buf_bytes};
  design_ = std::make_unique<npue::npu::Design>(dev, xclbin_path_, insts_path, buffers,
                                                "MLIR_AIE");

  const size_t n = static_cast<size_t>(kHeads) * static_cast<size_t>(kSeqPad) *
                   static_cast<size_t>(kHeadDim);
  q_bf_.resize(n);
  k_bf_.resize(n);
  v_bf_.resize(n);
  o_bf_.resize(n);

  std::printf("  fa attn    %s (%zu B, fnv1a %s)\n", xclbin_path_.c_str(),
             xclbin_bytes_, xclbin_hash_.c_str());
}

namespace {
void check_shape(const char *who, int64_t heads, int64_t head_dim, int64_t m_padded) {
  constexpr int64_t kHeads = 20, kHeadDim = 64, kSeqPad = 1536;
  if (heads != kHeads || head_dim != kHeadDim || m_padded != kSeqPad)
    throw std::runtime_error(std::string(who) + ": shape " + std::to_string(heads) + "/" +
                             std::to_string(head_dim) + "/" + std::to_string(m_padded) +
                             " does not match the kernel's fixed 20/64/1536");
}
}  // namespace

void FaAttention::dispatch_and_scatter(int64_t m_padded, int64_t t, int64_t d, int64_t heads,
                                       int64_t head_dim, float *out, FaPhases *phases) {
  constexpr int64_t kSeqPad = 1536;
  double t0 = now_s();
  std::memcpy(design_->host_ptr(0), q_bf_.data(), q_bf_.size() * sizeof(uint16_t));
  design_->sync_to_device(0);
  std::memcpy(design_->host_ptr(1), k_bf_.data(), k_bf_.size() * sizeof(uint16_t));
  design_->sync_to_device(1);
  std::memcpy(design_->host_ptr(2), v_bf_.data(), v_bf_.size() * sizeof(uint16_t));
  design_->sync_to_device(2);
  design_->dispatch_only();
  if (phases) phases->dispatch += now_s() - t0;

  t0 = now_s();
  design_->sync_from_device(3);
  std::memcpy(o_bf_.data(), design_->host_ptr(3), o_bf_.size() * sizeof(uint16_t));
  scatter_output(o_bf_.data(), kSeqPad, t, d, heads, head_dim, out);
  zero_pad_rows(out, t, m_padded, d);
  if (phases) phases->scatter += now_s() - t0;
}

void FaAttention::run(const float *qkv, int64_t m_padded, int64_t t, int64_t d,
                      int64_t heads, int64_t head_dim, float *out, FaPhases *phases) {
  check_shape("FaAttention::run", heads, head_dim, m_padded);
  constexpr int64_t kSeqPad = 1536;

  double t0 = now_s();
  repack_qkv(qkv, m_padded, kSeqPad, t, d, heads, head_dim, q_bf_.data(),
            k_bf_.data(), v_bf_.data());
  if (phases) phases->repack += now_s() - t0;

  dispatch_and_scatter(m_padded, t, d, heads, head_dim, out, phases);
}

// task 0180 Part A (OW_HOST_FAST=1): same contract as run(), except `qkv_c`
// is the qkv GEMM's device C buffer BEFORE the bias add (read-only, trap 27)
// -- the standalone host `qkv` copy and its separate add_bias() pass that the
// exact path (and run(), above) both need are skipped entirely; the bias add,
// the bf16 round and the head-first repack happen in one pass over `qkv_c`
// (repack_qkv_bias_fast, above), expected and tested bit-identical to
// add_bias(qkv) + repack_qkv().
void FaAttention::run_fast(const float *qkv_c, int64_t m_padded, int64_t t, int64_t d,
                           int64_t heads, int64_t head_dim, const float *bias, float *out,
                           FaPhases *phases) {
  check_shape("FaAttention::run_fast", heads, head_dim, m_padded);
  constexpr int64_t kSeqPad = 1536;

  double t0 = now_s();
  repack_qkv_bias_fast(qkv_c, m_padded, kSeqPad, t, d, heads, head_dim, bias, q_bf_.data(),
                       k_bf_.data(), v_bf_.data());
  if (phases) phases->repack += now_s() - t0;

  dispatch_and_scatter(m_padded, t, d, heads, head_dim, out, phases);
}

}  // namespace ow
