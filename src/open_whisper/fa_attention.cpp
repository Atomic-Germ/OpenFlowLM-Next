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
#pragma omp parallel for schedule(static)
  for (int64_t idx = 0; idx < work; ++idx) {
    const int64_t h = idx / t, t1 = idx % t;
    const float *row = qkv + t1 * stride + h * hd;
    const size_t off = static_cast<size_t>(h) * per_head + static_cast<size_t>(t1) * hd;
    bf16_fill(q_out + off, row, static_cast<size_t>(hd));
    bf16_fill(k_out + off, row + d, static_cast<size_t>(hd));
    bf16_fill(v_out + off, row + 2 * d, static_cast<size_t>(hd));
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
#pragma omp parallel for schedule(static)
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

void FaAttention::run(const float *qkv, int64_t m_padded, int64_t t, int64_t d,
                      int64_t heads, int64_t head_dim, float *out, FaPhases *phases) {
  constexpr int64_t kHeads = 20, kHeadDim = 64, kSeqPad = 1536;
  if (heads != kHeads || head_dim != kHeadDim || m_padded != kSeqPad)
    throw std::runtime_error(
        "FaAttention::run: shape " + std::to_string(heads) + "/" +
        std::to_string(head_dim) + "/" + std::to_string(m_padded) +
        " does not match the kernel's fixed 20/64/1536");

  double t0 = now_s();
  repack_qkv(qkv, m_padded, kSeqPad, t, d, heads, head_dim, q_bf_.data(),
            k_bf_.data(), v_bf_.data());
  if (phases) phases->repack += now_s() - t0;

  t0 = now_s();
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

}  // namespace ow
