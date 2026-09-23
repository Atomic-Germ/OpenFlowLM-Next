//===- fa_attention.hpp --------------------------------------*- C++ -*-===//
//
// open_whisper -- OPTIONAL bidirectional attention on the NPU, via AMD's
// MLIR-AIR fused FlashAttention example (externalrepos/mlir-air,
// programming_examples/flash_attention/kernel_fusion_based_whisper), built
// separately with its own toolchain (C:\air\airenv) into an xclbin+insts.bin
// pair that carries no design.json of this project's own. Selected by
// OW_ATTN=npu (default is host, host_ops.cpp's attention()); OW_FA_DIR names
// the directory holding air.xclbin + air.insts.bin.
//
// The kernel is fixed-shape: H=20, dk=dv=64, lq=lk=1536, non-causal, built
// for Whisper's exact geometry (Geometry::n_heads/head_dim/max_src_pos, and
// M=1536 every encoder layer GEMM already shares) -- there is no retiling
// here, only a repack from open_whisper's own [m_padded, 3*d] qkv layout into
// the kernel's head-first [H][seq_pad][head_dim] bf16 buffers, and back.
//
// SPDX-License-Identifier: MIT
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "npu_device.hpp"

namespace ow {

// Same shape as host_ops.hpp's AttnPhases, for the NPU path's own three
// stages: repacking Q/K/V into the kernel's layout, the dispatch itself
// (submit+wait, host-observed -- rule 1: never an NPU performance claim on
// its own), and reading O back and scattering it into the host layout.
struct FaPhases {
  double repack = 0, dispatch = 0, scatter = 0;
};

class FaAttention {
public:
  // `fa_dir` must hold air.xclbin and air.insts.bin, at the fixed shape
  // documented above (H=20, dk=dv=64, lq=lk=1536). Throws if either file is
  // missing or the buffer count the xclbin exposes is not 4 (Q, K, V, O).
  FaAttention(npue::npu::Device &dev, const std::string &fa_dir);

  // Same signature and same contract as host_ops.hpp's attention(): `qkv` is
  // [m_padded, 3*d] fp32 row-major (Q|K|V blocks of `d` columns each, head h
  // at columns h*head_dim..h*head_dim+head_dim within a block), `out` is
  // [m_padded, d] fp32 with rows [t, m_padded) zeroed on return. `m_padded`
  // and `t` must equal the kernel's own lq (1536) and valid_len (1500) --
  // checked, not assumed.
  void run(const float *qkv, int64_t m_padded, int64_t t, int64_t d,
          int64_t heads, int64_t head_dim, float *out, FaPhases *phases = nullptr);

  const std::string &xclbin_path() const { return xclbin_path_; }
  size_t xclbin_bytes() const { return xclbin_bytes_; }
  // FNV-1a 64-bit over the raw xclbin bytes, hex string. Not a cryptographic
  // hash -- just enough to prove which file on disk was actually loaded,
  // never the intention (CLAUDE.md: "report the value you read").
  const std::string &xclbin_fnv1a() const { return xclbin_hash_; }

private:
  std::unique_ptr<npue::npu::Design> design_;
  std::string xclbin_path_;
  size_t xclbin_bytes_ = 0;
  std::string xclbin_hash_;

  // Repack/scatter scratch, sized once at construction and reused across
  // layers -- same reasoning as Encoder's own s_* scratch members.
  std::vector<uint16_t> q_bf_, k_bf_, v_bf_, o_bf_;
};

}  // namespace ow
