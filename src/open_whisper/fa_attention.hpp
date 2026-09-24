//===- fa_attention.hpp --------------------------------------*- C++ -*-===//
//
// open_whisper -- OPTIONAL bidirectional attention on the NPU, via a fused
// FlashAttention kernel -- open_kernels/designs/whisper_fa (an IRON port of
// AMD's MLIR-AIR flash_attention/kernel_fusion_based example, built by
// open_kernels/export_whisper_kernels.py into <kernels_dir>/fa/) or an
// equivalent -- as an xclbin+insts.bin pair that carries no
// design.json of this project's own -- only fa.json (see fa_guards.hpp),
// which records what the build actually was rather than this engine
// assuming it, and is checked, not trusted.
//
// OW_ATTN selects: "auto" (default) uses it when found, else host; "npu"
// requires it, refusing otherwise; "host" never uses it. The kernel is
// found at <kernels_dir>/fa/ (air.xclbin, air.insts.bin, fa.json) unless
// OW_FA_DIR overrides the directory. See fa_guards.hpp for the parsing and
// the geometry guard, and encoder.cpp for where OW_ATTN is resolved.
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

#include "fa_guards.hpp"
#include "npu_device.hpp"

namespace ow {

// True iff `fa_dir` holds all three files a FlashAttention kernel needs
// (air.xclbin, air.insts.bin, fa.json) -- an existence probe only, used by
// OW_ATTN=auto/npu's resolution in encoder.cpp before committing to either
// path. Does not open or parse any of them.
bool fa_kernel_present(const std::string &fa_dir);

// Same shape as host_ops.hpp's AttnPhases, for the NPU path's own three
// stages: repacking Q/K/V into the kernel's layout, the dispatch itself
// (submit+wait, host-observed -- rule 1: never an NPU performance claim on
// its own), and reading O back and scattering it into the host layout.
struct FaPhases {
  double repack = 0, dispatch = 0, scatter = 0;
};

class FaAttention {
public:
  // `fa_dir` must hold air.xclbin, air.insts.bin and fa.json, at the fixed
  // shape documented above (H=20, dk=dv=64, lq=lk=1536, valid_len=1500).
  // Throws if any file is missing, if fa.json is malformed or incomplete
  // (fa_guards.hpp's read_fa_kernel_info), or if fa.json's own geometry does
  // not match this engine's (check_fa_geometry) -- never assumed from the
  // fixed shape alone.
  FaAttention(npue::npu::Device &dev, const std::string &fa_dir);

  // The parsed fa.json this instance was built from -- what the build
  // actually was, for the startup summary (engine_adapter.cpp).
  const FaKernelInfo &kernel_info() const { return info_; }

  // Same signature and same contract as host_ops.hpp's attention(): `qkv` is
  // [m_padded, 3*d] fp32 row-major (Q|K|V blocks of `d` columns each, head h
  // at columns h*head_dim..h*head_dim+head_dim within a block), `out` is
  // [m_padded, d] fp32 with rows [t, m_padded) zeroed on return. `m_padded`
  // and `t` must equal the kernel's own lq (1536) and valid_len (1500) --
  // checked, not assumed.
  void run(const float *qkv, int64_t m_padded, int64_t t, int64_t d,
          int64_t heads, int64_t head_dim, float *out, FaPhases *phases = nullptr);

  // task 0180 Part A: OW_HOST_FAST=1's path. Same contract as run(), except
  // `qkv_c` is the qkv GEMM's device C buffer BEFORE the bias add (read-only,
  // trap 27) and `bias` is the qkv bias to fuse in -- the caller no longer
  // needs its own biased fp32 `qkv` copy at all for this path. Expected and
  // tested bit-identical to run() called on add_bias(qkv_c).
  void run_fast(const float *qkv_c, int64_t m_padded, int64_t t, int64_t d,
               int64_t heads, int64_t head_dim, const float *bias, float *out,
               FaPhases *phases = nullptr);

  const std::string &xclbin_path() const { return xclbin_path_; }
  size_t xclbin_bytes() const { return xclbin_bytes_; }
  // FNV-1a 64-bit over the raw xclbin bytes, hex string. Not a cryptographic
  // hash -- just enough to prove which file on disk was actually loaded,
  // never the intention (CLAUDE.md: "report the value you read").
  const std::string &xclbin_fnv1a() const { return xclbin_hash_; }

private:
  // Shared tail of run()/run_fast(): upload q_bf_/k_bf_/v_bf_, dispatch, read
  // o_bf_ back and scatter it into `out`. Both callers have already filled
  // q_bf_/k_bf_/v_bf_ by the time this runs.
  void dispatch_and_scatter(int64_t m_padded, int64_t t, int64_t d, int64_t heads,
                            int64_t head_dim, float *out, FaPhases *phases);

  std::unique_ptr<npue::npu::Design> design_;
  std::string xclbin_path_;
  size_t xclbin_bytes_ = 0;
  std::string xclbin_hash_;
  FaKernelInfo info_;

  // Repack/scatter scratch, sized once at construction and reused across
  // layers -- same reasoning as Encoder's own s_* scratch members.
  std::vector<uint16_t> q_bf_, k_bf_, v_bf_, o_bf_;
};

}  // namespace ow
