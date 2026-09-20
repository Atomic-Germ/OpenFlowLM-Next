//===- kernels.hpp -------------------------------------------*- C++ -*-===//
//
// open_whisper -- locate and validate the whisper_gemm kernel set (one
// xclbin, seven instruction streams, over ONE hw_context) and drive it.
// SPDX-License-Identifier: MIT
//
#pragma once

#include <array>
#include <cstdint>
#include <memory>
#include <string>

#include "npu_device.hpp"
#include "open_qwen36/q4nx_file.hpp"

namespace ow {

enum class Op : size_t { Conv1 = 0, Conv2, Qkv, O, Fc1, Fc2, Xkv, Count };

struct StreamShape {
  int64_t M = 0, K = 0, N = 0;
  size_t instr_slot = 0;   // Design::bind_instr() argument
};

const char *op_name(Op op);

// The shape each stream must have: whisper-large-v3-turbo's geometry with its 1500 frames
// padded to 1536. open_kernels/designs/whisper_gemm/whisper_gemm.py's STREAMS is the other
// copy, and the exporter builds every stream from it.
StreamShape expected_shape(Op op);

// Refuse a stream whose recorded shape is not that one. design.json is DATA: the engine
// sizes its A and C buffers from the geometry it was compiled for, while run() takes its
// transfer sizes from these fields -- so a stale or hand-edited design.json would memcpy
// and DMA the wrong number of bytes instead of being refused. Free function, so the guard
// is reachable without a device (guards_test.cpp).
void check_stream_shape(const std::string &where, Op op, int64_t M, int64_t K, int64_t N);

// Refuse a tensor whose dtype is not BF16 before it is read as bf16 BITS. The decoder's
// raw loader keeps the bits rather than going through Q4nxFile::bf16(), which checks the
// dtype itself -- so without this any other two-byte dtype (F16, I16) satisfies a byte
// count and is then read as bf16: finite, plausible, wrong logits.
void require_bf16(const open_qwen36::Q4nxFile &f, const std::string &name);

// Finds the kernel set directory (OFLM_WHISPER_KERNELS_DIR env var, else
// <model_dir>/open_kernels, else throws), validates whisper_kernels.json
// against model_dir's config.json (hf_config_check), and validates
// design.json's b_layout against the tile tuple the caller tiled its weights
// with. Refuses rather than dispatching against a mismatched kernel set.
class KernelSet {
public:
  KernelSet(npue::npu::Device &dev, const std::string &kernels_dir_hint,
           const std::string &model_dir, int64_t weights_tile_k,
           int64_t weights_tile_n, int64_t weights_mac_s, int64_t weights_mac_t);

  // Resolve the directory the constructor would use, without opening the
  // device -- exposed so the CLI can print it.
  static std::string resolve_dir(const std::string &kernels_dir_hint,
                                 const std::string &model_dir);

  struct BLayout {
    int64_t tile_k = 0, tile_n = 0, mac_s = 0, mac_t = 0;
  };
  // Reads design.json's b_layout tuple, so Weights can tile with the SAME
  // tuple the KernelSet constructor will then check it against -- no device,
  // no xclbin load, just the tuple this depends on before it can be built.
  static BLayout read_b_layout(const std::string &kernels_dir);

  npue::npu::Design &design() { return *design_; }
  const StreamShape &shape(Op op) const {
    return shapes_[static_cast<size_t>(op)];
  }
  const std::string &dir() const { return dir_; }

  // Stage a tiled bf16 [K,N] operand once; returns the slot for run()'s
  // `b_slot`. `elems` is K*N (element count, not bytes).
  size_t stage_b(const uint16_t *tiled, size_t elems);

  // One GEMM dispatch. `a_bf16` is `shape(op).M * shape(op).K` bf16 values,
  // row-major [M,K] -- the caller must have already zero-filled any padded
  // rows, since a GEMM computes each output row independently and never
  // mixes rows, so padding is a per-row concern the kernel cannot see.
  // Returns a pointer into the design's own C buffer (fp32, row-major
  // [M,N]) valid until the next dispatch of ANY op on this KernelSet.
  const float *run(Op op, const uint16_t *a_bf16, size_t b_slot, double *t_in,
                   double *t_disp, double *t_out);

private:
  std::string dir_;
  std::unique_ptr<npue::npu::Design> design_;
  std::array<StreamShape, static_cast<size_t>(Op::Count)> shapes_{};
};

}  // namespace ow
