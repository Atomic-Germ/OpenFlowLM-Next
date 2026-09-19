//===- weights.hpp -------------------------------------------*- C++ -*-===//
//
// open_whisper -- load Whisper-V3-Turbo-OpenNPU2's model.open.safetensors and
// pre-tile every GEMM operand for the NPU kernel set (phase 2b, issue #72).
// SPDX-License-Identifier: MIT
//
// The container's B tensors are row-major [K, N] bf16 (utilities/q4nx-build/
// q4nx/open_whisper.py's whisper_tensors()) -- NOT pre-tiled, on purpose: the
// tiling depends on the kernel set's (tile_k, tile_n, mac_s, mac_t) tuple, and
// a container baked for one tuple would be silently wrong for another. This
// engine tiles at load, from whatever tuple kernels.cpp reads out of
// design.json.
//
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace ow {

// The pre-tiling npu_offload/gemm_rtp/npue.py's tile_b(order="k,n") performs,
// i.e. exactly:
//   [K,N] -> [kb][nb][tk/s][tn/t][s][t]   (kb outer, nb, then the mac sub-tile)
// Ported with attribution from NpuEmbeddings' src/open_npue/npue_pack.cpp
// (function `tile_b`, anonymous namespace, ~line 304) -- that symbol is not
// exported, so it is copied rather than linked (per the phase-2b brief).
// `mat` is row-major [K, N] float; output is bf16 bits (round-to-nearest-even,
// the same rounding tools/npue.py and the container's own writer use).
std::vector<uint16_t> tile_b(const float *mat, int64_t K, int64_t N, int64_t tk,
                             int64_t tn, int64_t mac_s = 8, int64_t mac_t = 8);

// fp32 -> bf16 bits, round-to-nearest-even.
uint16_t bf16_rne(float x);

struct LayerWeights {
  std::vector<uint16_t> qkv_B, o_B, fc1_B, fc2_B;      // tiled bf16
  std::vector<float> qkv_bias, o_bias, fc1_bias, fc2_bias;
  std::vector<float> ln1_w, ln1_b, ln2_w, ln2_b;
};

// Geometry this build actually implements -- whisper-large-v3-turbo's, and
// the only shape the shipped kernel set (whisper_gemm) was compiled for.
// A model_dir whose config.json disagrees is refused in Weights' constructor,
// before a single tensor is read: reading it anyway would tile weights of
// the wrong width against a kernel set built for another one, and the GEMM
// would return a plausible, wrong answer (this project's "fails open" class).
struct Geometry {
  static constexpr int64_t d_model = 1280;
  static constexpr int64_t n_enc = 32;
  static constexpr int64_t n_dec = 4;
  static constexpr int64_t n_heads = 20;
  static constexpr int64_t head_dim = 64;      // d_model / n_heads
  static constexpr int64_t ffn = 5120;
  static constexpr int64_t n_mel = 128;
  static constexpr int64_t max_src_pos = 1500;
};

class Weights {
public:
  // Throws on: a weights_manifest.json whose "format" is not
  // "oflm-open-whisper-v1", a config.json that does not match Geometry, or
  // any tensor missing / of the wrong shape.
  explicit Weights(const std::string &model_dir, int64_t tile_k, int64_t tile_n,
                   int64_t mac_s, int64_t mac_t);

  std::vector<uint16_t> conv1_B, conv2_B, xkv_B;        // tiled bf16
  std::vector<float> conv1_bias, conv2_bias, pos, ln_w, ln_b, xkv_bias;
  std::vector<LayerWeights> layers;   // size Geometry::n_enc

  // The tile tuple these weights were tiled with -- kernels.cpp compares it
  // against design.json's b_layout before trusting a single dispatch.
  int64_t tile_k, tile_n, mac_s, mac_t;
};

}  // namespace ow
