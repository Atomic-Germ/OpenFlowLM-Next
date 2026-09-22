//===- encoder.hpp -------------------------------------------*- C++ -*-===//
//
// open_whisper -- the Whisper-large-v3-turbo ENCODER: NPU GEMM for every
// matrix product, everything else (im2col, LayerNorm, GELU, attention, bias,
// residual, bf16 rounding) on the host in fp32. Phase 2b, issue #72. No
// decoder here -- see cli.cpp for the layer-by-layer gate this exists for.
// SPDX-License-Identifier: MIT
//
#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "host_ops.hpp"
#include "kernels.hpp"
#include "npu_device.hpp"
#include "weights.hpp"

namespace ow {

// Host wall-clock only (rule 1 of this project's NPU work: never an NPU
// performance claim). `npu_wait` is submit+wait for every GEMM dispatch --
// dominated by the hardware, but still a HOST observation of how long the
// call took, not a hardware trace.
struct Timers {
  double im2col = 0, bf16 = 0, layer_norm = 0, gelu = 0, bias = 0, residual = 0;
  double attention = 0;
  // attention(), split: the two GEMMs a kernel set could take, and the softmax
  // that stays on the host whatever happens to them. Only filled when
  // OW_ATTN_PHASES=1, because the split costs about 2% of the call.
  AttnPhases attn_phases;
  double npu_in = 0, npu_dispatch = 0, npu_out = 0;
  double total = 0;
};

// Called after each stage with the FIRST `rows` rows of a [*, cols]
// row-major fp32 buffer -- the caller (cli.cpp) compares these against
// float64 golden tensors. `name` is one of: "conv1", "conv2",
// "enc.hidden.<1..32>" (output of layer <name-1>), "enc.out",
// "dec.<0..3>.xk", "dec.<0..3>.xv" (golden's own names for cross-attention K/V).
using StageHook = std::function<void(const std::string &name, const float *data,
                                     int64_t rows, int64_t cols)>;

class Encoder {
public:
  // Throws on anything weights.hpp/kernels.hpp throw on: a container this
  // build does not recognise, a kernel set that does not match it, or no
  // usable NPU device.
  Encoder(const std::string &model_dir, const std::string &kernels_dir_hint);

  // Full forward pass over one 30 s window. `mel` is [128, 3000] row-major
  // (channel-major, as Whisper's feature extractor produces it).
  void encode(const float *mel, const StageHook &hook = nullptr);

  // Teacher forcing: run ONE encoder layer from an externally supplied
  // [1500, 1280] input (e.g. a golden enc.hidden.<i>), independent of
  // whatever encode() last computed. `output` is [1500, 1280].
  void run_layer_from(int64_t layer, const float *input_1500x1280, float *output);

  // Diagnostic: dispatch the SAME qkv GEMM `reps` times, cycling the staged B
  // slot across layers the way encode() does, and report any dispatch whose C
  // differs from that layer's first C. The array itself is bit-deterministic
  // (five identical harness dispatches agree byte for byte), so a mismatch
  // here is this engine's own dispatch path, not the hardware.
  int stress_qkv(int64_t reps, int64_t n_layers);

  // Rows [0,1500) of the cross-attention K|V for decoder layer `l`
  // (0..3), valid after encode(). d_model wide each; xkv() is the raw fused
  // [1500, 10240] = k0|v0|k1|v1|k2|v2|k3|v3.
  const std::vector<float> &xkv() const { return xkv_; }
  const std::vector<float> &enc_out() const { return enc_out_; }   // [1500,1280]

  const KernelSet &kernel_set() const { return *kernels_; }
  const Weights &weights() const { return *weights_; }

  Timers timers;

private:
  // Runs one encoder layer IN PLACE on a [Geometry::max_src_pos-rounded-up,
  // d_model] buffer already zero-padded beyond `real_rows`. Shared by
  // encode()'s chained path and run_layer_from()'s isolated one.
  void run_layer(int64_t layer, float *x /* [m_padded, d_model] */, int64_t real_rows,
                int64_t m_padded);

  std::unique_ptr<npue::npu::Device> device_;
  std::unique_ptr<Weights> weights_;
  std::unique_ptr<KernelSet> kernels_;

  // Staged B slots, filled once at construction.
  size_t conv1_slot_ = 0, conv2_slot_ = 0, xkv_slot_ = 0;
  struct LayerSlots { size_t qkv = 0, o = 0, fc1 = 0, fc2 = 0; };
  std::vector<LayerSlots> layer_slots_;

  std::vector<float> enc_out_;   // [1500, 1280], set by encode()
  std::vector<float> xkv_;       // [1500, 10240], set by encode()
};

}  // namespace ow
