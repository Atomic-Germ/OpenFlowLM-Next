/// \file vit.hpp
/// \brief The Qwen3.6 vision tower on the host CPU (issue #16 item 3, phase A).
///
/// Loads the shipped `vision_weight.q4nx` (bf16, every linear pre-tiled for the closed
/// engine's vision_mm kernel as [n/64][k/256][64][256] -- un-tiled here) and runs the
/// encoder in fp32: patch embed + interpolated positions, 27 blocks (LayerNorm, 2-D RoPE
/// attention over the whole image, GELU-tanh MLP), and the 2x2 patch merger. The output
/// rows replace the image tokens in the LM's prompt.
///
/// Reference: open_kernels/model/replica_vit.py (numpy, checked against transformers'
/// Qwen3VLVisionModel with the same weights to 9e-6). vit_test.cpp checks this port against
/// that reference's fixture. The NPU version (phase B) reuses the prefill GEMM.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace open_qwen36::vision {

struct VitConfig {
    int depth = 27, hidden = 1152, heads = 16, head_dim = 72, inter = 4304, out = 2048;
    int patch = 16, temporal = 2, merge = 2, npos = 2304, channels = 3;
    float eps = 1e-6f;
    /// From the model's config.json `vision_config` (FLM's QWEN3_6_MOE_VISION_* keys).
    static VitConfig from_model_dir(const std::string& model_dir);
    int patch_dim() const { return channels * temporal * patch * patch; }
};

/// y = x . W^T + b, W kept as bf16 [out, in] (un-tiled), b as f32.
struct Linear {
    std::vector<uint16_t> w;
    std::vector<float> b;
    int out = 0, in = 0;
};

struct VitBlock {
    std::vector<float> ln1_w, ln1_b, ln2_w, ln2_b;
    Linear qkv, proj, fc1, fc2;
};

struct VitWeights {
    Linear patch;                       // [hidden, C*T*P*P]
    std::vector<float> pos;             // [npos, hidden]
    std::vector<VitBlock> blocks;
    std::vector<float> merger_ln_w, merger_ln_b;
    Linear merger_fc1, merger_fc2;      // [4 hidden, 4 hidden], [out, 4 hidden]
};

/// Read and un-tile the container. ~0.85 GB resident as bf16.
VitWeights load_vit(const std::string& vision_q4nx_path, const VitConfig& cfg);

/// pixels: [grid_h * grid_w, patch_dim] f32 in the processor's patch order (merge-block
/// major, as modeling_*_image.cpp's reorder_patches emits). Returns
/// [grid_h * grid_w / merge^2, out] row-major.
std::vector<float> vit_forward(const VitConfig& cfg, const VitWeights& w, const float* pixels, int grid_h, int grid_w);

}  // namespace open_qwen36::vision
