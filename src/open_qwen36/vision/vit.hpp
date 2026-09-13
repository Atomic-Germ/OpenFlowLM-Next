/// \file vit.hpp
/// \brief The host-CPU vision towers (issue #16 item 3, phase A).
///
/// Two families share this file, because two thirds of it is family-independent (the
/// un-tiling, the linear, the patch order, the rotary tables, the merger's shape):
///
///   Qwen3VL  -- `vision_weight.q4nx`: patch embed + interpolated positions, LayerNorm
///               blocks, 2-D RoPE attention over the whole image, GELU-tanh MLP, 2x2
///               merger. Reference: open_kernels/model/replica_vit.py.
///   Qwen25VL -- `vision_weights.q4nx`: no position table, RMSNorm, SwiGLU, split q/k/v/o,
///               and attention inside square windows in all but the blocks named by
///               `fullatt_block_indexes`. The tower permutes its merge units so windows
///               are contiguous, runs the whole stack in that order and restores the row
///               order after the merger; the rotary table is permuted with the tokens.
///               Reference: open_kernels/model/replica_vit_qwen25.py.
///
/// Either way the container is bf16 with every linear pre-tiled for the closed engine's
/// vision_mm kernel as [n/64][k/256][64][256], zero-padded, which `untile` undoes. The
/// output rows replace the image tokens in the LM's prompt. vit_test.cpp checks each port
/// against its reference's fixture; the NPU version (phase B) reuses the prefill GEMM.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace open_qwen36::vision {

enum class VitFamily { Qwen3VL, Qwen25VL };

struct VitConfig {
    VitFamily family = VitFamily::Qwen3VL;
    int depth = 27, hidden = 1152, heads = 16, head_dim = 72, inter = 4304, out = 2048;
    int patch = 16, temporal = 2, merge = 2, npos = 2304, channels = 3;
    float eps = 1e-6f;
    int window = 0;             // window edge in pixels, Qwen2.5-VL only
    std::vector<int> fullatt;   // blocks that attend over the whole image, Qwen2.5-VL only
    /// From the model's config.json `vision_config`: OFLM's per-family prefixes
    /// (QWEN3_6_MOE_*, QWEN3_5_*) or the plain transformers keys.
    static VitConfig from_model_dir(const std::string& model_dir);
    /// The same reading, from config.json's text - the half that needs no container.
    static VitConfig from_config_text(const std::string& config_json);
    /// Qwen2.5-VL's windowed tower. Kept separate from from_config_text, which refuses a
    /// windowed vision_config because the tower it configures cannot run one.
    static VitConfig qwen25_from_model_dir(const std::string& model_dir);
    static VitConfig qwen25_from_config_text(const std::string& config_json);
    /// Whichever of the two the container asks for, by config.json's model_type. The
    /// engine calls this; the two readers above stay narrow so each keeps refusing the
    /// tower it cannot run (OPEN-VISION-VIT-CONFIG).
    static VitConfig for_model_dir(const std::string& model_dir);
    int patch_dim() const { return channels * temporal * patch * patch; }
    /// A window's edge in merge units: 112 px / 2 patches per unit / 14 px per patch = 4.
    int window_side() const { return window / merge / patch; }
};

/// Which merge unit goes where so that each window is contiguous, and where the attention
/// segments start (in patches - a merge unit is merge^2 patches and they share a window).
/// The padding is `side - n % side`, which is a whole empty window when n already divides;
/// those windows produce no tokens and no segment. Mirrors replica_vit_qwen25.window_index.
void window_index(const VitConfig& cfg, int grid_h, int grid_w, std::vector<int>& index, std::vector<int>& cu);

/// y = x . W^T + b, W kept as bf16 [out, in] (un-tiled), b as f32.
struct Linear {
    std::vector<uint16_t> w;
    std::vector<float> b;
    int out = 0, in = 0;
};

struct VitBlock {
    std::vector<float> ln1_w, ln1_b, ln2_w, ln2_b;   // the *_b stay empty under RMSNorm
    Linear qkv, proj, fc1, fc2;                      // Qwen3-VL
    Linear q, k, v, gate, up, down;                  // Qwen2.5-VL; proj is its o_proj
};

struct VitWeights {
    Linear patch;                       // [hidden, C*T*P*P]
    std::vector<float> pos;             // [npos, hidden]; empty on Qwen2.5-VL
    std::vector<VitBlock> blocks;
    std::vector<float> merger_ln_w, merger_ln_b;
    Linear merger_fc1, merger_fc2;      // [4 hidden, 4 hidden], [out, 4 hidden]
};

/// Read and un-tile the container, per cfg.family. ~0.85 GB resident as bf16 for the 35B,
/// ~1.3 GB for Qwen2.5-VL-3B.
VitWeights load_vit(const std::string& vision_q4nx_path, const VitConfig& cfg);

/// pixels: [grid_h * grid_w, patch_dim] f32 in the processor's patch order (merge-block
/// major, as modeling_*_image.cpp's reorder_patches emits). Returns
/// [grid_h * grid_w / merge^2, out] row-major, in that same order - the windowed tower's
/// permutation is undone before it returns.
std::vector<float> vit_forward(const VitConfig& cfg, const VitWeights& w, const float* pixels, int grid_h, int grid_w);

}  // namespace open_qwen36::vision
