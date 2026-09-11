/// \file manifest.hpp
/// \brief manifest.json: everything the open engine knows about a kernel set.
///
/// Written by open_kernels/export_qwen36_kernels.py from the family recipe
/// (open_kernels/recipes/manifest.py) beside the xclbins. The engine derives
/// every layout constant, context, kernel, per-layer program and packing law
/// from it -- there is no HID, no POOL_*, no "lx0" in the C++ -- and refuses
/// a model whose config.json disagrees with the manifest's `hf_config_check`.
///
/// Traces: OPEN-MANIFEST (specs/open-engine/spec.md).
#pragma once

#include <cstddef>
#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include "nlohmann/json.hpp"
#include "stream_patch.hpp"

namespace open_qwen36 {

/// One packing-plan op: which tensor lands at which byte offset in which
/// chunk order (open_kernels/recipes/pack.py is the same interpreter in NumPy).
struct PackOp {
    std::string op;                          ///< std_perm | q8_perm | expert_stripes | expert_down | put | conv_transpose | lmhead_q8 | transpose
    std::string tensor, up, gate;            ///< tensor names; "{l}" stands for the layer index
    uint64_t dst = 0;
    uint64_t cap = 0;                        ///< put: the slot's capacity
    uint64_t nch = 0, in_dim = 0, chunk0 = 0;                          ///< std_perm / q8_perm; in_dim also:
                                                                       ///< lmhead_q8, the hidden width
                                                                       ///< (q8_perm: nch counts POOL half-tiles,
                                                                       ///<  chunk0 counts SOURCE file chunks)
    uint64_t experts = 0, stripes = 0, stripe_bytes = 0, expert_bytes = 0;   ///< expert_stripes / expert_down
    uint64_t taps = 0, groups = 0, width = 0;                           ///< conv_transpose
    uint64_t chunk_bytes = 0;                                           ///< lmhead_q8 (the SOURCE chunk)
    uint64_t rows = 0, cols = 0, elem = 0;                              ///< transpose
    uint64_t dst_rows = 0;                                              ///< transpose: pad the
                                                                        ///< destination row to this
                                                                        ///< many values, tail zeroed
                                                                        ///< (0 = rows, no padding)
};

/// One verb of a layer type's (or the tail's) program.
struct Step {
    std::string op;                          ///< run | moeroute2
    std::string kernel;
    std::vector<std::string> args;           ///< run: buffer names (per-layer: pool consts act state; else globals)
    uint64_t act_off = 0;                    ///< moeroute2: the router record's offset in `act`
};

/// 0167/#32: the GEMM prefill route -- T tokens through a layer as 5
/// whole-array bf16 GEMM dispatches (q4_1 dequantised on-core) with q|k|v
/// FUSED into one ("qkv3"), T single-token attention dispatches between GEMM
/// A' and GEMM O, and HOST-side fp64 RMSNorm / residual / SwiGLU between
/// every GEMM stage (this route does NOT fuse norm/SwiGLU on-core; the
/// sequential production path does). `program` holds exactly 5 Steps in
/// FIXED order -- qkv3, o_proj, gate_proj, up_proj, down_proj -- each a plain
/// "run" against a per-layer weight buffer (named "gqkv3_w"/"go_w"/
/// "ggate_w"/"gup_w"/"gdown_w", built once at load_weights() time by Core,
/// see core.cpp) and a pair of GLOBAL scratch buffers ("gemm_x_hid"/
/// "gemm_x_ff" in, "gemm_y_qkv3"/"gemm_y_o"/"gemm_y_gate"/"gemm_y_up"/
/// "gemm_y_down" out) the manifest's own `globals` section sizes, exactly
/// like every other global. The attention half (kernel name fixed as "dxB")
/// is NOT a Step in `program` -- Core drives it directly (T attnpos-patched
/// dispatches through a GLOBAL "gact" T-wide scratch buffer, one shuttle
/// in/out per token) because it sits strictly between program[0] (qkv3) and
/// program[1] (o_proj), not appended to the list. Special-purpose and
/// Granite-only on purpose, not a generalized N-stage interpreter.
struct GemmBlockProgram {
    uint64_t t = 0;              ///< 0 = no gemm-block program for this layer type
    std::vector<Step> program;   ///< exactly 5 when t > 0: qkv3, o, gate, up, down (see above)
    // The handful of model constants this route's HOST-side math needs that
    // the rest of the manifest does not otherwise carry (RMSNorm eps; the
    // q/k/v attention-width split and FFN width; the T=1 "act" buffer's
    // AD_Q/AD_KVN/AD_OG byte offsets, open_kernels/recipes/dense.py's
    // DenseLayout). Not model constants baked into THIS file (core.hpp's own
    // rule) -- read from the manifest like everything else, just via new
    // fields instead of a generic Step.
    double eps = 0;
    uint64_t qw = 0, kvw = 0, ff = 0;
    uint64_t ad_q = 0, ad_kvn = 0, ad_og = 0;
};

struct LayerType {
    std::string name;
    uint64_t consts_bytes = 0, act_bytes = 0;
    std::string state_kind;                  ///< "linear" (a fixed-size state BO) | "kv" (max_ctx x state_row)
    uint64_t state_bytes = 0, state_row = 0;
    std::vector<Step> program;
    GemmBlockProgram gemm_block;
    std::vector<PackOp> pool, consts;
};

struct KernelDesc {
    std::string context;                     ///< name in Manifest::contexts
    std::string insts;                       ///< relative path of insts.bin
    std::string patch;                       ///< "" | moeroute2 | attnpos
    uint64_t window = 0;                     ///< attnpos: the sliding window (rows; 0 = every cached row)
};

/// A global sized max_ctx x row: the position record table(s).
struct RowGlobal {
    uint64_t per_row = 0;
    std::vector<double> inv_freq;            ///< its RoPE frequencies (rotary_dim / 2)
    double scale = 1.0;                      ///< on cos and sin (longrope's attention factor)
    uint64_t window = 0;                     ///< the row counts follow this window
    /// Phi-3's longrope: row r takes `long_inv_freq` once r >= switch_row, `inv_freq` before
    /// it -- HF's own per-call `seq_len = pos + 1 > original_max_position_embeddings` rule,
    /// applied per row since this engine computes one row per token as the context grows.
    /// Empty / kSwitchNever for every other family (row always takes `inv_freq`).
    std::vector<double> long_inv_freq;
    static constexpr uint64_t kSwitchNever = ~uint64_t{0};
    uint64_t switch_row = kSwitchNever;
};

struct Manifest {
    int version = 0;
    std::string family, spec_hash, build_key;
    size_t max_ctx_default = 0;
    // layout
    size_t hidden = 0, vocab = 0, real_vocab = 0;
    size_t chunk_bytes = 0, pool_bytes = 0, lmhead_pool_bytes = 0, lmhead_chunk_bytes = 0;
    size_t kv_row = 0, ptab_row = 0, rotary_dim = 0, rout_idx_off = 1024;
    double rope_theta = 0;
    std::vector<double> rope_inv_freq;       ///< per rotary pair (rotary_dim / 2 values; Llama 3's scaling is in here)
    bool has_moe = false;                    ///< layout.moe present (a family with routed experts)
    stream_patch::MoeGeometry moe;
    stream_patch::AttnGeometry attn;
    // the model and its programs
    std::vector<std::string> layers;         ///< per layer: a key of layer_types
    std::map<std::string, std::string> contexts;      ///< name -> relative path of final.xclbin
    std::map<std::string, KernelDesc> kernels;
    std::map<std::string, LayerType> layer_types;
    std::vector<Step> tail;
    std::map<std::string, uint64_t> globals;          ///< fixed-size global buffers (bytes)
    std::map<std::string, RowGlobal> per_row_globals; ///< globals sized max_ctx x row (the ptab(s))
    std::string embed_tensor, norm_tensor;
    std::vector<PackOp> lmhead_ops;          ///< pack.lm_head.ops into the lmpool global
    size_t norm_bytes = 0;
    nlohmann::json hf_config_check;
    /// What a key ABSENT from config.json means, for the keys of hf_config_check a config
    /// may omit (Phi-3's head_dim, partial_rotary_factor, rope_scaling, ...): check_model
    /// compares the expected value against this instead of refusing for the missing key.
    nlohmann::json hf_config_defaults = nlohmann::json::object();

    static Manifest load(const std::string& path);
    static Manifest parse(const nlohmann::json& j, const std::string& where);

    /// Throws naming the first key of config.json that disagrees with the manifest. A key
    /// config.json lacks takes its hf_config_defaults value when there is one, and is
    /// refused as lacking otherwise.
    void check_model(const nlohmann::json& config, const std::string& where) const;
    const LayerType& layer_type(size_t layer) const;
    /// Every file (relative to the kernel dir) the manifest names.
    std::vector<std::string> files() const;
};

}  // namespace open_qwen36
