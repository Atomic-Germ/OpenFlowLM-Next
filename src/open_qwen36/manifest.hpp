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
#include <memory>
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

struct LayerType {
    std::string name;
    uint64_t consts_bytes = 0, act_bytes = 0;
    std::string state_kind;                  ///< "linear" (a fixed-size state BO) | "kv" (max_ctx x state_row)
    uint64_t state_bytes = 0, state_row = 0;
    std::vector<Step> program;
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
    uint64_t window = 0;                     ///< the row counts follow this window
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
    /// The same kernel set for GGUF-direct weight files (f32-scale pool chunks,
    /// open_kernels/gguf_pool.py): a complete manifest of its own, parsed with
    /// this same code; the engine swaps it in when the model ships a .gguf.
    std::unique_ptr<Manifest> gguf;

    static Manifest load(const std::string& path);
    static Manifest parse(const nlohmann::json& j, const std::string& where);

    /// Throws naming the first key of config.json that disagrees with the manifest.
    void check_model(const nlohmann::json& config, const std::string& where) const;
    const LayerType& layer_type(size_t layer) const;
    /// Every file (relative to the kernel dir) the manifest names.
    std::vector<std::string> files() const;
};

}  // namespace open_qwen36
