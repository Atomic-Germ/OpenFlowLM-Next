/// \file dflash2.hpp
/// \brief The DFlash 2 block drafter on the NPU (OPEN-DRAFT-DFLASH2): the ternary drafter's matrices
///        on gemv_t2n (open_kernels/designs/gemv_t2n), everything between them on the host
///        (dflash2_host.hpp), the target's embedding and lm head borrowed from its Core.
///
/// Per committed target position p (inject): g = hidden_norm(fc(taps_p)), then every layer's
/// K = rope(k_norm(Wk g), p) and V = Wv g into the drafter's cache (f16 values, a sliding window).
/// Per draft (draft): the block [anchor, MASK x 7] at positions p0 .. p0 + 7 through the five
/// layers (norm -> conv_proj -> 2-tap conv -> q|k|v -> attention over [cache | block] -> o -> conv
/// -> residual -> norm -> conv_proj -> conv -> gate|up -> silu * up -> down -> conv -> residual),
/// the final norm, the target's head on rows 1..7, the selector lattice and its greedy walk; the
/// first k tokens are the draft.
#pragma once

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_hw_context.h"
#include "xrt/xrt_kernel.h"

#include "nlohmann/json.hpp"
#include "open_qwen36/core.hpp"
#include "open_qwen36/dflash2_host.hpp"
#include "open_qwen36/q4nx_file.hpp"

namespace open_qwen36 {
namespace dflash2 {

struct DraftConfig {
    size_t hidden = 0, layers = 0, nh = 0, kvh = 0, hd = 0, ff = 0, vocab = 0;
    size_t rank = 0, group = 0, window = 0, n_conv = 0;
    double eps = 1e-6, theta = 1e7;
    int mask_id = 0;
    std::vector<int> tap_layers;         ///< the target layers (0-based) whose OUTPUT is a tap
    size_t tap_width() const { return tap_layers.size() * hidden; }
};

struct PassTiming {
    double total_ms = 0, gemv_ms = 0, head_ms = 0, attn_ms = 0, select_ms = 0, host_ms = 0;
    double prep_ms = 0, sync_ms = 0, reduce_ms = 0;   ///< of host_ms: the tables, the y read-back, the sum over K slices
    int dispatches = 0;
    std::map<std::string, double> op_ms;   ///< dispatch time (start to done) by op
};

class Drafter {
public:
    /// `model_dir`: q4nx-build --dflash2-drafter's output; `kernel_dir`: the dflash2 family's kernel set.
    /// The target Core must have its weights loaded: its lmpool is the head this drafter streams.
    Drafter(const std::string& model_dir, const std::string& kernel_dir, Core& target, bool verbose = true);
    ~Drafter();
    Drafter(const Drafter&) = delete;
    Drafter& operator=(const Drafter&) = delete;

    void load_weights();
    /// Empty the cache (a new request).
    void reset();
    /// taps [n][tap_width] for target positions cached() .. cached() + n - 1, in order.
    void inject(const float* taps, size_t n);
    /// The cache holds positions [0, cached()); a draft's block starts there.
    long cached() const { return next_pos_; }
    /// Drop cached positions >= pos (rejected rows of a verify pass).
    void truncate(long pos);
    /// k (<= 7) tokens to follow `anchor`, whose position is cached(). `lat`, if given, gets the
    /// lattice of all seven rows.
    std::vector<int> draft(int anchor, size_t k, Lattice* lat = nullptr);

    const DraftConfig& config() const { return cfg_; }
    const PassTiming& last_pass() const { return timing_; }

private:
    struct Gemv {
        GemvPlan plan;
        std::unique_ptr<xrt::kernel> k;
        std::unique_ptr<xrt::bo> instr;
        uint32_t words = 0;
        xrt::bo tab, y;
        std::vector<uint8_t> rec;
        std::vector<float> xr;
    };

    DraftConfig cfg_;
    Core& target_;
    bool verbose_;
    std::string model_dir_, kernel_dir_;
    nlohmann::json man_;
    std::unique_ptr<Q4nxFile> file_;
    std::unique_ptr<xrt::hw_context> ctx_;
    std::map<std::string, Gemv> gemv_;       ///< by op: fc kv conv qkv o gu down head
    std::map<std::string, xrt::bo> w_;       ///< fc, kv
    std::vector<std::map<std::string, xrt::bo>> lw_;   ///< per layer: conv_attn qkv o conv_ffn gu down

    struct LayerConsts {
        std::vector<float> attn_norm, ffn_norm, q_norm, k_norm, attn_cb, ffn_cb;
    };
    std::vector<LayerConsts> lc_;
    std::vector<float> enc_norm_, out_norm_, sel_hidden_;
    std::vector<uint16_t> sel_prev_, sel_next_;   ///< bf16 [vocab][rank]

    // the cache: per layer, K and V rows [max_rows][kvh * hd], position p at row p - base_pos_
    std::vector<std::vector<float>> kc_, vc_;
    size_t max_rows_ = 0;
    long base_pos_ = 0, next_pos_ = 0;
    PassTiming timing_;

    void load_gemv(const std::string& op, const nlohmann::json& g);
    xrt::bo pack(const nlohmann::json& spec, int layer);
    /// out [8][N] = x [8][K] @ W^T: x through H/32 per 1024 block, the tables, one dispatch, the sum.
    void run_gemv(const std::string& op, xrt::bo& w, const float* x, float* out);
    void write_kv(int layer, long pos, const float* k, const float* v);
};

}  // namespace dflash2
}  // namespace open_qwen36
