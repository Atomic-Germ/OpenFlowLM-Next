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
#include "open_qwen36/spec.hpp"

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
    /// Empty the cache and begin it at position `pos` (a prompt longer than the window starts late).
    void restart(long pos);
    /// The first cached position.
    long base() const { return base_pos_; }
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

/// OPEN-DRAFT-DFLASH2 as an engine's spec::Drafter: keeps the drafter's cache in step with the history the
/// engine hands it, from the taps the target's own passes left (Core::tap_row), and proposes k tokens.
class Proposer : public spec::Drafter {
public:
    struct Stats {
        size_t drafts = 0, skipped = 0, restarts = 0, rows = 0, restart_rows = 0;
        double draft_ms = 0, inject_ms = 0, stage_ms = 0;   ///< the block pass; the cache rows; gathering the taps
        double restart_ms = 0;                              ///< of inject_ms: the calls that restarted the cache (the prompt)
        PassTiming last;                                    ///< the last block pass's breakdown
    };

    /// Turns the target's taps on (its weights must be loaded) and loads the drafter's weights.
    Proposer(Core& target, const std::string& model_dir, const std::string& kernel_dir, bool verbose = true);

    /// `next` sits at position fed.size(): the cache is brought to the history first.
    std::vector<int> propose(const std::vector<int>& fed, int next, size_t k) override;
    void observe(const int*, size_t) override {}   // the next propose() reads what the pass left from the taps
    /// A cold cache takes a long prompt's last `rows` positions only (the fill is ~0.85 ms a row); after
    /// `deepen_after` drafts it is refilled with the whole window (ColdFill). 0 = the whole window at once.
    void set_fill(size_t rows, size_t deepen_after = 4) { fill_.cold = rows; fill_.after = deepen_after; }
    /// Bring the cache to `fed` without drafting (what propose() does first). False: no context to draft from.
    bool sync(const std::vector<int>& fed);
    void keep_up(const std::vector<int>& fed) override;

    const Stats& stats() const { return stats_; }
    dflash2::Drafter& drafter() { return *d_; }

private:
    Core& core_;
    std::unique_ptr<dflash2::Drafter> d_;
    std::vector<int> ids_;          ///< the token at each cached position, from d_->base()
    std::vector<float> stage_;
    Stats stats_;
    ColdFill fill_;
    static constexpr size_t kStageRows = 64;
};

/// The shortest suffix match prompt lookup proposes from when it is in front of a draft model (the measured default).
constexpr int kStackLookupMin = 5;
constexpr size_t kStackFillRows = 256, kStackDeepenAfter = 4;

struct StackOptions {
    size_t fill_rows = 0;  ///< Proposer::set_fill: the cold fill (0 = the stack's default, kStackFillRows; alone: the window)
    size_t deepen_after = 0;   ///< likewise (0 = kStackDeepenAfter)
    int lookup_min = 0;    ///< overrides every lookup member's min_match (0 = the entry's, else the default)
    bool eager = false;    ///< the draft model catches up every pass lookup drafted, not when it next drafts
};

/// The drafter a speculative availability answer declares, on a core whose weights are loaded: the one member as
/// itself, several as a `spec::Stacked` in the declared order. `proposer` receives the dflash2 member, if any.
std::unique_ptr<spec::Drafter> make_stack(Core& core, const spec::Availability& av, const StackOptions& opt, bool verbose,
                                          Proposer** proposer = nullptr);

}  // namespace dflash2
}  // namespace open_qwen36
