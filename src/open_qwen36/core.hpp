/// \file core.hpp
/// \brief The resident open-kernel decode engine for Qwen3.6-MoE: device,
///        kernels, weights and per-layer state held for the process lifetime;
///        one `step()` per token.
///
/// This is the host half of the open path that phlegm ran as a batch `.cfg`
/// program (designs/layer_x driven by the config interpreter) and planned as
/// `OpenBackend` (npu-engine, `open-kernels-phase2-resident-driver.md`). It
/// has no dependency on the FLM app headers so it can be built and tested on
/// its own (cli.cpp); engine.hpp adapts it to the app's `causal_lm` seam.
///
/// Per token, per layer, two dispatches on ONE xclbin context (a context
/// switch restarts the cores, so a layer's two parts must share one):
///   part 0  norm -> projections -> DeltaNet / attention -> out -> norm(+res) -> router
///   host    read the router's top-8, re-point the expert fills (moeroute2)
///   part 1  the MoE block: 8 routed experts + shared expert + combine
/// then, when logits are wanted, the final norm and the q8 lm_head. The
/// attention layers share one instruction stream whose KV window length and
/// row / RoPE-record offsets are patched per token (attnpos).
///
/// Prefill is decode-as-prefill: the prompt goes through `step()` one token at
/// a time from zeroed state with logits skipped, which is exact for this
/// architecture (each layer's state update sees one token at a time) and is
/// the only prefill the open kernels have. Batch prefill needs kernels that do
/// not exist yet — see the plan.
#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_hw_context.h"
#include "xrt/xrt_kernel.h"

#include "open_qwen36/pools.hpp"
#include "open_qwen36/q4nx_file.hpp"
#include "stream_patch.hpp"

namespace open_qwen36 {

struct CoreConfig {
    std::string model_dir;   ///< holds config.json + model.q4nx (+ tokenizer files)
    std::string kernel_dir;  ///< holds lx0/ lx1/ ax0/ ax1/ ln/ lm_head_q8/, each final.xclbin + insts.bin
    int num_layers = -1;     ///< -1 = all of them; a prefix otherwise (testing)
    size_t max_ctx = 4096;   ///< KV rows per attention layer and RoPE records: the context capacity
    unsigned timeout_ms = 60000;  ///< per dispatch; 0 blocks
    bool verbose = true;
};

/// Everything a request needs to be resumed later (the app's checkpoint/restore).
struct Snapshot {
    int pos = 0;
    std::vector<std::vector<uint8_t>> states;  ///< per linear layer: conv state + S
    std::vector<std::vector<uint8_t>> kv;      ///< per attention layer: rows [0, pos)
};

struct StepTiming {
    double part0_ms = 0, part1_ms = 0, route_ms = 0, lmhead_ms = 0, total_ms = 0;
};

class Core {
public:
    /// Opens the device (or borrows `dev`), registers the xclbins and loads the
    /// instruction streams. Weights come with load_weights().
    Core(const CoreConfig& cfg, xrt::device* dev = nullptr);
    ~Core();
    Core(const Core&) = delete;
    Core& operator=(const Core&) = delete;

    /// Pack every layer's pools and consts straight into resident device
    /// buffers. Minutes on first touch of a 22 GB file, ~1 min warm.
    void load_weights(const std::function<void(int done, int total)>& progress = {});

    /// Start a new context: zero the linear states, position 0.
    void reset();
    /// One decode step for `token` at the current position. Logits (f32,
    /// VOCAB) are computed only when asked for; read them with logits().
    void step(int token, bool want_logits);
    const std::vector<float>& logits() const { return logits_host_; }

    int position() const { return pos_; }
    /// Test hook: place the next token at `pos` without decoding up to it.
    void seek(int pos);
    size_t max_ctx() const { return cfg_.max_ctx; }
    int num_layers() const { return nl_; }
    bool is_attention_layer(int l) const { return full_[l]; }
    const StepTiming& last_timing() const { return timing_; }

    Snapshot checkpoint() const;
    void restore(const Snapshot& s);

    /// One cached row of an attention layer's K or V (bf16[512]).
    void kv_row(int layer, int row, bool value, uint16_t* out512);

    const Q4nxFile& file() const { return *file_; }

private:
    struct Kern {
        std::string name;
        std::unique_ptr<xrt::kernel> k;
        std::unique_ptr<xrt::bo> instr;
        std::vector<uint32_t> words;
        std::vector<stream_patch::MoePatch> moe2;
        std::vector<stream_patch::AttnPatch> attn;
        uint32_t* iw() { return instr->map<uint32_t*>(); }
    };

    CoreConfig cfg_;
    std::unique_ptr<Q4nxFile> file_;
    int nl_ = 0, interval_ = 4;
    std::vector<bool> full_;
    bool has_lin_ = false, has_attn_ = false;

    std::unique_ptr<xrt::device> owned_dev_;
    xrt::device* dev_ = nullptr;
    std::unique_ptr<xrt::hw_context> ctx_x_, ctx_y_, ctx_l_, ctx_k_;
    Kern lx0_, lx1_, ax0_, ax1_, ln_, lm_;

    std::vector<xrt::bo> pools_, consts_, act_, state_;  // state_[l]: state (linear) or kv (attention)
    xrt::bo xres_, zero_, normw_, xresf_, hn_, logits_, ptab_, lmpool_;
    bool weights_loaded_ = false;
    int pos_ = 0;
    std::vector<float> logits_host_;
    StepTiming timing_;

    xrt::hw_context& load_context(std::unique_ptr<xrt::hw_context>& ctx, const std::string& design);
    void load_kernel(Kern& k, xrt::hw_context& ctx, const std::string& design);
    xrt::bo alloc(size_t bytes, const uint8_t* init = nullptr, size_t init_bytes = 0);
    double run(Kern& k, std::initializer_list<xrt::bo*> bufs);
    void route_and_run_part1(Kern& part1, xrt::bo& act, size_t rout_off, std::initializer_list<xrt::bo*> bufs);
    void log(const std::string& s) const;
};

}  // namespace open_qwen36
