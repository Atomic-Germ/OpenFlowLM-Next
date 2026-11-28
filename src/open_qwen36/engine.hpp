/// \file engine.hpp
/// \brief Open replacement for the closed `qwen3_6_moe_npu` engine: the app's
///        `causal_lm` seam, backed by open_qwen36::Core (the open kernels).
/// \note Everything above this seam — tokenizer, chat template, sampler,
///       server, prompt cache — is the app's own open code and drives this
///       exactly as it drives the closed engine. Images go through the vision
///       tower on the host CPU (vision/vit.hpp) and enter the model as embedding
///       vectors at their M-RoPE positions. Text prompts decode one token at a
///       time by default, which is exact but ~0.12 s per prompt token on the full
///       model; batched prefill (0167/#32) runs instead whenever the loaded kernel
///       set carries a GEMM-route block program (Core::gemm_block_t() > 0) and the
///       prompt is at least `prefill::kDenseBlockLoop` (96) tokens: T tokens per layer
///       as whole-array GEMM dispatches instead of T sequential steps.
///       OFLM_OPEN_GEMM_BLOCK=0 turns it off.
///
///       The number is 96, and a prompt of 64-95 engine tokens is STEPPED, not blocked.
///       Two thresholds exist and they answer different questions:
///
///         96  `prefill::kDenseBlockLoop` -- the dense sequential-to-block-loop
///             crossover. This is the one that decides the route here. Measured on
///             Qwen3.8-27B: fitted crossing 78, first clear win 92, applied 96.
///         64  `prefill::kLayerMajor` -- the crossover for the layer-major schedule,
///             which exists to amortise a routed expert's weight stream across a whole
///             prompt (measured on the Qwen3.6-35B MoE family). A dense set can run
///             layer-major but has nothing for it to amortise, so it is not the dense
///             default; `OFLM_OPEN_LAYER_MAJOR=1` forces it where the set is capable.
///
///       Both are in `prefill_thresholds.hpp`. OFLM_OPEN_GEMM_BLOCK_MIN overrides the
///       effective boundary at runtime for a diagnostic A/B; it does not change either
///       constant.
#pragma once

#include <memory>
#include <random>
#include <string>
#include <vector>

#include "buffer.hpp"
#include "causal_lm.hpp"
#include "device_runtime.hpp"
#include "lm_config.hpp"
#include "open_qwen36/core.hpp"
#include "open_qwen36/spec_lookup.hpp"
#include "open_qwen36/vision/vit.hpp"

namespace open_qwen36 {

class Engine : public causal_lm {
public:
    /// Opens the device contexts and instruction streams; weights follow with
    /// load_weights(). `MAX_L` sizes the KV cache (the context capacity).
    Engine(const LM_Config& config, oflm_rt::device* dev, int MAX_L);
    ~Engine() override;

    buffer<bf16> forward(int ids) override;
    buffer<bf16> prefill(std::vector<int>& ids, void* payload = nullptr) override;
    void set_context_length(int L) override;
    /// The argument is the closed reader; this engine reads the container
    /// itself (q4nx_file.hpp) and ignores it. See load_open_weights().
    void load_weights(Q4NX& q4nx) override;
    void load_open_weights();
    void update_max_length(uint32_t MAX_L) override;
    void clear_context() override;
    buffer<bf16> get_k_cache(int layer_idx, int idx) override;
    buffer<bf16> get_v_cache(int layer_idx, int idx) override;
    int get_current_context_length() override;
    int checkpoint() override;
    int restore() override;

    /// Where this model's open kernels are: OFLM_OPEN_KERNELS_DIR, else
    /// <model>/open_kernels, else <root>/xclbins/<model name>/open_kernels over
    /// EVERY root the closed path would consider (utils::xclbin_roots(): the
    /// configured root, the user-level oflm directory, the exe dir, the CWD, the
    /// installed bundle, the configured prefix) — a set under the user root and
    /// one shipped in the install tree are both reachable, whichever of the two
    /// find_xclbin_path() happens to return first. A kernel set is a manifest.json
    /// plus the files it names. Empty when none is found — the caller then keeps
    /// the closed engine.
    /// Resolve the kernel set. `how`, when given, receives which rule won.
    /// The caller logs it: a silently-chosen set produces perfectly valid
    /// output from kernels the user did not mean to run (#35).
    static std::string find_kernels(const LM_Config& config, std::string* how = nullptr);

    const Core& core() const { return *core_; }
    // OPEN-UNO-DECODE: needs both the kernel set's rows.lora_kernel and the model's uno.q4nx.
    bool uno_ok() const;
    // Leaves the cache holding every committed token but the last; committed token i sits at the return + 1 + i.
    int uno_cycle(int seed, std::vector<int>& out);
    /// Put the cache position at `pos`: the rows past it are dropped (a cycle cut short).
    void uno_seek(int pos);
    // OPEN-SPEC-LOOKUP: the switch (OFLM_SPECULATIVE=1) and the rows route; Uno joins when uno.q4nx is packed.
    bool spec_ok() const;
    void spec_begin();
    // uno_cycle's contract; false asks the caller for a plain step, so plain tokens come from the app's own sampler.
    bool spec_cycle(int seed, const std::vector<int>& history, std::vector<int>& out, int& p);
    /// Token ids whose logits read -inf from here on, so the sampler never picks them.
    void suppress_tokens(std::vector<int> ids) { suppressed_ = std::move(ids); }

private:
    std::vector<int> suppressed_;
    CoreConfig cfg_;
    oflm_rt::device* dev_;
    std::unique_ptr<Core> core_;
    Snapshot snapshot_;
    bool has_snapshot_ = false;
    std::vector<bf16> logits_;
    bool poisoned_ = false;
    std::mt19937 uno_rng_{0};                   ///< the draft rows' noise ids
    SpecPolicy spec_;
    bool spec_plain_ = false;                   ///< the last cycle was the caller's plain step: time it on the next call
    int uno_passes(int seed, std::vector<int>& out, double& draft_ms, double& verify_ms, size_t& accepted);
    vision::VitConfig vcfg_;
    std::unique_ptr<vision::VitWeights> vit_;   ///< loaded on the first image (~0.85 GB, ~3 s)
    void ensure_vit();
    /// 27.11: remember the caches so a block-prefill failure part-way through a prompt
    /// can be undone before the sequential path runs. Only taken when the prompt does
    /// not begin at position 0, because that case is recoverable by `reset()` for free.
    /// Returns false when no snapshot could be taken, which the caller reports rather
    /// than mistaking for success.
    bool save_pre_prompt_state();
    template <class Payload> buffer<bf16> prefill_images(std::vector<int>& ids, const Payload& p);

    buffer<bf16> logits_view();
    /// A kernel that timed out or aborted leaves the hardware context dead:
    /// every later submission fails. Mark the engine and rebuild it (contexts
    /// + weights, ~90 s) before the next request instead of failing forever.
    void ensure_alive();
    /// Print what failed before rethrowing. Core::run's message already names the
    /// kernel, the position and the ERT state, but on a streamed request that
    /// message only ever reaches an HTTP error body the streaming path does not
    /// write - so an intermittent dispatch failure used to show up as a client
    /// timeout and the one line that would diagnose it was thrown away.
    static void note_failure(const char* what);
    /// \brief Diagnostic: how many prefill calls this engine has served.
    /// Only read by the OFLM_OPEN_BLOCK_FAIL_INJECT arm, which needs "the second
    /// prompt of this session" to reach the snapshot path (position != 0).
    int prefill_calls_ = 0;

    template <class F> auto guarded(F&& f) -> decltype(f()) {
        ensure_alive();
        try {
            return f();
        } catch (const std::exception& e) {
            poisoned_ = true;
            note_failure(e.what());
            throw;
        } catch (...) {
            poisoned_ = true;
            note_failure("a non-std exception");
            throw;
        }
    }
};

}  // namespace open_qwen36
