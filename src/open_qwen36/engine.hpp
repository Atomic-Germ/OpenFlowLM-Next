/// \file engine.hpp
/// \brief Open replacement for the closed `qwen3_6_moe_npu` engine: the app's
///        `causal_lm` seam, backed by open_qwen36::Core (the open kernels).
/// \note Everything above this seam — tokenizer, chat template, sampler,
///       server, prompt cache — is the app's own open code and drives this
///       exactly as it drives the closed engine. What the closed engine still
///       does that this does not: the vision encoder (an image payload is
///       refused), and batched prefill (prompts are decoded one token at a
///       time, which is exact but ~0.3 s per prompt token on the full model).
#pragma once

#include <memory>
#include <string>
#include <vector>

#include "buffer.hpp"
#include "causal_lm.hpp"
#include "device_runtime.hpp"
#include "lm_config.hpp"
#include "open_qwen36/core.hpp"

namespace open_qwen36 {

class Engine : public causal_lm {
public:
    /// Opens the device contexts and instruction streams; weights follow with
    /// load_weights(). `MAX_L` sizes the KV cache (the context capacity).
    Engine(const LM_Config& config, flm_rt::device* dev, int MAX_L);
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

    /// Where this model's open kernels are: FLM_OPEN_KERNELS_DIR, else
    /// <model>/open_kernels, else <xclbin prefix>/xclbins/<model name>/open_kernels.
    /// Empty when none is found — the caller then keeps the closed engine.
    static std::string find_kernels(const LM_Config& config);

    const Core& core() const { return *core_; }

private:
    CoreConfig cfg_;
    flm_rt::device* dev_;
    std::unique_ptr<Core> core_;
    Snapshot snapshot_;
    bool has_snapshot_ = false;
    std::vector<bf16> logits_;
    bool poisoned_ = false;

    buffer<bf16> logits_view();
    /// A kernel that timed out or aborted leaves the hardware context dead:
    /// every later submission fails. Mark the engine and rebuild it (contexts
    /// + weights, ~90 s) before the next request instead of failing forever.
    void ensure_alive();
    template <class F> auto guarded(F&& f) -> decltype(f()) {
        ensure_alive();
        try { return f(); } catch (...) { poisoned_ = true; throw; }
    }
};

}  // namespace open_qwen36
