/// \file engine.hpp
/// \brief Open replacement for the closed `qwen3_6_moe_npu` engine: the app's
///        `causal_lm` seam, backed by open_qwen36::Core (the open kernels).
/// \note Everything above this seam — tokenizer, chat template, sampler,
///       server, prompt cache — is the app's own open code and drives this
///       exactly as it drives the closed engine. Images go through the vision
///       tower on the host CPU (vision/vit.hpp) and enter the model as embedding
///       vectors at their M-RoPE positions. Text prompts decode one token at a
///       time by default, which is exact but ~0.12 s per prompt token on the full
///       model; batched prefill (0167/#32) runs instead when the loaded kernel set
///       carries a GEMM-route block program (Core::gemm_block_t() > 0, currently
///       Granite only) AND OFLM_OPEN_GEMM_BLOCK=1 is set: T tokens per layer as 5
///       whole-array GEMM dispatches plus T attention dispatches, instead of T
///       sequential steps.
#pragma once

#include <memory>
#include <string>
#include <vector>

#include "buffer.hpp"
#include "causal_lm.hpp"
#include "device_runtime.hpp"
#include "lm_config.hpp"
#include "open_qwen36/core.hpp"
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

private:
    CoreConfig cfg_;
    oflm_rt::device* dev_;
    std::unique_ptr<Core> core_;
    Snapshot snapshot_;
    bool has_snapshot_ = false;
    std::vector<bf16> logits_;
    bool poisoned_ = false;
    vision::VitConfig vcfg_;
    std::unique_ptr<vision::VitWeights> vit_;   ///< loaded on the first image (~0.85 GB, ~3 s)
    void ensure_vit();
    template <class Payload> buffer<bf16> prefill_images(std::vector<int>& ids, const Payload& p);

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
