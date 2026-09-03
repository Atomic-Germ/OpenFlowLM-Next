/// \file gemma3_text_open.hpp
/// \brief Open Gemma3 text engine adapter implementing the causal_lm contract.
/// \author FastFlowLM Team
/// \date 2026-09-02
/// \note Wraps open_gemma3::Engine so it can replace the closed
/// `gemma_text_npu` library without changing AutoModel.
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "causal_lm.hpp"
#include "lm_config.hpp"
#include "npu_utils/npu_utils.hpp"
#include "open_gemma3/engine.hpp"

class Gemma3TextOpen : public causal_lm {
public:
    Gemma3TextOpen(LM_Config config, npu_xclbin_manager* npu, int MAX_L = 4096);
    ~Gemma3TextOpen() override = default;

    /// \brief Open weight loading from a model directory (safetensors).
    /// This replaces the closed Q4NX path entirely.
    bool load_model_dir(const std::string& model_dir);

    buffer<bf16> forward(int ids) override;
    buffer<bf16> prefill(std::vector<int>& ids, void* payload = nullptr) override;
    void set_context_length(int L) override;
    void update_max_length(uint32_t MAX_L) override;
    void clear_context() override;

    /// \brief Not implemented: the open engine loads from safetensors, not Q4NX.
    /// Retained only because the legacy causal_lm interface requires it.
    void load_weights(Q4NX& q4nx) override;

    /// \brief Not implemented: no consumer on the Gemma3 text path.
    buffer<bf16> get_k_cache(int layer_idx, int idx) override;
    /// \brief Not implemented: no consumer on the Gemma3 text path.
    buffer<bf16> get_v_cache(int layer_idx, int idx) override;
    /// \brief Not implemented: preemption snapshots are not supported yet.
    int checkpoint() override;
    /// \brief Not implemented: preemption snapshots are not supported yet.
    int restore() override;

    int get_current_context_length() override;

private:
    open_gemma3::Engine engine_;
    int MAX_L_ = 4096;
    bool loaded_ = false;

    static buffer<bf16> to_bf16(const std::vector<float>& v);
};
