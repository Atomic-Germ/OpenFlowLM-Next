/// \file gemma3_text_open.cpp
/// \brief Open Gemma3 text engine adapter implementation.
/// \author FastFlowLM Team
/// \date 2026-09-02

#include "open_gemma3/gemma3_text_open.hpp"

#include <cstdio>

Gemma3TextOpen::Gemma3TextOpen(LM_Config config, npu_xclbin_manager* npu, int MAX_L)
    : MAX_L_(MAX_L > 0 ? MAX_L : 4096) {
    (void)config;
    (void)npu;  // The open engine is CPU-only at this stage and needs no NPU.
}

bool Gemma3TextOpen::load_model_dir(const std::string& model_dir) {
    if (!engine_.load(model_dir)) {
        std::fprintf(stderr, "Gemma3TextOpen: failed to load %s\n", model_dir.c_str());
        return false;
    }
    if (!engine_.enable_cache(static_cast<size_t>(MAX_L_))) {
        std::fprintf(stderr, "Gemma3TextOpen: failed to allocate KV cache (%d positions)\n",
                     MAX_L_);
        return false;
    }
    loaded_ = true;
    return true;
}

buffer<bf16> Gemma3TextOpen::to_bf16(const std::vector<float>& v) {
    // Allocate an owning buffer and fill it in place. Do NOT build a buffer from
    // a std::vector: buffer(std::vector&&) is a shallow mapping that takes no
    // ownership, so a vector that dies on return leaves the logits pointing at
    // freed memory.
    buffer<bf16> out(v.size());
    for (size_t i = 0; i < v.size(); ++i) out[i] = static_cast<bf16>(v[i]);
    return out;
}

buffer<bf16> Gemma3TextOpen::forward(int ids) {
    if (!loaded_) return buffer<bf16>(0);
    return to_bf16(engine_.step({static_cast<int32_t>(ids)}));
}

buffer<bf16> Gemma3TextOpen::prefill(std::vector<int>& ids, void* payload) {
    (void)payload;  // Gemma3 text is text-only; no multimodal payload.
    if (!loaded_) return buffer<bf16>(0);
    std::vector<int32_t> ids32(ids.begin(), ids.end());
    return to_bf16(engine_.step(ids32));
}

void Gemma3TextOpen::set_context_length(int L) {
    if (L > 0) MAX_L_ = L;
    engine_.enable_cache(static_cast<size_t>(MAX_L_));
    engine_.clear_context();
}

void Gemma3TextOpen::update_max_length(uint32_t MAX_L) {
    if (MAX_L > static_cast<uint32_t>(MAX_L_)) {
        MAX_L_ = static_cast<int>(MAX_L);
        engine_.enable_cache(static_cast<size_t>(MAX_L_));
    }
}

void Gemma3TextOpen::clear_context() { engine_.clear_context(); }

int Gemma3TextOpen::get_current_context_length() {
    return static_cast<int>(engine_.context_length());
}

void Gemma3TextOpen::load_weights(Q4NX& q4nx) {
    (void)q4nx;
    std::fprintf(stderr,
                 "Gemma3TextOpen: load_weights(Q4NX&) is not implemented; the open engine "
                 "loads bf16 safetensors via load_model_dir()\n");
}

buffer<bf16> Gemma3TextOpen::get_k_cache(int layer_idx, int idx) {
    (void)layer_idx;
    (void)idx;
    std::fprintf(stderr, "Gemma3TextOpen: get_k_cache is not implemented\n");
    return buffer<bf16>(0);
}

buffer<bf16> Gemma3TextOpen::get_v_cache(int layer_idx, int idx) {
    (void)layer_idx;
    (void)idx;
    std::fprintf(stderr, "Gemma3TextOpen: get_v_cache is not implemented\n");
    return buffer<bf16>(0);
}

int Gemma3TextOpen::checkpoint() {
    std::fprintf(stderr, "Gemma3TextOpen: checkpoint is not implemented\n");
    return -1;
}

int Gemma3TextOpen::restore() {
    std::fprintf(stderr, "Gemma3TextOpen: restore is not implemented\n");
    return -1;
}
