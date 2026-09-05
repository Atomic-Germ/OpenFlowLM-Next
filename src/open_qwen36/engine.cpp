/// \file engine.cpp
/// \brief The open Qwen3.6-MoE engine behind the app's causal_lm seam (see engine.hpp).
#include "open_qwen36/engine.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <limits>
#include <stdexcept>

#include "utils/utils.hpp"

namespace open_qwen36 {

namespace fs = std::filesystem;
using namespace layout;

std::string Engine::find_kernels(const LM_Config& config) {
    auto complete = [](const fs::path& dir) {
        std::error_code ec;
        for (const char* d : {"lx0", "lx1", "ax0", "ax1", "ln", "lm_head_q8"})
            if (!fs::is_regular_file(dir / d / "final.xclbin", ec) || !fs::is_regular_file(dir / d / "insts.bin", ec)) return false;
        return true;
    };
    if (const char* env = std::getenv("FLM_OPEN_KERNELS_DIR")) {
        if (complete(env)) return env;
        std::fprintf(stderr, "open_qwen36: FLM_OPEN_KERNELS_DIR=%s lacks the six kernel pairs\n", env);
    }
    fs::path local = fs::path(config.model_path) / "open_kernels";
    if (complete(local)) return local.string();
    std::string prefix = config.exec_path;
    if (prefix.empty()) {
        try { prefix = utils::find_xclbin_path(); } catch (const std::exception&) { prefix.clear(); }
    }
    if (!prefix.empty()) {
        fs::path cand = fs::path(prefix) / "xclbins" / config.model_name / "open_kernels";
        if (complete(cand)) return cand.string();
    }
    return {};
}

Engine::Engine(const LM_Config& config, flm_rt::device* dev, int MAX_L) : dev_(dev) {
    cfg_.model_dir = config.model_path;
    cfg_.kernel_dir = find_kernels(config);
    if (cfg_.kernel_dir.empty())
        throw std::runtime_error("open_qwen36: no open kernels found for " + config.model_name +
                                 " (set FLM_OPEN_KERNELS_DIR or install xclbins/" + config.model_name + "/open_kernels)");
    cfg_.max_ctx = MAX_L > 0 ? static_cast<size_t>(MAX_L) : 4096;
    if (const char* tm = std::getenv("FLM_OPEN_TIMEOUT_MS")) cfg_.timeout_ms = static_cast<unsigned>(std::strtoul(tm, nullptr, 10));
    cfg_.verbose = std::getenv("FLM_OPEN_QUIET") == nullptr;
    core_ = std::make_unique<Core>(cfg_, dev_);
    logits_.assign(VOCAB, bf16(0.f));
}

Engine::~Engine() = default;

void Engine::load_weights(Q4NX&) { load_open_weights(); }

void Engine::load_open_weights() {
    core_->load_weights();
    core_->reset();
}

buffer<bf16> Engine::logits_view() {
    const std::vector<float>& lg = core_->logits();
    for (size_t i = 0; i < REAL_VOCAB; ++i) logits_[i] = bf16(lg[i]);
    // lm_head rows past the tokenizer's vocab are padding with undefined content
    for (size_t i = REAL_VOCAB; i < VOCAB; ++i) logits_[i] = bf16(-std::numeric_limits<float>::infinity());
    return buffer<bf16>(logits_.data(), logits_.size());
}

void Engine::ensure_alive() {
    if (!poisoned_) return;
    std::fprintf(stderr, "open_qwen36: a kernel failed on the last request; rebuilding the engine\n");
    core_.reset();
    core_ = std::make_unique<Core>(cfg_, dev_);
    core_->load_weights();
    core_->reset();
    has_snapshot_ = false;
    poisoned_ = false;
}

buffer<bf16> Engine::forward(int id) {
    return guarded([&] { core_->step(id, true); return logits_view(); });
}

buffer<bf16> Engine::prefill(std::vector<int>& ids, void* payload) {
    if (payload != nullptr)
        throw std::runtime_error("open_qwen36: the open engine has no vision path; images need the closed engine "
                                 "(FLM_QWEN36_ENGINE=closed)");
    if (ids.empty()) return logits_view();
    // Decode-as-prefill: exact for this architecture, one step per token, the
    // lm_head only for the last one (whose logits pick the first sampled token).
    return guarded([&] {
        for (size_t i = 0; i < ids.size(); ++i) core_->step(ids[i], i + 1 == ids.size());
        return logits_view();
    });
}

void Engine::set_context_length(int L) { update_max_length(static_cast<uint32_t>(L)); }

void Engine::update_max_length(uint32_t MAX_L) {
    if (MAX_L > core_->max_ctx())
        std::fprintf(stderr, "open_qwen36: context length %u exceeds the KV capacity %zu the engine was opened with; "
                             "capacity stays %zu\n", MAX_L, core_->max_ctx(), core_->max_ctx());
}

void Engine::clear_context() {
    ensure_alive();
    core_->reset();
    has_snapshot_ = false;
}

buffer<bf16> Engine::get_k_cache(int layer_idx, int idx) {
    buffer<bf16> out(512);
    if (!core_->is_attention_layer(layer_idx)) return out;
    core_->kv_row(layer_idx, idx, false, reinterpret_cast<uint16_t*>(out.data()));
    return out;
}

buffer<bf16> Engine::get_v_cache(int layer_idx, int idx) {
    buffer<bf16> out(512);
    if (!core_->is_attention_layer(layer_idx)) return out;
    core_->kv_row(layer_idx, idx, true, reinterpret_cast<uint16_t*>(out.data()));
    return out;
}

int Engine::get_current_context_length() { return core_->position(); }

int Engine::checkpoint() {
    if (poisoned_) return 0;
    snapshot_ = core_->checkpoint();
    has_snapshot_ = true;
    return snapshot_.pos;
}

int Engine::restore() {
    ensure_alive();
    if (!has_snapshot_) {
        core_->reset();
        return 0;
    }
    core_->restore(snapshot_);
    return snapshot_.pos;
}

}  // namespace open_qwen36
