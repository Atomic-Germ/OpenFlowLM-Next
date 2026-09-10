/// \file engine.cpp
/// \brief The open Qwen3.6-MoE engine behind the app's causal_lm seam (see engine.hpp).
#include "open_qwen36/engine.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "utils/utils.hpp"

namespace open_qwen36 {

namespace fs = std::filesystem;

std::string Engine::find_kernels(const LM_Config& config) {
    // A kernel set is a manifest.json plus every xclbin / insts.bin it names.
    auto complete = [](const fs::path& dir, std::string* why) {
        std::error_code ec;
        if (!fs::is_regular_file(dir / "manifest.json", ec)) { *why = "no manifest.json"; return false; }
        try {
            Manifest m = Manifest::load((dir / "manifest.json").string());
            for (const auto& f : m.files())
                if (!fs::is_regular_file(dir / f, ec)) { *why = "manifest names missing " + f; return false; }
        } catch (const std::exception& e) {
            *why = e.what();
            return false;
        }
        return true;
    };
    std::string why;
    if (const char* env = std::getenv("FLM_OPEN_KERNELS_DIR")) {
        if (complete(env, &why)) return env;
        std::fprintf(stderr, "open_qwen36: FLM_OPEN_KERNELS_DIR=%s is not a kernel set: %s\n", env, why.c_str());
    }
    fs::path local = fs::path(config.model_path) / "open_kernels";
    if (complete(local, &why)) return local.string();
    // Every xclbins root the closed path would consider, not just the first one
    // find_xclbin_path() happens to return: flm-add links a set under the user
    // root ($FLM_XCLBIN_PATH / ~/.config/flm) while the shipped sets live in the
    // install tree, and whichever root wins there would otherwise hide the other.
    std::vector<std::string> roots = utils::xclbin_roots();
    // config.exec_path is find_xclbin_path()'s single winner, already in the list above --
    // except under DEV_BUILD, where LM_Config hard-codes a relative tree.
    if (!config.exec_path.empty() &&
        std::find(roots.begin(), roots.end(), config.exec_path) == roots.end()) {
        roots.push_back(config.exec_path);
    }
    for (const std::string& r : roots) {
        fs::path cand = fs::path(r) / "xclbins" / config.model_name / "open_kernels";
        if (complete(cand, &why)) return cand.string();
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
    logits_.assign(core_->vocab(), bf16(0.f));
}

Engine::~Engine() = default;

void Engine::load_weights(Q4NX&) { load_open_weights(); }

void Engine::load_open_weights() {
    core_->load_weights();
    core_->reset();
}

buffer<bf16> Engine::logits_view() {
    const std::vector<float>& lg = core_->logits();
    const size_t real = core_->real_vocab(), vocab = core_->vocab();
    for (size_t i = 0; i < real; ++i) logits_[i] = bf16(lg[i]);
    // lm_head rows past the tokenizer's vocab are padding with undefined content
    for (size_t i = real; i < vocab; ++i) logits_[i] = bf16(-std::numeric_limits<float>::infinity());
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
    //
    // 0167/#32: FLM_OPEN_GEMM_BLOCK=1 selects the GEMM-route prefill block
    // (Core::step_gemm_block(), manifest.hpp's GemmBlockProgram) instead --
    // T tokens through every layer as 5 whole-array bf16 GEMM dispatches
    // (q4_1 dequantised on-core) plus T single-token attention dispatches,
    // instead of T sequential step() calls. Off by default so every existing
    // measurement is unaffected. The whole prompt runs in GT-wide blocks, the
    // tail padded with a repeated in-range id (hardware-proven exact: the
    // real columns' output does not depend on the padding), never falling
    // back to step(). A kernel set with no gemm_block program (GT == 0) runs
    // exactly the sequential path this engine always has.
    return guarded([&] {
        // Test the VALUE, not just presence: the docs say =1, and
        // FLM_OPEN_GEMM_BLOCK=0 switching the route ON is the kind of surprise
        // that gets diagnosed as a different bug entirely (review on #39).
        const char* gemm_block_env = std::getenv("FLM_OPEN_GEMM_BLOCK");
        if (gemm_block_env && std::string(gemm_block_env) == "1") {
            const size_t GT = core_->gemm_block_t();
            if (GT > 0) {
                size_t i = 0;
                while (i < ids.size()) {
                    const size_t t_real = std::min(GT, ids.size() - i);
                    std::vector<int> block(ids.begin() + static_cast<long>(i), ids.begin() + static_cast<long>(i + t_real));
                    block.resize(GT, block.empty() ? 0 : block.back());  // pad the tail with a repeated in-range id
                    core_->step_gemm_block(block, t_real, i + t_real == ids.size());
                    i += t_real;
                }
                return logits_view();
            }
        }
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
    buffer<bf16> out(core_->manifest().kv_row / 4);   // one K row: bf16[kv heads x head dim]
    if (!core_->is_attention_layer(layer_idx)) return out;
    core_->kv_row(layer_idx, idx, false, reinterpret_cast<uint16_t*>(out.data()));
    return out;
}

buffer<bf16> Engine::get_v_cache(int layer_idx, int idx) {
    buffer<bf16> out(core_->manifest().kv_row / 4);
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
