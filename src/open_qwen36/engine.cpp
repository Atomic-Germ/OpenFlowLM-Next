/// \file engine.cpp
/// \brief The open Qwen3.6-MoE engine behind the app's causal_lm seam (see engine.hpp).
#include "open_qwen36/engine.hpp"

#include <chrono>
#include <fstream>

#include "models/qwen3_5vl/qwen3_5vl_npu.hpp"       // qwen3_5vl_image_payload_t
#include "models/qwen3_6_moe/qwen3_6_moe_npu.hpp"   // qwen3_6_moe_image_payload_t
#include "nlohmann/json.hpp"

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

std::string Engine::find_kernels(const LM_Config& config, std::string* how) {
    auto chose = [how](const char* where) { if (how) *how = where; };
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
    if (const char* env = std::getenv("OFLM_OPEN_KERNELS_DIR")) {
        if (complete(env, &why)) { chose("OFLM_OPEN_KERNELS_DIR"); return env; }
        std::fprintf(stderr, "open_qwen36: OFLM_OPEN_KERNELS_DIR=%s is not a kernel set: %s\n", env, why.c_str());
    }
    fs::path local = fs::path(config.model_path) / "open_kernels";
    if (complete(local, &why)) { chose("beside the model"); return local.string(); }
    // Every xclbins root the closed path would consider, not just the first one
    // find_xclbin_path() happens to return: oflm-add links a set under the user
    // root ($OFLM_XCLBIN_PATH / ~/.config/oflm) while the shipped sets live in the
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
        if (complete(cand, &why)) { chose("an xclbins root"); return cand.string(); }
    }
    return {};
}

Engine::Engine(const LM_Config& config, oflm_rt::device* dev, int MAX_L) : dev_(dev) {
    cfg_.model_dir = config.model_path;
    std::string how;
    cfg_.kernel_dir = find_kernels(config, &how);
    if (cfg_.kernel_dir.empty())
        throw std::runtime_error("open_qwen36: no open kernels found for " + config.model_name +
                                 " (set OFLM_OPEN_KERNELS_DIR or install xclbins/" + config.model_name + "/open_kernels)");
    cfg_.max_ctx = MAX_L > 0 ? static_cast<size_t>(MAX_L) : 4096;
    if (const char* tm = std::getenv("OFLM_OPEN_TIMEOUT_MS")) cfg_.timeout_ms = static_cast<unsigned>(std::strtoul(tm, nullptr, 10));
    cfg_.verbose = std::getenv("OFLM_OPEN_QUIET") == nullptr;
    // Say which set won. Three rules can pick one, and every one of them yields
    // valid output -- so a set chosen against the user's intent looks exactly
    // like the right one. An A/B that silently ran identical kernels twice, and
    // read as "no measurable effect", is what this line exists to prevent (#35).
    if (cfg_.verbose)
        std::fprintf(stderr, "open_qwen36: kernels %s (%s)\n", cfg_.kernel_dir.c_str(), how.c_str());
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

void Engine::ensure_vit() {
    if (vit_) return;
    auto t0 = std::chrono::steady_clock::now();
    vcfg_ = vision::VitConfig::from_model_dir(cfg_.model_dir);
    std::string file = "vision_weight.q4nx";
    {
        std::ifstream cf(cfg_.model_dir + "/config.json");
        auto j = nlohmann::json::parse(cf, nullptr, false);
        if (j.is_object()) file = j.value("vision_model_weight", file);
    }
    vit_ = std::make_unique<vision::VitWeights>(vision::load_vit(cfg_.model_dir + "/" + file, vcfg_));
    std::fprintf(stderr, "open_qwen36: vision tower resident (%s, %d blocks) in %.1f s\n", file.c_str(), vcfg_.depth,
                 std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
}

buffer<bf16> Engine::prefill(std::vector<int>& ids, void* payload) {
    if (ids.empty()) return logits_view();
    if (payload == nullptr) {
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
            const char* gemm_block_env = std::getenv("OFLM_OPEN_GEMM_BLOCK");
            // A prompt that has had an image is on the (t, h, w) counter, and the
            // gemm-block route writes no position records - it would place these
            // tokens at the wrong positions, so stay sequential.
            if (gemm_block_env && std::string(gemm_block_env) == "1" && !core_->mrope_active()) {
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
    // Images: the app has expanded each one into grid_h * grid_w / 4 image tokens and
    // hands the preprocessed patches for the whole prompt. The vision tower (host CPU)
    // turns each image into that many embedding rows; each row is stepped through the
    // model as a hidden vector at its (t, h, w) position, text tokens continue from the
    // same counter (Core::mrope_*), and later chunks / generated tokens inherit it.
    if (!core_->has_mrope() || core_->image_token_id() < 0)
        throw std::runtime_error("open_qwen36: config.json carries no image_token_id / mrope_section for this model; "
                                 "images need the closed engine (OFLM_QWEN36_ENGINE=closed)");
    // The app hands its own family's payload struct (qwen3_6_moe_image_payload_t /
    // qwen3_5vl_image_payload_t -- the same fields); read it through the one matching the
    // kernel set's family.
    const std::string& fam = core_->manifest().family;
    if (fam == "qwen36moe") return prefill_images(ids, *static_cast<const qwen3_6_moe_image_payload_t*>(payload));
    if (fam == "qwen35") return prefill_images(ids, *static_cast<const qwen3_5vl_image_payload_t*>(payload));
    throw std::runtime_error("open_qwen36: family " + fam + " has no vision path");
}

template <class Payload>
buffer<bf16> Engine::prefill_images(std::vector<int>& ids, const Payload& p_) {
    const Payload* p = &p_;
    ensure_vit();
    const size_t hidden = core_->manifest().hidden, pd = static_cast<size_t>(vcfg_.patch_dim());
    std::vector<std::vector<float>> embs;
    size_t off = 0;
    for (const auto& im : p->images) {
        const size_t n = static_cast<size_t>(im.grid_h) * im.grid_w;
        if (off + n * pd > p->_data__processed.size())
            throw std::runtime_error("open_qwen36: the image payload holds fewer pixels than its grids need");
        std::vector<float> px(n * pd);
        for (size_t k = 0; k < n * pd; ++k) px[k] = static_cast<float>(p->_data__processed[off + k]);
        off += n * pd;
        auto t0 = std::chrono::steady_clock::now();
        embs.push_back(vision::vit_forward(vcfg_, *vit_, px.data(), im.grid_h, im.grid_w));
        if (embs.back().size() != n / 4 * hidden)
            throw std::runtime_error("open_qwen36: the vision tower's width does not match the model's hidden size");
        std::fprintf(stderr, "open_qwen36: image %dx%d patches -> %zu tokens in %.2f s\n", im.grid_h, im.grid_w, n / 4,
                     std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
    }
    return guarded([&] {
        size_t img = 0, j = 0;
        int64_t base = 0;
        for (size_t i = 0; i < ids.size(); ++i) {
            const bool last = i + 1 == ids.size();
            if (ids[i] != core_->image_token_id()) {
                core_->step(ids[i], last);
                continue;
            }
            if (img >= p->images.size())
                throw std::runtime_error("open_qwen36: more image tokens in the prompt than images in the payload");
            const auto& im = p->images[img];
            const int64_t gh = im.grid_h / 2, gw = im.grid_w / 2;    // the merged token grid
            if (j == 0) {
                core_->mrope_begin();
                base = core_->mrope_pos();
            }
            const int64_t mpos[3] = {base, base + static_cast<int64_t>(j) / gw, base + static_cast<int64_t>(j) % gw};
            core_->step_embed(embs[img].data() + j * hidden, last, mpos);
            if (++j == static_cast<size_t>(gh * gw)) {
                core_->mrope_advance(std::max(gh, gw));
                ++img;
                j = 0;
            }
        }
        if (img != p->images.size() || j != 0)
            std::fprintf(stderr, "open_qwen36: WARNING: %zu image(s) in the payload, %zu consumed by the prompt\n",
                         p->images.size(), img + (j ? 1 : 0));
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
