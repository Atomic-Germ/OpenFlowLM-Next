/// \file engine.cpp
/// \brief The open Qwen3.6-MoE engine behind the app's causal_lm seam (see engine.hpp).
#include "open_qwen36/engine.hpp"

#include <chrono>
#include <fstream>

#include "models/qwen3_5vl/qwen3_5vl_npu.hpp"       // qwen3_5vl_image_payload_t
#include "models/qwen3_6_moe/qwen3_6_moe_npu.hpp"   // qwen3_6_moe_image_payload_t
#include "models/qwen3vl/qwen3vl_npu.hpp"           // qwen3vl_image_payload_t
#include "models/qwen2vl/qwen2vl_npu.hpp"           // qwen2vl_image_payload_t
#include "nlohmann/json.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <filesystem>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "open_qwen36/prefill_route_reason.hpp"
#include "open_qwen36/prefill_thresholds.hpp"
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
    // A directory that exists but fails the check is named with the reason: otherwise the
    // caller reports "no open kernels installed" for a set that is there and incomplete
    // (a GGUF-only build has no q4nx-route xclbins, and the check counts both routes).
    auto incomplete = [](const fs::path& dir, const std::string& reason) {
        std::error_code ec;
        if (fs::exists(dir, ec))
            std::fprintf(stderr, "open_qwen36: %s is not a complete kernel set: %s\n", dir.string().c_str(), reason.c_str());
    };
    fs::path local = fs::path(config.model_path) / "open_kernels";
    if (complete(local, &why)) { chose("beside the model"); return local.string(); }
    incomplete(local, why);
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
        incomplete(cand, why);
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
    if (const char* pm = std::getenv("OFLM_OPEN_PREFILL_MODE")) cfg_.prefill_mode = pm;   // oflm --prefill-mode
    if (const char* pp = std::getenv("OFLM_OPEN_PREFILL_PRECISION"))                      // oflm --prefill-precision
        cfg_.prefill_precision = prefill_precision_switch::parse(pp);
    // Accept both the direct environment value and settings-system booleans.
    const std::string speculative = utils::getenv_oflm("OFLM_SPECULATIVE");
    cfg_.draft = speculative == "1" || speculative == "true" || speculative == "on";
    if (const char* sm = std::getenv("OFLM_SPEC_MODE")) {
        const std::string s = sm;
        using M = SpecPolicy::Mode;
        spec_.mode = s == "uno" ? M::uno : s == "lookup" ? M::lookup : s == "lookup-uno" ? M::lookup_uno
                   : s == "plain" ? M::plain : M::adaptive;
    }
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

void Engine::note_failure(const char* what) {
    std::fprintf(stderr, "open_qwen36: request failed: %s\n", what ? what : "(no message)");
    std::fflush(stderr);
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

bool Engine::save_pre_prompt_state() {
    try {
        snapshot_ = core_->checkpoint();
        return true;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "open_qwen36: cannot snapshot the pre-prompt state: %s\n", e.what());
        return false;
    }
}

buffer<bf16> Engine::forward(int id) {
    return guarded([&] { core_->step(id, true); return logits_view(); });
}

void Engine::ensure_vit() {
    if (vit_) return;
    auto t0 = std::chrono::steady_clock::now();
    vcfg_ = vision::VitConfig::for_model_dir(cfg_.model_dir);
    // `vision_weight.q4nx` for a container, `mmproj*.gguf` for a GGUF install; the
    // config's own `vision_model_weight` wins when it names one. See vision_weights_path.
    const std::string file = vision::vision_weights_path(cfg_.model_dir);
    vit_ = std::make_unique<vision::VitWeights>(vision::load_vit(cfg_.model_dir + "/" + file, vcfg_));
    std::fprintf(stderr, "open_qwen36: vision tower resident (%s, %d blocks) in %.1f s\n", file.c_str(), vcfg_.depth,
                 std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
}

buffer<bf16> Engine::prefill(std::vector<int>& ids, void* payload) {
    // Counted before the early return so the diagnostic's "second prompt of the session"
    // is a property of the request, not of whether it happened to carry tokens.
    ++prefill_calls_;
    if (ids.empty()) return logits_view();
    if (payload == nullptr) {
        // Decode-as-prefill: exact for this architecture, one step per token, the
        // lm_head only for the last one (whose logits pick the first sampled token).
        //
        // 0167/#32: when the kernel set carries a GEMM-route prefill block
        // (Core::step_gemm_block(), manifest.hpp's GemmBlockProgram) the prompt
        // runs on it -- T tokens through every layer as whole-array bf16 GEMM
        // dispatches (q4_1 dequantised on-core) instead of T sequential step()
        // calls. On by default (#109); OFLM_OPEN_GEMM_BLOCK=0 turns it off. The
        // whole prompt runs in GT-wide blocks, the tail padded with a repeated
        // in-range id (hardware-proven exact: the real columns' output does not
        // depend on the padding), never falling back to step(). A kernel set with
        // no gemm_block program (GT == 0) runs exactly the sequential path this
        // engine always has.
        return guarded([&] {
            // Only =0 turns it off: unset, =1 or anything else leaves it on.
            // getenv_oflm, not getenv, so a pre-rename FLM_* export still works (#41).
            const bool gemm_block_on = utils::getenv_oflm("OFLM_OPEN_GEMM_BLOCK") != "0";
            // A block costs its full-width GEMMs however few real tokens it holds, so a
            // short prompt is cheaper one token at a time. The crossover is measured on
            // hardware, never derived. Below it, everything is stepped.
            //
            // Above it there are two schedules and only one of them is the right default:
            //
            //   layer_major_preferred()  each layer once over the whole prompt. What this
            //                             buys is ONE routed-expert pass a layer, so an
            //                             expert's weights stream once per layer instead of
            //                             once per block -- the Qwen3.6-35B measurement.
            //                             It also pads the prompt to whole blocks
            //                             (`TOT = B * T`), which costs a full-width
            //                             dispatch for a short tail.
            //   the block loop below      whole 256-wide blocks with the tail STEPPED, one
            //                             pass per block. Phase 28 measured this against
            //                             layer-major on a dense kernel set (no routed
            //                             experts, so no expert amortisation to lose):
            //                             1.80x faster at 264 engine tokens, 1.42x at 520,
            //                             1.21x at 1032, 1.11x at 2056, never slower --
            //                             the whole gap is layer-major's padded tail, and
            //                             it decays as ceil(N/256) over that schedule's
            //                             pass count as N grows.
            //
            // The two crossover constants are NOT the same number and are not meant to be:
            // this one is `prefill::kDenseBlockLoop` (96) and applies to the block loop on a
            // dense set; `prefill::kLayerMajor` (64) belongs to the layer-major schedule and
            // the MoE families that measured it. What is true is that the block loop is the
            // schedule to use for every kernel set that does not carry routed experts, at
            // every length above THIS crossover. `layer_major_preferred()` is what keeps the
            // MoE sets on the schedule they were measured to want -- capability decides
            // whether a route can run, preference decides whether it should.
            //
            // The value counts tokens AS Engine::prefill receives them, not text tokens:
            // the chat template's offset depends on how its boundary merges with the first
            // text token (8, 10 and 11 have all been observed). Phase 27.4 sampled only
            // 1/8/32/64/128/192/255 and reported "the first sampled point where the block
            // wins", a grid artifact whose own table crosses near 81; the closure sweep
            // measured the flat block at 33.47 s against 425.5 ms/token + 156 ms
            // sequential, a crossing of ~78 with the first clear win at 92 (0.85).
            const size_t block_loop_crossover = prefill::kDenseBlockLoop;
            size_t min_prompt = block_loop_crossover;
            if (const char* mp = std::getenv("OFLM_OPEN_GEMM_BLOCK_MIN")) min_prompt = static_cast<size_t>(std::strtoul(mp, nullptr, 10));
            // Diagnostic: which route this request took, and how the prompt was cut. A
            // test must not infer this from elapsed time -- a short sequential prompt and
            // a padded block overlap in wall time, so timing cannot prove which ran. Two
            // numbers matter and they are different: `n` is the token count this function
            // received, `blocks` is how many full-width passes it becomes.
            const bool report_route = std::getenv("OFLM_OPEN_PREFILL_ROUTE") != nullptr;
            // A prompt that has had an image is on the (t, h, w) counter, and the
            // gemm-block route writes no position records - it would place these
            // tokens at the wrong positions, so stay sequential.
            // The dynamic half of the route diagnostic. It is deliberately per-request state
            // and not recomputed at the end: a block route that threw and was rewound leaves
            // the same switch, crossover and capability behind it as one that never ran, so
            // the static state cannot tell a recovered failure from a policy choice.
            prefill_route::Attempt attempt = prefill_route::Attempt::NotAttempted;
            if (gemm_block_on && ids.size() >= min_prompt && !core_->mrope_active()) {
                // B(2) of .claude/plans/prefill-parity.md: on a MoE kernel set, run the
                // whole prompt through each layer before the next, so the layer's expert
                // pass sees every token that wants an expert instead of one block's worth.
                // Bit-exact against the block loop below; OFLM_OPEN_LAYER_MAJOR=0 takes it,
                // and so does any kernel set layer_major_preferred() says not to use.
                if (core_->layer_major_preferred()) {
                    if (report_route)
                        std::fprintf(stderr, "prefill-route layer-major n=%zu blocks=1 tail=0\n",
                                     ids.size());
                    core_->step_gemm_prompt(ids, true);
                    return logits_view();
                }
                const size_t GT = core_->gemm_block_t();
                if (GT > 0) {
                    // How the prompt divides, decided before anything runs so the report
                    // is available even when the route then fails. The tail is padded to a
                    // block only when it is at least `min_prompt` long; below that it is
                    // stepped, because a block costs its full width regardless.
                    const size_t full_blocks = ids.size() / GT;
                    const size_t tail = ids.size() % GT;
                    const size_t padded_tail = (tail >= min_prompt) ? 1 : 0;
                    const size_t passes = full_blocks + padded_tail;
                    if (report_route)
                        std::fprintf(stderr,
                                     "prefill-route block n=%zu blocks=%zu tail=%zu padded=%zu passes=%zu\n",
                                     ids.size(), full_blocks, tail, padded_tail, passes);
                    // Diagnostic fault injection, for exercising the 27.11 handler on
                    // hardware rather than arguing about it from the source. Values:
                    //   before        fail the FIRST block, before it writes anything
                    //                 (case A: nothing has moved, so a reset is honest)
                    //   after         fail the SECOND block, after the first moved state
                    //                 (case B: the snapshot rewind has to undo block 0)
                    //   unrecoverable fail the second block AND break the rewind, so the
                    //                 throw-without-fallback branch is reachable (case C).
                    //                 It must NOT throw on pass 0: that is case A, where
                    //                 falling back is legitimate, and a version that threw
                    //                 there would "pass" case C without ever testing it.
                    //
                    // Both `after` and `unrecoverable` need a prompt that really reaches a
                    // second block, i.e. at least GT + min_prompt tokens -- a 300-token
                    // prompt is 256 plus a tail the crossover steers away from a block.
                    //
                    // `prefill_calls_` selects WHICH prefill of the session is broken, so
                    // the snapshot branch is reachable at all: a fresh prompt is at position
                    // 0, where reset() is free and no snapshot is taken, so breaking it can
                    // only ever exercise the reset branch. The snapshot branch is the one
                    // that has to undo a block that already ran, and it exists only for a
                    // prompt that starts at position != 0 -- the second prompt of a session.
                    const char* inject = std::getenv("OFLM_OPEN_BLOCK_FAIL_INJECT");
                    const std::string inject_mode = inject ? inject : "";
                    const char* inject_at = std::getenv("OFLM_OPEN_BLOCK_FAIL_AT");
                    const int armed_call = inject_at ? std::atoi(inject_at) : 0;
                    const bool armed = !inject_mode.empty() &&
                                       (!inject_at || prefill_calls_ >= armed_call);
                    auto fail_if = [&](size_t pass) {
                        if (!armed) return;
                        const bool first = inject_mode == "before";
                        const bool second = inject_mode == "after" || inject_mode == "unrecoverable";
                        if (pass == 0 && first)
                            throw std::runtime_error("injected block failure before the first block");
                        if (pass == 1 && second)
                            throw std::runtime_error("injected block failure after one block");
                    };
                    // 27.11: an UNEXPECTED failure on a route the kernel set advertises must
                    // not be disguised as "this set has no block route". If it happens
                    // before any block has run, nothing has been written to the caches and
                    // the sequential path below is exactly as correct as it would have been
                    // -- that is a genuine fallback. If it happens after a block has run,
                    // state has moved, so the caches are put back to where they were before
                    // the prompt and only then does the sequential path run. Continuing on
                    // top of a half-updated cache would silently produce the wrong answer
                    // with the right shape, which is the failure mode worth the extra code.
                    //
                    // The "where they were" is a snapshot, and taking one is a full readback
                    // of every layer's state, so it is not taken where it cannot be needed:
                    //
                    //   position 0      the reset state IS the pre-prompt state, so
                    //                   reset() undoes the prompt for free.
                    //   one block       the failure takes the whole prompt with it, so
                    //                   there is no half-applied cache to undo; if it did
                    //                   move state the throw below reports that rather
                    //                   than falling back onto a wrong cache.
                    //   multi-block     this is the case the plan names -- block 1 is in
                    //                   the caches and block 2 fails.
                    //
                    // So the common single-block path pays nothing, which is what 27.14
                    // requires, and the only path that can be left half-applied pays once.
                    const bool recover_by_reset = core_->position() == 0;
                    const bool multi_block = ids.size() > GT;
                    const bool have_snapshot =
                        !recover_by_reset && multi_block && save_pre_prompt_state();
                    auto rewind = [&]() -> bool {
                        // `unrecoverable` deliberately breaks the recovery too, so the
                        // throw-without-fallback branch can be seen on hardware instead of
                        // only read in the source.
                        if (armed && inject_mode == "unrecoverable") return false;
                        try {
                            if (recover_by_reset) { core_->reset(); return true; }
                            if (have_snapshot) { core_->restore(snapshot_); return true; }
                        } catch (...) {
                        }
                        return false;
                    };
                    size_t i = 0;
                    size_t pass = 0;
                    size_t tail_seq = 0;
                    bool ran = false;
                    try {
                        // The crossover decides the tail as well as the prompt, and it has
                        // to: a padded block costs its full width whatever it holds, so a
                        // 1-token tail on a 257-token prompt must NOT buy a second block
                        // pass. Blocks run only while at least `min_prompt` real tokens
                        // remain; anything shorter is stepped one at a time, which Phase 26
                        // proved produces the same state and the same token. Without this a
                        // 257-token prompt cost two passes instead of one plus one token.
                        while (i < ids.size()) {
                            const size_t remaining = ids.size() - i;
                            if (remaining < min_prompt) break;
                            const size_t t_real = std::min(GT, remaining);
                            std::vector<int> block(ids.begin() + static_cast<long>(i), ids.begin() + static_cast<long>(i + t_real));
                            block.resize(GT, block.empty() ? 0 : block.back());  // pad the tail with a repeated in-range id
                            fail_if(pass);
                            ran = true;
                            core_->step_gemm_block(block, t_real, i + t_real == ids.size());
                            i += t_real;
                            ++pass;
                        }
                        for (; i < ids.size(); ++i, ++tail_seq)
                            core_->step(ids[i], i + 1 == ids.size());
                        if (report_route && tail_seq)
                            std::fprintf(stderr, "prefill-route tail-sequential n=%zu tokens=%zu\n",
                                         ids.size(), tail_seq);
                        return logits_view();
                    } catch (const std::exception& e) {
                        // Only here, in the handler: a message emitted before the try would
                        // announce a failure on every run that succeeds.
                        if (ran && !rewind())
                            throw std::runtime_error(std::string(
                                "open_qwen36: block prefill failed after part of the prompt had "
                                "already run, and the pre-prompt state could not be restored: ") +
                                e.what());
                        std::fprintf(stderr,
                                     "open_qwen36: block prefill failed on a set that advertises it: %s\n"
                                     "open_qwen36: %s, then the sequential path\n", e.what(),
                                     recover_by_reset ? "cache reset"
                                                      : (have_snapshot ? "pre-prompt state restored"
                                                                       : "state NOT rewound"));
                        // Reaching here means the prompt WILL be stepped below from a state
                        // that was put back: the one case the telemetry calls `fallback`. The
                        // unrecoverable branch above throws instead of falling through, so a
                        // hard failure never reaches this assignment and is never reported as
                        // a successful fallback.
                        attempt = prefill_route::Attempt::FailedRecovered;
                        // fall through to the sequential path, from the restored state
                    }
                }
                // no route in this kernel set: fall through to the sequential path below
                // (this used to return here, leaving the prompt unprocessed -- #109)
            }
            // The sequential path reports itself the same way, including the reason the
            // block route was not taken, so a matrix can assert the policy rather than
            // infer it: below-crossover is a decision, disabled is a switch, no-program
            // is a capability, and mrope is a safety rule -- four different facts.
            if (report_route) {
                // `not-preferred` is capability without preference: the set CAN run
                // layer-major but has nothing for it to amortise, so the block loop is the
                // schedule to use. Reporting it separately from `fallback` matters, because
                // one is a decision and the other is a failure.
                //
                // `attempt` is the part that cannot be read off the static state: a block
                // route that threw and was rewound leaves the same switch, crossover and
                // capability behind it as one that never ran. `prefill_route::reason()`
                // gives the failure priority so a recovered crash is never reported as a
                // policy choice.
                const prefill_route::Policy policy =
                    !gemm_block_on                 ? prefill_route::Policy::Disabled
                    : ids.size() < min_prompt     ? prefill_route::Policy::BelowCrossover
                    : core_->mrope_active()       ? prefill_route::Policy::Mrope
                    : core_->gemm_block_t() == 0  ? prefill_route::Policy::NoBlockProgram
                    : (core_->layer_major_ok() && !core_->layer_major_preferred())
                          ? prefill_route::Policy::NotPreferred
                          : prefill_route::Policy::Enabled;
                const char* why = prefill_route::reason(attempt, policy);
                std::fprintf(stderr, "prefill-route sequential n=%zu blocks=0 tail=0 why=%s "
                                     "crossover=%zu layer-major-capable=%d\n",
                             ids.size(), why, min_prompt,
                             core_->layer_major_ok() ? 1 : 0);
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
                                 "images need the closed engine -- set this family's engine switch "
                                 "OFLM_<FAMILY>_ENGINE=closed (e.g. OFLM_GEMMA_ENGINE=closed)");
    // The app hands its own family's payload struct (qwen3_6_moe_image_payload_t /
    // qwen3_5vl_image_payload_t -- the same fields); read it through the one matching the
    // kernel set's family.
    const std::string& fam = core_->manifest().family;
    if (fam == "qwen36moe") return prefill_images(ids, *static_cast<const qwen3_6_moe_image_payload_t*>(payload));
    if (fam == "qwen35") return prefill_images(ids, *static_cast<const qwen3_5vl_image_payload_t*>(payload));
    // Qwen3-VL's decoder derives as plain Qwen3, so its kernel set is a qwen3 one and the
    // family string does not say "VL" - only the payload does.
    if (fam == "qwen3") return prefill_images(ids, *static_cast<const qwen3vl_image_payload_t*>(payload));
    // Qwen2.5-VL likewise: its decoder derives as plain qwen2 and shares a kernel set with
    // Qwen2.5-3B-Instruct, so the family says qwen2 and the windowed tower comes from the
    // container's own vision_config.
    if (fam == "qwen2") return prefill_images(ids, *static_cast<const qwen2vl_image_payload_t*>(payload));
    throw std::runtime_error("open_qwen36: family " + fam + " has no vision path");
}

template <class Payload>
buffer<bf16> Engine::prefill_images(std::vector<int>& ids, const Payload& p_) {
    const Payload* p = &p_;
    ensure_vit();
    const size_t hidden = core_->manifest().hidden, pd = static_cast<size_t>(vcfg_.patch_dim());
    std::vector<std::vector<float>> embs;
    // Qwen3-VL only: one [tokens, hidden] block per deepstack tap, per image. Empty for
    // every other family, and vit_forward_deepstack is exactly vit_forward when the
    // config lists no taps.
    std::vector<std::vector<std::vector<float>>> deeps;
    size_t off = 0;
    for (const auto& im : p->images) {
        const size_t n = static_cast<size_t>(im.grid_h) * im.grid_w;
        if (off + n * pd > p->_data__processed.size())
            throw std::runtime_error("open_qwen36: the image payload holds fewer pixels than its grids need");
        std::vector<float> px(n * pd);
        for (size_t k = 0; k < n * pd; ++k) px[k] = static_cast<float>(p->_data__processed[off + k]);
        off += n * pd;
        auto t0 = std::chrono::steady_clock::now();
        std::vector<std::vector<float>> deep;
        embs.push_back(vision::vit_forward_deepstack(vcfg_, *vit_, px.data(), im.grid_h, im.grid_w, &deep));
        for (const auto& f : deep)
            if (f.size() != n / 4 * hidden)
                throw std::runtime_error("open_qwen36: a deepstack feature is not [tokens, hidden]");
        deeps.push_back(std::move(deep));
        if (embs.back().size() != n / 4 * hidden)
            throw std::runtime_error("open_qwen36: the vision tower's width does not match the model's hidden size");
        std::fprintf(stderr, "open_qwen36: image %dx%d patches -> %zu tokens%s in %.2f s\n", im.grid_h, im.grid_w, n / 4,
                     deeps.back().empty() ? "" : (" + " + std::to_string(deeps.back().size()) + " deepstack").c_str(),
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
            // Gather this row's deepstack features into one contiguous block, feature k at
            // offset k * hidden, which is the order step_impl adds them in.
            std::vector<float> ds;
            if (!deeps[img].empty()) {
                ds.resize(deeps[img].size() * hidden);
                for (size_t k = 0; k < deeps[img].size(); ++k)
                    std::memcpy(ds.data() + k * hidden, deeps[img][k].data() + j * hidden, hidden * sizeof(float));
            }
            core_->step_embed(embs[img].data() + j * hidden, last, mpos, ds.empty() ? nullptr : ds.data(),
                              static_cast<int>(deeps[img].size()));
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
    buffer<bf16> out(core_->manifest().kv_row / (core_->manifest().attn_kv_pair ? 8 : 4));   // one K row: bf16[kv heads x head dim]
    if (!core_->is_attention_layer(layer_idx)) return out;
    core_->kv_row(layer_idx, idx, false, reinterpret_cast<uint16_t*>(out.data()));
    return out;
}

buffer<bf16> Engine::get_v_cache(int layer_idx, int idx) {
    buffer<bf16> out(core_->manifest().kv_row / (core_->manifest().attn_kv_pair ? 8 : 4));
    if (!core_->is_attention_layer(layer_idx)) return out;
    core_->kv_row(layer_idx, idx, true, reinterpret_cast<uint16_t*>(out.data()));
    return out;
}

int Engine::get_current_context_length() { return core_->position(); }

bool Engine::uno_ok() const { return core_ && core_->rows_l() && core_->has_draft(); }

int Engine::uno_passes(int seed, std::vector<int>& out, double& draft_ms, double& verify_ms, size_t& accepted) {
    const size_t L = core_->rows_l();
    std::uniform_int_distribution<int> noise(1, static_cast<int>(core_->uno_noise_bound()) - 1);   // the adapter's [1, N)
    std::vector<int> draft(L), am(L), av(L);
    const int p = core_->position();
    draft[0] = seed;
    for (size_t j = 1; j < L; ++j) draft[j] = noise(uno_rng_);
    core_->step_rows(draft.data(), am.data(), false, true);    // am[0] = c, am[1..] the drafts
    draft_ms = core_->rows_ms();
    core_->seek(p + 1);
    core_->step_rows(am.data(), av.data(), false, false);      // av[j] checks am[j + 1]
    verify_ms = core_->rows_ms();
    size_t k = 0;
    while (k + 1 < L && am[k + 1] == av[k]) ++k;
    for (size_t j = 0; j <= k; ++j) out.push_back(am[j]);
    out.push_back(av[k]);
    core_->seek(p + 2 + static_cast<int>(k));
    accepted = k;
    return p;
}

int Engine::uno_cycle(int seed, std::vector<int>& out) {
    return guarded([&] {
        double d = 0, v = 0;
        size_t k = 0;
        return uno_passes(seed, out, d, v, k);
    });
}

bool Engine::spec_ok() const { return core_ && cfg_.draft && core_->rows_l(); }

void Engine::spec_begin() {
    spec_.reset_estimates();
    spec_plain_ = false;
}

bool Engine::spec_cycle(int seed, const std::vector<int>& history, std::vector<int>& out, int& p) {
    return guarded([&] {
        const size_t L = core_->rows_l(), cap = core_->max_ctx();
        if (spec_plain_) spec_.record(SpecMethod::plain, 0, 0, core_->last_timing().total_ms, 0, 0);
        spec_plain_ = false;
        if (history.empty() || history.back() != seed)
            throw std::runtime_error("open_qwen36: spec_cycle's history must end with the seed");
        p = core_->position();
        std::vector<int> drafts;
        const int matched = spec_.uses_lookup() ? lookup_drafts(history.data(), history.size(), L - 1, 4, 1, drafts) : 0;
        // strict: the cycle's last seek lands on its next position, which must itself be inside the capacity
        const SpecMethod m = spec_.choose(matched, static_cast<size_t>(p) + L < cap,
                                          core_->has_draft() && static_cast<size_t>(p) + 1 + L < cap);
        if (m == SpecMethod::plain) {
            spec_plain_ = true;
            return false;
        }
        size_t k = 0;
        if (m == SpecMethod::uno) {
            double d = 0, v = 0;
            uno_passes(seed, out, d, v, k);
            spec_.record(m, matched, static_cast<int>(k), 0, v, d);
            return true;
        }
        std::vector<int> vr(L), av(L);
        vr[0] = seed;
        for (size_t j = 1; j < L; ++j) vr[j] = j - 1 < drafts.size() ? drafts[j - 1] : vr[j - 1];
        core_->step_rows(vr.data(), av.data(), false, false);
        while (k + 1 < L && vr[k + 1] == av[k]) ++k;    // draft k+1 checked by verify row k
        for (size_t j = 0; j <= k; ++j) out.push_back(av[j]);
        core_->seek(p + 1 + static_cast<int>(k));
        spec_.record(m, matched, static_cast<int>(k), 0, core_->rows_ms(), 0);
        return true;
    });
}

void Engine::uno_seek(int pos) {
    guarded([&] { core_->seek(pos); });
}

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
