/// \file test.cpp
/// \brief Validate the open Gemma3 text engine against the NumPy oracle.
/// \note Standalone by design: src/test/CMakeLists.txt unconditionally links the
/// closed q4_npu_eXpress/mha/dequant/gemm/lm_head stack, which this engine does
/// not use.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "nlohmann/json.hpp"
#include "open_gemma3/engine.hpp"
#include "open_gemma3/gemma3_text_open.hpp"
#include "tokenizer/tokenizer.hpp"
#include "utils/utils.hpp"
#include "utils/vm_args.hpp"

static float cosine(const std::vector<float>& a, const std::vector<float>& b) {
    double dot = 0.0, na = 0.0, nb = 0.0;
    for (size_t i = 0; i < a.size() && i < b.size(); ++i) {
        dot += static_cast<double>(a[i]) * b[i];
        na += static_cast<double>(a[i]) * a[i];
        nb += static_cast<double>(b[i]) * b[i];
    }
    if (na == 0.0 || nb == 0.0) return 0.0f;
    return static_cast<float>(dot / (std::sqrt(na) * std::sqrt(nb)));
}

int main(int argc, char* argv[]) {
    arg_utils::po::options_description desc("Allowed options");
    arg_utils::po::variables_map vm;
    desc.add_options()("model,m", arg_utils::po::value<std::string>()->required(), "Model dir");
    desc.add_options()("reference,r", arg_utils::po::value<std::string>()->required(),
                       "Reference fixture JSON");
    arg_utils::po::store(arg_utils::po::parse_command_line(argc, argv, desc), vm);

    const std::string model_dir = vm["model"].as<std::string>();
    const std::string ref_path = vm["reference"].as<std::string>();

    std::ifstream rf(ref_path);
    if (!rf) {
        std::fprintf(stderr, "cannot open reference: %s\n", ref_path.c_str());
        return 2;
    }
    nlohmann::json ref;
    rf >> ref;

    open_gemma3::Engine engine;
    if (!engine.load(model_dir)) {
        std::fprintf(stderr, "failed to load model: %s\n", model_dir.c_str());
        return 1;
    }
    if (!engine.enable_cache(1024)) {
        std::fprintf(stderr, "failed to allocate KV cache\n");
        return 1;
    }
    {
        // 2 tensors (K and V) x layers x positions x (num_key_value_heads*head_dim) floats.
        const double bytes = 2.0 * static_cast<double>(engine.num_layers()) *
                             static_cast<double>(engine.cache_capacity()) * 256.0 * sizeof(float);
        std::printf("KV cache: %zu layers x %zu positions (%.1f MB fp32)\n", engine.num_layers(),
                    engine.cache_capacity(), bytes / (1024.0 * 1024.0));
    }
    if (engine.vocab() == 0) {
        std::fprintf(stderr, "engine reports zero vocab\n");
        return 1;
    }

    int failures = 0;
    const auto& prompts = ref.at("prompts");

    // End-to-end check through the causal_lm adapter and the runtime tokenizer:
    // tokenize -> prefill -> greedy decode -> decode back to text. This is what
    // exercises the buffer<bf16> logit conversion the engine itself never sees.
    {
        Tokenizer tok(model_dir);
        Gemma3TextOpen adapter(*(new LM_Config()), nullptr, 512);
        if (!adapter.load_model_dir(model_dir)) {
            std::fprintf(stderr, "adapter load failed\n");
            return 1;
        }
        for (int pi = 0; pi < 2; ++pi) {
            // Use the oracle's token ids (which include BOS). The runtime
            // Tokenizer::encode does not prepend BOS itself; AutoModel supplies
            // it via the chat template, and the oracle records the full id list.
            std::vector<int> ids;
            for (const auto& id : prompts[pi].at("token_ids")) ids.push_back(id.get<int>());
            adapter.clear_context();

            buffer<bf16> logits = adapter.prefill(ids);
            std::printf("       [dbg] ids=");
            for (int v : ids) std::printf("%d ", v);
            std::printf("\n       [dbg] buffer size=%zu vocab=%zu\n", logits.size(),
                        open_gemma3::Engine().vocab());
            std::printf("       [dbg] buffer[0..4]=");
            for (size_t i = 0; i < 5 && i < logits.size(); ++i) {
                std::printf("%.4f ", static_cast<float>(logits[i]));
            }
            std::printf("\n       [dbg] buffer[9079]=%.4f\n",
                        static_cast<float>(logits[9079]));
            int first = 0;
            float best = -1e30f;
            for (size_t i = 0; i < logits.size(); ++i) {
                if (static_cast<float>(logits[i]) > best) {
                    best = static_cast<float>(logits[i]);
                    first = static_cast<int>(i);
                }
            }

            std::vector<int> gen;
            gen.push_back(first);
            for (int s = 0; s < 15; ++s) {
                buffer<bf16> step = adapter.forward(gen.back());
                int arg = 0;
                float b = -1e30f;
                for (size_t i = 0; i < step.size(); ++i) {
                    if (static_cast<float>(step[i]) > b) {
                        b = static_cast<float>(step[i]);
                        arg = static_cast<int>(i);
                    }
                }
                gen.push_back(arg);
            }

            const int expected = prompts[pi].at("argmax_token_id").get<int>();
            std::printf("[%s] adapter prompt %zu: first argmax %d (expected %d)\n",
                        first == expected ? "PASS" : "FAIL", static_cast<size_t>(pi), first,
                        expected);
            std::printf("       continuation: %s\n", tok.decode(gen).c_str());
            if (first != expected) ++failures;
        }
    }

    // Self-consistency: incremental decode through the KV cache must reproduce
    // the validated full-recompute path. This is what actually tests the cache,
    // including the hybrid sliding/global window logic.
    //
    // Only practical for short prompts here because decoding re-validates every
    // prefix, and the long prompt would need thousands of steps.
    int cache_failures = 0;
    int cache_checked = 0;
    for (size_t pi = 0; pi < prompts.size(); ++pi) {
        std::vector<int32_t> ids;
        for (const auto& id : prompts[pi].at("token_ids")) ids.push_back(id.get<int32_t>());
        if (ids.size() > 64) continue;

        const std::vector<float> want = engine.prefill(ids);

        engine.clear_context();
        std::vector<float> got;
        for (size_t i = 0; i < ids.size(); ++i) {
            got = engine.step({ids[i]});
        }
        ++cache_checked;

        const float cos = cosine(want, got);
        int arg_want = static_cast<int>(std::max_element(want.begin(), want.end()) - want.begin());
        int arg_got = static_cast<int>(std::max_element(got.begin(), got.end()) - got.begin());
        const bool ok = (cos > 0.9999f) && (arg_want == arg_got);
        if (!ok) ++cache_failures;
        std::printf("[%s] cache prompt %zu (%zu tokens): cosine %.6f argmax %s\n",
                    ok ? "PASS" : "FAIL", pi, ids.size(), cos,
                    arg_want == arg_got ? "ok" : "MISMATCH");
    }
    std::printf("\n%d/%d prompts: incremental decode matched full prefill\n",
                cache_checked - cache_failures, cache_checked);
    if (cache_failures) ++failures;

    for (size_t pi = 0; pi < prompts.size(); ++pi) {
        const auto& entry = prompts[pi];
        std::vector<int32_t> ids;
        for (const auto& id : entry.at("token_ids")) ids.push_back(id.get<int32_t>());

        std::vector<float> logits = engine.prefill(ids);

        const int expected_argmax = entry.at("argmax_token_id").get<int>();
        int got_argmax = static_cast<int>(
            std::max_element(logits.begin(), logits.end()) - logits.begin());

        // Compare against the oracle's top-k logits.
        double worst_abs = 0.0;
        int matched_top = 0;
        const auto& top = entry.at("top_logits");
        const int top_n = static_cast<int>(top.size());
        for (const auto& pair : top) {
            const int idx = pair[0].get<int>();
            const float expected = pair[1].get<float>();
            const float got = logits[static_cast<size_t>(idx)];
            worst_abs = std::max(worst_abs, static_cast<double>(std::fabs(got - expected)));
            if (std::fabs(got - expected) < 1e-2f) ++matched_top;
            (void)top_n;
        }

        // Greedy continuation: re-forward each step (no KV cache yet). This
        // validates the decode path before it is optimized.
        int cont_ok = -1;  // -1 = not present in the fixture
        if (entry.at("greedy_continuation").contains("ids")) {
            const auto& expected_ids = entry.at("greedy_continuation").at("ids");
            if (!expected_ids.empty()) {
                cont_ok = 0;
                std::vector<int32_t> running = ids;
                for (const auto& eid : expected_ids) {
                    std::vector<float> step = engine.prefill(running);
                    int step_argmax = static_cast<int>(
                        std::max_element(step.begin(), step.end()) - step.begin());
                    if (step_argmax == eid.get<int>()) ++cont_ok;
                    running.push_back(step_argmax);
                }
            }
        }

        const bool ok = (got_argmax == expected_argmax) && (matched_top == top_n) &&
                        (cont_ok < 0 || cont_ok == static_cast<int>(
                            entry.at("greedy_continuation").at("ids").size()));
        if (!ok) ++failures;

        std::printf("[%s] prompt %zu (%zu tokens): argmax %s (expected %d, got %d) "
                    "top-%d match %d/%d worst|diff| %.4f",
                    ok ? "PASS" : "FAIL", pi, ids.size(), ok ? "ok" : "MISMATCH",
                    expected_argmax, got_argmax, top_n, matched_top, top_n, worst_abs);
        if (cont_ok >= 0) {
            std::printf(" greedy %d/%zu", cont_ok,
                        entry.at("greedy_continuation").at("ids").size());
        }
        std::printf("\n");
    }

    std::printf("\n%d/%d prompts matched the NumPy oracle\n",
                static_cast<int>(prompts.size()) - failures, static_cast<int>(prompts.size()));
    return failures == 0 ? 0 : 1;
}
