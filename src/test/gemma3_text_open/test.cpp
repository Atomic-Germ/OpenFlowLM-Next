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
    if (engine.vocab() == 0) {
        std::fprintf(stderr, "engine reports zero vocab\n");
        return 1;
    }

    int failures = 0;
    const auto& prompts = ref.at("prompts");
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
