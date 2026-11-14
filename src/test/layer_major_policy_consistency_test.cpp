// Traces: OPEN-PREFILL-BATCH provenance. Host-only: no XRT, no device, no weights.
//
// `qwen38-production-baseline.json` is machine-readable provenance, so a wrong sentence in
// it is worse than a wrong sentence in a report: tooling can treat it as authoritative. It
// once described `block_loop_preferred` the wrong way round -- as though the flag made a
// dense set KEEP layer-major, when it is the flag that makes it PREFER the block loop.
//
// So this does not check that the JSON has a field, or that the field contains a phrase. It
// parses the structured policy block, feeds each case to the REAL C++ decision, and requires
// the JSON's recorded answer to equal what the code returns. If either side is edited alone,
// this fails.
//
// The path is supplied by CMake so the test can run from any build directory.
#include "open_qwen36/layer_major_policy.hpp"
#include "nlohmann/json.hpp"

#include <fstream>
#include <iostream>
#include <string>

using open_qwen36::layer_major::Facts;
using open_qwen36::layer_major::Force;
using open_qwen36::layer_major::preferred;

static int failures = 0;

#define CHECK(cond)                                                                    \
    do {                                                                               \
        if (!(cond)) {                                                                 \
            std::cerr << "FAIL " << __FILE__ << ":" << __LINE__ << ": " #cond << "\n"; \
            ++failures;                                                                \
        }                                                                              \
    } while (0)

#ifndef BASELINE_JSON
#error "BASELINE_JSON must name qwen38-production-baseline.json"
#endif

int main() {
    std::ifstream f(BASELINE_JSON);
    if (!f) {
        std::cerr << "cannot open " << BASELINE_JSON << "\n";
        return 1;
    }
    const auto doc = nlohmann::json::parse(f, nullptr, false);
    if (doc.is_discarded() || !doc.is_object()) {
        std::cerr << "baseline is not valid JSON\n";
        return 1;
    }
    const auto& routes = doc.at("routes");
    const auto& policy = routes.at("layer_major_policy");

    // ---------------------------------------------------------------- the two cases the
    // ---------------------------------------------------------------- schema must record

    // Dense + the validated preference: the block loop wins, so automatic layer-major is off.
    Facts dense_preferred;
    dense_preferred.capable = true;
    dense_preferred.block_loop_preferred = true;
    const bool code_dense_preferred = preferred(Force::Auto, dense_preferred);
    const bool json_dense_preferred = policy.at("qwen38_27b").at("prefers_layer_major");
    CHECK(code_dense_preferred == false);
    CHECK(json_dense_preferred == code_dense_preferred);

    // Dense without it: the set keeps what it shipped with, which is layer-major.
    Facts dense_shipped;
    dense_shipped.capable = true;
    dense_shipped.block_loop_preferred = false;
    CHECK(preferred(Force::Auto, dense_shipped) == true);
    CHECK(policy.at("dense").at("prefers_layer_major_when").get<std::string>().find(
              "block_loop_preferred is false") != std::string::npos);

    // Routed experts: layer-major, and it stays distinct from the dense rule.
    Facts moe;
    moe.capable = true;
    moe.has_routed_experts = true;
    moe.block_loop_preferred = true;   // the flag must not override the expert win
    const bool code_moe = preferred(Force::Auto, moe);
    const bool json_moe = policy.at("routed_experts").at("prefers_layer_major");
    CHECK(code_moe == true);
    CHECK(json_moe == code_moe);
    CHECK(code_moe != code_dense_preferred);   // the two preferences really are distinct

    // ---------------------------------------------------------------- the recorded default

    // The baseline's headline claim must be the answer the code gives for this model.
    CHECK(policy.at("qwen38_27b").at("block_loop_preferred") == true);
    CHECK(routes.at("dense_qwen38_default") == "block-loop");

    // ---------------------------------------------------------------- the override

    // Documented and implemented: an explicit request outranks the preference.
    CHECK(preferred(Force::On, dense_preferred) == true);
    CHECK(preferred(Force::Off, dense_shipped) == false);
    CHECK(policy.at("operator_override").at("nonzero").get<std::string>().find(
              "layer_major_ok()") != std::string::npos);

    // ---------------------------------------------------------------- the direction

    // The inversion this exists for, stated as an assertion: the flag REJECTS layer-major on
    // a dense set. If the JSON ever describes it as enabling layer-major, the sentence and
    // the code disagree and one of them is about to mislead somebody.
    const std::string meaning =
        policy.at("meaning_of_block_loop_preferred").get<std::string>();
    CHECK(meaning.find("block loop was MEASURED") != std::string::npos);
    CHECK(meaning.find("not a request to use layer-major") != std::string::npos);

    if (failures) {
        std::cerr << failures << " check(s) failed\n";
        return 1;
    }
    std::cout << "layer-major policy consistency: all checks passed\n";
    return 0;
}
