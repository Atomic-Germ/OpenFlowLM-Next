/// \file omp_policy_test.cpp
/// \brief Execute the production OpenMP startup helper without XRT or weights.
#include "open_qwen36/omp_policy.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#if defined(__linux__) && !defined(_WIN32)
#include <omp.h>
#endif

namespace {

int fail(const std::string& message) {
    std::fprintf(stderr, "FAIL: %s\n", message.c_str());
    return 1;
}

std::string env(const char* name) {
    const char* value = std::getenv(name);
    return value ? value : "<unset>";
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) return fail("test mode required");
    const std::string mode = argv[1];
    if (mode == "model") {
        struct Case { const char* family; const char* name; bool expected; };
        const Case cases[] = {
            {"qwen3.5", "Qwen3.8-27B-NPU2", true},
            {"qwen3.5", "Qwen3.8-27B-custom", true},
            {"qwen3.5", "Qwen3.5-9B-NPU2", false},
            {"qwen3.6-moe", "Qwen3.8-27B-NPU2", false},
            {"whisper-v3", "Whisper-large-v3-turbo", false},
            {nullptr, "Qwen3.8-27B-NPU2", false},
        };
        for (const auto& c : cases)
            if (open_qwen36::qwen38_model(c.family, c.name) != c.expected)
                return fail(std::string("model selection: ") + (c.family ? c.family : "null") + "/" + c.name);
        std::puts("PASS: model selection");
        return 0;
    }
    if (mode != "startup") return fail("unknown test mode");

    std::puts("ENTER");
    std::fflush(stdout);
    open_qwen36::reexec_with_passive(argv);
    std::printf("POLICY=%s\nGUARD=%s\n", env("OMP_WAIT_POLICY").c_str(), env("OFLM_OPEN_OMP_REEXEC").c_str());
    for (int i = 2; i < argc; ++i) std::printf("ARG=%s\n", argv[i]);
#if defined(__linux__) && !defined(_WIN32)
    omp_display_env(0);
#endif
    return 0;
}
