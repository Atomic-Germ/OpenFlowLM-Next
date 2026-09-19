/// \file test_npu_platform.cpp
/// \brief The npu_platform helpers (no NPU hardware, no catalog)
/// \author OpenFlowLM Team
/// \note The pure helpers ported from the FastFlowLM fork's
///       test_model_list_platform.cpp (MIT). The catalog-filtering half of
///       that suite is not ported: model_list has no supported_platforms /
///       platform_overrides keys yet, so there is nothing to test. It becomes
///       relevant when the catalog gains per-generation entries for the
///       family-xclbin distribution.
#include "utils/npu_platform.hpp"

#include <iostream>
#include <stdexcept>
#include <string>

#define TEST_REQUIRE(condition)                                                   \
    do {                                                                          \
        if (!(condition)) {                                                        \
            throw std::runtime_error(std::string("requirement failed: ") +       \
                                     #condition);                                  \
        }                                                                         \
    } while (false)

inline void RunTest(void (*test)(), const char* name) {
    try {
        test();
        std::cout << "PASS " << name << '\n';
    } catch (const std::exception& error) {
        std::cerr << "FAIL " << name << ": " << error.what() << '\n';
        std::exit(1);
    }
}

namespace {

void test_platform_helpers() {
    TEST_REQUIRE(utils::platform_from_columns(3) == utils::npu_platform::aie4);
    TEST_REQUIRE(utils::platform_from_columns(8) == utils::npu_platform::aie2p);
    TEST_REQUIRE(utils::is_known_column_count(3));
    TEST_REQUIRE(utils::is_known_column_count(8));
    TEST_REQUIRE(!utils::is_known_column_count(5));
    TEST_REQUIRE(utils::parse_platform("aie2p") == utils::npu_platform::aie2p);
    TEST_REQUIRE(utils::parse_platform("aie4") == utils::npu_platform::aie4);
    TEST_REQUIRE(!utils::parse_platform("medusa").has_value());
    TEST_REQUIRE(utils::parse_platform(utils::platform_id(utils::npu_platform::aie4)) ==
                 utils::npu_platform::aie4);
    TEST_REQUIRE(utils::default_npu_platform() == utils::npu_platform::aie2p);
}

void test_detect_without_npu_falls_back() {
    // No device and (normally) no override: never throws, always aie2p here.
    // The OFLM_PLATFORM override is process-cached, so this only pins the
    // no-throw contract, not a particular value.
    std::string detail;
    const auto platform = utils::detect_npu_platform(nullptr, &detail);
    TEST_REQUIRE(!detail.empty());
    TEST_REQUIRE(platform == utils::npu_platform::aie2p ||
                 platform == utils::npu_platform::aie4);
}

}  // namespace

int main() {
    RunTest(test_platform_helpers, "platform helpers");
    RunTest(test_detect_without_npu_falls_back, "detect without npu falls back");
    std::cout << "All npu_platform tests passed\n";
    return 0;
}
