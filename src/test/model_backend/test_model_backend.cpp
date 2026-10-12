/// \file test_model_backend.cpp
/// \brief The backend registry and the rules for picking a backend
/// \author OpenFlowLM Team
/// \note Ported from the FastFlowLM fork's test_model_backend.cpp (MIT):
///       oflm::backend replaces flm::backend, OFLM_BACKEND replaces
///       FLM_BACKEND, and the default id is oflm_npu (flm_npu stays as a
///       legacy alias in builtin_backends.cpp, not here).
/// \note  Deliberately free of NPU hardware and of any prebuilt engine library:
///        every backend here is a stub, so this builds and runs on Linux CI.
///        register_builtin_backends is stubbed out below for the same reason:
///        the real one lives in builtin_backends.cpp and pulls in every engine
///        header, and with them the prebuilt libraries.
#include "AutoModel/model_backend.hpp"

#include <cstdlib>
#include <exception>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>

#define TEST_REQUIRE(condition)                                                   \
    do {                                                                          \
        if (!(condition)) {                                                        \
            throw std::runtime_error(std::string("requirement failed: ") +       \
                                     #condition);                                  \
        }                                                                         \
    } while (false)

inline void RequireContains(std::string_view text, std::string_view expected) {
    if (text.find(expected) == std::string_view::npos) {
        throw std::runtime_error("expected '" + std::string(text) +
                                 "' to contain '" + std::string(expected) + "'");
    }
}

template <typename Exception = std::exception, typename Callable>
std::string RequireThrows(Callable&& callable) {
    try {
        callable();
    } catch (const Exception& error) {
        return error.what();
    }
    throw std::runtime_error("expected exception was not thrown");
}

inline void RunTest(void (*test)(), const char* name) {
    try {
        test();
        std::cout << "PASS " << name << '\n';
    } catch (const std::exception& error) {
        std::cerr << "FAIL " << name << ": " << error.what() << '\n';
        std::exit(1);
    }
}

using oflm::backend::BackendContext;
using oflm::backend::BackendRegistry;
using oflm::backend::BackendTraits;
using oflm::backend::kDefaultBackendId;
using oflm::backend::kLegacyBackendId;
using oflm::backend::ModelBackend;
using oflm::backend::resolve_backend_id;
using oflm::backend::supported_backends;

/// \brief the builtin set, emptied
/// \note These tests populate the registry themselves, so an empty set is both
///       enough and honest.
namespace oflm::backend {
void register_builtin_backends(BackendRegistry&) {}
}  // namespace oflm::backend

namespace {

/// \brief a backend that owns no engine
/// \note engine() is never called by these tests; nothing here has a causal_lm
///       to hand back, and building one needs an NPU.
class StubBackend final : public ModelBackend {
public:
    explicit StubBackend(std::string id) : id_(std::move(id)) {}
    causal_lm& engine() override {
        throw std::runtime_error("stub backend has no engine");
    }
    std::string id() const override { return id_; }

private:
    std::string id_;
};

oflm::backend::BackendFactory StubFactory(std::string id) {
    return [id = std::move(id)](const BackendContext&) {
        return std::make_unique<StubBackend>(id);
    };
}

/// \brief a registry that is not the process-wide one
BackendRegistry MakeRegistry() { return BackendRegistry(); }

nlohmann::ordered_json Entry(nlohmann::ordered_json details = nlohmann::ordered_json::object()) {
    nlohmann::ordered_json info = nlohmann::ordered_json::object();
    info["details"] = std::move(details);
    return info;
}

/// \brief scoped setenv/unsetenv for OFLM_BACKEND (and the FLM_BACKEND fallback)
class ScopedBackendEnv {
public:
    explicit ScopedBackendEnv(const char* value) {
#if defined(_WIN32)
        _putenv_s("OFLM_BACKEND", value ? value : "");
        _putenv_s("FLM_BACKEND", "");
#else
        if (value) ::setenv("OFLM_BACKEND", value, 1);
        else ::unsetenv("OFLM_BACKEND");
        ::unsetenv("FLM_BACKEND");
#endif
    }
    ~ScopedBackendEnv() {
#if defined(_WIN32)
        _putenv_s("OFLM_BACKEND", "");
        _putenv_s("FLM_BACKEND", "");
#else
        ::unsetenv("OFLM_BACKEND");
        ::unsetenv("FLM_BACKEND");
#endif
    }
};

void test_register_and_create() {
    auto registry = MakeRegistry();
    registry.register_backend("phi4", "oflm_npu", StubFactory("oflm_npu"));
    registry.register_backend("phi4", "corelib_aie4_gguf",
                              StubFactory("corelib_aie4_gguf"),
                              BackendTraits{false, false, 4096});

    TEST_REQUIRE(registry.has("phi4", "oflm_npu"));
    TEST_REQUIRE(!registry.has("phi4", "bogus"));
    TEST_REQUIRE(!registry.has("llama3", "oflm_npu"));

    // available() is sorted, which is what makes the error messages stable.
    const auto ids = registry.available("phi4");
    TEST_REQUIRE(ids.size() == 2);
    TEST_REQUIRE(ids[0] == "corelib_aie4_gguf");
    TEST_REQUIRE(ids[1] == "oflm_npu");
    TEST_REQUIRE(registry.available("llama3").empty());

    BackendContext context;
    auto backend = registry.create("phi4", "corelib_aie4_gguf", context);
    TEST_REQUIRE(backend != nullptr);
    TEST_REQUIRE(backend->id() == "corelib_aie4_gguf");
}

void test_traits_are_kept_per_backend() {
    auto registry = MakeRegistry();
    registry.register_backend("phi4", "oflm_npu", StubFactory("oflm_npu"));
    registry.register_backend("phi4", "corelib_aie4_gguf",
                              StubFactory("corelib_aie4_gguf"),
                              BackendTraits{false, false, 4096});

    // The defaults describe the OpenFlowLM NPU engines.
    const auto npu = registry.traits("phi4", "oflm_npu");
    TEST_REQUIRE(npu.needs_npu_xclbin);
    TEST_REQUIRE(npu.supports_preemption);
    TEST_REQUIRE(npu.max_context_length == 0);

    const auto corelib = registry.traits("phi4", "corelib_aie4_gguf");
    TEST_REQUIRE(!corelib.needs_npu_xclbin);
    TEST_REQUIRE(!corelib.supports_preemption);
    TEST_REQUIRE(corelib.max_context_length == 4096);
}

void test_backend_defaults() {
    StubBackend backend("oflm_npu");
    TEST_REQUIRE(backend.detail().empty());
    TEST_REQUIRE(backend.max_decode_length() == 0);
    TEST_REQUIRE(backend.supports_preemption());
    TEST_REQUIRE(backend.forwards_past_eos());
    TEST_REQUIRE(!backend.poisoned());
    TEST_REQUIRE(!backend.forced_eos_ids().has_value());
}

void test_duplicate_registration_is_rejected() {
    auto registry = MakeRegistry();
    registry.register_backend("phi4", "oflm_npu", StubFactory("first"));
    const std::string message = RequireThrows([&] {
        registry.register_backend("phi4", "oflm_npu", StubFactory("second"));
    });
    RequireContains(message, "already registered");

    BackendContext context;
    TEST_REQUIRE(registry.create("phi4", "oflm_npu", context)->id() == "first");

    RequireThrows([&] { registry.register_backend("", "oflm_npu", StubFactory("x")); });
    RequireThrows([&] { registry.register_backend("phi4", "", StubFactory("x")); });
    RequireThrows([&] { registry.register_backend("phi4", "x", nullptr); });
}

void test_replace_backend_is_the_test_seam() {
    auto registry = MakeRegistry();
    registry.register_backend("phi4", "oflm_npu", StubFactory("real"));
    registry.replace_backend("phi4", "oflm_npu", StubFactory("stub"));

    BackendContext context;
    TEST_REQUIRE(registry.create("phi4", "oflm_npu", context)->id() == "stub");
    TEST_REQUIRE(registry.available("phi4").size() == 1);

    // It also registers a backend that was not there before.
    registry.replace_backend("llama3", "oflm_npu", StubFactory("fresh"));
    TEST_REQUIRE(registry.create("llama3", "oflm_npu", context)->id() == "fresh");
}

void test_unknown_id_names_what_exists() {
    auto registry = MakeRegistry();
    registry.register_backend("phi4", "oflm_npu", StubFactory("oflm_npu"));

    BackendContext context;
    const std::string message =
        RequireThrows([&] { registry.create("phi4", "bogus", context); });
    RequireContains(message, "bogus");
    RequireContains(message, "oflm_npu");

    const std::string empty =
        RequireThrows([&] { registry.create("llama3", "oflm_npu", context); });
    RequireContains(empty, "(none)");
}

void test_supported_backends_falls_back() {
    // No key at all: what the entry has always run on.
    TEST_REQUIRE(supported_backends(Entry()) ==
                 std::vector<std::string>{kDefaultBackendId});

    // execution_backend alone still narrows the entry to itself.
    const auto corelib = Entry({{"execution_backend", "corelib_aie4_gguf"}});
    TEST_REQUIRE(supported_backends(corelib) ==
                 std::vector<std::string>{"corelib_aie4_gguf"});

    // An explicit list wins over execution_backend.
    auto both = corelib;
    both["supported_backends"] = {"oflm_npu", "corelib_aie4_gguf"};
    TEST_REQUIRE(supported_backends(both).size() == 2);

    auto malformed = Entry();
    malformed["supported_backends"] = nlohmann::ordered_json::array();
    RequireThrows([&] { supported_backends(malformed); });
    malformed["supported_backends"] = {1, 2};
    RequireThrows([&] { supported_backends(malformed); });
    RequireThrows([&] {
        supported_backends(Entry({{"execution_backend", 7}}));
    });
}

void test_legacy_alias_is_documented() {
    // The pre-rename id exists so shared catalogs keep resolving; it is wired
    // in builtin_backends.cpp, so here we only pin the constant.
    TEST_REQUIRE(std::string(kLegacyBackendId) == "flm_npu");
    TEST_REQUIRE(std::string(kDefaultBackendId) == "oflm_npu");
}

void test_resolution_precedence() {
    auto& registry = BackendRegistry::instance();
    registry.replace_backend("phi4", "oflm_npu", StubFactory("oflm_npu"));
    registry.replace_backend("phi4", "corelib_aie4_gguf",
                             StubFactory("corelib_aie4_gguf"));

    auto info = Entry();
    info["supported_backends"] = {"oflm_npu", "corelib_aie4_gguf"};

    std::string source;
    // 4. nothing says anything -> the default
    TEST_REQUIRE(resolve_backend_id("phi4", info, "", &source) == kDefaultBackendId);
    TEST_REQUIRE(source == "default");

    // 3. the catalog
    auto catalog = info;
    catalog["details"]["execution_backend"] = "corelib_aie4_gguf";
    TEST_REQUIRE(resolve_backend_id("phi4", catalog, "", &source) ==
                 "corelib_aie4_gguf");
    TEST_REQUIRE(source == "model catalog");

    {
        // 2. OFLM_BACKEND beats the catalog
        ScopedBackendEnv env("oflm_npu");
        TEST_REQUIRE(resolve_backend_id("phi4", catalog, "", &source) == "oflm_npu");
        TEST_REQUIRE(source == "OFLM_BACKEND");

        // 1. --backend beats both
        TEST_REQUIRE(resolve_backend_id("phi4", catalog, "corelib_aie4_gguf",
                                        &source) == "corelib_aie4_gguf");
        TEST_REQUIRE(source == "--backend");
    }

    // An empty OFLM_BACKEND is the same as an unset one.
    ScopedBackendEnv empty(nullptr);
    TEST_REQUIRE(resolve_backend_id("phi4", info, "", &source) == kDefaultBackendId);
}

void test_resolution_rejects_with_a_readable_message() {
    auto& registry = BackendRegistry::instance();
    registry.replace_backend("phi4", "oflm_npu", StubFactory("oflm_npu"));

    // Registered for the family, but this entry does not allow it.
    auto info = Entry();
    info["supported_backends"] = {"oflm_npu"};
    const std::string not_allowed = RequireThrows(
        [&] { resolve_backend_id("phi4", info, "corelib_aie4_gguf"); });
    RequireContains(not_allowed, "--backend");
    RequireContains(not_allowed, "It supports: oflm_npu");

    // Allowed by the entry, but this build does not have it.
    auto allowed = Entry();
    allowed["supported_backends"] = {"oflm_npu", "not_built"};
    const std::string not_built =
        RequireThrows([&] { resolve_backend_id("phi4", allowed, "not_built"); });
    RequireContains(not_built, "not compiled into this build");
    RequireContains(not_built, "oflm_npu");

    // The env var gets named in the message too, so the user can find it.
    ScopedBackendEnv env("bogus");
    const std::string from_env =
        RequireThrows([&] { resolve_backend_id("phi4", info, ""); });
    RequireContains(from_env, "OFLM_BACKEND");
}

}  // namespace

int main() {
    RunTest(test_register_and_create, "register and create");
    RunTest(test_traits_are_kept_per_backend, "traits are kept per backend");
    RunTest(test_backend_defaults, "backend policy defaults");
    RunTest(test_duplicate_registration_is_rejected, "duplicate registration is rejected");
    RunTest(test_replace_backend_is_the_test_seam, "replace_backend is the test seam");
    RunTest(test_unknown_id_names_what_exists, "unknown id names what exists");
    RunTest(test_supported_backends_falls_back, "supported_backends falls back");
    RunTest(test_legacy_alias_is_documented, "legacy alias is documented");
    RunTest(test_resolution_precedence, "resolution precedence");
    RunTest(test_resolution_rejects_with_a_readable_message,
            "resolution rejects with a readable message");
    std::cout << "All model backend tests passed\n";
    return 0;
}
