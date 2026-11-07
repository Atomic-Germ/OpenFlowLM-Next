/// \file test_stubs.cpp
/// \brief Utils stubs for unit tests — replaces utils.cpp to avoid XRT dependency
/// \note Config stubs (split_tag, deep_merge, get_merged_model_entry) were removed.
///       Tests that need them should compile config_store.cpp instead.
#include <string>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

namespace utils {

std::string getenv_oflm(const char* oflm_name) {
    if (const char* v = std::getenv(oflm_name)) {
        if (*v) return std::string(v);
    }
    if (std::strncmp(oflm_name, "OFLM_", 5) == 0) {
        std::string legacy = std::string("FLM_") + (oflm_name + 5);
        if (const char* v = std::getenv(legacy.c_str())) {
            if (*v) return std::string(v);
        }
    }
    return std::string();
}

std::string find_shipped_registry() {
    // Honour the same variables the real one does, so a test can point the
    // loader at a registry on disk and exercise the whole load path.
    if (const char* p = getenv("OFLM_CONFIG_PATH")) {
        if (*p) return p;
    }
    throw std::runtime_error("stub: set OFLM_CONFIG_PATH to a registry for this test");
}

std::string find_user_settings() {
    if (const char* p = getenv("OFLM_SETTINGS_PATH")) {
        if (*p) return p;
    }
    return std::string("/tmp") + "/settings.json";
}

std::string get_models_directory() {
    return std::string("/tmp");
}

std::string get_executable_directory() {
    return std::string(".");
}

std::string get_user_directory() {
    return std::string("/tmp");
}

int get_server_port(int user_port) {
    return user_port > 0 ? user_port : 52625;
}

} // namespace utils
