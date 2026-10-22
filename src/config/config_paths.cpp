/// \file config_paths.cpp
/// \brief The paths oflm reads and writes, and where they point right now.
#include "config/config_paths.hpp"

#include "utils/utils.hpp"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <nlohmann/json.hpp>

namespace oflm::config {

namespace {

// Where the models live: the model store, plus the subdirectory the
// registry names. The registry says "models", so the models are in
// <model store>/models -- which is why OFLM_MODEL_PATH points at the
// directory that HOLDS the models, not at the models themselves.
std::string resolve_model_root() {
    std::string sub = "models";
    try {
        const std::string registry = utils::find_shipped_registry_quiet();
        std::ifstream file(registry);
        if (file) {
            nlohmann::json j = nlohmann::json::parse(file);
            sub = j.value("model_path", std::string("models"));
        }
    } catch (const std::exception&) {
        // The registry is not readable: the default subdirectory is the best
        // answer there is, and the registry row says so itself.
    }
    return (std::filesystem::path(utils::get_models_directory()) / sub).string();
}

} // namespace

std::vector<PathEntry> resolve_paths() {
    std::vector<PathEntry> out;

    // A path that the environment sets is shown as the environment has it, so
    // that it can be compared with `echo $OFLM_...`. A path the environment
    // does not set is shown as the one oflm resolved to, which is what the
    // user wants to know then.
    auto add = [&](const char* var, const char* what, const std::string& resolved,
                   const std::string& note = std::string()) {
        PathEntry e;
        e.var = var;
        e.what = what;
        e.note = note;
        e.value = resolved;
        const std::string env = utils::getenv_oflm(var);
        if (!env.empty() && env == resolved) {
            e.from_env = true;
        } else if (!env.empty() && !note.empty()) {
            // The variable points somewhere else than the path oflm reads
            // (the models are the one: the variable holds the directory that
            // HOLDS the models tree, not the tree itself). The value of the
            // variable is the one that counts.
            e.value = env;
            e.from_env = true;
        }
        out.push_back(e);
    };

    // The registry of models.
    try {
        add("OFLM_CONFIG_PATH", "the model registry", utils::find_shipped_registry_quiet());
    } catch (const std::exception&) {
        add("OFLM_CONFIG_PATH", "the model registry", "not found");
    }

    // The model info: sizes, hashes, download lists.
    try {
        add("OFLM_MODELINFO_PATH", "the model info", utils::find_model_info_quiet());
    } catch (const std::exception&) {
        add("OFLM_MODELINFO_PATH", "the model info", "not found");
    }

    // The root that holds the xclbins/ tree.
    try {
        add("OFLM_XCLBIN_PATH", "the xclbins/ tree", utils::find_xclbin_path());
    } catch (const std::exception&) {
        add("OFLM_XCLBIN_PATH", "the xclbins/ tree", "not found");
    }

    // The root that holds the models/ tree. The variable holds the directory
    // that HOLDS the models, so the note says where the models are.
    add("OFLM_MODEL_PATH", "the models/ tree", utils::get_models_directory(),
        "the models are in " + resolve_model_root());

    // The user's settings: the file oflm writes when a value is set.
    add("OFLM_SETTINGS_PATH", "your settings file", utils::find_user_settings());

    return out;
}

void print_paths(std::ostream& os) {
    const std::vector<PathEntry> paths = resolve_paths();

    size_t var_width = 0;
    size_t value_width = 0;
    for (const auto& p : paths) {
        var_width = std::max(var_width, p.var.size());
        value_width = std::max(value_width, p.value.size());
    }
    // A long path is not made shorter, but it does not push the other
    // columns off the screen either.
    const size_t max_value_width = 52;

    os << "Paths in use:" << std::endl;
    for (const auto& p : paths) {
        std::string value = p.value;
        if (value.size() > max_value_width) {
            value = value.substr(0, max_value_width - 3) + "...";
        }
        os << "  " << std::left << std::setw(static_cast<int>(var_width) + 2) << p.var
           << std::setw(static_cast<int>(value_width) + 2) << value
           << std::left << p.what;
        if (!p.note.empty()) os << " (" << p.note << ")";
        os << "  " << (p.from_env ? "(env)" : "(default)") << std::endl;
    }
}

} // namespace oflm::config
