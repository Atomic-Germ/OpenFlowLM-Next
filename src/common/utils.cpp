/// \file utils.cpp
/// \brief utils class
/// \author OpenFlowLM Team
/// \date 2025-06-24
/// \version 0.9.24
/// 
/// \note This file contains some utility functions for the OpenFlowLM project.
#include "utils/utils.hpp"
#include <algorithm>
#include <cstring>
#include <filesystem>
#include <cstdlib>
#include <set>

#ifdef _WIN32
#include <windows.h>
#include <shlobj.h>
#else
#include <unistd.h>
#include <limits.h>
#endif

namespace utils {

std::string getenv_oflm(const char* oflm_name) {
    if (const char* v = std::getenv(oflm_name)) {
        if (*v) return std::string(v);
    }
    // "OFLM_FOO" -> "FLM_FOO". The rename dropped nothing else, so the inverse is
    // to remove the leading 'O'; a name that is not OFLM_-prefixed has no legacy form.
    if (std::strncmp(oflm_name, "OFLM_", 5) != 0) return std::string();
    const std::string legacy = std::string("FLM_") + (oflm_name + 5);
    const char* v = std::getenv(legacy.c_str());
    if (!v || !*v) return std::string();
    static std::set<std::string> warned;          // once per variable, not once per read
    if (warned.insert(legacy).second) {
        std::cerr << "[OFLM]  " << legacy << " is the name used before the oflm rename. "
                  << "It still works; set " << oflm_name << " instead." << std::endl;
    }
    return std::string(v);
}

/// The user-level directories the pre-rename releases used, for the same reason
/// user_oflm_directories() exists: a model store is gigabytes and must not have to
/// move for an upgrade. Searched AFTER every oflm location, never before.
std::vector<std::string> legacy_flm_directories() {
    std::vector<std::string> v;
#ifdef _WIN32
    v.push_back((std::filesystem::path(get_user_directory()) / ".flm").string());
    v.push_back((std::filesystem::path(get_user_directory()) / ".config" / "flm").string());
#else
    v.push_back(get_user_directory() + "/flm");
#endif
    return v;
}

std::string find_model_list() {
    std::string install_prefix = CMAKE_INSTALL_PREFIX;

    // 1. Check OFLM_CONFIG_PATH environment variable
    const std::string env_path = getenv_oflm("OFLM_CONFIG_PATH");
    if (!env_path.empty()) {
        if (std::filesystem::exists(env_path)) {
            std::cerr << "[OFLM]  Using custom model list path: " << env_path << std::endl;
            return env_path;
        }
    }

    // Portable development-tree location (next to the executable, then CWD).
    std::string exe_dir = get_executable_directory();
    std::string exe_relative_path = exe_dir + "/model_list.json";
    if (std::filesystem::exists(exe_relative_path)) {
        return exe_relative_path;
    }
    if (std::filesystem::exists("model_list.json")) {
        return "model_list.json";
    }

    // Relocatable installed bundle, independent of its original prefix.
    std::string bundle_path = exe_dir + "/../share/oflm/model_list.json";
    if (std::filesystem::exists(bundle_path)) {
        return bundle_path;
    }

    // Legacy configured prefix.
    std::string installed_path = install_prefix + "/share/oflm/model_list.json";
    if (std::filesystem::exists(installed_path)) {
        return installed_path;
    }

    // If not found, throw an error
    throw std::runtime_error("model_list.json not found. Please set OFLM_CONFIG_PATH or place it next to the executable.");
}

std::string find_model_info() {
    std::string install_prefix = CMAKE_INSTALL_PREFIX;

    // 1. Check OFLM_MODELINFO_PATH environment variable
    const std::string env_path = getenv_oflm("OFLM_MODELINFO_PATH");
    if (!env_path.empty()) {
        if (std::filesystem::exists(env_path)) {
            std::cerr << "[OFLM]  Using custom model info path: " << env_path << std::endl;
            return env_path;
        }
    }

    // 2. Stay next to an explicitly configured model_list.json. A relocated
    // install points OFLM_CONFIG_PATH at its own share/oflm; without this the
    // lookup falls through to the baked-in prefix below and we end up sizing
    // and hash-checking downloads against a different (stale) revision.
    const std::string config_path = getenv_oflm("OFLM_CONFIG_PATH");
    if (!config_path.empty()) {
        std::filesystem::path sibling =
            std::filesystem::path(config_path).parent_path() / "model_info.json";
        if (std::filesystem::exists(sibling)) {
            return sibling.string();
        }
    }

#ifndef _WIN32
    // Linux: Portable
    // if (std::filesystem::exists("model_list.json")) {
    //     return "model_list.json";
    // }
    std::string exe_dir = get_executable_directory();
    std::string exe_relative_path = exe_dir + "/model_info.json";
    if (std::filesystem::exists(exe_relative_path)) {
        return exe_relative_path;
    }

    // Relocatable installed bundle, independent of its original prefix.
    std::string bundle_path = exe_dir + "/../share/oflm/model_info.json";
    if (std::filesystem::exists(bundle_path)) {
        return bundle_path;
    }

    // Linux: install
    std::string installed_path = install_prefix + "/share/oflm/model_info.json";
    if (std::filesystem::exists(installed_path)) {
        return installed_path;
    }
#else
    // Windows: Check relative to executable
    std::string exe_dir = get_executable_directory();
    std::string exe_relative_path = exe_dir + "\\model_info.json";
    if (std::filesystem::exists(exe_relative_path)) {
        return exe_relative_path;
    }
#endif

    // If not found, throw an error
    throw std::runtime_error("model_info.json not found. Please set OFLM_MODELINFO_PATH or place it next to the executable.");
}

namespace {

/// A configured root may be given with or without its trailing "xclbins"
/// component; the callers of `find_xclbin_path` always append it themselves.
std::string strip_xclbins(std::string path) {
    std::filesystem::path p(path);
    if (p.filename().empty()) p = p.parent_path();   // a trailing separator
    if (p.filename() == "xclbins") return p.parent_path().string();
    return path;
}

/// The user-level directories for THIS project, newest first (#30). Returned as
/// a list rather than a single path so that a directory made either way is
/// found: #30 asks for %USERPROFILE%\.oflm on Windows, while oflm-add's own
/// `Path.home() / ".config" / <name>` lands in <profile>/.config/oflm. On POSIX
/// get_user_directory() already ends in .config, so one entry covers it. Scanning
/// them costs one stat each.
///
/// This list subsumes the former user_oflm_directory(), which the oflm rename left
/// returning a path already in here -- dead code, and the reason nobody noticed the
/// legacy root had stopped being searched (#41).
std::vector<std::string> user_oflm_directories() {
    std::vector<std::string> v;
#ifdef _WIN32
    v.push_back((std::filesystem::path(get_user_directory()) / ".oflm").string());
    v.push_back((std::filesystem::path(get_user_directory()) / ".config" / "oflm").string());
#else
    v.push_back(get_user_directory() + "/oflm");
#endif
    return v;
}

/// The roots `find_xclbin_path` has always walked, in its order. Kept separate so that
/// widening the OPEN path's search (xclbin_roots below) cannot move which root the CLOSED
/// path picks: it returns exactly one, and every closed kernel is loaded relative to it.
std::vector<std::string> closed_path_roots() {
    std::vector<std::string> c;
    const std::string env_path = getenv_oflm("OFLM_XCLBIN_PATH");
    if (!env_path.empty()) c.push_back(strip_xclbins(env_path));
    std::string exe_dir = get_executable_directory();
    c.push_back(exe_dir);                       // portable development tree
    c.push_back(".");                           // then the CWD
    c.push_back(exe_dir + "/../share/oflm");     // relocatable installed bundle
    c.push_back(CMAKE_XCLBIN_PREFIX);           // legacy configured prefix
    return c;
}

} // namespace

std::vector<std::string> xclbin_roots() {
    std::vector<std::string> candidates;

    // The user-level roots first: oflm-add installs a model's kernels under one of these,
    // and the shipped sets live in the install tree below. A lookup that stops at the
    // first root (find_xclbin_path) can only ever see one of the two.
    const std::string env_path = getenv_oflm("OFLM_XCLBIN_PATH");
    if (!env_path.empty()) candidates.push_back(strip_xclbins(env_path));
    // Beside an explicitly configured model_list.json, the way find_model_info stays
    // beside it: a user registry and its kernels live in one directory.
    const std::string config_path = getenv_oflm("OFLM_CONFIG_PATH");
    if (!config_path.empty()) {
        candidates.push_back(std::filesystem::path(config_path).parent_path().string());
    }
    // The directories oflm-add uses when neither variable is exported, new
    // before legacy (#30) -- an existing install keeps working with no
    // migration and no copying of multi-gigabyte weights.
    //
    // The second line used to be user_flm_directory(), i.e. the LEGACY root, and the
    // oflm rename turned it into user_oflm_directory() -- a strict subset of the list
    // above it. The line became dead and the legacy root stopped being searched, while
    // the comment went on promising it. legacy_flm_directories() restores what the
    // comment says (#41).
    for (const std::string& d : user_oflm_directories()) candidates.push_back(d);
    for (const std::string& d : legacy_flm_directories()) candidates.push_back(d);

    // The embedding design sets (BERT-*/gemm_rtp) ship here, separately from the
    // chat-model xclbin tree that CMAKE_XCLBIN_PREFIX points at. Listing it makes
    // the pre-built sets discoverable by geometry/family with no manual symlink.
    candidates.push_back("/usr/local/share/flm");

    for (const std::string& c : closed_path_roots()) candidates.push_back(c);

    std::vector<std::string> roots;
    for (const std::string& c : candidates) {
        if (c.empty()) continue;
        std::error_code ec;
        if (!std::filesystem::exists(c + "/xclbins", ec)) continue;
        if (std::find(roots.begin(), roots.end(), c) == roots.end()) roots.push_back(c);
    }
    return roots;
}

std::string find_xclbin_path() {
    for (const std::string& c : closed_path_roots()) {
        if (!c.empty() && std::filesystem::exists(c + "/xclbins")) return c;
    }
    throw std::runtime_error("xclbins not found. Please set OFLM_XCLBIN_PATH or place it next to the executable.");
}

std::string get_executable_directory() {
#ifdef _WIN32
    char buffer[MAX_PATH];
    GetModuleFileNameA(NULL, buffer, MAX_PATH);
    std::string exe_path(buffer);
    size_t last_slash = exe_path.find_last_of("\\");
    if (last_slash != std::string::npos) {
        return exe_path.substr(0, last_slash);
    }
    return ".";
#else
    char buffer[PATH_MAX] = {0};
    ssize_t len = readlink("/proc/self/exe", buffer, sizeof(buffer) - 1);
    if (len > 0) {
        buffer[len] = '\0';
        std::string exe_path(buffer);
        size_t last_slash = exe_path.find_last_of("/");
        if (last_slash != std::string::npos) {
            return exe_path.substr(0, last_slash);
        }
    }
    return ".";
#endif
}

std::string get_user_directory() {
#ifdef _WIN32
    char buffer[MAX_PATH];
    if (SUCCEEDED(SHGetFolderPathA(NULL, CSIDL_PROFILE, NULL, 0, buffer))) {
        return std::string(buffer);
    }
    // Fallback to current directory if user folder cannot be found
    return ".";
#else
    const char* home = std::getenv("HOME");
    if (home && *home) {
        return std::string(home) + "/.config";
    }
    return ".";
#endif
}

///@brief get_server_port gets the server port from environment variable OFLM_SERVE_PORT
///@return the server port, default is 52625 if environment variable is not set
int get_server_port(int user_port) {
    if (user_port > 0 && user_port <= 65535) {
        return user_port;
    }
    else {
        // OFLM_SERVE_PORT, or the FLM_SERVE_PORT the pre-rename installer wrote (#41).
        const std::string port_env = getenv_oflm("OFLM_SERVE_PORT");
        if (!port_env.empty()) {
            try {
                int port = std::stoi(port_env);
                if (port > 0 && port <= 65535) {
                    return port;
                }
            }
            catch (const std::exception&) {
                // Invalid port number, use default
            }
        }
    }

    return 52625; // Default port
}

///@brief get_models_directory gets the models directory from environment variable or defaults to user/.oflm/models on Windows or ~/.config/oflm on Linux
///@return the models directory path
std::string get_models_directory() {
    // OFLM_MODEL_PATH, or the FLM_MODEL_PATH the pre-rename installer wrote (#41).
    const std::string custom_path = getenv_oflm("OFLM_MODEL_PATH");
    if (!custom_path.empty()) return custom_path;

    // No variable set: the oflm directory, then the pre-rename one if it exists and the
    // oflm one does not. A model store is gigabytes; an upgrade must not orphan it.
    std::string user_dir = get_user_directory();
#ifdef _WIN32
    const std::string oflm_dir = user_dir + "\\.oflm";
#else
    const std::string oflm_dir = user_dir + "/oflm";
#endif
    std::error_code ec;
    if (std::filesystem::is_directory(oflm_dir, ec)) return oflm_dir;
    for (const std::string& d : legacy_flm_directories()) {
        if (std::filesystem::is_directory(d, ec)) {
            std::cerr << "[OFLM]  using the pre-rename model directory " << d
                      << "; move it to " << oflm_dir << " when convenient." << std::endl;
            return d;
        }
    }
    return oflm_dir;
}

} // end of namespace utils
