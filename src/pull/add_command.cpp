/// \file add_command.cpp
/// \brief Process handoff from `oflm add` to the bundled installer.
#include "add_command.hpp"

#include "utils/utils.hpp"

#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

#ifdef _WIN32
#include <process.h>
#else
#include <cerrno>
#include <cstring>
#include <unistd.h>
#endif

namespace add_command {
namespace {

std::filesystem::path find_bundled_script() {
    if (const char* configured = std::getenv("OFLM_ADD_SCRIPT")) {
        if (*configured && std::filesystem::is_regular_file(configured)) {
            return configured;
        }
    }

    const std::filesystem::path exe_dir = utils::get_executable_directory();
    const std::vector<std::filesystem::path> candidates = {
        exe_dir / "oflm-add" / "oflm-add.py",                         // portable install
        exe_dir / ".." / "share" / "oflm" / "oflm-add" / "oflm-add.py",
        exe_dir / ".." / ".." / "utilities" / "oflm-add" / "oflm-add.py",
#ifdef OFLM_ADD_SOURCE_SCRIPT
        OFLM_ADD_SOURCE_SCRIPT,
#endif
    };
    for (const auto& candidate : candidates) {
        std::error_code ec;
        if (std::filesystem::is_regular_file(candidate, ec)) {
            return std::filesystem::weakly_canonical(candidate, ec);
        }
    }
    return {};
}

int spawn(const std::vector<std::string>& args) {
    std::vector<char*> raw;
    raw.reserve(args.size() + 1);
    for (const auto& arg : args) raw.push_back(const_cast<char*>(arg.c_str()));
    raw.push_back(nullptr);
#ifdef _WIN32
    const intptr_t rc = _spawnvp(_P_WAIT, raw[0], raw.data());
    return rc == -1 ? 1 : static_cast<int>(rc);
#else
    execvp(raw[0], raw.data());
    std::cerr << "Error: could not start " << args[0] << ": "
              << std::strerror(errno) << std::endl;
    return 1;
#endif
}

} // namespace

int run(int argc, char* argv[]) {
    const auto script = find_bundled_script();
    std::vector<std::string> args;
    if (!script.empty()) {
        const auto exe = std::filesystem::path(utils::get_executable_directory()) /
#ifdef _WIN32
                         "oflm.exe";
        _putenv_s("OFLM_EXECUTABLE", exe.string().c_str());
#else
                         "oflm";
        setenv("OFLM_EXECUTABLE", exe.string().c_str(), 1);
#endif
        const char* configured_python = std::getenv("OFLM_PYTHON");
#ifdef _WIN32
        args.emplace_back(configured_python && *configured_python ? configured_python : "python");
#else
        args.emplace_back(configured_python && *configured_python ? configured_python : "python3");
#endif
        args.push_back(script.string());
    } else {
        // Keep developer installs useful when oflm-add was installed separately.
        args.emplace_back("oflm-add");
    }
    for (int i = 0; i < argc; ++i) args.emplace_back(argv[i]);

    const int rc = spawn(args);
    if (rc != 0 && script.empty()) {
        std::cerr << "Error: oflm-add is not bundled and was not found on PATH. "
                     "Reinstall OpenFlowLM or set OFLM_ADD_SCRIPT." << std::endl;
    }
    return rc;
}

} // namespace add_command
