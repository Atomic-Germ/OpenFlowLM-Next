/// \file pack_command.cpp
/// \brief Process handoff from `oflm pack` to the bundled Q4NX builder.
#include "pack_command.hpp"

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

namespace pack_command {
namespace {

/// The bundled launcher, or the package directory when there is no launcher.
///
/// The launcher is preferred and is what an install ships: it is the same
/// `q4nx-build` that sits in the bin directory, so `oflm pack` and `q4nx-build`
/// cannot drift -- one environment setup, written once. The package directory
/// is the fallback for a tree where the launcher was not installed, and then
/// python3 is invoked directly with PYTHONPATH pointed at it.
struct Builder {
    std::filesystem::path launcher;   // a script to run as-is; empty if none
    std::filesystem::path package;    // a directory holding the `q4nx` package
};

Builder find_builder() {
    Builder b;
    if (const char* configured = std::getenv("OFLM_PACK_SCRIPT")) {
        if (*configured && std::filesystem::is_regular_file(configured)) {
            b.launcher = std::filesystem::weakly_canonical(configured);
            return b;
        }
    }
    const std::filesystem::path exe_dir = utils::get_executable_directory();
    const std::vector<std::filesystem::path> launchers = {
        exe_dir / "q4nx-build",                                           // installed beside oflm
        exe_dir / "q4nx-build.sh",
    };
    for (const auto& candidate : launchers) {
        std::error_code ec;
        if (std::filesystem::is_regular_file(candidate, ec)) {
            b.launcher = std::filesystem::weakly_canonical(candidate, ec);
            return b;
        }
    }
    const std::vector<std::filesystem::path> packages = {
        exe_dir / ".." / "share" / "oflm" / "utilities" / "q4nx-build",
        exe_dir / ".." / ".." / "utilities" / "q4nx-build",
#ifdef OFLM_Q4NX_PACKAGE_DIR
        OFLM_Q4NX_PACKAGE_DIR,
#endif
    };
    for (const auto& candidate : packages) {
        std::error_code ec;
        if (std::filesystem::is_directory(candidate / "q4nx", ec)) {
            b.package = std::filesystem::weakly_canonical(candidate, ec);
            return b;
        }
    }
    return b;
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
    const Builder b = find_builder();
    std::vector<std::string> args;

    if (!b.launcher.empty()) {
        args.push_back(b.launcher.string());
    } else if (!b.package.empty()) {
        const char* configured_python = std::getenv("OFLM_PYTHON");
        if (configured_python && *configured_python) {
            args.emplace_back(configured_python);
        } else {
#ifdef _WIN32
            args.emplace_back("python");
#else
            args.emplace_back("python3");
#endif
        }
        args.emplace_back("-c");
        args.emplace_back("import sys; from q4nx.cli import main; sys.exit(main())");
#ifdef _WIN32
        _putenv_s("PYTHONPATH", b.package.string().c_str());
#else
        const char* existing = std::getenv("PYTHONPATH");
        setenv("PYTHONPATH",
               (b.package.string() + (existing && *existing ? std::string(":") + existing : ""))
                   .c_str(),
               1);
#endif
    } else {
        // Not bundled: the standalone builder may still be on PATH, which is
        // what a developer tree looks like before `cmake --install`.
        args.emplace_back("q4nx-build");
    }

    for (int i = 0; i < argc; ++i) args.emplace_back(argv[i]);
    const int rc = spawn(args);
    if (rc != 0 && b.launcher.empty() && b.package.empty()) {
        std::cerr << "Error: the Q4NX builder is not bundled and q4nx-build was not "
                     "found on PATH. Reinstall OpenFlowLM, or set OFLM_PACK_SCRIPT to "
                     "the tool." << std::endl;
    }
    return rc;
}

} // namespace pack_command
