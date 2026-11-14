/// \file dev_command.cpp
/// \brief Developer/toolchain workflows for `oflm dev`.
#include "dev_command.hpp"

#include "utils/utils.hpp"

#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>
#include <algorithm>

#ifdef _WIN32
#include <process.h>
#else
#include <cerrno>
#include <cstring>
#include <unistd.h>
#endif

namespace dev_command {
namespace fs = std::filesystem;

namespace {

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

const char* python_exe() {
    const char* configured = std::getenv("OFLM_PYTHON");
    if (configured && *configured) return configured;
    if (const char* v = std::getenv("VIRTUAL_ENV")) {
        static std::string p;
        p = std::string(v) + "/bin/python3";
        if (fs::is_regular_file(p)) return p.c_str();
    }
#ifdef _WIN32
    return "python";
#else
    return "python3";
#endif
}

fs::path find_repo_root() {
    if (const char* e = std::getenv("OFLM_REPO_ROOT")) {
        if (*e && fs::is_directory(e)) return fs::weakly_canonical(fs::path(e));
    }
    // Walk up from cwd and from exe dir looking for utilities/build-new-spec.py
    std::vector<fs::path> starts;
    std::error_code ec;
    starts.push_back(fs::current_path(ec));
    const auto exe_dir = fs::path(utils::get_executable_directory());
    if (!exe_dir.empty()) starts.push_back(exe_dir);
    for (auto p : starts) {
        for (int i = 0; i < 8 && !p.empty(); ++i) {
            if (fs::is_regular_file(p / "utilities" / "build-new-spec.py", ec))
                return fs::weakly_canonical(p, ec);
            if (fs::is_regular_file(p / "open_kernels" / "recipes" / "spec.py", ec))
                return fs::weakly_canonical(p, ec);
            auto parent = p.parent_path();
            if (parent == p) break;
            p = parent;
        }
    }
    // fallback: cwd
    return fs::current_path(ec);
}

void print_help() {
    std::cout
        << "Usage: oflm dev <command> [args...]\n\n"
        << "Developer/toolchain workflows (source/build environments).\n\n"
        << "Commands:\n"
        << "  --build-new-spec [args...]   Build and validate a new kernel spec/geometry\n"
        << "                               (wraps utilities/build-new-spec.py)\n"
        << "      e.g. oflm dev --build-new-spec --repo mradermacher/Qwen2.5-1.5B-Instruct-i1-GGUF\n"
        << "           oflm dev --build-new-spec --spec qwen2.5-1.5b-instruct-i1-gguf\n"
        << "  --clean-specs [--dry-run]    Remove failed/candidate spec artifacts from\n"
        << "                               open_kernels/recipes/candidates/ (and report)\n"
        << "  -h, --help                   Show this help\n\n"
        << "Notes:\n"
        << "  - Kernel build/gating requires the NPU toolchain (XRT, mlir-aie/Peano)\n"
        << "    and, for validation, NPU hardware. `cmake --workflow --preset linux-default`\n"
        << "    and utilities/env.sh set up the environment in source trees.\n"
        << "  - Set OFLM_REPO_ROOT to point at the source tree if auto-detection fails.\n";
}

int clean_specs(const fs::path& repo, bool dry_run) {
    const fs::path cand = repo / "open_kernels" / "recipes" / "candidates";
    std::error_code ec;
    if (!fs::is_directory(cand, ec)) {
        std::cout << "No candidates directory: " << cand.string() << "\n";
        return 0;
    }
    int n = 0;
    for (const auto& e : fs::directory_iterator(cand, ec)) {
        if (!e.is_regular_file(ec)) continue;
        const auto p = e.path();
        std::cout << (dry_run ? "would remove " : "remove ") << p.string() << "\n";
        if (!dry_run) {
            fs::remove(p, ec);
            if (ec) {
                std::cerr << "  failed: " << ec.message() << "\n";
                return 1;
            }
        }
        ++n;
    }
    if (n == 0) std::cout << "No candidate specs to clean in " << cand.string() << "\n";
    else std::cout << (dry_run ? "Would remove " : "Removed ") << n << " candidate spec(s)\n";
    return 0;
}

} // namespace

int run(int argc, char* argv[]) {
    if (argc <= 0) {
        print_help();
        return 1;
    }
    std::string cmd = argv[0];
    if (cmd == "-h" || cmd == "--help" || cmd == "help") {
        print_help();
        return 0;
    }

    const fs::path repo = find_repo_root();

    if (cmd == "--build-new-spec" || cmd == "build-new-spec" || cmd == "new-spec") {
        const fs::path script = repo / "utilities" / "build-new-spec.py";
        if (!fs::is_regular_file(script)) {
            std::cerr << "Error: build-new-spec.py not found at " << script.string()
                      << "\nSet OFLM_REPO_ROOT to the OpenFlowLM source tree." << std::endl;
            return 1;
        }
        std::vector<std::string> args;
        args.emplace_back(python_exe());
        args.push_back(script.string());
        for (int i = 1; i < argc; ++i) args.emplace_back(argv[i]);
        return spawn(args);
    }

    if (cmd == "--clean-specs" || cmd == "clean-specs") {
        bool dry_run = false;
        for (int i = 1; i < argc; ++i) {
            std::string a = argv[i];
            if (a == "--dry-run" || a == "-n") dry_run = true;
            else if (a == "-h" || a == "--help") { print_help(); return 0; }
            else {
                std::cerr << "Unknown option for clean-specs: " << a << std::endl;
                return 1;
            }
        }
        return clean_specs(repo, dry_run);
    }

    std::cerr << "Unknown dev command: " << cmd << "\n\n";
    print_help();
    return 1;
}

} // namespace dev_command
