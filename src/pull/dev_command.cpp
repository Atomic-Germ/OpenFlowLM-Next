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
#include <sstream>

#ifdef _WIN32
#include <process.h>
#else
#include <cerrno>
#include <cstring>
#include <unistd.h>
#include <sys/stat.h>
#endif

namespace dev_command {
namespace fs = std::filesystem;

namespace {

struct ToolchainStatus {
    bool xrt = false;
    bool peano = false;
    bool mlir_aie = false;
    bool npu_dev = false;
    std::string xrt_root;
    std::string peano_dir;
    std::string mlir_aie_dir;
};

int spawn(const std::vector<std::string>& args, const fs::path& cwd = {}) {
    std::vector<char*> raw;
    raw.reserve(args.size() + 1);
    for (const auto& arg : args) raw.push_back(const_cast<char*>(arg.c_str()));
    raw.push_back(nullptr);
#ifdef _WIN32
    if (!cwd.empty()) _chdir(cwd.string().c_str());
    const intptr_t rc = _spawnvp(_P_WAIT, raw[0], raw.data());
    return rc == -1 ? 1 : static_cast<int>(rc);
#else
    if (!cwd.empty()) {
        if (chdir(cwd.string().c_str()) != 0) {
            std::cerr << "Error: chdir " << cwd.string() << ": " << std::strerror(errno) << "\n";
            return 1;
        }
    }
    execvp(raw[0], raw.data());
    std::cerr << "Error: could not start " << args[0] << ": "
              << std::strerror(errno) << std::endl;
    return 1;
#endif
}

std::string python_exe(const fs::path& repo) {
    const char* configured = std::getenv("OFLM_PYTHON");
    if (configured && *configured) return configured;
    if (const char* v = std::getenv("VIRTUAL_ENV")) {
        fs::path p = fs::path(v) / "bin" / "python3";
        if (fs::is_regular_file(p)) return p.string();
    }
    // Prefer repo ironvenv if present
    fs::path iv = repo / "ironvenv" / "bin" / "python3";
    if (fs::is_regular_file(iv)) return iv.string();
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
    return fs::current_path(ec);
}

bool has_cmd(const char* name) {
    std::string cmd = "command -v ";
    cmd += name;
    cmd += " >/dev/null 2>&1";
    return std::system(cmd.c_str()) == 0;
}

ToolchainStatus detect_toolchain(const fs::path& repo) {
    ToolchainStatus s;
    const char* xrt = std::getenv("XILINX_XRT");
    s.xrt_root = (xrt && *xrt) ? xrt : "/opt/xilinx/xrt";
    s.xrt = fs::exists(fs::path(s.xrt_root) / "bin" / "xrt-smi") || has_cmd("xrt-smi");
    // Peano / mlir-aie in ironvenv
    fs::path iv = repo / "ironvenv";
    fs::path peano = iv / "lib" / "python3.11" / "site-packages" / "llvm-aie";
    if (!fs::exists(peano)) {
        // try other python versions
        std::error_code ec;
        if (fs::is_directory(iv / "lib", ec)) {
            for (auto& e : fs::directory_iterator(iv / "lib", ec)) {
                auto cand = e.path() / "site-packages" / "llvm-aie";
                if (fs::exists(cand)) { peano = cand; break; }
            }
        }
    }
    fs::path mlir = iv / "lib" / "python3.11" / "site-packages" / "mlir_aie";
    if (!fs::exists(mlir)) {
        std::error_code ec;
        if (fs::is_directory(iv / "lib", ec)) {
            for (auto& e : fs::directory_iterator(iv / "lib", ec)) {
                auto cand = e.path() / "site-packages" / "mlir_aie";
                if (fs::exists(cand)) { mlir = cand; break; }
            }
        }
    }
    s.peano = fs::exists(peano);
    s.mlir_aie = fs::exists(mlir);
    s.peano_dir = peano.string();
    s.mlir_aie_dir = mlir.string();
#ifdef __linux__
    s.npu_dev = fs::exists("/dev/accel/accel0") || fs::exists("/dev/xdna0");
#else
    s.npu_dev = false;
#endif
    return s;
}

void print_toolchain_hint(const fs::path& repo, bool for_build) {
    auto s = detect_toolchain(repo);
    bool missing = !s.xrt || !s.mlir_aie || !s.peano;
    if (!missing && (!for_build || s.npu_dev)) return;
    std::cerr << "\n[oflm dev] Toolchain/dependency check:\n";
    std::cerr << "  XRT:      " << (s.xrt ? ("ok (" + s.xrt_root + ")") : "not found") << "\n";
    std::cerr << "  mlir-aie: " << (s.mlir_aie ? "ok (ironvenv)" : "not found") << "\n";
    std::cerr << "  Peano:    " << (s.peano ? "ok (ironvenv)" : "not found") << "\n";
    if (for_build) {
        std::cerr << "  NPU dev:  " << (s.npu_dev ? "ok" : "not detected (/dev/accel/accel0)") << "\n";
    }
    if (missing) {
        std::cerr << "\nKernel builds need the NPU toolchain (XRT + mlir-aie/Peano). In a source tree:\n"
                  << "  - Install XRT and the NPU driver (see docs/BUILD.md / docs/linux-getting-started.md)\n"
                  << "  - Run: cmake --workflow --preset linux-default   (sets up ironvenv/toolchain)\n"
                  << "  - Or:  source utilities/env.sh\n"
                  << "If you installed OpenFlowLM from packages, install the development package\n"
                  << "(e.g. openflowlm-devel / openflowlm-dev) for the kernel toolchain and dev utilities.\n";
    } else if (for_build && !s.npu_dev) {
        std::cerr << "\nNPU device not detected. Kernel compilation can run without it, but GATE/validation\n"
                  << "and on-device tests require NPU hardware (/dev/accel/accel0).\n";
    }
    std::cerr << "\n";
}

void print_help() {
    std::cout
        << "Usage: oflm dev <command> [args...]\n\n"
        << "Developer/toolchain workflows for OpenFlowLM (source/build environments).\n\n"
        << "Commands:\n"
        << "  build-new-spec [args...]   Build and validate a new kernel spec/geometry\n"
        << "                             (wraps utilities/build-new-spec.py)\n"
        << "      e.g. oflm dev build-new-spec --repo mradermacher/Qwen2.5-1.5B-Instruct-i1-GGUF\n"
        << "           oflm dev build-new-spec --spec qwen2.5-1.5b-instruct-i1-gguf\n"
        << "           oflm dev build-new-spec --dry-run --repo <hf-repo>\n"
        << "  probe-sizes [args...]      Probe GGUF repos for served/would-build/unreadable\n"
        << "                             geometries (wraps utilities/probe-family-sizes.py)\n"
        << "      e.g. oflm dev probe-sizes --builtin --jobs 4\n"
        << "  export-kernels [args...]   Build/export open kernel xclbin sets\n"
        << "                             (wraps utilities/export-kernels.py)\n"
        << "      e.g. oflm dev export-kernels --specs qwen3-8b --skip-bert --skip-diffusion\n"
        << "           (use -j/--jobs to limit parallelism if builds are flaky)\n"
        << "  op-gate [args...]          Validate missing catalogue points by running fixtures\n"
        << "                             (wraps utilities/op-gate.py)\n"
        << "      e.g. oflm dev op-gate --spec qwen3-8b\n"
        << "  run-kernel <cfg> [args]    Run open_kernels harness (run_kernel) on a .cfg fixture\n"
        << "                             (builds harness if needed; OFLM_RUN_KERNEL overrides)\n"
        << "  clean-specs [--dry-run]    Remove failed/candidate specs from\n"
        << "                             open_kernels/recipes/candidates/\n"
        << "  clean-kernels [--sets]     Remove kernel export scratch (and optionally exported sets)\n"
        << "  env-check                  Show toolchain/environment status\n"
        << "  help                       Show this help\n\n"
        << "Aliases accepted: --build-new-spec, --probe-sizes, --export-kernels, --op-gate,\n"
        << "                  --clean-specs, --env-check, new-spec, run-kernel, harness, run_kernel.\n\n"
        << "Notes:\n"
        << "  - Geometry is the identity: new sizes of supported families are specs + builds,\n"
        << "    not new kernel code. `build-new-spec` runs SPEC->BUILD->GATE->CATALOGUE->REBUILD->REGISTER.\n"
        << "  - GATE/validation needs NPU hardware; compilation needs XRT + mlir-aie/Peano.\n"
        << "  - In source trees, `cmake --workflow --preset linux-default` or `source utilities/env.sh`\n"
        << "    prepares the toolchain. From packages, install openflowlm-devel/openflowlm-dev.\n"
        << "  - Large-K geometries may hit standalone fixture L1 limits during gating; treat those\n"
        << "    as fixture/layout validation gaps to investigate (not host RAM limits).\n";
}

int run_script(const fs::path& repo, const std::string& rel, int argc, char* argv[], bool buildy) {
    const fs::path script = repo / rel;
    if (!fs::is_regular_file(script)) {
        std::cerr << "Error: " << rel << " not found at " << script.string()
                  << "\nSet OFLM_REPO_ROOT to the OpenFlowLM source tree." << std::endl;
        if (buildy) print_toolchain_hint(repo, buildy);
        return 1;
    }
    // If this is a build/gate/export flow, warn early when toolchain looks missing
    if (buildy) print_toolchain_hint(repo, buildy);
    std::vector<std::string> args;
    args.emplace_back(python_exe(repo));
    args.push_back(script.string());
    for (int i = 0; i < argc; ++i) args.emplace_back(argv[i]);
    return spawn(args, repo);
}

int env_check(const fs::path& repo) {
    auto s = detect_toolchain(repo);
    std::cout << "repo:      " << repo.string() << "\n";
    std::cout << "python:    " << python_exe(repo) << "\n";
    std::cout << "XRT:       " << (s.xrt ? ("ok (" + s.xrt_root + ")") : "missing") << "\n";
    std::cout << "mlir-aie:  " << (s.mlir_aie ? ("ok (" + s.mlir_aie_dir + ")") : "missing (ironvenv)") << "\n";
    std::cout << "Peano:     " << (s.peano ? ("ok (" + s.peano_dir + ")") : "missing (ironvenv)") << "\n";
    std::cout << "NPU dev:   " << (s.npu_dev ? "present (/dev/accel*)" : "not detected") << "\n";
    const fs::path envsh = repo / "utilities" / "env.sh";
    std::cout << "env.sh:    " << (fs::is_regular_file(envsh) ? envsh.string() : "not found") << "\n";
    if (!s.xrt || !s.mlir_aie || !s.peano) {
        print_toolchain_hint(repo, true);
        return 1;
    }
    std::cout << "\nToolchain looks ready for kernel builds" << (s.npu_dev ? " and NPU validation." : " (NPU device not detected).") << "\n";
    return 0;
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

fs::path find_or_build_run_kernel(const fs::path& repo) {
    if (const char* e = std::getenv("OFLM_RUN_KERNEL")) {
        fs::path p(e);
        if (fs::is_regular_file(p)) return fs::weakly_canonical(p);
    }
    std::vector<fs::path> candidates = {
        repo / "build" / "op-gate" / "harness" / "run_kernel",
        repo / "open_kernels" / "harness" / "out" / "run_kernel",
        repo / "build" / "dev-harness" / "run_kernel",
    };
    // pick newest attn-gate harness if present
    std::error_code ec;
    fs::path attn_gate = repo / "build" / "attn-gate";
    fs::file_time_type best_time{};
    fs::path best;
    if (fs::is_directory(attn_gate, ec)) {
        for (const auto& ent : fs::directory_iterator(attn_gate, ec)) {
            if (!ent.is_directory(ec)) continue;
            fs::path rk = ent.path() / "harness" / "run_kernel";
            if (!fs::is_regular_file(rk, ec)) continue;
            auto t = fs::last_write_time(rk, ec);
            if (best.empty() || t > best_time) { best = rk; best_time = t; }
        }
    }
    if (!best.empty()) candidates.insert(candidates.begin(), best);
    for (const auto& c : candidates) {
        if (fs::is_regular_file(c, ec)) return fs::weakly_canonical(c, ec);
    }

    const fs::path harness_src = repo / "open_kernels" / "harness";
    if (!fs::is_directory(harness_src, ec)) return {};
    const fs::path build_dir = repo / "build" / "dev-harness";
    std::cout << "[oflm dev] Building run_kernel harness in " << build_dir.string() << "...\n";
    fs::create_directories(build_dir, ec);
    // Configure/build (best-effort). Uses system CMake/XRT; users may need env.sh.
    {
        std::vector<std::string> cfg = {"cmake", "-S", harness_src.string(), "-B", build_dir.string(), "-DCMAKE_BUILD_TYPE=Release"};
        if (spawn(cfg, repo) != 0) return {};
    }
    {
        std::vector<std::string> bld = {"cmake", "--build", build_dir.string(), "-j", "8"};
        if (spawn(bld, repo) != 0) return {};
    }
    fs::path out = build_dir / "run_kernel";
    if (fs::is_regular_file(out, ec)) return fs::weakly_canonical(out, ec);
#ifdef _WIN32
    out = build_dir / "run_kernel.exe";
    if (fs::is_regular_file(out, ec)) return fs::weakly_canonical(out, ec);
#endif
    return {};
}

int run_kernel_cmd(const fs::path& repo, int argc, char* argv[]) {
    for (int i = 0; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "-h" || a == "--help") {
            std::cout << "Usage: oflm dev run-kernel <run.cfg> [-- more harness args]\n\n"
                      << "Runs the open_kernels NPU harness (open_kernels/harness/run_kernel).\n"
                      << "Builds it if missing (requires CMake and XRT). Set OFLM_RUN_KERNEL to\n"
                      << "use a specific binary. Relative paths in the cfg resolve against the cfg's\n"
                      << "directory (same as running run_kernel directly).\n";
            return 0;
        }
    }
    if (argc <= 0) {
        std::cerr << "Error: run-kernel requires a cfg path (and any harness args).\n";
        return 1;
    }
    const fs::path rk = find_or_build_run_kernel(repo);
    if (rk.empty()) {
        std::cerr << "Error: could not find or build run_kernel harness.\n"
                  << "Ensure XRT/CMake are available (source utilities/env.sh in source trees)\n"
                  << "or set OFLM_RUN_KERNEL=/path/to/run_kernel.\n";
        print_toolchain_hint(repo, true);
        return 1;
    }
    std::vector<std::string> args;
    args.push_back(rk.string());
    for (int i = 0; i < argc; ++i) args.emplace_back(argv[i]);
    std::cout << "[oflm dev] " << rk.string();
    for (int i = 0; i < argc; ++i) std::cout << " " << argv[i];
    std::cout << "\n";
    return spawn(args, repo);
}

std::string norm(std::string c) {
    // accept --foo and foo
    while (c.rfind("--", 0) == 0) c = c.substr(2);
    return c;
}

} // namespace

int run(int argc, char* argv[]) {
    if (argc <= 0) {
        print_help();
        return 1;
    }
    std::string cmd = norm(argv[0]);
    if (cmd == "-h" || cmd == "help") {
        print_help();
        return 0;
    }

    const fs::path repo = find_repo_root();

    if (cmd == "build-new-spec" || cmd == "new-spec") {
        return run_script(repo, "utilities/build-new-spec.py", argc - 1, argv + 1, true);
    }
    if (cmd == "probe-sizes") {
        return run_script(repo, "utilities/probe-family-sizes.py", argc - 1, argv + 1, false);
    }
    if (cmd == "export-kernels") {
        return run_script(repo, "utilities/export-kernels.py", argc - 1, argv + 1, true);
    }
    if (cmd == "op-gate") {
        return run_script(repo, "utilities/op-gate.py", argc - 1, argv + 1, true);
    }
    if (cmd == "run-kernel" || cmd == "run_kernel" || cmd == "harness") {
        return run_kernel_cmd(repo, argc - 1, argv + 1);
    }
    if (cmd == "env-check" || cmd == "check-env" || cmd == "env") {
        return env_check(repo);
    }
    if (cmd == "clean-specs") {
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

    std::cerr << "Unknown dev command: " << argv[0] << "\n\n";
    print_help();
    return 1;
}

} // namespace dev_command
