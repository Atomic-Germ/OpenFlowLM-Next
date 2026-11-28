---
title: "Build System"
description: "Configure, compile, test, and package the OpenFlowLM engine and open NPU kernels with CMake presets."
icon: "build"
weight: 10
---

OpenFlowLM compiles both the `oflm` binary and open MLIR-AIE / IRON NPU kernel sets (`.xclbin`, `.elf`, `insts.bin`, and `manifest.json`) from source via CMake presets. Open kernels are built locally rather than checked into git (remaining prebuilt `.xclbin` folders in `src/xclbins/` are inherited legacy artifacts being phased out; Nix builds strip them via [#244](https://github.com/Atomic-Germ/OpenFlowLM-Next/issues/244)).

## Two Preset Scopes

<div class="oflm-two-col">
  <div class="oflm-card">

### 1. Repository Root (Full Distribution) {#scope-repo-root}

Builds the engine, exports all open NPU kernels (bootstrapping `ironvenv`), and installs `/opt/openflowlm`:

```bash
cmake --preset linux-default
cmake --build --preset linux-default -j$(nproc)
```

Prepare host dependencies in [Before the First Build](/docs/install/linux/packages/#before-the-first-build).

  </div>
  <div class="oflm-card">

### 2. `src/` Subdirectory (Engine Only) {#scope-src-subdir}

Compiles only the C++/Rust executable. Use on Windows 11 ([`src/WinSetup.md`](https://github.com/Atomic-Germ/OpenFlowLM-Next/blob/main/src/WinSetup.md)) or when iterating on individual kernels manually.

  </div>
</div>

> [!IMPORTANT]
> Both the repo root and `src/` define `linux-default`. Check `pwd` before running `cmake --preset`. On Nix/NixOS, run inside `nix develop` ([Nix & NixOS](/docs/install/linux/nix/)).

---

## Root CMake Presets

| Preset | Kind | Behavior |
|---|---|---|
| `linux-default` (`fedora-default`) | Configure / Workflow | Engine + all open NPU kernels (`OFLM_BUILD_KERNELS=ON`) + utilities |
| `linux-debug` (`fedora-debug`) | Configure | Engine only with `Debug` symbols (`OFLM_BUILD_KERNELS=OFF`) |
| `linux-portable` | Configure | Full distribution bundling userspace XRT and XDNA libraries |
| `windows-default` | Configure | Visual Studio 17 2022 engine build (`src/` also has `windows-vs18`) |
| `linux-package` | Workflow | Configure + Build + CTest + emit `.rpm` and `.tar.gz` in `build/packages/` |
| `linux-package-deb` / `rpm` / `tgz` | CPack | Emit `.deb` (on Debian/Ubuntu or `./build_in_docker.sh`), `.rpm`, or `.tar.gz` |

Builds auto-detect `mold` (GCC 12+ or Clang) and `ccache` for C/C++ (disable via `-DOFLM_USE_MOLD=OFF -DOFLM_USE_CCACHE=OFF`). Keep `build/` intact for incremental Rust Cargo builds.

---

## Common Build Flows

```bash
# Full Linux build + install to /opt/openflowlm
cmake -B build --preset linux-default
cmake --build build -j
sudo cmake --install build

# Fast C++ engine iteration (skip kernel export)
cmake --preset linux-debug && cmake --build --preset linux-debug

# Build only selected kernel specs (or a single set within a spec)
cmake -B build --preset linux-default -DOFLM_KERNEL_SPECS=qwen35-9b
cmake -B build --preset linux-default -DOFLM_KERNEL_SPECS=qwen3-4b:ax0

# Single-command RPM + tarball package workflow
cmake --workflow --preset linux-package

# Run registered CTest smoke & compatibility tests
ctest --preset linux-default --output-on-failure
```

### Kernel Export (`utilities/export-kernels.py`)

CMake's `export_kernels` target compiles offline LLM/VLM/MoE specs (`open_kernels/export_qwen36_kernels.py` -> `src/xclbins/<model>/open_kernels/`) and device-dependent BERT families (`npu_offload/gemm_rtp/export_gemm_rtp.py` -> `src/xclbins/BERT-h*/`).

```bash
source utilities/env.sh
python utilities/export-kernels.py --specs qwen3-4b,gemma3-4b
```

| Flag | Effect |
|---|---|
| `--specs <list>` / `--skip-specs <list>` | Build or skip comma-separated spec names |
| `--only <set>` / `-j <N>` | Build one set in a spec or parallelize sets within a spec |
| `--skip-bert` / `--bert-only` | Omit BERT sets (for hosts without `/dev/accel/accel0`) or build BERT only |
| `--force` / `--check <DIR>` | Ignore `build_key` cache or verify instruction streams against `DIR` |

### Docker & Windows Builds

```bash
# Build Ubuntu .deb (with NPU passthrough and oflm-ccache volume)
./build_in_docker.sh
```

On Windows 11, configure from a Visual Studio Developer PowerShell (`vcvars64.bat`) and build kernels from a `mlir-aie` checkout pinned at tag `v1.4.2` (with LLVM `bin` on `PATH`):

```powershell
cmake --preset windows-default
cmake --build --preset windows-default

# BERT kernels on Windows (do not run concurrent exports in one checkout):
cd C:\dev\mlir-aie; . .\iron_env.ps1
cd <repo>; .\npu_offload\gemm_rtp\build.ps1
```

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `third_party/tokenizers-cpp does not contain a CMakeLists.txt` | Run `git submodule update --init --recursive` |
| Missing `Boost`, `xrt`, or `libdrm/drm.h` | Install distro `-dev`/`-devel` packages in [Before the First Build](/docs/install/linux/packages/#before-the-first-build) |
| Undefined `std::__1::...` linker symbols | Reconfigure with GCC (`libstdc++`): `CC=gcc CXX=g++ cmake --fresh --preset linux-default` |
| `mmap(...) failed (err=-11)` | Set `ulimit -l` (`memlock`) to `unlimited` ([Packages](/docs/install/linux/packages/)) |
| `Kernel not loaded` at runtime | Build `linux-default` (not `linux-debug`) and check `oflm config show` |
| `No module named 'pyxrt'` | Match Python to system XRT bindings (`python3-xrt` / `xrt-base` / `$XILINX_XRT/python`) |
