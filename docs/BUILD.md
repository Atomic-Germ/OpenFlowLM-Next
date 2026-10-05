# Build System

> "The code that matters." -- Keep it simple, clear, and focused.

OpenFlowLM uses a **single-command CMake workflow** for building from source. The build system handles both the executable and kernel exports automatically.

---

## Quick Start
Building the executable is not the whole job. The `oflm` binary also needs a
set of compiled NPU kernels - the `.xclbin` and `insts.bin` files, known as
design sets. Without them the binary starts up fine and then refuses to load
any model, naming the set it could not find. The kernels are not checked in;
they are compiled from sources in this repository.

There are two ways to get both, and they are not interchangeable.

**The short way, Linux only.** From the top of the repository, a single preset
builds the engine and all the NPU kernels together, and sets up the kernel
toolchain for itself if it isn't already there:

```bash
cmake --preset linux-default
cmake --build --preset linux-default
```

That is the path the [README](../README.md) describes in full, and on Linux it
is the one to use.

**The longer way, a step at a time.** From inside `src/`, a different set of
presets builds the executable on its own, and you build the kernels yourself
afterwards. This is the only option on Windows, where the kernel build does not
run. It is also the one you want while you are changing kernels and don't want
to rebuild everything each time. The rest of this page covers it.

Both directories contain a preset called `linux-default` and the two do
different things, so where you run the command from matters.

**On Nix / NixOS.** A flake ships the same two artifacts — the engine package
and a package of open NPU kernel xclbins — plus dev shells, a NixOS module and
a standalone `nix-shell` entry point. See [NIX.md](../NIX.md). Everything below
still applies to a `nix develop` shell; the flake just supplies the toolchain
(mlir-aie, Peano, XRT + the amdxdna plugin) instead of a distro package.

---

## Presets Overview

All builds use CMake presets in `CMakePresets.json`. Presets define configure, build, and test steps.

### Configure Presets

| Preset | Purpose | Description |
|---|---|---|
| `linux-default` | Full distribution | Builds executable + open NPU kernels + bundled utilities (default) |
| `linux-debug` | Debug development | Engine only, kernels OFF (fast iteration) |
| `fedora-default` | Fedora release | Alias of `linux-default` (XRT discovery is common to all Linux presets) |
| `fedora-debug` | Fedora debug | Alias of `linux-debug` |
| `linux-portable` | Portable bundle | Bundles XRT/XDNA libraries |
| `windows-default` | Windows build | Visual Studio 17 2022 generator (engine + kernels) |

These are the **repository-root** presets, which build the full distribution
(you also need one from `src/` when building from inside `src/`).

### Build Presets

| Preset | Description |
|---|---|
| `linux-default` | Build the configured preset |
| `linux-debug` | Debug build |
| `linux-portable` | Portable build |
| `windows-default` | Windows build |

### Test Presets

| Preset | Description |
|---|---|
| `linux-default` | Runs the four registered tests: `oflm_smoke`, `openai_compat`, `OPEN-VISION-IMAGE-READ`, `bench_embed` |

Test presets exist only in the repository-root `CMakePresets.json`. Run
`ctest --preset linux-default` from the repository root.

### Package Presets

| Preset | Output |
|---|---|
| `linux-package` | `.rpm` + `.tar.gz` (one command) |
| `linux-package-tgz` | `.tar.gz` distribution |
| `linux-package-deb` | `.deb` package (build on Debian/Ubuntu or in a Debian container) |
| `linux-package-rpm` | `.rpm` package |

### Workflow Presets

| Preset | Steps |
|---|---|
| `linux-default` | Configure + Build + Test (one command) |
| `linux-package` | Configure + Build + Test + Package RPM/TGZ (one command) |

---

## Build Flows

### 1. Full Distribution Build

Builds the executable **and** exports all open NPU kernels.

```bash
cmake -B build --preset linux-default
cmake --build build -j
cmake --install build
```

**What builds:**
- `oflm` executable
- Engine shared libraries
- Open kernel xclbins (all families)
- Model registry files
- Bundled utilities (`oflm-test`, `q4nx-build`) and their launchers

**Output:**
- Binary in `build/src/oflm`
- Kernels in `src/xclbins/`
- Installed to `/opt/openflowlm`
- `/usr/bin/oflm` symlink + `/etc/profile.d/openflowlm.sh`, so `oflm` is on
  `PATH` immediately after install with no shell-rc editing. (Set
  `-DOFLM_INSTALL_PATH_PLUMBING=OFF` to skip these for an engine-only dev
  install.)

**One command for engine, kernels, tests, and packages:**

```bash
cmake --workflow --preset linux-package
```

This configures, builds, tests, and emits the `.rpm` and portable `.tar.gz` into
`build/packages/`. Note that `linux-package-deb` and `linux-package-tgz` are
*package* presets, not workflow presets -- drive them with
`cmake --build --preset linux-package-deb && cpack --preset linux-package-deb`,
not with `cmake --workflow`. Packages depend on the system XRT (`xrt-base`
on Fedora, `libxrt-npu2` on Ubuntu), matching the runtime-prerequisite flow.

`.deb` is intentionally **not** part of this preset: a DEB is only valid when
built on Debian/Ubuntu, since the engine binary carries the build host's glibc,
FFmpeg and Boost sonames. Produce it with `linux-package-deb` on an Ubuntu host
or inside a Debian container.

### 2. Engine-Only Build

Fast development iteration without kernel compilation.

```bash
cmake --preset linux-debug
cmake --build --preset linux-debug
```

**What builds:**
- `oflm` executable
- Engine shared libraries (XRT or HRX)
**What doesn't build:**
- Open NPU kernel xclbins (export requires ironvenv)

**Use case:** Development iteration, testing engine logic without kernel builds.

### 3. Build Specific Kernel Specs

Export only specific kernel families.

```bash
# Build only Qwen3.5 4B kernel
cmake -B build --preset linux-default -DOFLM_KERNEL_SPECS=qwen35-4b

# Build specific kernel composition
cmake -B build --preset linux-default -DOFLM_KERNEL_SPECS=qwen3-4b:ax0
```

**Available specs:**
- `qwen3-4b` -- Qwen3 dense (all sizes)
- `qwen25-3b` -- Qwen2.5 dense 3B
- `gemma3-4b`, `gemma3-12b` -- Gemma 3 dense
- `llama31-8b` -- Llama 3.1 8B
- `hy-mt2-7b` -- Hy-MT2-7B
- `granite42-3b` -- IBM Granite 4.2 3B
- `qwen35-9b` -- Qwen3.5 dense (all sizes)
- `qwen36-35b-a3b` -- Qwen3.6-MoE 35B-A3B
- `phi4-mini-4b` -- Phi-4-mini
- `lfm2-1.2b` -- LFM2 1.2B
- `minicpm5-2b` -- MiniCPM 5 2B

### 4. Build Open NPUE Kernels Only

Build only the BERT embedding kernels (open_npue).

```bash
cmake --preset linux-debug
cmake --build --preset linux-debug   # engine only, no kernel export

cmake --preset linux-default -DOFLM_KERNEL_SPECS=qwen3-4b
cmake --build --preset linux-default  # now builds that kernel spec too
```

**Note:** Open NPUE kernels require the NPU present on the build host.

---

## Kernel Export Details

Kernels are exported as part of the CMake build via the `export_kernels` CMake target.

### The Export Script

`utilities/export-kernels.py` orchestrates kernel exports:

1. **Open kernels** (open_qwen36 + dense families)
   - Compiled by `open_kernels/export_qwen36_kernels.py`
   - One command per recipe spec
   - Compile-only; needs no NPU device
   - Output: `src/xclbins/<model>/open_kernels/`

2. **Open NPUE kernels** (BERT embedding design sets)
   - Compiled by `npu_offload/gemm_rtp/export_gemm_rtp.py`
   - One command per family in `npu_offload/gemm_rtp/families.json`
   - Needs pyxrt and an installed NPU
   - Output: `src/xclbins/<family>/` (e.g. `BERT-h768-bfp16`)

### Export Flags

The `export_kernels.py` script supports:

| Flag | Description |
|---|---|
| `--specs <list>` | Comma-separated spec names (empty = all) |
| `--skip-specs <list>` | Comma-separated spec names to skip (a recipe with a known gap) |
| `--force` | Rebuild even when build cache is current |
| `--skip-bert` | Skip the open_npue BERT sets (they need an NPU at build time) |
| `--bert-only` | Build only the open_npue BERT sets |
| `-j, --jobs N` | Kernel sets to build in parallel within a spec (default 1) |
| `--only` | Build only one spec/family |
| `--check DIR` | Verify against reference directory |

To build a single kernel within a spec, use the underlying exporter directly --
`open_kernels/export_qwen36_kernels.py` takes `--only` and `--check`:

**Example:**
```bash
python utilities/export-kernels.py --specs qwen3-4b,gemma3-4b
```

Each flag is forwarded to `open_kernels/export_qwen36_kernels.py` where it
applies (`--only`, `--check`, `-j`), so that script takes them too.

### Kernel Build Requirements

| Component | Requirements |
|---|---|
| **Open kernels** | ironvenv (mlir-aie + Peano), XRT on PATH, NPU not required |
| **Open NPUE** | ironvenv, XRT with pyxrt, NPU present |

---

## Build Targets

CMake provides custom targets for fine-grained control:

### `export_kernels`

Export all open NPU kernels during the build.

```bash
cmake --build build --target export_kernels
```

### `oflm`

Build the executable.

```bash
cmake --build build --target oflm
```

### `install`

Install the build to the configured prefix.

```bash
cmake --install build
```

or

```bash
cmake --build --target install
```

---

## Testing

### Smoke Test

```bash
ctest --preset linux-default
```

Runs `oflm list` to verify the binary, engine libraries, and model registry all load.

### Full Test Suite

```bash
ctest --preset linux-default --output-on-failure
```

### Test Specific Tests

```bash
ctest --preset linux-default --test-name-pattern <pattern>
```

---

### Docker build

Build fast and simple with Docker.

Simply run(linux-package preset is used by default):

```bash
./build_in_docker.sh
```

Or name a workflow preset:

```bash
./build_in_docker.sh linux-default
```

The script builds the `openflowlm-build:ubuntu26` image, passes
`/dev/accel/accel0` into the container, enables XRT memory locking, and
persists the NPU cache directory (`./npu-cache`). To produce a `.deb`, run the
`linux-package-deb` package preset inside the container:

```bash
docker run --rm -it --device=/dev/accel/accel0 --cap-add=IPC_LOCK \
  --ulimit memlock=-1:-1 -v "$PWD:/code" openflowlm-build:ubuntu26 \
  cmake --preset linux-package-deb && cmake --build --preset linux-package-deb \
  && cpack --preset linux-package-deb
```

The host must have the AMD XDNA driver installed and expose:

```text
/dev/accel/accel0
```

Verify NPU access with:

```bash
docker run --rm -it \
  --device=/dev/accel/accel0 \
  --cap-add=IPC_LOCK \
  --ulimit memlock=-1:-1 \
  openflowlm-build:ubuntu26 \
  xrt-smi examine
```

---

## Windows Build

Windows builds use Visual Studio. Run from a Visual Studio developer environment.

```powershell
# Set up environment (the windows-default preset uses the VS 17 2022 generator;
# for Visual Studio 18 2026 use the windows-vs18 preset instead)
& "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"

# Configure and build
cmake --preset windows-default
cmake --build --preset windows-default
```

**Note:** the commands above build the engine only. The NPU kernels themselves
(both the embedding and the language-model sets) build natively on Windows
too -- see "2. The NPU kernels" below, and mlir-aie's own
[`docs/buildHostWinNative.md`](https://github.com/Xilinx/mlir-aie/blob/main/docs/buildHostWinNative.md)
for the toolchain setup (a downloaded XRT SDK zip and `iron_setup.py`; no WSL,
no source build), pinned to mlir-aie **v1.4.2**, which this tree's designs are
written against. `utilities/export-kernels.py`'s own orchestration is
Linux-only (its path handling is POSIX-specific), not the build it drives.

---

## Development Workflow

### Quick Development
The binary lands in `src/build/oflm.exe`, with `model_list.json`,
`model_info.json` and the engine DLLs copied beside it by the build — it will
not start without those, and Windows reports a missing DLL as a silent exit
before `main()`. The build also creates an `xclbins` junction beside the binary
and warns if `OFLM_XCLBIN_PATH` is not set -- set it to that junction's path if
kernel lookup fails.

### Linux

[`linux-getting-started.md`](linux-getting-started.md) has the rest: the
`apt install` line for the development packages this build needs, and the
driver and XRT setup.

Other presets in the repository-root `CMakePresets.json`: `linux-portable`.
Two more exist only in `src/CMakePresets.json`: `linux-snap` and
`windows-vs18` (the Visual Studio 18 2026 generator).

---

## 2. The NPU kernels

There are two sets to build: the ones the embedding models use, and the ones
the language models use. Both need the IRON toolchain **dot-sourced** into the
shell first:

```powershell
cd C:\dev\mlir-aie; . .\iron_env.ps1        # the leading dot is required
```

On Linux, activate the equivalent `mlir-aie` virtualenv (`ironvenv`), with
`xclbinutil` and `aiebu-asm` on `PATH` — both come from XRT, not from the
mlir-aie wheel.

Setting that checkout up on Windows, if you don't have one, is mlir-aie's own
[`docs/buildHostWinNative.md`](https://github.com/Xilinx/mlir-aie/blob/main/docs/buildHostWinNative.md):
extract the XRT SDK zip to `C:\Xilinx\XRT`, clone mlir-aie, run
`python utils\iron_setup.py`, which writes the `iron_env.ps1` above. Two things
that guide does not tell you, both of which bite:

- **Check the clone out at tag `v1.4.2` first.** This tree's IRON designs use
  1.4.2's API — `ironvenv-requirements.txt` pins `mlir_aie==1.4.2` — and
  `iron_setup.py` reads the checkout: a tag installs its matching release wheel,
  while a checkout left on `main` installs the rolling one, whose API `dx.py`
  fails to import against (`cannot import name 'TaskGroup'`).
- The published Windows `llvm-aie` wheel ships no `llvm-objcopy.exe`, which
  `iron_setup.py` needs to repair that wheel; it falls back to one on `PATH`.
  `winget install LLVM.LLVM` and putting its `bin` on `PATH` satisfies it.

### Embedding models (`open_npue`)

```powershell
cd <repo>
.\npu_offload\gemm_rtp\build.ps1
```

Five families, roughly 3–4 minutes each, so budget about 20 minutes. Already
built families are skipped; `-Force` rebuilds and `-Only <name>` does one. The
build flags live in `families.json`, not in the script — one machine-readable
source, verified by `check_design_sets.py`.

**Do not run two families concurrently.** The IRON build cache is shared and
matches on content, so two families that share a geometry will delete each
other's work; the script takes a lock and refuses rather than letting that
happen.

### LLM models (the open engine)

```bash
# 1. Configure
cmake --preset linux-debug

# 2. Build
cmake --build --preset linux-debug

# 3. Run
oflm list
oflm run <model>
```

### Iteration Loop

1. Make changes in `src/`
2. Test with `oflm run <model>`
3. Run `oflm list` to verify kernel resolution
4. Commit and push

---

## Contributing to Build System

### When to Contribute

1. **Adding new specs** -- Add to `open_kernels/recipes/specs/`
2. **Fixing build issues** -- Report and fix in CMakeLists.txt
3. **Adding new presets** -- Update `CMakePresets.json`
4. **Improving documentation** -- Update this file

### Branch Naming

```
feat/added-new-spec
fix/build-kernel-issue
docs/build-process
chore/cmake-configuration
```

### Testing Build Changes

```bash
# Test build process
cmake --build build --preset linux-default
ctest --preset linux-default

# Test specific preset
cmake --build --preset linux-debug
```

---

## Common Issues

### "Kernel not loaded"

- Check the kernel spec was actually built. `src/xclbins/` already contains
  closed-source prebuilt directories, so the presence of that directory tells
  you nothing; look for `src/xclbins/<model>/open_kernels*/` or
  `src/xclbins/BERT-h*/`, which are the build outputs (and are gitignored).
- Run `oflm list` to see which kernel set was loaded.
- Verify `model_list.json` includes your model, and that `oflm` was not built
  with `linux-debug` -- that preset sets `OFLM_BUILD_KERNELS=OFF`, so no
  xclbins are produced and every model will report a missing design set.

### "Build fails on Windows"

- Run in Visual Studio developer environment
- Use `vcvars64.bat` to set include paths
- Check CMakeLists.txt for platform-specific flags

### "Test passes locally but fails in CI"

- Check environment variables
- Verify kernel toolchain is active
- Check file paths are absolute
Three rules can pick that directory — `OFLM_OPEN_KERNELS_DIR`, then a set beside
the model, then an `xclbins` root — and all three produce valid output. If you
have just rebuilt kernels and want to be sure the new ones ran, read that line.

---

## Quick Links

- [Branch naming convention](contributing/code-contributions.md)
- [Code contributions](contributing/code-contributions.md)
- [Kernel contributions](contributing/kernel-contributions.md)
- [Testing](contributing/test-contributions.md)
- [Documentation](contributing/doc-contributions.md)
- [Tools](contributing/tool-contributions.md)
- The Windows and embedding-set instructions above were run on this repository;
  the Linux presets and the LLM kernel export are transcribed from the build
  scripts' own documented usage.
- The install layout and the environment variables were renamed with the
  executable: `share/oflm`, `lib/oflm`, `OFLM_*`. The user model directory is
  `~/.oflm/models` on Windows and `~/.config/oflm/models` on Linux (older
  installs use `~/oflm/models`). A pre-rename install still works — `~/.flm` is searched after the oflm locations, and any
  variable read through `utils::getenv_oflm` accepts its old `FLM_*` name and
  prints a one-line notice naming the new one. `OFLM_OPEN_KERNELS_DIR` is the
  exception: the open engine reads it with plain `getenv`, so `FLM_OPEN_KERNELS_DIR`
  does nothing.
