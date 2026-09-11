# Build System

<<<<<<< HEAD
> "The code that matters." -- Keep it simple, clear, and focused.

OpenFlowLM uses a **single-command CMake workflow** for building from source. The build system handles both the executable and kernel exports automatically.

---

## Quick Start
=======
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

---

## Presets Overview

All builds use CMake presets in `CMakePresets.json`. Presets define configure, build, and test steps.

### Configure Presets

| Preset | Purpose | Description |
|---|---|---|
| `linux-default` | Full distribution | Builds executable + open NPU kernels (default) |
| `linux-debug` | Debug development | Engine only, kernels OFF (fast iteration) |
| `linux-portable` | Portable bundle | Bundles XRT/XDNA libraries |
| `windows-default` | Windows build | Visual Studio build (engine only; kernels Linux-only) |

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
| `linux-default` | Smoke test (`oflm list`) |

### Package Presets

| Preset | Output |
|---|---|
| `linux-package-tgz` | `.tar.gz` distribution |
| `linux-package-deb` | `.deb` package |
| `linux-package-rpm` | `.rpm` package |

### Workflow Presets

| Preset | Steps |
|---|---|
| `linux-default` | Configure + Build + Test (one command) |

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

**Output:**
- Binary in `build/bin/oflm`
- Kernels in `src/xclbins/`
- Installed to `/opt/openflowlm`

### 2. Engine-Only Build

Fast development iteration without kernel compilation.

```bash
cmake -B build --preset linux-debug
cmake --build build
cmake --install --preset linux-debug
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
- `qwen3-4b` -- Qwen3 dense 4B (all sizes)
- `gemma3-4b` -- Gemma3 dense 4B
- `llama-8b` -- Llama 3.1 8B
- `hy-mt2-7b` -- Hy-MT2-7B
- `granite-3b` -- IBM Granite 4.2 3B
- `qwen35-4b` -- Qwen3.5 dense 4B
- `qwen36-moe` -- Qwen3.6-MoE

### 4. Build Open NPUE Kernels Only

Build only the BERT embedding kernels (open_npue).

```bash
cmake -B build --preset linux-debug  # First, build engine only
cmake -B --build --preset linux-default  # Then build kernels
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
   - Output: `src/xclbins/<family>/`

### Export Flags

The `export_kernels.py` script supports:

| Flag | Description |
|---|---|
| `--specs <list>` | Comma-separated spec names (empty = all) |
| `--force` | Rebuild even when build cache is current |
| `--only` | Build only one spec/family |
| `--check DIR` | Verify against reference directory |

**Example:**
```bash
python utilities/export-kernels.py --specs qwen3-4b,gemma3-4b --only qwen3-4b:ax0
```

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

## Windows Build

Windows builds use Visual Studio. Run from a Visual Studio developer environment.

```powershell
# Set up environment
& "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat"

# Configure and build
cmake -B build --preset windows-default
cmake --build build
```

**Note:** Windows builds are engine-only; NPU kernel export is Linux-only.

---

## Development Workflow

### Quick Development
=======
The binary lands in `src/build/oflm.exe`, with `model_list.json`,
`model_info.json` and the engine DLLs copied beside it by the build — it will
not start without those, and Windows reports a missing DLL as a silent exit
before `main()`.

### Linux

[`linux-getting-started.md`](linux-getting-started.md) has the rest: the
`apt install` line for the development packages this build needs, and the
driver and XRT setup.

Other presets: `linux-portable`, `linux-snap`, `windows-vs18`.

---

## 2. The NPU kernels

There are two sets to build: the ones the embedding models use, and the ones
the language models use. Both need the IRON toolchain **dot-sourced** into the
shell first:

```powershell
cd C:\dev\mlir-aie; . .\iron_env.ps1        # the leading dot is required
```

On Linux, activate the equivalent `mlir-aie` virtualenv (`ironenv`), with
`xclbinutil` and `aiebu-asm` on `PATH` — both come from XRT, not from the
mlir-aie wheel.

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

- Check kernels were exported to `src/xclbins/`
- Run `oflm list` to see which kernel set was loaded
- Verify `model_list.json` includes your model

### "Build fails on Windows"

- Run in Visual Studio developer environment
- Use `vcvars64.bat` to set include paths
- Check CMakeLists.txt for platform-specific flags

### "Test passes locally but fails in CI"

- Check environment variables
- Verify kernel toolchain is active
- Check file paths are absolute

---

## Quick Links

- [Branch naming convention](../contributing/code-contributions.md)
- [Code contributions](../contributing/code-contributions.md)
- [Kernel contributions](../contributing/kernel-contributions.md)
- [Testing](../contributing/test-contributions.md)
- [Documentation](../contributing/doc-contributions.md)
- [Tools](../contributing/tool-contributions.md)
