# Building from source

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

## 1. The executable on its own

### Prerequisites

- Git, Ninja, and CMake 3.25 or newer
- a C++20 compiler: MSVC on Windows, GCC or Clang on Linux

### Windows

**Run it from a Visual Studio developer environment**, not a plain shell. The
presets do not set the MSVC include paths themselves, and without them the
build fails deep inside a vendored dependency on a missing `<cstdint>` — an
error that names a third-party header and not the real cause.

```powershell
# a "x64 Native Tools Command Prompt", or in an existing shell:
& "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat"

cd src
cmake --preset windows-default
cmake --build build
```

The binary lands in `src/build/oflm.exe`, with `model_list.json`,
`model_info.json` and the engine DLLs copied beside it by the build — it will
not start without those, and Windows reports a missing DLL as a silent exit
before `main()`.

### Linux

```bash
cd src
cmake --preset linux-default     # installs to /opt/openflowlm
cmake --build build
sudo cmake --install build       # optional
```

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
python open_kernels/export_qwen36_kernels.py [--model-dir DIR | --spec FILE] [--out DIR]
```

It derives the model's shape from its `config.json`, builds every kernel set
the recipe names, and writes `final.xclbin` + `insts.bin` into
`src/xclbins/<model>/open_kernels/` — where `src/open_qwen36/engine.cpp` looks
for them. Beside them it writes `manifest.json` (everything the engine reads)
and `toolchain.json` (the mlir-aie and Peano versions, this tree's git commit,
and a sha256 per file, so a binary can be traced back to its source).

`--only`, `--no-build`, `--force` and `--check DIR` are there for iterating on
one set.

---

## 3. Check it works

```
oflm --version
oflm list
```

The engine prints which kernel set it resolved when it loads a model:

```
open_qwen36: kernels .../<model>/open_kernels (beside the model)
```

Three rules can pick that directory — `FLM_OPEN_KERNELS_DIR`, then a set beside
the model, then an `xclbins` root — and all three produce valid output. If you
have just rebuilt kernels and want to be sure the new ones ran, read that line.

---

## Notes

- The Windows and embedding-set instructions above were run on this repository;
  the Linux presets and the LLM kernel export are transcribed from the build
  scripts' own documented usage.
- `~/.flm`, `share/flm`, `lib/flm` and the `FLM_*` environment variables keep
  their names — those are install layout and contract, not the executable.
