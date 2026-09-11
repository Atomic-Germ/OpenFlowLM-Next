# Building from source

**There are two things to build, and the executable alone is not enough.**

| | what it is | without it |
|---|---|---|
| **the executable** | `oflm` (`oflm.exe` on Windows) | nothing to run |
| **the AIE design sets** (`.xclbin` + `insts.bin`) | the NPU kernels the open engine dispatches | the binary starts and then **refuses to load a model**, naming the missing set |

The design sets are **not** checked in — they are compiled from the sources in
this repository, and building them needs a second toolchain (IRON / MLIR-AIE)
that the executable's build does not use. This is the part that surprises
people, so it comes first in every section below.

---

## 1. The executable

### Prerequisites

- Git, CMake ≥ 3.22, Ninja
- a C++20 compiler — MSVC on Windows, GCC or Clang on Linux

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

## 2. The AIE design sets

Both kinds need the IRON toolchain **dot-sourced** into the shell first:

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
