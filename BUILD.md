# Building OpenFlowLM

Single-command CMake workflow. From a clean recursive clone, configure, build,
test, install, and package with preset-based commands from the **repository root**.

```bash
cmake --preset linux-default
cmake --build --preset linux-default
ctest --preset linux-default
cmake --install build
```

Or engine + kernels + tests + packages (RPM + TGZ) in one command:

```bash
cmake --workflow --preset linux-package
```

Install lands in `/opt/openflowlm`, with `/usr/bin/oflm` symlink +
`/etc/profile.d/openflowlm.sh`, so `oflm` is on `PATH` with no shell-rc edits.
(Set `-DOFLM_INSTALL_PATH_PLUMBING=OFF` to skip for an engine-only dev install.)

Kernel exports are owned by the build, not by hand. `-default` presets set
`OFLM_BUILD_KERNELS=ON` and run `utilities/export-kernels.py` for every spec in
`open_kernels/recipes/specs/` (incremental per-spec `build_key` cache).
`-debug` presets set it `OFF`. On Windows it is always `OFF`. What lands in
`src/xclbins/` after a `-default` build is the distribution asset set — build
it and ship it, do not re-run the export by hand.

Sources: [docs/BUILD.md](docs/BUILD.md) (full reference),
[README §Building](README.md#%EF%B8%8F-building-from-source),
[NIX.md](NIX.md), [docs/linux-getting-started.md](docs/linux-getting-started.md),
[docs/docs/install_lin.md](docs/docs/install_lin.md).

## 1. Prerequisites

- Git, CMake >= 3.25, C++20 compiler (GCC/Clang/MSVC), Ninja (recommended).
- NPU runtime stack (needed to *run*; the `open_kernels` families compile
  without a device, the BERT embedding sets need one):
  - AMD XDNA2 NPU (Strix / Strix Halo / Kraken / Gorgon Point). XDNA1 (7000 /
    8000 / 200-series) is not supported.
  - `amdxdna` kernel driver — in-tree since kernel 6.17, else `amdxdna-dkms`.
  - NPU firmware >= 1.1.0.0.
  - XRT userspace (`/opt/xilinx/xrt`), including `pyxrt` (ships for Python
    3.11, so the kernel toolchain venv is pinned to 3.11). Provides
    `xclbinutil`, `aiebu-asm`, `xrt-smi`.
  - Memlock `unlimited` (`ulimit -l`; set in `/etc/security/limits.conf`).
- Kernel toolchain (`ironvenv` with mlir-aie + Peano, Python 3.11) — **created
  automatically** by the build if absent. `third_party/mlir-aie` is cloned
  automatically (best-effort, for `toolchain.json` metadata). IRON designs pin
  mlir-aie **v1.4.2** (`ironvenv-requirements.txt`, `mlir_aie==1.4.2`).
- Ubuntu 24.04/25.10: `sudo add-apt-repository ppa:lemonade-team/stable &&
  sudo apt update && sudo apt install libxrt-npu2 amdxdna-dkms`, then reboot.
- Arch: `sudo pacman -S linux-headers linux-firmware-other xrt
  xrt-plugin-amdxdna` + `amdxdna-dkms` from the AUR, reboot; confirm with
  `modinfo -F filename amdxdna` (want `updates/dkms`) and `xrt-smi examine`.
- Validate any setup with `oflm validate` (also `oflm validate --json`:
  `kernel_ok`, `amd_device_found`, `all_fw_ok`, `enough_cols`, `memlock_ok`,
  `runtime_ok`, `ready`). If validate passes but `oflm run` fails with
  `No such device`, XRT lacks its amdxdna plugin.

## 2. Presets

All builds use `CMakePresets.json`. Both the repo root and `src/` contain a
preset called `linux-default` that does different things — where you run the
command matters. Root presets build the full distribution; `src/` presets
build the executable alone.

| Configure preset | Purpose |
|---|---|
| `linux-default` | Full distribution: engine + open NPU kernels + bundled utilities |
| `linux-debug` | Engine only, kernels OFF (fast iteration) |
| `fedora-default` / `fedora-debug` | Aliases of the above |
| `linux-portable` | Bundles XRT/XDNA libraries into the install tree |
| `windows-default` | VS 17 2022 generator (engine only); `windows-vs18` (VS 18 2026) lives in `src/` only |

Build presets mirror the configure names; `ctest --preset linux-default`
runs `oflm_smoke`, `openai_compat`, `OPEN-VISION-IMAGE-READ`, `bench_embed`.
Package presets: `linux-package` (RPM+TGZ), `linux-package-tgz`,
`linux-package-deb` (build on Debian/Ubuntu only — the binary carries the
host glibc/FFmpeg/Boost sonames), `linux-package-rpm`. Workflow presets:
`linux-default` (configure+build+test), `linux-package`
(configure+build+test+package RPM/TGZ).

## 3. Build flows

### Full distribution (Linux — the one to use)

```bash
cmake -B build --preset linux-default
cmake --build build -j
cmake --install build
```

Builds: `oflm` + engine libs + all open kernel xclbins + model registry +
bundled utilities (`oflm-test`, `q4nx-build`). Binary at `build/src/oflm`,
kernels in `src/xclbins/`, installed to `/opt/openflowlm`.

One command for engine + kernels + tests + RPM/TGZ into `build/packages/`:

```bash
cmake --workflow --preset linux-package
```

`.deb` is not part of it — drive with `cmake --build --preset
linux-package-deb && cpack --preset linux-package-deb` on Ubuntu/Debian.
Packages depend on system XRT (`xrt-base` Fedora, `libxrt-npu2` Ubuntu).
No `linux-static` preset exists — vendoring XRT/XDNA driver from source is
unsupported; use `linux-portable` to bundle the libraries instead:

```bash
cmake --preset linux-portable
cmake --build --preset linux-portable -j$(nproc)
sudo cmake --install build
```

### Engine-only (fast iteration)

```bash
cmake --preset linux-debug
cmake --build --preset linux-debug
```

No xclbins are produced — every model will report a missing design set. For
kernel work, rebuild one spec on top instead of everything (see below).

### Specific kernel specs

```bash
cmake -B build --preset linux-default -DOFLM_KERNEL_SPECS=qwen35-9b
cmake -B build --preset linux-default -DOFLM_KERNEL_SPECS=qwen3-4b:ax0
python utilities/export-kernels.py --specs qwen3-4b,gemma3-4b
```

Twelve specs in `open_kernels/recipes/specs/*.json`, all built by default:
`qwen3-4b`, `qwen35-9b`, `qwen36-35b-a3b`, `gemma3-4b`, `gemma3-12b`,
`llama31-8b`, `hy-mt2-7b`, `granite42-3b`, `qwen25-3b`, `lfm2-1.2b`,
`phi4-mini-4b`, `minicpm5-2b`. Open kernels need ironvenv + XRT on PATH, no
NPU; open NPUE (BERT sets) need pyxrt + NPU present. Useful flags on
`utilities/export-kernels.py`: `--skip-specs`, `--force`, `--skip-bert`
(no-NPU builds), `--bert-only`, `-j N`, `--only`, `--check DIR`
(forwarded to `open_kernels/export_qwen36_kernels.py`).

### Fine-grained targets

```bash
cmake --build build --target export_kernels   # kernels only
cmake --build build --target oflm             # executable only
cmake --install build                         # install
```

## 4. Nix

Full docs: [NIX.md](NIX.md), [nix/README.md](nix/README.md). Flake outputs:
`#oflm` (engine + dense kernels), `#oflm-with-bert` (needs NPU at build),
`#openflowlm-open-kernels`, `#openflowlm-open-kernels-with-bert`,
dev shells `#oflm` (C++/CMake + XRT/amdxdna) and `#open-kernels` (mlir-aie /
IRON toolchain).

```bash
nix develop .#open-kernels
python utilities/export-kernels.py            # dense + BERT (needs /dev/accel*)
python utilities/export-kernels.py --bert-only
nix run .#oflm -- --help
```

`nix/` packages the release tarball (see [RELEASE.md](RELEASE.md)); it does
no source build and no `/usr/bin` plumbing. `oflm-test`/`q4nx-build` need
their own pip env (`openai`; `gguf numpy torch einops safetensors
huggingface-hub`). NixOS module: `programs.openflowlm = { enable = true;
enableNPU = true; }` (set `enableNPU = false` on NPU-less build hosts).

## 5. Docker

```bash
./build_in_docker.sh                 # linux-package-deb by default
./build_in_docker.sh linux-package   # or name a workflow preset
docker compose up -d --build
curl --fail http://127.0.0.1:52625/api/version
```

Builds `openflowlm-build:ubuntu26`, passes `/dev/accel/accel0` in, enables
XRT memory locking, persists `./npu-cache`. Host needs the amdxdna driver
and `/dev/accel/accel0` (`xrt-smi examine` to check). Compose `runtime`
target installs the existing DEB — keep exactly one `openflowlm*.deb` in
`build/packages`. Container toolchain is `/opt/ironvenv` (`OFLM_VENV_DIR`),
outside the `/code` mount, so a host `ironvenv` can't shadow it.

## 6. Windows

Run from a VS developer prompt (`vcvars64.bat`). Engine only:

```powershell
cmake --preset windows-default
cmake --build --preset windows-default
```

Binary lands in `src/build/oflm.exe` with `model_list.json`,
`model_info.json`, engine DLLs beside it (missing DLL = silent exit before
`main()`); set `OFLM_XCLBIN_PATH` to the `xclbins` junction if lookup fails.
NPU kernels also build natively on Windows (see
[docs/BUILD.md §2](docs/BUILD.md#2-the-npu-kernels) + mlir-aie's
`docs/buildHostWinNative.md`, pinned to mlir-aie **v1.4.2**; needs the XRT
SDK zip at `C:\Xilinx\XRT`, `python utils\iron_setup.py`, and LLVM on PATH
for the missing `llvm-objcopy.exe`). Only the `export-kernels.py` driver is
Linux-only. Embedding sets (`.\npu_offload\gemm_rtp\build.ps1`, ~20 min for
5 families, never two concurrently — shared IRON cache) and LLM sets both
apply; dot-source `iron_env.ps1` first (`cd C:\dev\mlir-aie; . .\iron_env.ps1`).

## 7. Releases / packaging

A release is a tag; see [RELEASE.md](RELEASE.md). Kernels (`src/xclbins/*`)
and Windows XRT inputs (`prebuilts/win/`) are staged on NPU machines onto a
`release/X.Y` branch (`utilities/release/stage-prebuilts.sh`,
`stage-prebuilts-win.ps1`), verified with `verify-prebuilts.py`, then tagged
`vX.Y.Z`. DEB/RPM/TGZ + MSI + Nix all build in CI from the tag. Fedora-native
RPM is a known gap (Ubuntu-built RPM targets Fedora 41+/RHEL 10+).

## 8. Common issues

- **"Kernel not loaded"**: `src/xclbins/` holds closed prebuilts, so look for
  `src/xclbins/<model>/open_kernels*/` or `src/xclbins/BERT-h*/` (gitignored
  build outputs). `oflm list` shows which set loaded and which search path won
  (`OFLM_OPEN_KERNELS_DIR` > beside-model > `xclbins` root). `linux-debug`
  builds no kernels by design.
- **Windows build fails**: use the VS developer env + `vcvars64.bat`.
- **Tests pass locally, fail in CI**: env vars, kernel toolchain active,
  absolute paths.
- **`oflm validate` runtime check fails / `xrt-smi examine` shows 0 devices**:
  install the XRT amdxdna plugin for your distro.
- **Arch firmware**: don't force the 1.1 `npu.sbin.zst` symlink on stock 6.19
  (protocol mismatch); use `amdxdna-dkms` or a newer kernel, verify firmware
  `1.1.x` via `oflm validate`.

---

### 💡 Trick: Building XRT / Firmware Without the 98GB Vitis Download

If you need to build or modify the host XRT runtime and require the Embedded Resource Trainer (ERT) firmware, you do not need to download the massive AMD Vitis installer. XRT only uses Vitis to access the MicroBlaze GCC compiler.

You can bypass the AMD bloatware entirely by faking the Vitis environment:

1. **Install the standalone MicroBlaze compiler and required packages:**
   ```bash
   # On RHEL / CentOS / Rocky Linux:
   sudo dnf install -y \
     gcc-microblaze-linux-gnu \
     binutils-microblaze-linux-gnu \
     gcc-c++-microblaze-linux-gnu \
     libuuid-devel \
     ocl-icd-devel \
     pybind11-devel \
     pybind11-json-devel \
     libstdc++-static \
     glibc-static \
     rpmbuild \
     curl-devel \
     fftw3-devel \
     libavcodec-free-devel \
     libavformat-free-devel \
     libavutil-free-devel \
     libswscale-free-devel \
     libswresample-free-devel \
     readline-devel \
     libdrm-devel \
     boost-devel \
     rapidjson-devel \
     systemtap-sdt-devel \
     rust \
     cargo \
     git \
     cmake \
     ninja-build
   ```

   *(Note: Ensure you install `ocl-icd-devel` rather than alternative OpenCL development headers, or the build may fail on OpenCL runtime hooks).*

2. **Create a fake Vitis directory structure:**
   ```bash
   mkdir -p ~/fake_vitis/gnu/microblaze/lin/bin
   ```

3. **Symlink the lightweight compiler into your dummy directory:**
   ```bash
   ln -s /usr/bin/microblaze-linux-gnu-gcc ~/fake_vitis/gnu/microblaze/lin/bin/microblaze-xilinx-elf-gcc
   ln -s /usr/bin/microblaze-linux-gnu-g++ ~/fake_vitis/gnu/microblaze/lin/bin/microblaze-xilinx-elf-g++
   ```

4. **Point your environment to the fake path before building:**
   ```bash
   export XILINX_VITIS=~/fake_vitis
   git clone https://github.com/Xilinx/XRT.git --recursive
   cd XRT/build
   ./build.sh -npu -cmake-flags DXRT_ENABLE_USDT=OFF -j $(nproc)
   ```

CMake will now find the required compiler binaries, allowing you to compile the ERT firmware smoothly without ever touching the official AMD installer.
