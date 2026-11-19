---
title: "Packages"
description: "Install DEB, RPM, or portable tar.gz packages and configure AMD XDNA drivers on Linux."
icon: "inventory_2"
weight: 10
aliases: ["/docs/install_lin/"]
---

Install native `.deb`, `.rpm`, or portable `.tar.gz` packages into `/opt/openflowlm` and configure the `amdxdna` kernel driver and XRT runtime on `x86_64` Linux.

> [!TIP]
> Running containers or Nix? Follow the [Docker](/docs/install/linux/docker/) or [Nix & NixOS](/docs/install/linux/nix/) guides. Check supported Ryzen AI processors in [Hardware & Quickstart](/docs/install/requirements/).

## System Requirements

| Component | Requirement | Status |
|---|---|---|
| Ryzen AI 300 / 400 / Max 300 / Z2 Extreme | AMD XDNA2 NPU (`/dev/accel/accel*`) | Supported |
| Ryzen 7040 / 8040 / 200-series | Phoenix, Hawk Point (XDNA1) | Unsupported |
| NPU firmware | `1.1.0.0` or newer | Required |
| Kernel driver | Linux `6.17+` in-tree `amdxdna`, or `amdxdna-dkms` | Required |
| Userspace runtime | AMD XRT + `libxrt_driver_xdna.so.2` (or bundled `.tar.gz`) | Required |
| Locked memory | `ulimit -l unlimited` for NPU memory buffers | Required |
| Python | Python `3.8+` (used by `oflm add`) | Required |

## 1. Driver and XRT Setup {#step-1-driver-and-xrt-setup}

### Ubuntu 24.04, 25.10, and 26.04

1. Add the Lemonade XRT PPA ([`ppa:lemonade-team/stable`](https://launchpad.net/~lemonade-team/+archive/ubuntu/stable)) and install the XRT runtime and DKMS driver:

    ```bash
    sudo add-apt-repository ppa:lemonade-team/stable
    sudo apt update
    sudo apt install libxrt-npu2 amdxdna-dkms
    ```

    On Ubuntu 26.04 or kernels `6.17+` with matching `1.1+` firmware, the built-in `amdxdna` kernel module already provides the driver.

2. Raise `memlock` if `ulimit -l` is not `unlimited`, then reboot:

    ```bash
    printf '*    soft    memlock    unlimited\n*    hard    memlock    unlimited\n' \
      | sudo tee /etc/security/limits.d/99-npu-memlock.conf
    sudo reboot
    ```

### Fedora 44

1. Enable [`darrencocco/ryzen-ai-npu`](https://copr.fedorainfracloud.org/coprs/darrencocco/ryzen-ai-npu/) and install userspace XRT (`--allowerasing` swaps `OpenCL-ICD-Loader` for `ocl-icd`, keeping `libOpenCL.so.1`; Fedora 44 includes `amdxdna` in the kernel):

    ```bash
    sudo dnf copr enable darrencocco/ryzen-ai-npu
    sudo dnf install --allowerasing xrt-base xrt-plugin-amdxdna
    ```

2. Raise PAM and `systemd --user` `memlock` limits (prevents XRT `mmap ... err=-11` errors), log out and back in, and verify with `xrt-smi examine`:

    ```bash
    printf '*    soft    memlock    unlimited\n*    hard    memlock    unlimited\n' \
      | sudo tee /etc/security/limits.d/99-npu-memlock.conf
    sudo mkdir -p /etc/systemd/user.conf.d
    printf '[Manager]\nDefaultLimitMEMLOCK=infinity\n' \
      | sudo tee /etc/systemd/user.conf.d/99-npu-memlock.conf
    ```

### Arch Linux

1. Install kernel headers, firmware, and XRT, then install `amdxdna-dkms` from the AUR:

    ```bash
    sudo pacman -Syu linux-headers linux-firmware-other xrt xrt-plugin-amdxdna
    # After installing amdxdna-dkms from AUR:
    sudo dkms autoinstall -k "$(uname -r)"
    sudo depmod -a && sudo reboot
    ```

2. After rebooting, confirm `modinfo -F filename amdxdna` points under `updates/dkms`, verify `ulimit -l` reports `unlimited`, and run `xrt-smi examine`.

    > [!WARNING]
    > On Arch Linux 6.19, do not symlink `npu.sbin.zst` to 1.1 firmware on the stock kernel driver. Use `amdxdna-dkms` (which loads `npu_7.sbin.zst`) and confirm `oflm validate` reports firmware `1.1.x`.

### Build the Driver from Source (Fallback)

On distributions without prebuilt XRT/XDNA packages (and where the portable `.tar.gz` plus Linux `6.17+` in-tree `amdxdna` is not enough), build [`amd/xdna-driver`](https://github.com/amd/xdna-driver) from source:

1. Clone the upstream repository with its XRT submodule and install host build dependencies:

    ```bash
    git clone --recursive https://github.com/amd/xdna-driver.git
    cd xdna-driver
    sudo ./tools/amd_xdna_deps.sh
    ```

2. Build and install the XRT base package into `/opt/xilinx/xrt`:

    ```bash
    cd xrt/build
    ./build.sh -npu -opt
    sudo apt install ./Release/*-xrt.deb   # or sudo dnf install ./Release/*-xrt.rpm
    ```

3. Build and install the `amdxdna` plugin and DKMS module:

    ```bash
    cd ../../build
    ./build.sh -release
    sudo apt install ./Release/*-amdxdna*.deb   # or sudo dnf install ./Release/*-amdxdna*.rpm
    ```

4. Set `memlock` to `unlimited`, reboot, and confirm `/opt/xilinx/xrt/bin/xrt-smi examine` lists your Ryzen AI NPU.

## 2. Install OpenFlowLM Packages {#step-2-install-openflowlm-packages}

Download the package for your distribution from [GitHub Releases](https://github.com/Atomic-Germ/OpenFlowLM-Next/releases):

```bash
# Debian / Ubuntu (.deb)
sudo apt install ./openflowlm*.deb

# Fedora 41+ / RHEL 10+ (.rpm, glibc >= 2.39)
sudo dnf install ./openflowlm*.rpm
```

Packages install into `/opt/openflowlm` and place `oflm`, `oflm-add`, and `oflm-test` on `PATH` via `/usr/bin/oflm` and `/etc/profile.d/openflowlm.sh`.

### Portable Tarball (`.tar.gz`, Bundles XRT)

Use the portable archive on distributions without `.deb` / `.rpm` or system XRT packages. It bundles userspace XRT and XDNA libraries and unpacks to `/opt/openflowlm`:

```bash
tar xf openflowlm-<version>-Linux.tar.gz
sudo cp -r openflowlm-<version>-Linux/opt/openflowlm /opt/
export PATH=/opt/openflowlm/bin:$PATH
```

## 3. Validate and Run {#step-3-validate-and-run}

Check that OpenFlowLM and XRT can open your NPU:

```bash
xrt-smi examine
oflm validate
oflm validate --json
```

`oflm validate` verifies `/dev/accel/accel*`, firmware version, `memlock`, and the XRT runtime (`Device runtime: NPU opened`). With `--json`, confirm `"ready": true`. If `runtime_ok` is `false`, verify `libxrt_driver_xdna.so.2` with `xrt-smi examine`.

Add and run a model:

```bash
oflm add unsloth/Qwen3.5-4B-GGUF
oflm run qwen3.5:4b
```

For containerized `oflm serve` deployments, see [Docker](/docs/install/linux/docker/).

## Before the First Build

Install a prebuilt `.deb`, `.rpm`, or `.tar.gz` release above unless you are compiling OpenFlowLM from source (see [Build System](/docs/contributing/build/) for contributor presets). Before building from source, prepare four prerequisites:

1. **Submodules:** Clone recursively (`git clone --recursive https://github.com/Atomic-Germ/OpenFlowLM-Next.git`) or run `git submodule update --init --recursive`.
2. **Development headers** (including `libdrm-dev` / `libdrm-devel`):

    ```bash
    # Fedora 44 (with darrencocco/ryzen-ai-npu COPR enabled)
    sudo dnf install gcc-c++ cmake ninja-build mold ccache git cargo rust patchelf uv \
        boost-devel libcurl-devel fftw-devel ffmpeg-free-devel \
        readline-devel ncurses-devel libdrm-devel xrt-base-devel

    # Ubuntu / Debian (with ppa:lemonade-team/stable enabled)
    sudo apt install build-essential cmake ninja-build mold ccache pkg-config git \
        curl wget doxygen python3 python3-dev python3-pip python3-venv cargo rustc \
        clang lld patchelf rpm dpkg-dev file fakeroot uuid-dev \
        libboost-all-dev libcurl4-openssl-dev libfftw3-dev libreadline-dev \
        libncurses-dev libavformat-dev libavcodec-dev libavutil-dev libswscale-dev \
        libswresample-dev libssl-dev zlib1g-dev libdrm-dev libudev-dev \
        libxrt2 libxrt-npu2 libxrt-dev libxrt-utils libxrt-utils-npu python3-xrt
    ```

3. **Compiler:** Use GCC (`gcc`/`g++`) or Clang targeting `libstdc++` (reset CMake if switching compilers: `CC=gcc CXX=g++ cmake --fresh --preset linux-default`).
4. **NPU access during full builds:** Ensure `/dev/accel/accel0` is accessible and `ulimit -l` is `unlimited`, then run `cmake --preset linux-default && cmake --build --preset linux-default -j$(nproc) && sudo cmake --install build`.
