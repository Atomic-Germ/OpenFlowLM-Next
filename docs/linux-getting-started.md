# Linux Getting Started Guide

This guide will help you get started with OpenFlowLM on Linux, including setup for various distributions and NPU (Neural Processing Unit) support.

## Supported Distributions
- Ubuntu 24.04 LTS
- Ubuntu 25.10
- Ubuntu 26.04
- Fedora 44
- Arch Linux
- Other (Generic Linux)

---

## Prerequisites
- `amdxdna` driver (in-tree since kernel 6.17, or via `amdxdna-dkms`)
- NPU firmware version 1.1.0.0 or later
- Python 3.8+
- XRT stack from AMD

---

## System Preparation


### Ubuntu (24.04, 25.10)

#### 1. Add the AMD XRT PPA (Required for NPU/XDNA)
The AMD XRT stack is a prerequisite for NPU support. Add AMD's PPA:
```sh
sudo add-apt-repository ppa:lemonade-team/stable
sudo apt update
```
See [lemonade-team/stable PPA](https://launchpad.net/~lemonade-team/+archive/ubuntu/stable) for details.

#### 2. Install XRT and NPU Drivers
```sh
sudo apt install libxrt-npu2 amdxdna-dkms
```

#### 3. Reboot
```sh
sudo reboot
```

#### 4. Install OpenFlowLM
- Download the package for your distribution from the
  [Releases page](https://github.com/Atomic-Germ/OpenFlowLM/releases):

```sh
# Debian / Ubuntu (.deb)
sudo apt install ./openflowlm*.deb

# Fedora / RHEL (.rpm) -- needs glibc 2.39+ (Fedora 41+, RHEL 10+)
sudo dnf install ./openflowlm*.rpm

# Any Linux, no package manager (bundles XRT/XDNA; expects /opt/openflowlm)
tar xf openflowlm-<version>-Linux.tar.gz
sudo cp -r openflowlm-<version>-Linux/opt/openflowlm /opt/
export PATH=/opt/openflowlm/bin:$PATH
```

#### 5. (NPU) Check memlock limit
- Run:
   ```sh
   ulimit -l
   ```
- If not `unlimited`, add to `/etc/security/limits.conf`:
   ```
   *    soft    memlock    unlimited
   *    hard    memlock    unlimited
   ```
- Reboot system

---

### Arch Linux

Arch requires both the kernel-side `amdxdna` driver and the XRT userspace plugin. `oflm validate` opens `/dev/accel/accel0` through the DRM ioctls and then separately asks the device runtime to open the NPU, and `oflm run` uses XRT (`xrt::device(0)`) -- so both layers must be working, and `validate` reports on both.

#### 1. Install the runtime packages

```sh
sudo pacman -Syu
sudo pacman -S linux-headers linux-firmware-other xrt xrt-plugin-amdxdna
```

Install `amdxdna-dkms` from the AUR using your preferred AUR workflow, then rebuild the module for the running kernel if needed:

```sh
sudo dkms autoinstall -k "$(uname -r)"
sudo depmod -a
sudo reboot
```

If you need to rebuild a specific DKMS version, check `dkms status` and use that version explicitly.

If you run a non-default kernel, install the matching headers package instead, for example `linux-zen-headers` for `linux-zen`.

#### 2. Confirm the DKMS driver is selected

After rebooting, make sure `modinfo` resolves to the DKMS module rather than the in-tree kernel module:

```sh
modinfo -F filename amdxdna
```

Expected output should contain `updates/dkms`, for example:

```text
/lib/modules/6.19.13-arch1-1/updates/dkms/amdxdna.ko.zst
```

If it points under `kernel/drivers/accel/amdxdna/`, the stock kernel driver is still being used. Recheck matching headers, the DKMS build status, `sudo depmod -a`, and reboot.

#### 3. Confirm XRT sees the NPU

```sh
xrt-smi examine
```

The output should list an NPU under the device table. If `oflm validate` reports that the device runtime cannot open the NPU, XRT usually cannot see it. Confirm `xrt-plugin-amdxdna` is installed and that `xrt-smi examine` lists the device before trying `oflm run` again.

#### 4. Firmware note for Linux 6.19

Some Arch `linux-firmware-other` versions include both `npu.sbin.1.0.0.63.zst` and `npu.sbin.1.1.2.64.zst` for `17f0_10`. On the stock 6.19 in-tree `amdxdna` driver, forcing `npu.sbin.zst` to the 1.1 firmware can make the NPU disappear because the driver expects the older firmware protocol. The DKMS driver can use the protocol-7 firmware through `npu_7.sbin.zst`.

In short: if 1.1 firmware breaks probing on stock 6.19, do not keep forcing the `npu.sbin.zst` symlink. Use `amdxdna-dkms` or a kernel with the newer `amdxdna` driver, then verify `oflm validate` reports firmware version `1.1.x`.

---

### Fedora 44

The stock Fedora kernel carries the `amdxdna` driver (in-tree since 6.17), so
only the userspace side needs installing.

#### 1. Install XRT and the NPU plugin

Fedora's own repositories have no XRT. The third-party COPR
[`darrencocco/ryzen-ai-npu`](https://copr.fedorainfracloud.org/coprs/darrencocco/ryzen-ai-npu/)
packages it as `xrt-base`, the name the OpenFlowLM `.rpm` depends on:

```sh
sudo dnf copr enable darrencocco/ryzen-ai-npu
sudo dnf install --allowerasing xrt-base xrt-plugin-amdxdna
```

`--allowerasing` is there because `xrt-base` requires the `ocl-icd` OpenCL
loader, and a default Fedora install has the other one, `OpenCL-ICD-Loader`,
which conflicts with it. Both provide the same `libOpenCL.so.1` and read the
same `/etc/OpenCL/vendors/`, so dnf swapping one for the other loses nothing;
check that the transaction removes nothing else. The COPR also has
`amdxdna-kmod`, which you don't need with the in-tree driver.

#### 2. Raise the memlock limit

XRT locks the NPU's buffers into memory, and the default limit is 8 MB. With
it, everything that touches the NPU fails, `xrt-smi examine` included:

```text
mmap(addr=..., len=67108864, ...) failed (err=-11): Resource temporarily unavailable
```

```sh
printf '*    soft    memlock    unlimited\n*    hard    memlock    unlimited\n' \
  | sudo tee /etc/security/limits.d/99-npu-memlock.conf
```

Log out and back in, then run `ulimit -l` in a new terminal. If it still says
`8192`, the terminal was started by systemd's user manager, which applies its
own default; raise that too and log in again:

```sh
sudo mkdir -p /etc/systemd/user.conf.d
printf '[Manager]\nDefaultLimitMEMLOCK=infinity\n' \
  | sudo tee /etc/systemd/user.conf.d/99-npu-memlock.conf
```

#### 3. Confirm XRT sees the NPU

```sh
xrt-smi examine
```

The device table should list the NPU (`NPU Strix`, `NPU Strix Halo`,
`NPU Krackan`).

---

### Build System

For the full build system with CMake presets, see [docs/BUILD.md](BUILD.md).

#### Before the first build

Set up the NPU runtime for your distribution first (above). A from-source
build then needs three more things.

**The submodules.** The tokenizer library is a git submodule. Clone with
`--recursive`, or in a checkout made without it:

```sh
git submodule update --init --recursive
```

Without it, the configure stops with `third_party/tokenizers-cpp does not
contain a CMakeLists.txt file`.

**The development packages.**

Fedora 44:

```sh
sudo dnf install gcc-c++ cmake ninja-build mold ccache git cargo rust patchelf uv \
    boost-devel libcurl-devel fftw-devel ffmpeg-free-devel \
    readline-devel ncurses-devel libdrm-devel xrt-base-devel
```

`xrt-base-devel` comes from the COPR above. `uv` is optional; the kernel build
uses it, when it is there, to set up its Python venv. `ffmpeg-free-devel` is
enough to build; RPM Fusion's `ffmpeg-devel` adds the codecs Fedora leaves
out.

Ubuntu / Debian (the list the Docker builder installs on Ubuntu 26.04, with
XRT from the PPA above):

```sh
sudo apt install build-essential cmake ninja-build mold ccache pkg-config git \
    curl wget doxygen python3 python3-dev python3-pip python3-venv cargo rustc \
    clang lld patchelf rpm dpkg-dev file fakeroot uuid-dev \
    libboost-all-dev libcurl4-openssl-dev libfftw3-dev libreadline-dev \
    libncurses-dev libavformat-dev libavcodec-dev libavutil-dev libswscale-dev \
    libswresample-dev libssl-dev zlib1g-dev libdrm-dev libudev-dev \
    libxrt2 libxrt-npu2 libxrt-dev libxrt-utils libxrt-utils-npu python3-xrt
```

`libdrm-dev` / `libdrm-devel` is easy to miss: the configure does not check for
it, and the build stops partway with `fatal error: libdrm/drm.h: No such file
or directory`.

**A compiler that matches the system libraries.** Use GCC, or Clang with
GCC's C++ library (libstdc++). The distribution's Boost and XRT are built
against libstdc++, and a compiler that uses LLVM's libc++ instead (`zig c++`,
or a Clang set up that way) won't link against them. CMake keeps the first
compiler it found, so after installing or switching compilers, configure from
scratch:

```sh
CC=gcc CXX=g++ cmake --fresh --preset linux-default
```

**The NPU, at build time.** The embedding-model (BERT) kernel sets are built on
the device, so the build host needs the NPU working and the memlock limit
raised, as for running `oflm`.

**Quick build:**
```sh
cmake --preset linux-default
cmake --build --preset linux-default
cmake --install --preset linux-default
```

**Native package and portable tarball, in one command:**
```sh
cmake --workflow --preset linux-package   # configure + build + test + RPM/TGZ
```

`.deb` is built separately (`linux-package-deb`) on Debian/Ubuntu or in a
Debian container, because the engine binary carries the build host's glibc,
FFmpeg and Boost sonames.

The install goes to `/opt/openflowlm`; the package adds `/usr/bin/oflm` and
`/etc/profile.d/openflowlm.sh` so `oflm`, `oflm-test`, and `q4nx-build` are on
`PATH` with no shell-rc editing.

**Engine-only (fast iteration):**
```sh
cmake --preset linux-debug
cmake --build --preset linux-debug
```

**Build specific kernels:**
```sh
cmake --preset linux-default -DOFLM_KERNEL_SPECS=qwen3-4b
```

---

## Validating NPU Setup

To validate your NPU setup, run:
```sh
oflm validate
```
You should see output similar to this, with more or fewer debug lines per
your driver configuration:
```
[Linux]  Kernel: 7.3.0-0.rc4.260925g165768bb7026.42.fc46.x86_64
[Linux]  NPU: /dev/accel/accel0 with 8 columns
[Linux]  NPU FW Version: 1.1.2.64
[Linux]  amdxdna version: 0.10
[Linux]  Memlock Limit: infinity
251 (139936403511488): PID(35118): Created pcidev (0000:c2:00.1)
843716 (139936403511488): PID(35118): Opened /dev/accel/accel0 as 3, sysfs: /sys/bus/pci/devices/0000:c2:00.1
873802 (139936403511488): PID(35118): Expanding BO from 0 to 67108864
14209799 (139936403511488): PID(35118): Created expandable AMDXDNA_BO_DEV_HEAP: hdl=1 sz=0x4000000 paddr=0x4000000 vaddr=0x7f4558000000 uptr=0x0
14230859 (139936403511488): PID(35118): Created device (0000:c2:00.1) ...
14251958 (139936403511488): PID(35118): Destroying device (0000:c2:00.1) ...
14258370 (139936403511488): PID(35118): Destroying AMDXDNA_BO_DEV_HEAP: hdl=1 sz=0x4000000 paddr=0x4000000 vaddr=0x7f4558000000 uptr=0x0
17945498 (139936403511488): PID(35118): Closed 3
[Linux]  Device runtime: NPU opened
18243027 (139936403511488): PID(35118): Destroying pcidev (0000:c2:00.1)
```

Note that the NPU line reports the AIE **column count**, and that
`oflm validate` prints both a DRM-level check and a
`Device runtime: NPU opened` line. If the runtime line is missing or reads
`ERROR ... the device runtime cannot open it`, XRT could not load its NPU
plugin (`libxrt_driver_xdna.so.2`, built from
https://github.com/amd/xdna-driver). `xrt-smi examine` listing 0 devices is the
same symptom. If validation passes but running a model still fails, check XRT
separately:

```sh
xrt-smi examine
```

Use `oflm validate --json` to see each check as a field (`kernel_ok`,
`amd_device_found`, `all_fw_ok`, `enough_cols`, `memlock_ok`, 
`runtime_ok`, and the aggregate `ready`).

---
### Run with Docker Compose

On a Linux host with Docker Compose and the AMD XDNA driver, first build the
Ubuntu DEB (including the kernel exports), then start the API:

```bash
./build_in_docker.sh
docker compose up -d --build
curl --fail http://127.0.0.1:52625/api/version
```

The Dockerfile has separate `builder` and `runtime` targets. The build script
uses `builder`; Compose uses `runtime`, which installs the existing DEB and its
runtime dependencies into Ubuntu 26.04. No source checkout or compiler is mounted
in the running container. Keep exactly one `openflowlm*.deb` in `build/packages`
when building the runtime image; move older packages elsewhere. A package built
on another distribution may have incompatible dependencies. Kernel compilation
stays in the device-enabled build container because it needs the NPU.

Download a supported model and inspect the running service:

```bash
docker compose run --rm oflm pull llama3.2:1b
docker compose exec oflm /opt/openflowlm/bin/oflm list
docker compose logs -f oflm
docker compose down
```

The API listens at `http://127.0.0.1:52625` (OpenAI-compatible base URL:
`http://127.0.0.1:52625/v1`). The healthcheck checks `/api/version`; it does not
validate model inference. Models persist in the `./oflm-data` Docker volume mount and
the NPU cache in `./npu-cache`.

Optional environment variables, also accepted in a Compose `.env` file:

| Variable | Default | Purpose |
| --- | --- | --- |
| `OFLM_BIND_ADDRESS` | `127.0.0.1` | Host interface for the API (`0.0.0.0` exposes it to the network) |
| `OFLM_PORT` | `52625` | Published host port |
| `OFLM_NPU_DEVICE` | `/dev/accel/accel0` | Host NPU device passed into the container |

After rebuilding the DEB, run `docker compose up -d --build` to update the
runtime image and recreate the service. For the build environment alone, use
`docker build --target builder -t openflowlm-build:ubuntu26 .`.