---
layout: docs
title: Get Started (Linux)
nav_order: 2
has_children: false
---

# Linux NPU Support

This article will teach you how to run LLMs on your **AMD XDNA2 NPU** on Linux using **OpenFlowLM**.  
Get set up and then show us what you build!

**Date:** March 5, 2026  
**Authors:** [Lemonade-server🍋](https://lemonade-server.ai/) and OpenFlowLM contributors

## 📢 OpenFlowLM Linux Support

[OpenFlowLM](https://github.com/Atomic-Germ/OpenFlowLM) is a lightweight LLM runtime optimized for **AMD NPUs**.  
Today, OpenFlowLM is adding support for **Ubuntu, Arch, and other distros** to enable **fast, low-power LLMs** on **Ryzen™ AI PCs that run Linux**.

This article will help you:

- Understand **Linux NPU support status** and required platform versions
- Install the **OFLM + driver stack** for your distribution
- Validate your setup with `oflm validate`
- Fix common **firmware, driver, and memlock issues**

---

## ⚙️ Hardware Requirements

### Supported processors

OpenFlowLM on Linux requires an **AMD XDNA2 NPU**.

| Ryzen AI family | Codename | Status |
|---|---|---|
| Max 300-series | Strix Halo | Supported |
| 300-series | Kraken Point, Strix Point | Supported |
| 400-series | Gorgon Point | Supported |
| Z2 Extreme | Handheld devices | Supported |

> **Note:** Ryzen AI 7000 / 8000 / 200-series chips have **XDNA1**, which is **not supported**.

---

## 🧰 Software Requirements

### Runtime stack

The NPU requires specific firmware, kernel version, driver, and runtime software to function.  
The quickstart guide below will help you install these requirements.

| Item | Requirement |
|---|---|
| NPU firmware | Version 1.1.0.0 or later |
| Kernel + driver | Kernel **6.17+** with `amdxdna` (in-tree), or `amdxdna-dkms` |
| Runtime | OpenFlowLM installed |
| Memlock limit | Must be high enough for NPU execution |

---

## 🚀 Quickstart

## Supported Distributions
- Ubuntu 24.04 LTS
- Ubuntu 25.10
- Ubuntu 26.04
- Arch Linux
- Other (Generic Linux)

---

## 1. Prerequisites
- `amdxdna` driver (in-tree since kernel 6.17, or via `amdxdna-dkms`)
- NPU firmware version 1.1.0.0 or later
- Python 3.8+
- XRT stack from AMD

---

## 2. System Preparation

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

# Fedora / RHEL (.rpm)
sudo dnf install ./openflowlm*.rpm
```

> Pick the package that matches your distribution. The engine binary carries
> the build host's glibc, FFmpeg and Boost sonames, so a `.deb` is only valid on
> Debian/Ubuntu and an `.rpm` only on Fedora/RHEL. Both are built on
> `ubuntu-24.04` in CI.

##### Portable `.tar.gz`

If you would rather not install packages, the release also publishes a relocatable
tarball:

```sh
tar xf openflowlm-<version>-Linux.tar.gz
sudo cp -r openflowlm-<version>-Linux/opt/openflowlm /opt/
export PATH=/opt/openflowlm/bin:$PATH
```

The tarball unpacks to `openflowlm-<version>-Linux/opt/openflowlm/`, so it
installs to the same `/opt/openflowlm` prefix as the packages, with the
`profile.d` script and the `usr/bin/oflm` symlink already in the tree. It is
not relocatable -- it expects `/opt/openflowlm`, so put it there and add the
`bin` directory to `PATH` (the bundled `etc/profile.d/openflowlm.sh` does this
for login shells).

The tarball bundles the XRT/XDNA libraries, so it does not need system XRT. It
does still need the kernel `amdxdna` driver and the NPU firmware. The `.rpm`
additionally requires glibc 2.39 or newer (Fedora 41+, RHEL 10+).

#### 5. (NPU) Check memlock limit
- Run:
   ```sh
   ulimit -l
   ```
- If not `unlimited`, add to `/etc/security/limits.conf`:

   `*    soft    memlock    unlimited`   
   `*    hard    memlock    unlimited`
- Reboot system

---

### Ubuntu 26.04, Arch, and Others

For Ubuntu 26.04 and other distributions, check this [Linux NPU setup guide](https://lemonade-server.ai/flm_npu_linux.html).

#### Arch Linux

Arch users need the kernel driver, matching kernel headers, XRT, and the AMD XDNA XRT plugin:

```sh
sudo pacman -Syu
sudo pacman -S linux-headers linux-firmware-other xrt xrt-plugin-amdxdna
```

Install `amdxdna-dkms` from the AUR using your preferred AUR workflow, then reboot. If needed, rebuild the DKMS module for the running kernel:

```sh
sudo dkms autoinstall -k "$(uname -r)"
sudo depmod -a
sudo reboot
```

If you need to rebuild a specific DKMS version, check `dkms status` and use that version explicitly.

After rebooting, confirm the DKMS module is selected:

```sh
modinfo -F filename amdxdna
```

The path should contain `updates/dkms`. If it points under `kernel/drivers/accel/amdxdna/`, the stock in-tree driver is still being used.

Then confirm XRT can see the NPU:

```sh
xrt-smi examine
```

If `oflm validate` passes but `oflm run` fails with `No such device with index '0'`, XRT does not see a device. Make sure `xrt-plugin-amdxdna` is installed and `xrt-smi examine` lists the NPU.

> **Arch firmware note:** Some `linux-firmware-other` versions ship both 1.0 and 1.1 NPU firmware for `17f0_10`. On stock Linux 6.19, forcing `npu.sbin.zst` to 1.1 firmware can make the NPU disappear because the in-tree driver expects the older firmware protocol. Use `amdxdna-dkms` or a newer kernel that supports the protocol-7 firmware path, then verify `oflm validate` reports firmware `1.1.x`.

---

### Building from Source

1. Clone the repository:
   ```sh
   git clone https://github.com/Atomic-Germ/OpenFlowLM.git
   cd OpenFlowLM
   ```
2. Build and install from the **repository root** (the root presets are what
   produce the documented `/opt/openflowlm` layout):

   ```sh
   cmake --preset linux-default
   cmake --build --preset linux-default -j$(nproc)
   sudo cmake --install build
   ```

   See [docs/BUILD.md](https://github.com/Atomic-Germ/OpenFlowLM/blob/main/docs/BUILD.md)
   for the full set of presets.

#### Advanced Build Options

**Portable Build with Bundled XRT/XDNA**

To bundle the XRT/XDNA libraries into the install tree instead of depending on
system XRT, use the `linux-portable` preset:

```sh
cmake --preset linux-portable
cmake --build --preset linux-portable -j$(nproc)
sudo cmake --install build
```

There is no `linux-static` preset -- a truly static build that vendors XRT and
the XDNA driver from source is not currently supported.

---

## 3. Validating NPU Setup

To validate your NPU setup, run:
```sh
oflm validate
```
You should see output similar to:
```
[Linux]  Kernel: 6.19.13-arch1-1
[Linux]  NPU: /dev/accel/accel0 with 4 columns
[Linux]  NPU FW Version: 1.1.2.64
[Linux]  amdxdna version: <driver version>
[Linux]  Memlock Limit: infinity
[Linux]  Device runtime: NPU opened
```

The NPU line reports the AIE column count. `oflm validate` performs two checks:
it opens `/dev/accel/accelN` through the DRM ioctls, then asks the device
runtime to open the NPU the way `oflm run` does. If the runtime check fails it
prints `ERROR ... the device runtime cannot open it`; run `xrt-smi examine` and install the XRT AMD XDNA plugin for your distribution. Use
`oflm validate --json` to see each check as a field (`kernel_ok`,
`amd_device_found`, `all_fw_ok`, `enough_cols`, `memlock_ok`, `runtime_ok`, 
and the aggregate `ready`).

---

## 📚 Additional resources

- [Lemonade-server🍋](https://lemonade-server.ai/)
- [Lemonade GitHub issues](https://github.com/lemonade-ai/lemonade/issues)
- [Lemonade Discord](https://discord.com/invite/jtWZdMJ8ee)

---
