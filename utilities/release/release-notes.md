<!--
  Release notes for the GitHub release. The workflow publishes this file as the
  release body, so it is edited IN THE TAG, like everything else a release
  claims: a release whose notes were written afterwards is a release whose
  notes describe something slightly different from what shipped.

  Keep the generated facts (the table below) and the prose. Delete the
  instructions in this comment before tagging.
-->

## Install

<!-- Pick the one for the reader; keep all four, they are the supported set. -->

| Platform | Package | Install |
| --- | --- | --- |
| Fedora / RHEL | `openflowlm-<version>-1.x86_64.rpm` | `sudo dnf install ./openflowlm-<version>-1.x86_64.rpm` |
| Debian / Ubuntu | `openflowlm_<version>_amd64.deb` | `sudo apt install ./openflowlm_<version>_amd64.deb` |
| Any Linux | `openflowlm-<version>-Linux.tar.gz` | `tar xf … && sudo cp -r openflowlm-<version>-Linux/opt/openflowlm /opt/` |
| Windows | `oflm-setup.msi` | `msiexec /i oflm-setup.msi` |
| Nix | `openflowlm-<version>-nix.tar.gz` | `nix build "https://github.com/Atomic-Germ/OpenFlowLM-Next/releases/download/v<version>/openflowlm-<version>-nix.tar.gz#openflowlm"` |

The RPM is built on Ubuntu 24.04, so it needs **glibc 2.39 or newer** (Fedora
41+, RHEL 10+). Older distributions: use the TGZ, which is the same tree with
no dependency resolution.

XRT (the AMD NPU runtime, `libxrt-npu2` on Debian/Ubuntu, `xrt-base` on Fedora)
and the Ryzen AI NPU driver are **not** bundled. The packages declare the
dependency; the driver does not have a package on every distribution, so
install it first:

<https://ryzenai.docs.amd.com/en/latest/inst.html#install-npu-drivers>

Verify with:

```bash
oflm --version
oflm list
```

## What is in this release

<!-- Bullet per user-visible change. Model additions get the model tag; kernel
     and engine changes get the PR. Keep it to what a user would notice. -->

## Models

<!-- One line per added or newly-supported model, with `oflm add <tag>`. -->

## Known issues

<!-- Anything the release workflow could not check: a skipped MSI, a kernel set
     built on a different toolchain than the engine, a model whose weights are
     not published yet. Say it here rather than in an issue later. -->
