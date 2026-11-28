---
title: "Linux"
description: "Install OpenFlowLM on Linux via DEB or RPM packages, portable tarball, Docker, or Nix and NixOS."
icon: "terminal"
weight: 20
---

Install OpenFlowLM on `x86_64` Linux via native packages, portable archives, Docker containers, or Nix and NixOS:

<div class="oflm-two-col">
  <div class="oflm-card">

**[Packages →](/docs/install/linux/packages/)**

Configure the `amdxdna` NPU driver and XRT runtime on Ubuntu, Fedora, or Arch Linux, and install `.deb`, `.rpm`, or portable `.tar.gz` releases.

  </div>
  <div class="oflm-card">

**[Docker →](/docs/install/linux/docker/)**

Run `oflm serve` in Docker or Docker Compose with `/dev/accel/accel0` NPU passthrough and persistent model volumes.

  </div>
  <div class="oflm-card">

**[Nix & NixOS →](/docs/install/linux/nix/)**

Run `oflm` via `nix run`, enable the declarative `programs.openflowlm` NixOS module with `amdxdna` driver setup, or add OpenFlowLM to Home Manager.

  </div>
</div>

## Linux Installation Options

| Distribution / Workflow | Format | Guide |
|---|---|---|
| Ubuntu 24.04+ / Debian | `.deb` package + Lemonade XRT PPA | [Packages](/docs/install/linux/packages/) |
| Fedora 41+ / RHEL 10+ | `.rpm` package + Ryzen AI COPR | [Packages](/docs/install/linux/packages/) |
| Arch Linux & other distros | Portable `.tar.gz` (bundles userspace XRT) | [Packages](/docs/install/linux/packages/) |
| Containerized server | `docker run` or Docker Compose (`oflm serve`) | [Docker](/docs/install/linux/docker/) |
| Nix & NixOS (`x86_64-linux`) | Flake (`nix run`), `programs.openflowlm`, or prebuilt Nix tarball | [Nix & NixOS](/docs/install/linux/nix/) |
