---
title: "Semantic Versioning"
description: "Version numbering rules, pre-1.0 conventions, and CMake preset version bumps for releases."
icon: "tag"
weight: 70
---

OpenFlowLM releases follow [Semantic Versioning](https://semver.org) (`MAJOR.MINOR.PATCH`, with optional `-beta.N` or `+build.N` suffixes).

## Increment Rules

| Component | When to Increment | Example |
|---|---|---|
| `MAJOR` | Breaking CLI, HTTP API, or container format changes (post-1.0) | `1.2.3 -> 2.0.0` |
| `MINOR` | Backward-compatible features, model families, flags, or endpoints *(may include breaking changes while `0.x`)* | `0.4.0 -> 0.5.0` |
| `PATCH` | Backward-compatible bug, accuracy, build, or documentation fixes | `0.4.0 -> 0.4.1` |

`1.0.0` marks the first stable public contract; during `0.x`, minor bumps may carry breaking changes.

## Where `OFLM_VERSION` Lives

`OFLM_VERSION` in both root `CMakePresets.json` and `src/CMakePresets.json` is the single version source for `oflm version` (`__OFLM_VERSION__`) and CPack installers.

Tagging `v<version>` builds the `.deb`, `.rpm`, `.tar.gz`, Windows `.msi`, and prebuilt Nix tarball release assets. Follow [`RELEASE.md`](https://github.com/Atomic-Germ/OpenFlowLM-Next/blob/main/RELEASE.md) when cutting a release.
