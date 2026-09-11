# Semantic Versioning

OpenFlowLM follows [Semantic Versioning](https://semver.org) (SemVer) for its
releases. This document is the canonical reference for what a version number
means and how to bump it.

## The format

```
MAJOR.MINOR.PATCH
```

For example, `0.1.0` is major `0`, minor `1`, patch `0`.

Optionally, a pre-release or build suffix may be appended after a `-` or `+`
(e.g. `1.0.0-beta.1`, `1.0.0+build.5`). Pre-release (`-`) sorts before the
corresponding release; build metadata (`+`) does not affect precedence.

## What each number means

Given a version `MAJOR.MINOR.PATCH`, increment the:

1. **MAJOR** version when you make incompatible API or behaviour changes —
   anything that breaks existing users: a renamed CLI, a removed flag, a changed
   default, an output format change, or a model container format that old
   binaries can no longer read.

2. **MINOR** version when you add functionality in a backward-compatible manner —
   a new model family, a new flag, a new endpoint, a new optional feature. The
   `0.1.0` -> `0.2.0` kind of bump.

3. **PATCH** version when you make backward-compatible bug fixes — a crash fix,
   a wrong number, a broken link, a build fix. `0.1.0` -> `0.1.1`.

When a number is incremented, the ones to its right reset to zero: `1.2.3` ->
`1.3.0` (minor) -> `2.0.0` (major).

## The `0.x` pre-1.0 convention

Before `1.0.0`, the public API is treated as unstable. The rule is:

- `0.MAJOR` — while the major is `0`, **anything may change at any time**.
  `0.1.0` -> `0.2.0` may contain breaking changes. This is expected and normal;
  it is the whole point of the leading zero.

- `1.0.0` marks the first stable API: from then on, breaking changes **require**
  a major bump.

OpenFlowLM is currently in the `0.x` range. Breaking changes during `0.x` are
expressed as minor bumps (`0.1.0` -> `0.2.0`), not major bumps.

## How OpenFlowLM applies it

The single source of truth for the version is `OFLM_VERSION` in
`CMakePresets.json` (both the repository-root preset and the one under `src/`).
It is baked into the binary at build time (`__OFLM_VERSION__`) and used for the
packages (CPack).

A release is therefore one commit that:

1. Bumps `OFLM_VERSION` to the correct next version.
2. Tags the commit `v<version>` (e.g. `v0.1.0`).
3. Produces the installers from that tag.

A breaking change (renamed binary, removed flag, format change) means a minor
bump while the major is `0`, and a major bump once the major is `>= 1`.

## Why this matters

Version numbers that follow SemVer are a promise. A consumer can read `1.3.1`
and know, without reading the changelog, that upgrading from `1.3.0` is safe,
and that jumping to `2.0.0` is not. Skipping the discipline — bumping the patch
for a breaking change, or the major for a trivial one — destroys that promise
and forces everyone to read every release note by hand.
