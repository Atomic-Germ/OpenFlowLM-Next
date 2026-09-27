# Cutting a release

A release is a tag. `.github/workflows/release.yml` builds every artifact from
it and publishes them as one GitHub release.

```
v1.2.3 pushed (from the staging branch, not from main)
  │
  ├─ verify      the tag is a version this tree can package, and the kernels
  │              in it were built from this source
  ├─ linux       DEB + RPM + TGZ          (one ubuntu-24.04 build)
  ├─ windows     oflm-setup.msi            (windows-2022, if XRT inputs were staged)
  ├─ nix         the flake, built to prove it works
  └─ publish     one GitHub release, checksummed
```

GitHub has no NPU, so the kernels and the Windows XRT inputs cannot be built
there. They are built on machines that have one, committed to a long-lived
`staging` branch, and the release tag is cut from that branch. `main` stays
binary-free: it holds the version and the notes, and each cycle merges
`main` into `staging`, never the other way.

## The two things a release needs that CI cannot do

| Built on an NPU machine | Where it lands | Why CI cannot |
| --- | --- | --- |
| `src/xclbins/*/open_kernels*` and `src/xclbins/BERT-h*` | `staging`, by `utilities/release/stage-prebuilts.sh` | The open kernel sets compile without a device, but the BERT design sets allocate NPU tensors (`device="npu"`), so the whole export needs pyxrt and a card. |
| `prebuilts/win/` (XRT headers and `xrt_coreutil.lib`) | `staging`, by `utilities/release/stage-prebuilts-win.ps1` | The NPU *driver* is the runtime. The import library is made from the driver's own DLL, and the headers come from XRT with one generated header filled in by hand. |

Both scripts write one section of `prebuilts/manifest.json` (schema 2) and
leave the other section alone, so they can run in either order, any number of
times. Everything else -- the engine, the tests, all five package formats --
is built in CI from the tag.

## One-time setup: the `staging` branch

A long-lived branch that is `main` plus the prebuilts. Create it once:

```bash
git checkout main && git pull
git checkout -b staging
git push -u origin staging
```

At the start of each release cycle, merge `main` into it. Do not merge
`staging` back into `main`: that would put the binaries in `main`, which is
the thing this layout exists to avoid. The release tag is therefore not an
ancestor of `main`. That is expected.

## The release

### 1. On `main`: bump the version, write the notes, commit

`OFLM_VERSION` in **both** `CMakePresets.json` and `src/CMakePresets.json` is
the single source of truth
([docs/semantic-versioning.md](docs/semantic-versioning.md)). Both staging
scripts and the release workflow refuse a tree where the two disagree, or
where the value is not `X.Y.Z`.

Edit `utilities/release/release-notes.md` in the same commit: the workflow
publishes that file as the release body.

```bash
# OFLM_VERSION in CMakePresets.json and src/CMakePresets.json
# utilities/release/release-notes.md: what changed, models, known issues
git commit -am 'release: 1.2.3'
git push origin main
```

### 2. Merge `main` into `staging`

```bash
git checkout staging && git pull
git merge main
git push origin staging
```

### 3. On each NPU machine: stage that platform's binaries

The scripts refuse to run on any branch but `staging`, and they refuse a
working tree that has changes outside the paths they own. They fetch
`origin/staging` first, build, update only their own section of the manifest,
commit, and push. If the other machine pushed while this one was building,
the push is rejected and the script merges and re-applies its own section.
Do not force-push.

On the Linux machine with the NPU:

```bash
source ironvenv/bin/activate
git checkout staging && git pull
utilities/release/stage-prebuilts.sh
```

That runs `utilities/export-kernels.py --force` (every open kernel spec plus
the BERT design sets) unless `--no-build` is passed. The collection rule is
`.gitignore` itself, asked with `git check-ignore --no-index` so a re-run
still finds kernels that are already tracked on `staging`. A new family is
picked up with no edit to the script.

On the Windows machine with the Ryzen AI driver:

```powershell
git checkout staging
git pull
utilities\release\stage-prebuilts-win.ps1
```

The recipe (and why `xrt/detail/version-slim.h` has to be generated) is in
[src/WinSetup.md](src/WinSetup.md).

Before tagging, check that both sides landed:

```bash
utilities/release/verify-prebuilts.py
```

It is the same check the release workflow runs. A source-only commit on
`staging` after the kernels were built fails it: re-run the staging script
rather than tagging kernels built for a different engine.

### 4. Tag it, from `staging`

```bash
git checkout staging && git pull
git tag -a v1.2.3 -m 'OpenFlowLM 1.2.3'
git push origin v1.2.3
```

Only a plain `vX.Y.Z` tag triggers the workflow. A `-rc` suffix is not
packageable: an RPM `%VERSION` and a DEB `Version` field cannot carry one.

Do not tag `main`. It has no binaries, and `verify` will refuse the tag.

### 5. Watch it

The workflow builds all five formats and publishes on success. If a job fails,
nothing is published: re-run the failed workflow, or re-run from
**Actions → Release → Run workflow** with `dry_run` to rebuild without
publishing.

| Input | Effect |
| --- | --- |
| `dry_run` | Build every artifact, publish nothing. |
| `skip_prebuilts` | Package without the staged open kernel sets. The result loads no open model. |
| `skip_windows` | Publish Linux only. The MSI job is also skipped, on its own, when no Windows inputs were staged. |
| `tag` | Re-run for a different tag without moving the current one. |

### 6. After the release

The repository's `nix/pin.json` still points at the *previous* release, because
updating it would be a commit after the tag. Each release carries its own flake
(`openflowlm-<version>-nix.tar.gz`), so this is optional -- but if you want
`nix build github:Atomic-Germ/OpenFlowLM-Next#openflowlm` to mean the newest
release, copy the pin out of the asset in a follow-up commit on `main`, then
merge `main` into `staging` at the start of the next cycle.

## What each format is, and what it costs

| Format | Built by | Notes |
| --- | --- | --- |
| DEB | `cpack -G DEB` on ubuntu-24.04 | Native. `Depends: libxrt-npu2`. |
| RPM | `cpack -G RPM` on ubuntu-24.04 | **Binds the build host's ABI**: glibc 2.39+, the apt FFmpeg sonames. Fedora 41+ / RHEL 10+. |
| TGZ | `cpack -G TGZ` | The same tree, no dependency resolution, for anywhere else. |
| MSI | WiX on windows-2022 | Needs `prebuilts/win/` from the Windows NPU machine. |
| Nix | `nix/default.nix` | Packages the TGZ; see [nix/README.md](nix/README.md). |

A Fedora-native RPM is a known gap. Until it is scripted, the Ubuntu-built RPM
is a working RPM for Fedora 41+ / RHEL 10+ and nothing older.

## If a release goes wrong

The packages are not the source of truth: the tag is. Fix forward.

* **A kernel set is wrong** (a model produces bad numbers, an xclbin fails to
  load): re-run `stage-prebuilts.sh` on the NPU machine (it rebuilds and
  commits), then tag a patch version. Do not edit the xclbins by hand.
* **A package is missing something**: `verify-package.sh` is the checklist, and
  the release workflow runs it. Reproduce locally from the tag
  (`git checkout v1.2.3`) with
  `cmake --preset linux-default -DOFLM_BUILD_KERNELS=OFF && cpack`. The kernels
  are already in that tree.
* **A release was published broken**: `gh release delete v1.2.3 --yes`, fix, and
  tag a new patch version. Do not re-tag a published version: the RPM and DEB
  keep the same filename and the same version, and a user who already installed
  it will not get your fix.
