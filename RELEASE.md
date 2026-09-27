# Cutting a release

A release is a tag. `.github/workflows/release.yml` builds every artifact from
it and publishes them as one GitHub release.

```
v1.2.3 pushed
  │
  ├─ verify      the tag is a version this tree can package, and the presets agree
  ├─ prebuilts   fetch the NPU-built kernels, check them against the tag
  ├─ linux       DEB + RPM + TGZ          (one ubuntu-24.04 build)
  ├─ windows     oflm-setup.msi            (windows-2022)
  ├─ nix         the flake, built to prove it works
  └─ publish     one GitHub release, checksummed
```

The interesting part is `prebuilts`. GitHub has no NPU, so the kernels cannot
be built there. They are built on a machine that has one and parked on a
dedicated branch until a release claims them.

## The two things a release needs that CI cannot do

| Built on an NPU machine, by hand | Why CI cannot |
| --- | --- |
| `src/xclbins/*/open_kernels*` and `src/xclbins/BERT-h*` | The open kernel sets compile without a device, but the BERT design sets allocate NPU tensors (`device="npu"`), so the whole export needs pyxrt and a card. |
| `xrt_coreutil.lib` and the XRT headers, for the MSI | The NPU *driver* is the runtime. The import library is made from the driver's own DLL, and the headers come from XRT with one generated header filled in by hand. |

Everything else -- the engine, the tests, all five package formats -- is built
in CI from the tag.

## One-time setup: the `npu-prebuilts` branch

An orphan branch: no common ancestor with `main`, rewritten (force-pushed) on
every release, so main's history never grows by a few hundred megabytes of
xclbins. Create it once, from any clone:

```bash
git fetch origin npu-prebuilts || git checkout --orphan npu-prebuilts
git rm -rf --cached . ; rm -rf ./* ; git commit --allow-empty -m 'prebuilts: orphan root'
git push origin npu-prebuilts
git checkout -
```

The branch keeps the two newest `prebuilts/<version>/` bundles and prunes older
ones, so an older tag can still be rebuilt but the branch does not grow forever.

## The release

### 1. On the NPU machine: build the kernels

From the commit that is about to be tagged, on the machine with the NPU:

```bash
source ironvenv/bin/activate
utilities/release/stage-prebuilts.sh --version 1.2.3
```

That runs `utilities/export-kernels.py --force` (every open kernel spec plus
the BERT design sets), collects **everything under `src/xclbins` that git
ignores**, writes a manifest, and force-pushes `npu-prebuilts`. It prints the
branch commit; that is the ref the release will claim.

The collection rule is `.gitignore` itself (`git ls-files --others --ignored
--exclude-standard -- src/xclbins`), so a new kernel family is picked up with no
edit to the script.

To build the MSI's inputs as well, on the **Windows** machine with the Ryzen AI
driver installed, first:

```powershell
utilities\release\stage-prebuilts-win.ps1 -Dest C:\path\to\bundle
```

then, on the Linux NPU machine, pointing the same `--dest` at it:

```bash
utilities/release/stage-prebuilts.sh --version 1.2.3 --dest /path/to/bundle --require-windows
```

`--require-windows` fails rather than producing a bundle without them, so a
release cannot quietly ship an MSI that will not run. The recipe (and why
`xrt/detail/version-slim.h` has to be generated) is in
[src/WinSetup.md](src/WinSetup.md).

### 2. Write the notes, bump the version, commit

`OFLM_VERSION` in **both** `CMakePresets.json` and `src/CMakePresets.json` is the
single source of truth ([docs/semantic-versioning.md](docs/semantic-versioning.md)).
The release workflow refuses a tag that disagrees with either.

Edit `utilities/release/release-notes.md` in the same commit: the workflow
publishes that file as the release body, so notes written after the tag describe
something slightly different from what shipped.

```bash
# OFLM_VERSION in CMakePresets.json and src/CMakePresets.json
# utilities/release/release-notes.md: what changed, models, known issues
git commit -am 'release: 1.2.3'
```

### 3. Tag it

```bash
git tag -a v1.2.3 -m 'OpenFlowLM 1.2.3'
git push origin v1.2.3
```

Only a plain `vX.Y.Z` tag triggers the workflow. A `-rc` suffix is not
packageable: an RPM `%VERSION` and a DEB `Version` field cannot carry one, and a
release that cannot be packaged is not a release.

### 4. Watch it

The workflow builds all five formats and publishes on success. If a job fails,
nothing is published: re-run the failed workflow, or re-run from
**Actions → Release → Run workflow** with `dry_run` to rebuild without
publishing.

Useful flags for a re-run:

| Input | Effect |
| --- | --- |
| `dry_run` | Build every artifact, publish nothing. The way to test a release. |
| `skip_prebuilts` | Build without the NPU kernels. The result loads no open model; `verify-package.sh` fails, which is the point. |
| `skip_windows` | Publish Linux only, when there is no MSI to publish. |
| `tag` | Re-run for a different tag without moving the current one. |

### 5. After the release

The repository's `nix/pin.json` still points at the *previous* release, because
updating it would be a commit after the tag. Each release carries its own flake
(`openflowlm-<version>-nix.tar.gz`), so this is optional -- but if you want
`nix build github:Atomic-Germ/OpenFlowLM-Next#openflowlm` to mean the newest
release, copy the pin out of the asset in a follow-up commit.

## What each format is, and what it costs

| Format | Built by | Notes |
| --- | --- | --- |
| DEB | `cpack -G DEB` on ubuntu-24.04 | Native. `Depends: libxrt-npu2`. |
| RPM | `cpack -G RPM` on ubuntu-24.04 | **Binds the build host's ABI**: glibc 2.39+, the apt FFmpeg sonames. Fedora 41+ / RHEL 10+. |
| TGZ | `cpack -G TGZ` | The same tree, no dependency resolution, for anywhere else. |
| MSI | `src/wix/build.ps1` on windows-2022 | Needs the prebuilt XRT headers and import library. |
| Nix | `nix/default.nix` | Packages the TGZ; see [nix/README.md](nix/README.md). |

A Fedora-native RPM is a known gap, and the honest version of it is: build
`linux-default` in a Fedora 41+ container (XRT from `xrt-base`, the
FFmpeg/Boost/FFTW `-devel` packages), then `cpack -G RPM` there. Until that is
scripted, the Ubuntu-built RPM is a working RPM for Fedora 41+ / RHEL 10+ and
nothing older.

## If a release goes wrong

The packages are not the source of truth: the tag is. Fix forward.

* **A kernel set is wrong** (a model produces bad numbers, an xclbin fails to
  load): the kernels are the only part of a release that is not built from the
  tag, so this is a prebuilt problem. Re-run `stage-prebuilts.sh --no-build`
  after fixing the design, re-tag as a patch version.
* **A package is missing something**: `verify-package.sh` is the checklist, and
  the release workflow runs it. Reproduce locally with
  `utilities/release/fetch-prebuilts.sh --version 1.2.3` followed by
  `cmake --preset linux-default -DOFLM_BUILD_KERNELS=OFF && cpack`.
* **A release was published broken**: `gh release delete v1.2.3 --yes`, fix, and
  tag a new patch version. Do not re-tag a published version: the RPM and DEB
  keep the same filename and the same version, and a user who already installed
  it will not get your fix.
