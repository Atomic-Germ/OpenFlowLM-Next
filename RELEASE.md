# Cutting a release

A release is a tag. `.github/workflows/release.yml` builds every artifact from
it and publishes them as one GitHub release.

```
v1.2.3 pushed (from release/1.2, not from main)
  │
  ├─ verify      the tag is a version this tree can package, and the kernels
  │              in it were built from this source
  ├─ linux       DEB + RPM + TGZ          (one ubuntu-24.04 build)
  ├─ windows     oflm-setup.msi            (windows-2022, if XRT inputs were staged)
  ├─ nix         the flake, built to prove it works
  └─ publish     one GitHub release, checksummed
```

GitHub has no NPU, so the kernels and the Windows XRT inputs cannot be built
there. They are built on machines that have one and committed to a release
branch, `release/X.Y`, cut from `main` once per minor version. The release tag
is cut from that branch. `main` stays binary-free: nothing is ever merged from a
release branch back into it.

Why a branch per release rather than one long-lived branch: the release checks
that the kernels were built from exactly the `open_kernels/` and `npu_offload/`
trees in the tag. A long-lived branch that takes all of `main` each cycle takes
whatever is in flight with it, and every merge that touches those directories
invalidates the kernels. A release branch is cut once, staged once, and after
that changes only by fixes cherry-picked from `main`.

## The two things a release needs that CI cannot do

| Built on an NPU machine | Where it lands | Why CI cannot |
| --- | --- | --- |
| `src/xclbins/*/open_kernels*` and `src/xclbins/BERT-h*` | the release branch, by `utilities/release/stage-prebuilts.sh` | The open kernel sets compile without a device, but the BERT design sets allocate NPU tensors (`device="npu"`), so the whole export needs pyxrt and a card. |
| `prebuilts/win/` (XRT headers and `xrt_coreutil.lib`) | the release branch, by `utilities/release/stage-prebuilts-win.ps1` | The NPU *driver* is the runtime. The import library is made from the driver's own DLL, and the headers come from XRT with one generated header filled in by hand. |

Both scripts write one section of `prebuilts/manifest.json` (schema 2) and
leave the other section alone, so they can run in either order, any number of
times. Everything else -- the engine, the tests, all five package formats --
is built in CI from the tag.

Both scripts work out the branch from `OFLM_VERSION` (`0.1.0` -> `release/0.1`)
and refuse to run anywhere else. `--branch` (Linux), `-Branch` (Windows) or the
`OFLM_STAGING_BRANCH` environment variable override it.

## The release

### 1. On `main`: the version

`OFLM_VERSION` in **both** `CMakePresets.json` and `src/CMakePresets.json` is
the single source of truth
([docs/semantic-versioning.md](docs/semantic-versioning.md)). Both staging
scripts and the release workflow refuse a tree where the two disagree, or
where the value is not `X.Y.Z`.

A new minor or major version is bumped on `main`, through a PR, before its
branch is cut. (A patch version is bumped on the release branch; see
[Patch releases](#patch-releases).)

### 2. Cut the release branch from `main`

Once per `MAJOR.MINOR`:

```bash
git fetch origin
git switch -c release/1.2 --no-track origin/main
git push -u origin release/1.2
```

From here on the branch changes only by:

* the two staging scripts,
* the release notes,
* fixes that have already merged to `main`, cherry-picked with
  `git cherry-pick -x <sha>`.

Never merge `main` into it, and never merge it into `main`. The release tag is
therefore not an ancestor of `main`. That is expected.

### 3. Write the release notes, on the branch

The workflow publishes `utilities/release/release-notes.md` as the release
body. On `main` it stays the template; each release branch fills it in.

```bash
# utilities/release/release-notes.md: what changed, models, known issues
git commit -am 'release notes: 1.2.3'
git push
```

### 4. On each NPU machine: stage that platform's binaries

The scripts refuse a working tree that has changes outside the paths they own.
They fetch the branch first, build, update only their own section of the
manifest, commit, and push. If the other machine pushed while this one was
building, the push is rejected and the script merges and re-applies its own
section. Do not force-push.

On the Linux machine with the NPU:

```bash
source ironvenv/bin/activate
git fetch origin && git switch release/1.2 && git pull
utilities/release/stage-prebuilts.sh
```

That runs `utilities/export-kernels.py --force` (every open kernel spec plus
the BERT design sets) unless `--no-build` is passed. The collection rule is
`.gitignore` itself, asked with `git check-ignore --no-index` so a re-run
still finds kernels that are already tracked on the branch. A new family is
picked up with no edit to the script.

On the Windows machine with the Ryzen AI driver:

```powershell
git fetch origin
git switch release/1.2
git pull
powershell -ExecutionPolicy Bypass -File utilities\release\stage-prebuilts-win.ps1
```

`-ExecutionPolicy Bypass` because Windows' default policy refuses to run an
unsigned script. Windows PowerShell 5.1 (what a stock install has) and pwsh 7
both work.

The recipe (and why `xrt/detail/version-slim.h` has to be generated) is in
[src/WinSetup.md](src/WinSetup.md).

Before going further, check that both sides landed:

```bash
utilities/release/verify-prebuilts.py
```

It is the same check the release workflow runs. A fix cherry-picked after the
kernels were built that touches `open_kernels/` or `npu_offload/` fails it:
re-run the Linux script rather than tagging kernels built for a different
engine.

### 5. Dry run

Build every artifact from the branch, as the version it is about to become,
without publishing:

```bash
gh workflow run release.yml --ref release/1.2 -f tag=v1.2.3 -f dry_run=true
```

Or **Actions → Release → Run workflow**, with *Use workflow from* set to the
release branch, `tag` set to `v1.2.3` and `dry_run` checked. A dry run builds
the commit the branch points at; the `tag` input only names the version, so the
tag does not have to exist yet.

### 6. Tag it, from the release branch

```bash
git switch release/1.2 && git pull
git tag -a v1.2.3 -m 'OpenFlowLM 1.2.3'
git push origin v1.2.3
```

Only a plain `vX.Y.Z` tag triggers the workflow. A `-rc` suffix is not
packageable: an RPM `%VERSION` and a DEB `Version` field cannot carry one.

Do not tag `main`. It has no binaries, and `verify` will refuse the tag.

### 7. Watch it

The workflow builds all five formats and publishes on success. If a job fails,
nothing is published: re-run the failed workflow, or re-run from
**Actions → Release → Run workflow**.

| Input | Effect |
| --- | --- |
| `dry_run` | Build every artifact, publish nothing. Builds the ref picked in *Use workflow from*, so it works before the tag exists. |
| `skip_prebuilts` | Package without the staged open kernel sets. The result loads no open model. |
| `skip_windows` | Publish Linux only. The MSI job is also skipped, on its own, when no Windows inputs were staged. |
| `tag` | Re-run for a different tag without moving the current one. With `dry_run`, the version to build as. |

### 8. After the release

Keep the branch: patch releases come from it.

The repository's `nix/pin.json` still points at the *previous* release, because
updating it would be a commit after the tag. Each release carries its own flake
(`openflowlm-<version>-nix.tar.gz`), so this is optional -- but if you want
`nix build github:Atomic-Germ/OpenFlowLM-Next#openflowlm` to mean the newest
release, copy the pin out of the asset in a follow-up PR to `main`.

## Patch releases

1. The fix lands on `main` through a PR, like any other.
2. Cherry-pick it onto the release branch: `git cherry-pick -x <sha>`.
3. On the branch, bump `OFLM_VERSION` in both presets to the patch version and
   update the release notes. Commit.
4. Re-stage both sides: the manifest records the version each side was staged
   for. On Linux, `stage-prebuilts.sh --no-build` is enough unless the fix
   touched `open_kernels/` or `npu_offload/`; then run it without the flag.
5. `verify-prebuilts.py`, dry run, tag, as above.

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
  load): fix it on `main`, cherry-pick, re-run `stage-prebuilts.sh` on the
  release branch (it rebuilds and commits), then tag a patch version. Do not
  edit the xclbins by hand.
* **A package is missing something**: `verify-package.sh` is the checklist, and
  the release workflow runs it. Reproduce locally from the tag
  (`git checkout v1.2.3`) with
  `cmake --preset linux-default -DOFLM_BUILD_KERNELS=OFF`. The kernels are
  already in that tree. Note that a bare `cpack` produces **TGZ only** --
  `CPACK_GENERATOR` defaults to `TGZ`, so all three formats need the loop the CI
  uses:

  ```sh
  for gen in DEB RPM TGZ; do cpack -G "$gen"; done
  ```
* **A release was published broken**: `gh release delete v1.2.3 --yes`, fix, and
  tag a new patch version. Do not re-tag a published version: the RPM and DEB
  keep the same filename and the same version, and a user who already installed
  it will not get your fix.
