---
name: openflowlm-packaging
description: Build and ship the OpenFlowLM Linux distribution (engine + open xclbins + bundled utilities) as RPM/TGZ (and DEB on Debian/Ubuntu). Use when producing a release, changing the install prefix or PATH handling, editing src/CMakeLists.txt install/CPack rules or CMakePresets package/workflow presets, or debugging "no manual steps" install problems (oflm not on PATH, user model added with oflm-add not visible, missing xclbins). For cutting a tag-driven release (MSI, prebuilt kernels, Nix), see the openflowlm-release skill.
---

# OpenFlowLM packaging

One command builds and packages everything:

```bash
cmake --workflow --preset linux-package   # configure + build + test + RPM/TGZ
```

Artifacts land in `build/packages/`. `linux-default` is the "everything" configure
preset (engine + `OFLM_BUILD_KERNELS=ON` + `OFLM_BUILD_UTILITIES=ON`); the
`linux-package` package preset emits `["RPM","TGZ"]` in one cpack run.
Per-format presets remain: `linux-package-{rpm,tgz}` plus `linux-package-deb`.

## XRT detection is common to all Linux presets

`linux-default` and `linux-debug` inherit a hidden `common-xrt` preset that sets
`XILINX_XRT`, prepends `/opt/xilinx/xrt/{lib64,lib}/pkgconfig` to
`PKG_CONFIG_PATH`, and prepends `/opt/xilinx/xrt` to `CMAKE_PREFIX_PATH`.
`fedora-default`/`fedora-debug` are now thin aliases of the Linux presets. This
is why `pkg_check_modules(XRT xrt)` succeeds on Fedora, where `/opt/xilinx/xrt`
is not a default pkg-config search path; nonexistent paths are ignored elsewhere.

## What ships

The runtime package (`openflowlm` on both formats) contains:

- `/opt/openflowlm/bin/oflm` + engine `.so` (`lib/xrt`) + `share/oflm/{model_list,model_info}.json`
- `share/oflm/xclbins/` — tracked closed sets + open sets built by `export_kernels`
- `share/oflm/{oflm-add,open_kernels/recipes}` — end-user model installer + spec derivation
- `share/oflm/utilities/{oflm-test,q4nx-build}` + launchers in `bin/` (`OFLM_BUILD_UTILITIES`)
- `/usr/bin/oflm` symlink and `/etc/profile.d/openflowlm.sh` (`OFLM_INSTALL_PATH_PLUMBING`)

The companion package (`openflowlm-devel` RPM / `openflowlm-dev` DEB) contains
the kernel-build toolchain under `/opt/openflowlm/share/oflm/devel/`:

- `utilities/{env.sh,env.fish,export-kernels.py}`, `ironvenv-requirements.txt`
- the full `open_kernels/` source tree (recipes, designs, specs, packers)
- `npu_offload/gemm_rtp/` + `npu_offload/m5-eltwise/kernels/` (BERT/embedding sets)
- `src/model_list.json` — the marker `env.sh` walks up to find its root
- `README.md` explaining that `/opt` is not writable and the tree must be copied
- `/etc/yum.repos.d/_copr:copr.fedorainfracloud.org:abn:amd-npu.repo` — the
  `abn/amd-npu` COPR that provides `xrt-base`, `xrt-npu`,
  `xrt-plugin-amdxdna`, `xrt-devel` and `python3-xrt` on Fedora. (The
  `xrt-devel` subpackage is the real name; older docs said `xrt-base-devel`,
  which does not exist.)

The TGZ is one archive containing both payloads: CPack component install is
enabled only for RPM and DEB, so TGZ keeps its single-archive behaviour.

### The two-package split (CPack components)

`src/CMakeLists.txt` sets `CMAKE_INSTALL_DEFAULT_COMPONENT_NAME "runtime"`, so
every existing `install()` is runtime and only the devel block says
`COMPONENT devel`. `CPACK_COMPONENTS_ALL "runtime;devel"`,
`CPACK_RPM_COMPONENT_INSTALL ON` / `CPACK_DEB_COMPONENT_INSTALL ON`, plus:

- `CPACK_RPM_MAIN_COMPONENT "runtime"` keeps the runtime RPM named
  `openflowlm` (otherwise it becomes `openflowlm-runtime`).
- `CPACK_RPM_DEVEL_PACKAGE_REQUIRES "openflowlm = %{version}-%{release}, ..."`
  pins the exact runtime NEVR and names all five COPR packages.
- DEB: `CPACK_DEBIAN_{RUNTIME,DEVEL}_PACKAGE_NAME`, plus
  `CPACK_DEBIAN_ENABLE_COMPONENT_DEPENDS ON` and
  `CPACK_COMPONENT_DEVEL_DEPENDS runtime` to get
  `Depends: openflowlm (= <version>)`; `CPACK_DEBIAN_DEVEL_PACKAGE_DEPENDS`
  adds `libxrt-dev, python3-xrt`.
- `/etc/yum.repos.d` is in `CPACK_RPM_EXCLUDE_FROM_AUTO_FILELIST_ADDITION` so
  the devel RPM does not claim to own a directory that `fedora-repos` owns.

A package cannot enable a repository in the transaction that installs from it,
so the COPR must be enabled first; the `.repo` file the devel package installs
is the same one `dnf copr enable abn/amd-npu` writes, so it does not duplicate
the definition.

The `open_kernels` copy is an `install(CODE)` walk, not `install(DIRECTORY)`
with `PATTERN "build*" EXCLUDE`: that pattern also eats real sources
`build_design.py` and `build_mlir.sh`. The walk skips only *directory*
components matching `^build`, `__pycache__`, `^\.staging`, `.locks`, and files
matching `\.(pyc|xclbin|o|elf|prj)$` — an AIE `.o`/`.elf` in the package makes
RPM's `brp-strip` abort. It uses `file(COPY)` (keeps permissions, does **not**
prepend `$ENV{DESTDIR}`) and fails configure/pkg if it copied zero files.

`utilities/release/verify-package.sh` routes `openflowlm-devel-*` /
`openflowlm-dev_*` to `check_devel` + `check_devel_requires` and runs
`check_common` only on the runtime; the TGZ is checked against both payloads.
It also asserts the runtime RPM `Requires: xrt-plugin-amdxdna` (the engine
`dlopen()`s the plugin, so `AUTOREQ` cannot derive it).

## Invariants that keep it "no manual steps"

1. Install prefix is `/opt/openflowlm`, which is **not** on PATH. The package
   writes `/usr/bin/oflm` (so `oflm`, `oflm-test`, `q4nx-build` resolve in any
   shell) and `/etc/profile.d/openflowlm.sh` (login shells).
2. Both are **outside `CMAKE_INSTALL_PREFIX`**, so they are written by
   `install(CODE)` under `$ENV{DESTDIR}`. Gotcha: `file(INSTALL)` prepends
   `$ENV{DESTDIR}` itself (do not add it to `DESTINATION`), but
   `file(MAKE_DIRECTORY)` and `cmake -E create_symlink` do not.
3. `CPACK_SET_DESTDIR=ON`; CPack packs everything staged under the package
   root, which is how `/usr/bin` and `/etc/profile.d` end up owned by the
   package. `CPACK_PACKAGE_RELOCATABLE=OFF` (the absolute symlink is
   incompatible with relocation).
4. XRT is a **system** dependency, not bundled: RPM auto-`Requires` picks up
   `libxrt_coreutil.so.2` (provided by `xrt-base` on Fedora), DEB names
   `libxrt-npu2`. `CPACK_DEBIAN_PACKAGE_SHLIBDEPS` turns on only when
   `/var/lib/dpkg/status` is present **and non-empty** — on Fedora it is a
   0-byte file, so shlibdeps is skipped (the explicit `Depends` is the fallback)
   and CMake prints why at configure time.
5. **Do not ship a DEB built on Fedora.** The engine binds the build host's ABI:
   Fedora links `libm.so.6(GLIBC_2.43)`, `libavcodec.so.62`, Boost 1.90, none of
   which exist on Ubuntu 24.04 (glibc 2.39 / FFmpeg 6 / Boost 1.83). That is why
   `linux-package` emits only `["RPM","TGZ"]`; build the DEB on Debian/Ubuntu or
   in a Debian container via `linux-package-deb`.
6. The engine-lib glob must not pick up backups:
   `file(GLOB ... "*.so" "*.so.*")` + `list(FILTER ... EXCLUDE REGEX "\\.bak")`.
   `src/lib/xrt/libq4_npu_eXpress.so.bak-20260826` is tracked in git but must
   never be installed.

## Runtime auto-discovery (why user installs work with no env vars)

- `utils::find_model_list()` reads the **user registry**
  (`~/.config/oflm/model_list.json`, written by `oflm-add`) before the shipped
  copy, so a just-added model is visible without `OFLM_CONFIG_PATH`.
  Caveat: the user registry is a snapshot-plus-additions; a model added to the
  shipped registry *after* the user's last `oflm-add` is not visible until the
  next `oflm add` refreshes the snapshot. If that matters, merge the two files
  in `model_list` rather than preferring one.
- The closed engines take exactly one xclbin root. `utils::find_xclbin_root_for(model)`
  (used by `LM_Config::_resolve_paths`) picks the root carrying
  `<root>/xclbins/<model>`: shipped models resolve from the install tree,
  user-added models from `~/.config/oflm/xclbins/<model>` symlinks.
  `find_xclbin_path()` keeps its old system-first single-winner behavior for the
  open embedding/`open_npue` family lookups.
- `open_qwen36::Engine::find_kernels` already scans every `utils::xclbin_roots()`.

## Verify a build

```bash
cmake --preset linux-default && cmake --build --preset linux-default
ctest --preset linux-default
cpack --preset linux-package          # build/packages/*.rpm + *.tar.gz
```

No `PKG_CONFIG_PATH` export needed: `common-xrt` supplies it (the preset's XRT
line should read `Found XRT via pkg-config`).

Inspect: `rpm -qlp`/`dpkg-deb -c` for `/usr/bin/oflm`, `/etc/profile.d/openflowlm.sh`,
utilities, and **no** `.bak`; `rpm -qpR` for `libxrt_coreutil.so.2` and
`xrt-plugin-amdxdna`. A cpack run emits two RPMs and two DEBs (`openflowlm` and
`openflowlm-devel` / `openflowlm-dev`) and one TGZ. The full
`cmake --workflow --preset linux-package` additionally rebuilds the kernels
(cached) and runs the tests in one shot; `git status` should show `src/xclbins`
manifests only if a recipe actually changed.

For a *release* build, note the two differences: the release workflow configures
with `OFLM_BUILD_KERNELS=OFF` and the kernels are already in the tag (committed
to the release branch by `stage-prebuilts.sh`, not fetched), and it checks the result
with `utilities/release/verify-package.sh build/packages --version <v>`, which
asserts the open-kernel and BERT xclbins are actually present. A local
`OFLM_BUILD_KERNELS=OFF` build of `main` legitimately fails that check — see
the `openflowlm-release` skill and [RELEASE.md](../../../../RELEASE.md).

## Known blockers on this host

- `linux-default` with `OFLM_BUILD_KERNELS=ON` needs the kernel toolchain
  (`ironvenv`, mlir-aie + Peano, XRT, NPU present). `export-kernels.py` creates
  `ironvenv` automatically; the `open_npue` BERT sets additionally need the NPU
  on the build host.
- Engine-only validation (fast): configure with `OFLM_BUILD_KERNELS=OFF` and a
  writable prefix, then run `cmake --install DESTDIR=...` and `cpack`.
