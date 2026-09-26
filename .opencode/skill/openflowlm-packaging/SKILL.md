---
name: openflowlm-packaging
description: Build and ship the OpenFlowLM Linux distribution (engine + open xclbins + bundled utilities) as RPM/TGZ (and DEB on Debian/Ubuntu). Use when producing a release, changing the install prefix or PATH handling, editing src/CMakeLists.txt install/CPack rules or CMakePresets package/workflow presets, or debugging "no manual steps" install problems (oflm not on PATH, user model added with oflm-add not visible, missing xclbins).
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

- `/opt/openflowlm/bin/oflm` + engine `.so` (`lib/xrt`) + `share/oflm/{model_list,model_info}.json`
- `share/oflm/xclbins/` — tracked closed sets + open sets built by `export_kernels`
- `share/oflm/{oflm-add,open_kernels/recipes}` — end-user model installer + spec derivation
- `share/oflm/utilities/{oflm-test,q4nx-build}` + launchers in `bin/` (`OFLM_BUILD_UTILITIES`)
- `/usr/bin/oflm` symlink and `/etc/profile.d/openflowlm.sh` (`OFLM_INSTALL_PATH_PLUMBING`)

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
utilities, and **no** `.bak`; `rpm -qpR` for `libxrt_coreutil.so.2`. The full
`cmake --workflow --preset linux-package` additionally rebuilds the kernels
(cached) and runs the tests in one shot; `git status` should show `src/xclbins`
manifests only if a recipe actually changed.

## Known blockers on this host

- `linux-default` with `OFLM_BUILD_KERNELS=ON` needs the kernel toolchain
  (`ironvenv`, mlir-aie + Peano, XRT, NPU present). `export-kernels.py` creates
  `ironvenv` automatically; the `open_npue` BERT sets additionally need the NPU
  on the build host.
- Engine-only validation (fast): configure with `OFLM_BUILD_KERNELS=OFF` and a
  writable prefix, then run `cmake --install DESTDIR=...` and `cpack`.
