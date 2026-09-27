# The MSI job stopped here

There are no XRT headers in this tag, so there is nothing on this runner to
compile `oflm.exe` against.

`xrt_coreutil.lib` and the XRT headers are the two Windows build inputs that
exist only on a machine with the Ryzen AI NPU driver installed. The driver *is*
the NPU runtime. They are staged into the repository, on the `staging` branch,
by the same machine that has the driver:

```powershell
git checkout staging
git pull
utilities\release\stage-prebuilts-win.ps1
```

It makes the import library out of the driver's own `xrt_coreutil.dll` and
checks out the XRT headers (generating `xrt/detail/version-slim.h`, which does
not exist in a checkout and without which every XRT header fails to compile).
The recipe and the reasons are in `src/WinSetup.md`.

Then tag from `staging` (not from `main`, which has no binaries) and re-run
the release. The Linux kernels, if already staged, are left alone: each script
owns one section of `prebuilts/manifest.json`.

## Cutting a release without the MSI

Re-run this workflow with `skip_windows` set. The MSI job is also skipped on
its own when the tag has no `platforms.windows` section. Everything else is
published, and the release notes should say the MSI is missing rather than
leaving people to discover it.
