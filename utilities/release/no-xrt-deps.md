# The MSI job stopped here

There are no XRT headers in the prebuilts bundle, so there is nothing on this
runner to compile `oflm.exe` against.

`xrt_coreutil.lib` and the XRT headers are the two Windows build inputs that
exist only on a machine with the Ryzen AI NPU driver installed. The driver *is*
the NPU runtime, and neither the headers nor the import library are in this
repository, which is the same reason the kernels come from a prebuilt bundle
rather than from CI.

## Fixing it

On the Windows machine with the NPU driver installed:

```powershell
cd C:\path\to\OpenFlowLM-Next
utilities\release\stage-prebuilts-win.ps1 -Dest C:\path\to\bundle
```

It makes the import library out of the driver's own `xrt_coreutil.dll` and
checks out the XRT headers (generating `xrt/detail/version-slim.h`, which does
not exist in a checkout and without which every XRT header fails to compile).
The recipe and the reasons are in `src/WinSetup.md`.

Then on the Linux NPU machine, from the release commit:

```bash
utilities/release/stage-prebuilts.sh --dest /path/to/bundle --require-windows
```

which assembles the bundle, verifies it, and pushes `npu-prebuilts`. Re-run the
release workflow; the kernels are cached in that branch, so only the kernels
rebuild if the source has not changed.

## Cutting a release without the MSI

Re-run this workflow with `skip_windows` set. Everything else is published, and
the release notes should say the MSI is missing rather than leaving people to
discover it.
