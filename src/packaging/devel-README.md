# OpenFlowLM development toolchain

Installed by `openflowlm-devel` (RPM) and `openflowlm-dev` (DEB). The runtime
package ships the finished xclbins and the engine; this is the part of the
source tree that *builds* kernel sets.

The directory is laid out so the tools' own path discovery works unchanged —
`env.sh` walks up to a directory holding `src/model_list.json`, and
`export-kernels.py` treats its parent's parent as the repository root:

```
utilities/env.sh              XRT + ROCm + mlir-aie/Peano + project paths
utilities/env.fish            the same, for fish
utilities/export-kernels.py   builds every kernel set
ironvenv-requirements.txt     the mlir-aie + Peano pins
open_kernels/                 recipes, designs, specs, packers
npu_offload/gemm_rtp/         the BERT (embedding) design sets
npu_offload/m5-eltwise/       kernels the BERT gemm design compiles in
src/model_list.json           env.sh's root marker
```

## Using it

The export writes build products beside the sources, and `/opt` is not writable
for a normal user, so copy the tree somewhere writable first:

```sh
cp -a /opt/openflowlm/share/oflm/devel ~/oflm-toolchain
source ~/oflm-toolchain/utilities/env.sh
python3 ~/oflm-toolchain/utilities/export-kernels.py
```

`env.sh` creates and hardcodes nothing: it finds XRT, ROCm and the mlir-aie
venv where they are installed (`utilities/env.sh --check` reports what it
found). `export-kernels.py` creates `ironvenv/` from
`ironvenv-requirements.txt` on the first run, using `uv` when it is present.

The sets land in `~/oflm-toolchain/src/xclbins/<model>/`. Sourcing `env.sh` in
that tree points `oflm` at them (`OFLM_XCLBIN_PATH`); `oflm add` installs a
model and links it to the family set.

## What the toolchain still needs

mlir-aie and Peano arrive in the venv (PyPI wheels). XRT does not: it comes
from the `abn/amd-npu` COPR, whose repository definition ships with this
package as
`/etc/yum.repos.d/_copr:copr.fedorainfracloud.org:abn:amd-npu.repo`. The
compiler side comes from Fedora:

```sh
sudo dnf install gcc-c++ cmake ninja-build xrt-devel python3
```

A package cannot enable a repository in the transaction that installs packages
from it, so enable the COPR first (`sudo dnf copr enable abn/amd-npu`) and
install `openflowlm-devel` after it. Installing this package is what keeps the
COPR enabled afterwards, and it fails loudly at install time if
`xrt-base`/`xrt-devel` cannot be resolved.
