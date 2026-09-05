r"""Where the design fixtures find their inputs, and how they name paths in run.cfg.

Two kinds of path used to be hard-coded in the make_test.py scripts (phlegm's
`C:/caps` capture tree and its `C:/code/phlegm/tools/open-kernels` checkout).
They now come from here:

- run.cfg paths are RELATIVE to the design directory. `run_kernel` resolves
  relative paths against the cfg's own directory, so `build/final.xclbin`,
  `x.bin` and `../deltanet/build/insts.bin` work from any checkout, on either
  side of a WSL/Windows split.
- Captured FLM buffers (weights sliced out of the closed engine's device
  buffers) are not in this repo. `OPEN_KERNELS_CAPS` names the directory that
  holds them; a generator that needs one and can't find it stops with a
  message instead of a traceback. `OPEN_KERNELS_CAPS_HOST` is the same
  directory as the *driver* sees it, for the WSL-generator / Windows-driver
  split (e.g. CAPS=/mnt/c/caps, CAPS_HOST=C:/caps). It defaults to CAPS.
- The chain harnesses (`*_chain/make_*.py`) additionally import phlegm's
  `tools/kernel-interp` (PHLEGM_KERNEL_INTERP) and a model container
  (MODEL_Q4NX). Their in-repo successor is `open_kernels/model/`.

Usage from a design directory:

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    import fixture_paths as FX
    W = np.fromfile(FX.caps("m0d/000118.bo"), np.uint8)      # read it here
    cfg = [f"buf pool 536870912 {FX.caps_cfg('m0d/000118.bo')}"]  # name it for the driver
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent            # open_kernels/
DESIGNS = ROOT / "designs"


def _fail(msg: str) -> "None":
    sys.exit(f"{Path(sys.argv[0]).name}: {msg}")


def caps(rel: str) -> Path:
    """A captured FLM buffer, for reading by the generator."""
    root = os.environ.get("OPEN_KERNELS_CAPS")
    if not root:
        _fail(f"needs the captured FLM buffer {rel!r}, which is not in this repo. "
              "Set OPEN_KERNELS_CAPS to the directory holding the captures "
              "(phlegm's was C:/caps; /mnt/c/caps under WSL), or use a design with a "
              "synthetic fixture (gemv_q4, ln, silu_mul, npu_offload/matmul).")
    p = Path(root) / rel
    if not p.is_file():
        _fail(f"{p} not found (OPEN_KERNELS_CAPS={root})")
    return p


def caps_cfg(rel: str) -> str:
    """The same capture as a path for run.cfg (what the driver opens)."""
    root = os.environ.get("OPEN_KERNELS_CAPS_HOST") or os.environ.get("OPEN_KERNELS_CAPS")
    if not root:
        _fail(f"needs OPEN_KERNELS_CAPS (and OPEN_KERNELS_CAPS_HOST if the driver runs on "
              f"another OS) to name the captured buffer {rel!r} in run.cfg.")
    return (Path(root) / rel).as_posix()


def kernel_interp() -> Path:
    """phlegm's tools/kernel-interp (decode_step.py, q4nx.py, build_pools.py).

    Only the *_chain harnesses need it; they are phlegm's step-by-step chains,
    kept for the record. The fp64 reference for this tree is model/replica.py."""
    root = os.environ.get("PHLEGM_KERNEL_INTERP")
    if not root or not (Path(root) / "decode_step.py").is_file():
        _fail("needs phlegm's tools/kernel-interp (PHLEGM_KERNEL_INTERP=<dir containing "
              "decode_step.py>). This chain harness predates open_kernels/model/, which "
              "is the in-repo way to score the kernels against an fp64 reference.")
    return Path(root)


def model_q4nx() -> str:
    """The .q4nx container the chain harnesses load through kernel-interp."""
    p = os.environ.get("MODEL_Q4NX")
    if not p or not Path(p).is_file():
        _fail("needs MODEL_Q4NX=<path to the model's .q4nx> (FLM keeps models under "
              "~/.flm/models/<model>/).")
    return p


def env_dir(var: str, what: str) -> str:
    """A directory that must come from the environment (no developer default)."""
    p = os.environ.get(var)
    if not p or not Path(p).is_dir():
        _fail(f"needs {var}=<{what}>")
    return Path(p).as_posix()
