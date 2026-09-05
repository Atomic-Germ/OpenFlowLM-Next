r"""ROT13 on the NPU -- toolchain smoke test (design only, no dispatch).

Adapted from vegah/LLMNpuTest designs/rot13 (Apache-2.0, see LICENSE.LLMNpuTest):
same kernel, same dataflow; the dispatch half is dropped because we build in
WSL (no NPU) and run from Windows through phlegm's own XRT shim
(`open-qwen-npu npu <config>` with the `run` directive).

Build:  see ../../build_design.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

import aie.iron as iron
from aie.iron import CompileTime, In, ObjectFifo, Out, Program, Runtime, Worker
from aie.iron.kernel import ExternalFunction

HERE = Path(__file__).parent

# Must match the template instantiation in rot13.cc. IRON's declared arg_types
# are cosmetic at the kernel boundary, so a mismatch compiles clean and hangs.
TILE = 1024


def _include_dirs() -> list[str]:
    from aie.iron.kernels._common import _detect_arch, _include_dirs as base
    from aie.utils import config

    inc = base()
    root = Path(config.cxx_header_path()) / "aie_kernels"
    inc.append(str(root))
    inc.append(str(root / _detect_arch()))
    return inc


@iron.jit(aiecc_flags=["--alloc-scheme=basic-sequential"])
def rot13(text_in: In, text_out: Out, *, tile: CompileTime[int] = TILE):
    tile_ty = np.ndarray[(tile,), np.dtype[np.int8]]

    kernel = ExternalFunction(
        f"rot13_{tile}",
        source_file=str(HERE / "rot13.cc"),
        arg_types=[tile_ty, tile_ty],
        include_dirs=_include_dirs(),
    )

    of_in = ObjectFifo(tile_ty, name="txt_in", depth=2)
    of_out = ObjectFifo(tile_ty, name="txt_out", depth=2)

    def core_body(rx, tx, fn):
        a = rx.acquire(1)
        b = tx.acquire(1)
        fn(a, b)
        rx.release(1)
        tx.release(1)

    worker = Worker(
        core_body, fn_args=[of_in.cons(), of_out.prod(), kernel], stack_size=0xD00
    )

    def sequence(src, dst, rx_prod, tx_cons):
        rx_prod.fill(src)
        tx_cons.drain(dst, wait=True)

    rt = Runtime(sequence, [tile_ty, tile_ty, of_in.prod(), of_out.cons()])
    return Program(iron.get_current_device(), rt, workers=[worker]).resolve_program()


DESIGN = rot13
SPECIALIZE = {"tile": TILE}
