r"""npu_host: run_kernel's model from Python (pyxrt), for multi-kernel chain tests.

    npu = Npu()
    fa = npu.kernel_set("fa", <set dir>)            # final.xclbin -> one hardware context
    st = fa.stream("r512_attn_dbl")                  # insts_<name>.bin -> instruction BO
    x = npu.buf("X", nbytes); x.write(arr, row=0)    # host-only BO; bf16/uint16 numpy in/out
    st.run(x.view(offset, nbytes), ...)              # opcode 3, blocking wait (it sleeps)
    h = st.start(...); ...; h.wait()                 # queued: same-context runs go in order

Buffers are XRT host-only BOs; `view` makes a sub-buffer of the ROOT allocation (XRT
does not compose sub-buffers of sub-buffers: the host pointer and the device address
would pick up different offsets). Needs the IRON env (`C:\dev\mlir-aie\iron_env.ps1`),
which puts C:\Xilinx\XRT\python (pyxrt) on PYTHONPATH.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pyxrt

TO_DEV = pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE
FROM_DEV = pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE
COMPLETED = pyxrt.ert_cmd_state.ERT_CMD_STATE_COMPLETED


class Buf:
    def __init__(self, npu: "Npu", name: str, nbytes: int):
        self.name, self.nbytes = name, nbytes
        self.bo = pyxrt.bo(npu.dev, nbytes, pyxrt.bo.host_only, npu.group_kernel.group_id(3))
        self.host = np.frombuffer(self.bo.map(), dtype=np.uint8)
        self._views = {}

    def view(self, offset: int = 0, nbytes: int | None = None):
        nbytes = self.nbytes - offset if nbytes is None else nbytes
        if offset == 0 and nbytes == self.nbytes:
            return self.bo
        assert offset + nbytes <= self.nbytes, (self.name, offset, nbytes, self.nbytes)
        key = (offset, nbytes)
        if key not in self._views:
            self._views[key] = pyxrt.bo(self.bo, nbytes, offset)
        return self._views[key]

    def write(self, arr: np.ndarray, offset: int = 0) -> None:
        b = np.ascontiguousarray(arr).view(np.uint8).reshape(-1)
        self.host[offset:offset + b.size] = b
        self.bo.sync(TO_DEV, b.size, offset)

    def zero(self) -> None:
        self.host[:] = 0
        self.bo.sync(TO_DEV, self.nbytes, 0)

    def load(self, path, offset: int = 0) -> None:
        """Read a file straight into the mapped BO (no intermediate copy)."""
        n = Path(path).stat().st_size
        assert offset + n <= self.nbytes, (self.name, path, n, self.nbytes)
        with open(path, "rb") as f:
            f.readinto(memoryview(self.host)[offset:offset + n])
        self.bo.sync(TO_DEV, n, offset)

    def read(self, dtype=np.uint16, offset: int = 0, count: int | None = None) -> np.ndarray:
        itemsize = np.dtype(dtype).itemsize
        count = (self.nbytes - offset) // itemsize if count is None else count
        self.bo.sync(FROM_DEV, count * itemsize, offset)
        return self.host[offset:offset + count * itemsize].view(dtype).copy()


class Stream:
    def __init__(self, kset: "KernelSet", name: str, path: Path):
        self.kset, self.name = kset, name
        self.words = np.fromfile(path, dtype=np.uint32)
        self.nwords = int(self.words.size)
        self.instr = pyxrt.bo(kset.npu.dev, self.words.nbytes, pyxrt.bo.cacheable,
                              kset.kernel.group_id(1))
        self._upload()

    def _upload(self) -> None:
        self.instr.write(self.words.tobytes(), 0)
        self.instr.sync(TO_DEV, self.words.nbytes, 0)

    def patch(self, words: list[int], value: int) -> None:
        """Set instruction words (e.g. an RTP value) -- only while no run of it is queued."""
        self.words[words] = value
        self._upload()

    def start(self, *bos):
        """Queue a run without waiting (runs queued on one hardware context execute in
        order; wait on the last before using another context -- queued across contexts,
        they hang the array)."""
        return self.kset.kernel(3, self.instr, self.nwords, *bos)

    def run(self, *bos) -> float:
        t0 = time.perf_counter()
        st = self.start(*bos).wait()
        ms = (time.perf_counter() - t0) * 1e3
        if st != COMPLETED:
            raise RuntimeError(f"{self.kset.name}/{self.name}: {st} after {ms:.1f} ms")
        self.kset.npu.log.append((self.kset.name, self.name, ms))
        return ms


class KernelSet:
    def __init__(self, npu: "Npu", name: str, directory: Path):
        self.npu, self.name, self.dir = npu, name, Path(directory)
        xb = pyxrt.xclbin(str(self.dir / "final.xclbin"))
        npu.dev.register_xclbin(xb)
        self.ctx = pyxrt.hw_context(npu.dev, xb.get_uuid())
        self.kernel = pyxrt.kernel(self.ctx, "MLIR_AIE")
        self._streams = {}

    def stream(self, name: str) -> Stream:
        if name not in self._streams:
            self._streams[name] = Stream(self, name, self.dir / f"insts_{name}.bin")
        return self._streams[name]


class Npu:
    def __init__(self):
        self.dev = pyxrt.device(0)
        self.sets: dict[str, KernelSet] = {}
        self.log: list[tuple[str, str, float]] = []

    @property
    def group_kernel(self):
        return next(iter(self.sets.values())).kernel

    def kernel_set(self, name: str, directory) -> KernelSet:
        self.sets[name] = KernelSet(self, name, directory)
        return self.sets[name]

    def buf(self, name: str, nbytes: int) -> Buf:
        return Buf(self, name, nbytes)
