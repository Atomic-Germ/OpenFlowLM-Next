r"""Pack the 35B's weights at load time, straight into the harness's XRT buffers -- no pool files.

The lax decode used to read ~21 GB of pre-packed files (make_decode.py's pool_L*.bin,
pool_lmhead.bin, consts_*.bin, the raw embedding table), which lived in /tmp and did not survive
a reboot. Here the driver packs every buffer from the model's own `model.q4nx` with the same
recipes/pack.py code make_decode.py uses (bit-identical), and hands the bytes to the harness over
a pipe: the program says `buf pool7 <size>` then `fdload pool7 <fd>`, and the harness read()s the
pipe straight into the BO's host mapping. Nothing is written to disk or /dev/shm.

Packing runs in `workers` forked processes; each task (a layer's pool + consts, the lm_head pool,
the embedding table, ...) is packed by whichever worker took it, which then waits for its turn
(the pipe carries the items in program order) and write()s the bytes into the pipe itself -- no
pickling of 512 MB arrays back to a parent. At most `workers` tasks are in flight, so the host
RAM beyond the BOs themselves is ~workers x 0.7 GB. The container is read with one pread per
tensor rather than through its mmap, which is what keeps a cold page cache (after a reboot)
near the disk's speed.

Only the lax decode's buffers are packed here (lax_decode_cfg.py's GLOBALS subset, one pool +
consts per layer, the embedding table); the zero / zeroed-state buffers need no bytes at all
(`buf` zeroes), and the fp64 reference stays make_decode.py's job.
"""
from __future__ import annotations

import fcntl
import multiprocessing as mp
import os
import queue
import re
import subprocess
import threading
from pathlib import Path


HERE = Path(__file__).resolve().parent
HARNESS = HERE.parent / "harness" / "build" / "run_kernel"
DEFAULT_MODEL_DIR = Path.home() / ".config" / "flm" / "models" / "Qwen3.6-35B-A3B-NPU2"
F_SETPIPE_SZ = 1031


class Model:
    """make_decode.py --requant's spec / manifest / plan / container, without the reference."""

    def __init__(self, model_dir: Path, max_ctx: int = 4096):
        import sys
        # --requant: every q8 projection becomes q4_1 in the pool -- the lax kernels are built that
        # way. Read when the spec is derived, so set it before recipes.load runs.
        os.environ["OPEN_KERNELS_FORCE_Q4_1"] = "1"
        for p in (str(HERE.parent), str(HERE)):
            if p not in sys.path:
                sys.path.insert(0, p)
        from q4nx import Q4NX
        from recipes.load import spec_from_model_dir
        from recipes.manifest import manifest
        md = Path(model_dir)
        self.spec = spec_from_model_dir(md)
        self.m = manifest(self.spec, max_ctx)
        self.max_ctx = max_ctx
        m = self.m
        self.plan = {"pool_bytes": m["pack"]["pool_bytes"], "chunk_bytes": m["pack"]["chunk_bytes"],
                     "layer_types": {lt: d["pack"] for lt, d in m["layer_types"].items()},
                     "lm_head": m["pack"]["lm_head"], "embed": m["pack"]["embed"], "norm": m["pack"]["norm"]}
        self.q = Q4NX(md / "model.q4nx")
        q8_pat = [re.compile(re.escape(op["tensor"]).replace(r"\{l\}", r"\d+"))
                  for d in self.plan["layer_types"].values() for op in d["pool"] + d["consts"]
                  if op.get("op") == "q8_perm"]
        self.q.native_q8 = lambda n: any(p.fullmatch(n) for p in q8_pat)
        self.q.hidden = self.spec.hidden
        self.types = list(self.spec.layer_types)
        self.q.raw = self._pread_raw

    def kinds(self) -> str:
        """'l' (linear attention, lax_l) / 'f' (full attention, lax_a) per layer."""
        return "".join("f" if lt == "full_attention" else "l" for lt in self.types)

    def head_lines(self, designs: Path) -> list[str]:
        """xclbin / kernelx lines of the final norm (ln) and lm_head (lm) kernels."""
        c = []
        for kn in ("ln", "lm"):
            kd = self.m["kernels"][kn]
            bdir = Path(designs) / self.m["builds"][kd["build"]]["build_dir"]
            if not (bdir / "final.xclbin").is_file():
                raise FileNotFoundError(f"{bdir}/final.xclbin: no {kn} build under --designs {designs}")
            c += [f"xclbin {kd['context']} {bdir}/final.xclbin", f"kernelx {kn} {kd['context']} {bdir}/insts.bin"]
        return c

    def size(self, name: str) -> int:
        g = self.m["globals"][name]
        return g["per_row"] * self.max_ctx if isinstance(g, dict) else g

    def consts_bytes(self, l: int) -> int:
        return self.m["layer_types"][self.types[l]]["buffers"]["consts"]

    def embed_range(self) -> tuple[int, int]:
        t = self.q.tensors["model.embed_tokens.weight"]
        assert t["dtype"] == "BF16", t
        o0, o1 = t["data_offsets"]
        return self.q.data_base + o0, self.q.data_base + o1

    def _pread_raw(self, name):
        """Q4NX.raw by pread instead of slicing the mmap: on a cold page cache (after a reboot) a
        slice faults the mapping in with the small mmap readahead, and eight packers doing that at
        once ran at a fraction of the disk's speed; one large pread per tensor streams it."""
        t = self.q.tensors[name]
        a, b = (self.q.data_base + o for o in t["data_offsets"])
        fd = self.q.f.fileno()
        parts = []
        while a < b:
            d = os.pread(fd, min(b - a, 1 << 30), a)
            if not d:
                raise EOFError(f"{self.q.path}: short read of {name}")
            parts.append(d)
            a += len(d)
        return parts[0] if len(parts) == 1 else b"".join(parts)

    def build(self, key: tuple):
        """The bytes of one streamed item (a numpy array or a memoryview)."""
        from recipes import pack as PK
        from q4nx import f32_to_bf16
        kind = key[0]
        if kind == "pool":
            return PK.build_layer_pool(self.plan, self.types[key[1]], self.q, key[1])
        if kind == "consts":
            return PK.build_consts(self.plan, self.types[key[1]], self.q, key[1], self.consts_bytes(key[1]))
        if kind == "lmpool":
            return PK.build_lmhead_pool(self.plan, self.q)
        if kind == "normw":
            return f32_to_bf16(self.q.bf16(self.plan["norm"]["tensor"]))
        if kind == "ptab":
            s, g = self.spec, self.m["globals"]["ptab"]
            return PK.ptab(self.max_ctx, s.rotary_dim, s.rope_theta, g["per_row"], g.get("inv_freq", s.rope_inv_freq()),
                           g.get("window", 0), g.get("scale", 1.0), g.get("long_inv_freq"), g.get("switch_row"))
        if kind == "embed":
            return self.q.raw("model.embed_tokens.weight")
        if kind == "bytes":
            return key[1]
        raise ValueError(f"unknown item {key!r}")


# ---- streaming: forked workers pack items and write them into the pipe in program order

_G: dict = {}


def _write_all(fd: int, data) -> int:
    mv = memoryview(data).cast("B") if not isinstance(data, (bytes, bytearray)) else memoryview(data)
    n = len(mv)
    while mv:
        mv = mv[os.write(fd, mv):]
    return n


def _work(t: int) -> int:
    """Task t: pack its items, wait for its turn, write them, pass the turn on."""
    g = _G
    datas = []
    for i in g["tasks"][t]:
        key, size = g["items"][i]
        data = g["model"].build(key)
        n = memoryview(data).nbytes
        if n != size:
            raise RuntimeError(f"item {i} {key[:2]}: packed {n} B, the program expects {size}")
        datas.append(data)
    with g["cond"]:
        g["cond"].wait_for(lambda: g["turn"].value == t)
    n = sum(_write_all(g["wfd"], d) for d in datas)
    with g["cond"]:
        g["turn"].value = t + 1
        g["cond"].notify_all()
    return n


def _tasks(items, big: int = 64 << 20) -> list[list[int]]:
    """Items -> tasks: every big item starts one and the small ones after it (a layer's consts
    after its pool) ride along, so a worker never sits on its turn holding only a small item."""
    tasks: list[list[int]] = []
    for i, (_, size) in enumerate(items):
        if size >= big or not tasks:
            tasks.append([i])
        else:
            tasks[-1].append(i)
    return tasks


class Stream:
    """A pipe the harness reads with `fdload <buf> <fd>`, fed by `workers` packing processes.

    The workers are forked here, before the harness starts, so they hold none of its stdin /
    stdout pipes and the harness inherits only the read end (`pass_fds=(stream.rfd,)`)."""

    def __init__(self, model: Model, items: list[tuple[tuple, int]], workers: int = 8):
        self.rfd, wfd = os.pipe()
        try:
            fcntl.fcntl(wfd, F_SETPIPE_SZ, 1 << 20)
        except OSError:
            pass
        ctx = mp.get_context("fork")
        _G.update(model=model, wfd=wfd, turn=ctx.Value("i", 0), cond=ctx.Condition(), items=items,
                  tasks=_tasks(items))
        # a worker must not hold the read end: were the harness to die, its writes would then
        # block on a full pipe instead of failing with EPIPE
        self.pool = ctx.Pool(workers, initializer=os.close, initargs=(self.rfd,))
        os.close(wfd)                       # only the workers write; EOF once they are gone
        self.n = len(_G["tasks"])
        self.res = None

    def start(self, on_error=None):
        """Call once the harness holds the read end: stream every item, in order."""
        os.close(self.rfd)
        self.res = self.pool.map_async(_work, range(self.n), chunksize=1, error_callback=on_error)

    def wait(self) -> int:
        n = sum(self.res.get())
        self.pool.close()
        self.pool.join()
        return n


class Npu:
    """A `run_kernel -` session: write program lines, read what they print. With a Stream, the
    harness inherits its pipe and the program's `fdload` lines read the packed weights from it."""

    def __init__(self, lines: list[str], stream: Stream | None = None):
        fds = (stream.rfd,) if stream else ()
        self.p = subprocess.Popen([str(HARNESS), "-"], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  text=True, bufsize=1, pass_fds=fds)
        # drain stdout on a thread: a long prompt is written before anything is read back, and
        # a full stdout pipe would otherwise block the harness while we block on its stdin
        self.q: queue.Queue = queue.Queue()
        self.out: list[str] = []
        threading.Thread(target=self._read, daemon=True).start()
        if stream:
            stream.start(on_error=lambda e: self.p.kill())
        self.send(lines + ["tick"])
        self.wait("tick")
        if stream:
            stream.wait()

    def send(self, lines):
        self.p.stdin.write("\n".join(lines) + "\n")
        self.p.stdin.flush()

    def _read(self):
        for line in self.p.stdout:
            self.q.put(line)
        self.q.put(None)

    def wait(self, prefix: str) -> str:
        """The next output line starting with `prefix` (an ERROR line raises)."""
        while (line := self.q.get()) is not None:
            self.out.append(line)
            if line.startswith(prefix):
                return line
            if line.startswith(("ERROR", "RUN FAILED")):
                raise RuntimeError(line.strip() + "".join(self._rest()))
        raise RuntimeError(f"harness exited (code {self.p.wait()})")

    def _rest(self):
        try:
            while (line := self.q.get(timeout=0.5)) is not None:
                yield line
        except queue.Empty:
            return

    def close(self):
        self.p.stdin.close()
        return self.p.wait()
