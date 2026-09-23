#!/usr/bin/env python3
r"""Check that the two `lax` control texts really do share one xclbin, and that the
merged design still fits the array.

    python3 designs/layer_x/check_lax_merge.py build_lax_l build_lax_a

Asserts, from the builds alone (no device):

  1. `final.xclbin` has the same size for both kinds and differs only in the
     UUID/timestamp metadata (its per-tile ELFs are byte-identical) -- so one
     `hw_context` created from either xclbin can host both `insts.elf` streams.
  2. Every per-tile program is byte-identical between the two kinds.
  3. The unified main program is exactly the standalone `lx` main program's size
     (16272 B at the 35B), i.e. the second layer type cost the core nothing.
  4. The shim DMA budget: <= 2 MM2S and <= 2 S2MM per `ShimNOCTile`, <= 16 of each
     across the 8 columns.
  5. The `w{c}` MM2S channels are identical for both kinds (one queue map serves both).

Exits non-zero if any of these fails.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

QUEUE = {0: 0x1D214, 1: 0x1D21C}
LX_MAIN_BYTES = 16272  # the standalone lx main core (build_emitter), for the size check


def _elf_size(elf: Path) -> int:
    """The first PT_LOAD's file size -- the core program's size."""
    import struct
    data = elf.read_bytes()
    assert data[:4] == b"\x7fELF", elf
    e_phoff, = struct.unpack_from("<I", data, 28)
    e_phentsize, e_phnum = struct.unpack_from("<HH", data, 42)
    for i in range(e_phnum):
        off = e_phoff + i * e_phentsize
        p_type, = struct.unpack_from("<I", data, off)
        if p_type == 1:  # PT_LOAD
            p_filesz, = struct.unpack_from("<I", data, off + 16)
            return p_filesz
    raise AssertionError(f"no PT_LOAD in {elf}")


def _shim_allocs(build: Path) -> tuple[dict[int, int], dict[str, int], dict[str, int]]:
    mlir = next(p for p in (build / "final.prj" / "input_with_addresses.mlir",
                            build / "final.prj" / "aie.mlir") if p.exists())
    text = mlir.read_text()
    wch = {int(m.group(1)): int(m.group(2)) for m in re.finditer(
        r"aie\.shim_dma_allocation\s+@w(\d+)_shim_alloc\([^,]+,\s*MM2S,\s*(\d+)\)", text)}
    per_shim: dict[str, int] = {}
    counts: dict[str, int] = {"MM2S": 0, "S2MM": 0}
    for tile, dirn in re.findall(r"aie\.shim_dma_allocation\s+@\w+\((%\w+),\s*(MM2S|S2MM),", text):
        per_shim[tile + " " + dirn] = per_shim.get(tile + " " + dirn, 0) + 1
        counts[dirn] += 1
    return wch, per_shim, counts


def main() -> int:
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    a, b = (Path(p) for p in sys.argv[1:3])
    problems: list[str] = []

    # 1. the xclbins: same size, tiny metadata-only difference
    xa, xb = a / "final.xclbin", b / "final.xclbin"
    da, db = xa.read_bytes(), xb.read_bytes()
    if len(da) != len(db):
        problems.append(f"xclbin sizes differ: {len(da)} vs {len(db)}")
    else:
        diff = sum(1 for x, y in zip(da, db) if x != y)
        print(f"xclbin: {len(da)} B, {diff} differing bytes (UUID/timestamp metadata)")
        if diff > 400:
            problems.append(f"{diff} differing xclbin bytes -- more than metadata")

    # 2. per-tile programs byte-identical
    pa, pb = a / "final.prj", b / "final.prj"
    na = sorted(p.relative_to(pa) for p in pa.glob("elfs_*/*.elf"))
    nb = sorted(p.relative_to(pb) for p in pb.glob("elfs_*/*.elf"))
    if na != nb:
        problems.append("different core sets between kinds")
    for rel in na:
        if (pa / rel).read_bytes() != (pb / rel).read_bytes():
            problems.append(f"core program differs between kinds: {rel}")
    print(f"core programs: {len(na)} tiles, byte-identical across kinds")

    # 3. the unified main program is the standalone lx size
    main = pa / "elfs_main_core_0_2" / "elfs_main_core_0_2.elf"
    size = _elf_size(main)
    print(f"main program: {size} B (standalone lx {LX_MAIN_BYTES} B)")
    if size != LX_MAIN_BYTES:
        problems.append(f"main program {size} != lx's {LX_MAIN_BYTES}")

    # 4./5. shim budget and the queue map
    wa, per_a, ca = _shim_allocs(a)
    wb, per_b, cb = _shim_allocs(b)
    print(f"shim DMA: kind=l {ca}, kind=a {cb}")
    for name, per, c in (("l", per_a, ca), ("a", per_b, cb)):
        if c["MM2S"] > 16 or c["S2MM"] > 16:
            problems.append(f"kind={name}: more than 16 shim DMA channels: {c}")
        if per and max(per.values()) > 2:
            over = {k: v for k, v in per.items() if v > 2}
            problems.append(f"kind={name}: more than 2 DMA channels on a shim: {over}")
    print(f"w channels: kind=l {[wa.get(c) for c in range(8)]}, kind=a {[wb.get(c) for c in range(8)]}")
    if wa != wb:
        problems.append("the w{c} channels differ between kinds -- one queue map cannot serve both")
    qmap = [QUEUE[wa[c]] for c in range(8)] if len(wa) == 8 else []
    print("queue map:", " ".join(f"{v:#x}" for v in qmap))

    if problems:
        print("\nFAIL")
        for p in problems:
            print("  -", p)
        return 1
    print("\nOK -- one xclbin, both control texts, budget intact")
    return 0


if __name__ == "__main__":
    sys.exit(main())
