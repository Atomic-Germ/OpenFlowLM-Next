"""Program memory per core, and the symbols that fill it, for an IRON build directory.

    python utilities/core_sizes.py <build_dir> [--tiles 2_3,3_3,4_3,5_3] [--symbols]

An AIE core has 16,384 bytes of program memory and the build fails late -- inside
`_XAie_LoadProgMemSection` -- when a core's `.text` passes it, so the number to watch is
the `.text` of `<build>/final.prj/elfs_main_core_<col>_<row>`, one per core.

`--symbols` adds the per-symbol breakdown from the core ELF's symbol table. Size by
SYMBOL, never by object file: the weak `attn_row_impl` and `vexpN<N>` bodies appear in
every object that uses them and are deduplicated at link, so an object's `.text` double
counts them and has already produced one wrong prediction (attn-block-rb2.md).
"""
from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path

PROG_MEM = 16384


def sections(elf: bytes) -> dict[str, int]:
    """{name: size} of an ELF32 little-endian file's sections."""
    if elf[:4] != b"\x7fELF" or elf[4] != 1:
        raise ValueError("not a 32-bit ELF")
    e_shoff, = struct.unpack_from("<I", elf, 0x20)
    e_shentsize, e_shnum, e_shstrndx = struct.unpack_from("<HHH", elf, 0x2E)
    def sh(i):
        o = e_shoff + i * e_shentsize
        name, _typ, _flags, _addr, off, size, link, _info, _al, entsize = struct.unpack_from("<10I", elf, o)
        return name, off, size, link, entsize
    _, stroff, _, _, _ = sh(e_shstrndx)
    out = {}
    for i in range(e_shnum):
        name, _off, size, _link, _es = sh(i)
        end = elf.index(b"\0", stroff + name)
        out[elf[stroff + name:end].decode()] = size
    return out


def symbols(elf: bytes) -> list[tuple[str, int]]:
    """[(name, size)] of the sized symbols, largest first."""
    e_shoff, = struct.unpack_from("<I", elf, 0x20)
    e_shentsize, e_shnum, e_shstrndx = struct.unpack_from("<HHH", elf, 0x2E)
    hdrs = []
    for i in range(e_shnum):
        o = e_shoff + i * e_shentsize
        hdrs.append(struct.unpack_from("<10I", elf, o))
    out: list[tuple[str, int]] = []
    for name, typ, _fl, _ad, off, size, link, _info, _al, entsize in hdrs:
        if typ != 2 or not entsize:           # SHT_SYMTAB
            continue
        stroff = hdrs[link][4]
        for i in range(size // entsize):
            o = off + i * entsize
            st_name, _val, st_size, _info2, _other, _shndx = struct.unpack_from("<IIIBBH", elf, o)
            if not st_size:
                continue
            end = elf.index(b"\0", stroff + st_name)
            out.append((elf[stroff + st_name:end].decode(), st_size))
    out.sort(key=lambda t: -t[1])
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("build")
    ap.add_argument("--tiles", default=None, help="comma-separated col_row (default: every core found)")
    ap.add_argument("--symbols", action="store_true")
    ap.add_argument("--top", type=int, default=25)
    a = ap.parse_args()

    prj = Path(a.build) / "final.prj"
    if not prj.is_dir():
        prj = Path(a.build)
    want = a.tiles.split(",") if a.tiles else None
    found = sorted(prj.glob("elfs_main_core_*/*.elf")) or sorted(prj.glob("elfs_main_core_*.elf"))
    if not found:
        print(f"no core ELFs under {prj}")
        return 1
    for f in found:
        tile = f.stem[len("elfs_main_core_"):]
        if want and tile not in want:
            continue
        b = f.read_bytes()
        text = sections(b).get(".text", 0)
        print(f"core {tile}: .text {text:>6} / {PROG_MEM}   {PROG_MEM - text:>5} free")
        if a.symbols:
            for name, size in symbols(b)[:a.top]:
                print(f"    {size:>6}  {name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
