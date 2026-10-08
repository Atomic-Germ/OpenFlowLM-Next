r"""Emit an IRON design's MLIR (no compile), optionally with ObjectFifos lowered, to inspect
channel, BD and lock use before aiecc's later passes fail on them.

    python emit_mlir.py designs/dit_conv/dit_conv.py out.mlir [--lower]

Same conventions as build_design.py (DESIGN + SPECIALIZE, env for the spec). --lower runs
aie-opt's place-tiles and ObjectFifo transform, then prints each memtile's DMA channels
with their BD counts and the even/odd BD pools (aie2p memtiles give channels 0/2/4 BDs
0-23 and channels 1/3/5 BDs 24-47).
"""

from __future__ import annotations

import importlib.util
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

import aie.iron as iron
from aie.iron.device import from_name


def main() -> int:
    src, out = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
    iron.set_current_device(from_name("npu2", n_cols=None))
    spec = importlib.util.spec_from_file_location(src.stem, src)
    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(src.parent))
    spec.loader.exec_module(mod)
    out.write_text(str(mod.DESIGN.specialize(**mod.SPECIALIZE).compilable.generate_mlir()))
    print(f"wrote {out}")
    if "--lower" not in sys.argv:
        return 0
    low = out.with_suffix(".lowered.mlir")
    subprocess.run(["aie-opt", "--aie-place-tiles", "--aie-objectFifo-stateful-transform",
                    str(out), "-o", str(low)], check=True)
    text = low.read_text()
    for m in re.finditer(r"aie\.memtile_dma\((%[\w]+)\)", text):
        body = text[m.end():text.find("aie.end\n", m.end())]
        cnt, cur = Counter(), None
        for line in body.splitlines():
            d = re.search(r"aie\.dma_start\((\w+), (\d+)", line)
            if d:
                cur = (d.group(1), int(d.group(2)))
            elif "aie.dma_bd(" in line and cur:
                cnt[cur] += 1
        even = sum(v for (_, ch), v in cnt.items() if ch % 2 == 0)
        odd = sum(v for (_, ch), v in cnt.items() if ch % 2)
        chans = ", ".join(f"{d[:4]}{c}:{v}" for (d, c), v in sorted(cnt.items()))
        print(f"{m.group(1)}: {chans}  | even {even}/24, odd {odd}/24")
    print(f"wrote {low}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
