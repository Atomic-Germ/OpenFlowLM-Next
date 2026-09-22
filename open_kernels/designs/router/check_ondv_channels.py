#!/usr/bin/env python3
"""Re-derive the per-column MM2S queue addresses from a built design and check them
against designs/router/ondv_ctrl.h's kOndvQueue.

The on-device router addresses each column's w-channel task queue by its shim register
(MM2S ch0 -> 0x1D214, ch1 -> 0x1D21C). Which channel each @w{c} lands on is a property
of the design's shim budget, not of the model, so it is pinned in the kernel header and
checked here: if the design's fifo set changes, this fails before the packets push the
wrong column's queue.

    python3 designs/router/check_ondv_channels.py designs/layer_x/build_ondv_lx1

Exits non-zero on a mismatch.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

QUEUE = {0: 0x1D214, 1: 0x1D21C}


def parse(path: Path) -> dict[int, int]:
    """@w{c}_shim_alloc(%shim_noc_tile_{c}_0, MM2S, {ch}) -> {c: queue register}."""
    text = path.read_text()
    out = {}
    pat = re.compile(r"aie\.shim_dma_allocation\s+@w(\d+)_shim_alloc\([^,]+,\s*MM2S,\s*(\d+)\)")
    for m in pat.finditer(text):
        out[int(m.group(1))] = QUEUE[int(m.group(2))]
    return out


def header_queue(header: Path) -> list[int]:
    """kOndvQueue in ondv_ctrl.h, as a list indexed by column."""
    text = header.read_text()
    m = re.search(r"kOndvQueue\[kOndvCores\]\s*=\s*\{([^}]*)\}", text)
    if not m:
        raise SystemExit("check_ondv_channels: kOndvQueue not found in the header")
    return [int(re.sub(r"[uUlL]+$", "", v.strip()), 0) for v in m.group(1).split(",") if v.strip()]


def main() -> int:
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    build = Path(sys.argv[1])
    mlir = next((p for p in (build / "final.prj" / "input_with_addresses.mlir",
                             build / "final.prj" / "aie.mlir") if p.exists()), None)
    if mlir is None:
        raise SystemExit(f"check_ondv_channels: no mlir under {build}/final.prj")
    got = parse(mlir)
    header = Path(__file__).with_name("ondv_ctrl.h")
    want = header_queue(header)
    if len(got) != len(want):
        print(f"check_ondv_channels: {len(got)} w-allocations found, {len(want)} expected "
              f"(from {header})")
        for c in range(len(want)):
            print(f"  c{c}: design={got.get(c, '?'):#x} header={want[c]:#x}")
        return 1
    bad = [(c, got[c], want[c]) for c in sorted(got) if got[c] != want[c]]
    if bad:
        print(f"check_ondv_channels: MISMATCH in {mlir}")
        for c, g, w in bad:
            print(f"  c{c}: design={g:#x} header={w:#x}")
        return 1
    print(f"check_ondv_channels: OK -- kOndvQueue matches {mlir.name} "
          f"({', '.join(f'{c}:{got[c]:#x}' for c in sorted(got))})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
