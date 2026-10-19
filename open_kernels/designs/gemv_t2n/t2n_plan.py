"""gemv_t2n's work plan: which core streams which bands over which K slice (pure Python, no IRON).

The design (gemv_t2n.py) turns a plan into its instruction stream and the recipe (recipes/dflash2.py)
writes the same plan into manifest.json, where the engine reads it to build the tables and add up y.
"""
from __future__ import annotations

TOKENS = 8            # rows per dispatch: two token quads, the mmul's four rows each
CHUNK = 2176          # one unpadded t2 chunk: 32 rows x 256 K
BAND_ROWS = 64
REC = 1216            # one (k-tile, 128-group, quad) table record: x 1024 | row sums 128 | shift 64
HDR = 64              # the table element's header: i32 nb, i32 nkt
Y_BAND = 512          # floats per band: [half 2][quad 2][field 4][tok 4][p 8]


def tab_bytes(kt_max: int) -> int:
    raw = HDR + kt_max * 4 * REC
    return (raw + 2047) // 2048 * 2048          # whole 2 KB runs: the fill's inner DMA dimension


def slices(nkt: int, kt_max: int) -> list[tuple[int, int]]:
    """K split into the fewest slices of at most kt_max k-tiles, sizes within one of each other."""
    ns = -(-nkt // kt_max)
    q, r = divmod(nkt, ns)
    out, kt0 = [], 0
    for s in range(ns):
        n = q + (1 if s < r else 0)
        out.append((kt0, n))
        kt0 += n
    return out


def plan(n: int, k: int, n_cores: int = 8, kt_max: int = 8, nb_max: int = 1023) -> dict:
    """Items per core, each {slice, b0, nb, kt0, nkt, tab, y}: tab is the item's table element index,
    y its first float in the output. Work (bands x k-tiles) is split evenly over the cores in
    slice-major order, so a core holds at most a few items."""
    if n % BAND_ROWS or k % 256:
        raise ValueError(f"gemv_t2n: N {n} must be whole 64-row bands and K {k} whole 256-K tiles")
    nb_all, nkt_all = n // BAND_ROWS, k // 256
    sl = slices(nkt_all, kt_max)
    total = nb_all * nkt_all
    bounds = [round(total * (c + 1) / n_cores) for c in range(n_cores)]
    owner = []                                   # per (slice, band) unit, in order: the core
    done, c = 0, 0
    for s, (_, nkt) in enumerate(sl):
        for b in range(nb_all):
            while c < n_cores - 1 and done >= bounds[c]:
                c += 1
            owner.append((s, b, c))
            done += nkt
    items: list[list[dict]] = [[] for _ in range(n_cores)]
    for s, b, c in owner:
        cur = items[c][-1] if items[c] else None
        if cur and cur["slice"] == s and cur["b0"] + cur["nb"] == b and cur["nb"] < nb_max:
            cur["nb"] += 1
        else:
            kt0, nkt = sl[s]
            items[c].append({"slice": s, "b0": b, "nb": 1, "kt0": kt0, "nkt": nkt})
    tab, y = 0, 0
    for r in range(max(len(i) for i in items)):
        for c in range(n_cores):
            if r < len(items[c]):
                it = items[c][r]
                it["tab"], it["y"] = tab, y
                tab += 1
                y += it["nb"] * Y_BAND
    return {"n": n, "k": k, "tokens": TOKENS, "kt_max": kt_max, "tab_bytes": tab_bytes(kt_max),
            "n_tabs": tab, "y_floats": y, "slices": sl, "items": items}
