r"""Compare per-layer residual-stream dumps: which layer diverges, not just whether the output does.

Both sides are produced by tools in this tree:

  utilities/ref-dump/dump_ref --model <gguf> --ids <csv> --out <dir>/ref
      -> <dir>/ref_l<layer>.bin   f32[tokens][hidden]  (llama.cpp's l_out-<il>)

  OFLM_OPEN_LAYER_NAN=1 OFLM_OPEN_LAYER_DUMP=<dir>/npu \
  open_qwen36_cli --model <dir> --kernels <dir>/open_kernels --ids <csv> \
      --max-tokens 1 --prefill-logits
      -> <dir>/npu_<seq>_l<layer>.bin   f32[hidden], seq = position * layers + layer

  python utilities/quant-compare/cmp_layers.py <dir>/ref <dir>/npu [--per-position]

Prints, for every layer, the correlation and relative L2 error between the
engine's residual stream after that layer and llama.cpp's, over the positions
present in both. The first layer whose correlation collapses is the one to
debug; every layer after it inherits the error.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np


def ref_meta(prefix: Path) -> tuple[int, int, int]:
    meta = prefix.parent / (prefix.name + "_meta.txt")
    if not meta.exists():
        sys.exit(f"missing {meta}; regenerate with utilities/ref-dump/dump_ref")
    words = meta.read_text().split()
    kv = dict(zip(words[0::2], words[1::2]))
    return int(kv["tokens"]), int(kv["hidden"]), int(kv["layers"])


def npu_layer_dumps(prefix: Path, layers: int) -> dict[tuple[int, int], Path]:
    """(position, layer) -> file, from <prefix>_<seq>_l<layer>.bin, seq = pos*layers + layer."""
    pat = re.compile(re.escape(prefix.name) + r"_(\d+)_l(\d+)\.bin$")
    out: dict[tuple[int, int], Path] = {}
    for p in prefix.parent.iterdir():
        m = pat.match(p.name)
        if not m:
            continue
        seq, layer = int(m.group(1)), int(m.group(2))
        out[(seq // layers, layer)] = p
    return out


def corr(a: np.ndarray, b: np.ndarray) -> float:
    if a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def rel_l2(a: np.ndarray, b: np.ndarray) -> float:
    d = float(np.linalg.norm(a - b))
    n = float(np.linalg.norm(a))
    return d / n if n else float("nan")


def main() -> int:
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    ref_prefix, npu_prefix = Path(sys.argv[1]), Path(sys.argv[2])
    per_position = "--per-position" in sys.argv[3:]

    tokens, hidden, layers = ref_meta(ref_prefix)
    dumps = npu_layer_dumps(npu_prefix, layers)
    if not dumps:
        sys.exit(f"no layer dumps matching {npu_prefix}_<seq>_l<layer>.bin")
    positions = sorted({pos for pos, _ in dumps})
    missing = [l for l in range(layers) if not any((p, l) in dumps for p in positions)]
    if missing:
        print(f"warning: no engine dump for layers {missing[:8]}{'...' if len(missing) > 8 else ''}")

    print(f"{len(positions)} positions x {layers} layers (hidden {hidden})")
    print(f"{'layer':>5}  {'corr worst':>10}  {'corr med':>10}  {'relL2 worst':>11}  {'relL2 med':>10}")
    first_bad = None
    for layer in range(layers):
        cs, rs = [], []
        for pos in positions:
            f = dumps.get((pos, layer))
            if f is None:
                continue
            ref_file = ref_prefix.parent / f"{ref_prefix.name}_l{layer}.bin"
            ref = np.fromfile(ref_file, np.float32)
            if ref.size < (pos + 1) * hidden:
                continue
            r = ref[pos * hidden:(pos + 1) * hidden].astype(np.float64)
            g = np.fromfile(f, np.float32).astype(np.float64)
            if g.size != hidden:
                print(f"warning: layer {layer} pos {pos}: engine dump has {g.size} floats, expected {hidden}")
                continue
            cs.append(corr(r, g))
            rs.append(rel_l2(r, g))
        if not cs:
            print(f"{layer:>5}  (no comparable dumps)")
            continue
        cw, cm = np.nanmin(cs), np.nanmedian(cs)
        rw, rm = np.nanmax(rs), np.nanmedian(rs)
        print(f"{layer:>5}  {cw:>10.6f}  {cm:>10.6f}  {rw:>11.6f}  {rm:>10.6f}")
        if first_bad is None and (cm < 0.98 or rm > 0.20):
            first_bad = layer

    if first_bad is not None:
        print(f"\nfirst layer outside tolerance: {first_bad}")
        if per_position:
            layer = first_bad
            for pos in positions:
                f = dumps.get((pos, layer))
                if f is None:
                    continue
                ref = np.fromfile(ref_prefix.parent / f"{ref_prefix.name}_l{layer}.bin", np.float32)
                r = ref[pos * hidden:(pos + 1) * hidden].astype(np.float64)
                g = np.fromfile(f, np.float32).astype(np.float64)
                print(f"  pos {pos}: corr {corr(r, g):.6f}  relL2 {rel_l2(r, g):.6f}  "
                      f"|ref| {np.linalg.norm(r):.3f}  |got| {np.linalg.norm(g):.3f}")
    else:
        print("\nevery layer within tolerance")
    return 0 if first_bad is None else 1


if __name__ == "__main__":
    sys.exit(main())
