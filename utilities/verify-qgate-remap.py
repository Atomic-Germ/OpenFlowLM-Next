#!/usr/bin/env python3
"""Sanity-check the qgate remap indices in a Qwen3.5/3.6 MoE manifest.

For every full-attention layer's two q_proj pool ops (Q half + gate half), this
recomputes the file row-block each pool chunk reads under pools.cpp's remap and
checks that it is in bounds for the GGUF tensor's row count. An out-of-bounds
row block silently reads zeros (gguf_chunk breaks at `row >= rows`), which is
the failure mode that zeroes an entire projection half.

Usage:
    python utilities/verify-qgate-remap.py <model_dir> [<gguf_path>]

If <gguf_path> is omitted, the script looks for model.gguf beside the manifest.
"""
from __future__ import annotations
import json, sys
from pathlib import Path

try:
    import gguf
except ImportError:
    gguf = None


def tensor_rows(gf, name: str) -> int | None:
    if gf is None:
        return None
    for ti in gf.tensors:
        if ti.name == name:
            dims = list(ti.shape)
            return int(dims[1]) if len(dims) > 1 else int(dims[0])
    # fallback: any tensor ending in the requested suffix
    for ti in gf.tensors:
        if ti.name.endswith(name.split('.')[-2] + '.' + name.split('.')[-1]) or ti.name.endswith(name.rsplit('.',1)[-1]):
            # only accept if it's a q_proj weight
            if 'q_proj.weight' in ti.name:
                dims = list(ti.shape)
                return int(dims[1]) if len(dims) > 1 else int(dims[0])
    return None


def check_op(op: dict, total_rows: int | None) -> list[str]:
    msgs = []
    nch = op["nch"]; in_dim = op["in_dim"]; chunk0 = op.get("chunk0", 0)
    ncol = in_dim // 256
    hpb = (op.get("qgate_rows") or 0) // 32
    gate = bool(op.get("qgate_gate", False))
    rb0 = chunk0 // ncol if (hpb and ncol) else 0
    # std_perm is identity-ish for index bounds; bounds only depend on chunk0 + c
    max_rb = 0
    for c in range(nch):
        fc = chunk0 + c  # ignoring std_perm (monotonic-ish enough for bounds)
        if hpb:
            r = fc // ncol - rb0
            kt = fc % ncol
            fc = ((r // hpb) * 2 * hpb + (r % hpb) + (hpb if gate else 0)) * ncol + kt
        rb = fc // ncol
        max_rb = max(max_rb, rb)
        if total_rows is not None and 32 * rb >= total_rows:
            msgs.append(f"  OUT OF BOUNDS: pool chunk {c} -> file row-block {rb} (row {32*rb} >= {total_rows})")
    if total_rows is not None:
        msgs.append(f"  max file row-block {max_rb} (row {32*max_rb}) of {total_rows} rows  -> {'OK' if 32*(max_rb+1) <= total_rows else 'OVERFLOW'}")
    return msgs


def main() -> int:
    model_dir = Path(sys.argv[1])
    man_path = model_dir / "open_kernels" / "manifest.json"
    if not man_path.exists():
        # maybe pointed at an xclbins dir
        man_path = model_dir / "manifest.json"
    if not man_path.exists():
        sys.exit(f"no manifest.json under {model_dir}")
    man = json.loads(man_path.read_text())

    gguf_path = Path(sys.argv[2]) if len(sys.argv) > 2 else None
    if gguf_path is None:
        cand = model_dir / "model.gguf"
        if cand.exists():
            gguf_path = cand
    gf = None
    if gguf_path is not None and gguf is not None and gguf_path.exists():
        gf = gguf.GGUFReader(str(gguf_path))
        print(f"GGUF: {gguf_path}")

    gguf_section = man.get("gguf", man)
    lt = gguf_section["layer_types"].get("full_attention", {})
    pool = (lt.get("pack") or {}).get("pool", [])
    q_ops = [o for o in pool if "q_proj" in o.get("tensor", "")]
    if not q_ops:
        print("no q_proj pool ops found (not a gated full-attention layer type?)")
        return 0

    print(f"full_attention q_proj ops: {len(q_ops)}")
    for op in q_ops:
        name = op["tensor"]
        rows = tensor_rows(gf, name.replace("model.layer.{l}.", "").replace("{l}", "0"))
        # tensor name in manifest uses {l}; gguf uses layer index
        # try a few common forms
        if rows is None and gf is not None:
            for ti in gf.tensors:
                if ti.name.endswith("self_attn.q_proj.weight"):
                    dims = list(ti.shape) if hasattr(ti, "shape") else list(ti.dims)
                    rows = int(dims[1]) if len(dims) > 1 else int(dims[0])
                    break
        print(f"\nop: chunk0={op.get('chunk0')} nch={op['nch']} in_dim={op['in_dim']} "
              f"qgate_rows={op.get('qgate_rows')} qgate_gate={op.get('qgate_gate')}")
        if rows is not None:
            print(f"  gguf tensor rows: {rows}")
        for m in check_op(op, rows):
            print(m)
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__); sys.exit(2)
    sys.exit(main())
