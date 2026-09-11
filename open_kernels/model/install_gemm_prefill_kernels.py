r"""0167/#32: build a SELF-CONTAINED kernel directory carrying the GEMM
prefill route (manifest.hpp's GemmBlockProgram) alongside the existing
sequential ("dx") path, WITHOUT touching the installed
~/Documents/oflm/models/Granite-4.2-3B-NPU2/open_kernels -- so a baseline
re-run (no env var) is untouched, and the new route is opt-in via
OFLM_OPEN_KERNELS_DIR. That variable is checked FIRST by Engine::find_kernels()
(engine.cpp), before the model-dir copy -- so a rebuilt kernel set here is
never silently shadowed by a stale one already installed next to the model.

Copies the production dx/ln/lm_head_q4 kernel set verbatim, adds the 5 new
GEMM contexts (already built, real per-layer weights, hardware-validated)
plus dxB (already built, the single-token attention dispatch,
designs/dense/dx_attn.py, unmodified) as new contexts/kernels, and adds ONE
new key to layer_types.dense: "gemm_block" -- exactly the schema
manifest.cpp parses.

    python open_kernels/model/install_gemm_prefill_kernels.py [--dest DIR]
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DESIGNS = HERE.parent / "designs"
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

from recipes import dense as QR  # noqa: E402
from recipes.load import spec_from_model_dir  # noqa: E402

DEFAULT_MODEL_DIR = Path.home() / ".oflm" / "models" / "Granite-4.2-3B-NPU2"
SRC_KERNELS = Path.home() / "Documents" / "oflm" / "models" / "Granite-4.2-3B-NPU2" / "open_kernels"
GQP = DESIGNS / "gemm_q4_prefill"
DENSE = DESIGNS / "dense"
T = 256


def copy_build(build_dir: Path, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    for name in ("final.xclbin", "insts.bin"):
        src = build_dir / name
        if not src.exists():
            raise FileNotFoundError(f"missing {src}")
        shutil.copy2(src, dest / name)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR))
    ap.add_argument("--src-kernels", default=str(SRC_KERNELS), help="the existing installed sequential kernel set to copy from")
    ap.add_argument("--dest", default=str(HERE / "out_gemm_prefill_kernels"))
    a = ap.parse_args()

    md = Path(a.model_dir)
    src = Path(a.src_kernels)
    dest = Path(a.dest)

    print(f"model dir:   {md}")
    print(f"src kernels: {src}")
    print(f"dest:        {dest}")

    spec = spec_from_model_dir(md)
    L, G = QR.layout(spec), QR.geometry(spec)
    hid, ff = spec.hidden, spec.intermediate
    qw, kvw = G.QW, G.KVW
    n_qkv3 = qw + 2 * kvw
    assert n_qkv3 == 3584 and hid == 2560 and ff == 8192, (n_qkv3, hid, ff)  # this route's builds are fixed at these widths

    # ---- fresh copy of the production sequential kernel set (dx, ln, lm_head_q4) ----
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(src, dest)

    # ---- add the GEMM route's kernels: dxB (the single-token attention
    # dispatch, dx_attn.py, unmodified) + 4 new GEMM contexts (real per-layer
    # weights) ----
    copy_build(DENSE / "build_granite_h2560_dxB", dest / "dxB")
    copy_build(GQP / "build_qkv3_t256", dest / "gqkv3")
    copy_build(GQP / "build_qkv_t256", dest / "gqo")       # o_proj: same [hid,hid] shape as the original qkv probe build
    copy_build(GQP / "build_up_gate_t256", dest / "ggu")   # gate_proj AND up_proj share this context
    copy_build(GQP / "build_down_proj_t256", dest / "gd")

    manifest_path = dest / "manifest.json"
    m = json.loads(manifest_path.read_text())

    m["contexts"].update({
        "dxB": "dxB/final.xclbin",
        "gqkv3": "gqkv3/final.xclbin",
        "gqo": "gqo/final.xclbin",
        "ggu": "ggu/final.xclbin",
        "gd": "gd/final.xclbin",
    })
    m["kernels"].update({
        "dxB": {"context": "dxB", "insts": "dxB/insts.bin", "patch": "attnpos", "window": 0},
        "gqkv3": {"context": "gqkv3", "insts": "gqkv3/insts.bin"},
        "gqo": {"context": "gqo", "insts": "gqo/insts.bin"},
        "ggu": {"context": "ggu", "insts": "ggu/insts.bin"},
        "gd": {"context": "gd", "insts": "gd/insts.bin"},
    })
    m["globals"].update({
        "gact": T * L.AD_BYTES,
        "gemm_x_hid": hid * T * 2,
        "gemm_x_ff": ff * T * 2,
        "gemm_y_qkv3": n_qkv3 * T * 4,
        "gemm_y_o": hid * T * 4,
        "gemm_y_gate": ff * T * 4,
        "gemm_y_up": ff * T * 4,
        "gemm_y_down": hid * T * 4,
    })
    m["layer_types"]["dense"]["gemm_block"] = {
        "t": T,
        "eps": spec.norm_eps,
        "qw": qw, "kvw": kvw, "ff": ff,
        "ad_q": L.AD_Q, "ad_kvn": L.AD_KVN, "ad_og": L.AD_OG,
        "program": [
            {"op": "run", "kernel": "gqkv3", "args": ["gqkv3_w", "gemm_x_hid", "gemm_y_qkv3"]},
            {"op": "run", "kernel": "gqo", "args": ["go_w", "gemm_x_hid", "gemm_y_o"]},
            {"op": "run", "kernel": "ggu", "args": ["ggate_w", "gemm_x_hid", "gemm_y_gate"]},
            {"op": "run", "kernel": "ggu", "args": ["gup_w", "gemm_x_hid", "gemm_y_up"]},
            {"op": "run", "kernel": "gd", "args": ["gdown_w", "gemm_x_ff", "gemm_y_down"]},
        ],
    }

    manifest_path.write_text(json.dumps(m, indent=1))
    print(f"wrote {manifest_path}")
    print(f"AD_Q={L.AD_Q} AD_KVN={L.AD_KVN} AD_OG={L.AD_OG} AD_BYTES={L.AD_BYTES} "
          f"qw={qw} kvw={kvw} ff={ff} n_qkv3={n_qkv3} eps={spec.norm_eps}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
