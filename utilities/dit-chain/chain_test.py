r"""chain_test: one whole FLUX.2 [klein] 4B double block and single block on the NPU.

    . C:\dev\mlir-aie\iron_env.ps1
    python utilities\dit-chain\chain_test.py --kernels C:\dev\klein-kernels --goldens C:\dev\ditref-out\goldens_512

Every op of the block runs on the NPU -- dit_gemm (the linears, with the SwiGLU as its
epilogue), dit_ew (LayerNorm + modulate, the residual updates, q/k RMSNorm + RoPE) and
dit_fa (attention) --
in the buffer layout export_dit_kernels.py's streams encode, from the block input
utilities/dit-ref/capture_goldens.py captured out of diffusers. The host only packs
weights, fills buffers and waits.

The check is on the block's update (output - input), which the residual would otherwise
hide. Gate (the plan's: "materially higher than predicted is a bug"): the NPU's update may
be at most GATE_RATIO times further from diffusers' than the CPU emulation of the same
arithmetic is (capture_goldens' npu_*). A layout or sequencing bug is O(1); numerics
land within a few percent of the emulation. Also printed: the floor that rounding the
block's output to bf16 alone puts on the update comparison -- in deep single blocks the
residual is large next to the update, so it is several percent.

The modulation vectors are arranged per dispatch here (a run of 2-3 vectors with one vector
of slack each side, as dit_ew reads them); in the engine the modulation GEMM's output
order does it.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "open_kernels" / "harness"))
sys.path.insert(0, str(ROOT / "open_kernels" / "designs" / "dit_gemm"))
sys.path.insert(0, str(ROOT / "open_kernels" / "designs" / "dit_ew"))
from npu_host import Npu  # noqa: E402
from pack import interleave_swiglu, pack_b  # noqa: E402
from make_test import qk_params  # noqa: E402

H, EL, L_TXT = 3072, 3072, 512
GATE_RATIO = 1.25    # NPU-vs-diffusers may be at most this times emulation-vs-diffusers


class Golden:
    def __init__(self, d: Path, block: str):
        self.d = d / block

    def __call__(self, name) -> np.ndarray:
        return np.load(self.d / f"{name}.npy")

    def w(self, name) -> np.ndarray:
        return np.load(self.d / "w" / f"{name.removesuffix('.weight')}.npy")


def bf(a) -> np.ndarray:
    return np.asarray(a, dtype=np.float32).astype(bfloat16)


def run_block_chain(npu, sets, g: Golden, kind: str, R: int, T_img: int):
    T = T_img + L_TXT
    gm, fa, ew = sets["gemm"], sets["fa"], sets["ew"]
    B = lambda n, cols: npu.buf(n, T * cols * 2)  # noqa: E731
    X, Zn, AO = B("X", H), B("Zn", H), B("AO", H)
    dummy = npu.buf("dummy", EL * 2)
    rows = lambda buf, cols, r0, n: buf.view(r0 * cols * 2, n * cols * 2)  # noqa: E731
    parts = {"txt": (0, L_TXT), "img": (L_TXT, T_img), "all": (0, T)}

    def pbuf(name, vecs):
        p = npu.buf(name, (len(vecs) + 2) * EL * 2)
        p.write(np.concatenate([np.zeros(EL, bfloat16)] + [bf(v) for v in vecs]
                               + [np.zeros(EL, bfloat16)]))
        return p

    def weight(name, W, swiglu_from=None):   # W: [out, in] -> packed [in, out]
        Wt = W.astype(np.float32).T
        if swiglu_from is not None:   # MLP-in columns into the epilogue's order
            Wt = interleave_swiglu(Wt, swiglu_from)
        pk = pack_b(Wt)
        b = npu.buf(name, pk.nbytes)
        b.write(pk)
        return b

    def gemm(stream, a_view, w, c_view):
        gm.stream(stream).run(a_view, w.bo, c_view)

    def ew_run(stream, a, b, p, y, z):
        ew.stream(stream).run(a, b if b is not None else dummy.bo, p.bo,
                              y, z if z is not None else dummy.bo)

    if kind == "double":
        x_in = np.concatenate([g("in_encoder")[0], g("in_hidden")[0]])
        mods = {"img": np.split(g("mod_img").reshape(-1), 6),
                "txt": np.split(g("mod_txt").reshape(-1), 6)}  # shift/scale/gate msa, mlp
        QKV, O, FF = B("QKV", 3 * H), B("O", H), B("FF", 6 * H)
        wts = {
            "txt_qkv": np.concatenate([g.w(f"attn.add_{c}_proj.weight") for c in "qkv"]),
            "img_qkv": np.concatenate([g.w(f"attn.to_{c}.weight") for c in "qkv"]),
            "txt_out": g.w("attn.to_add_out.weight"), "img_out": g.w("attn.to_out.0.weight"),
            "txt_ffin": g.w("ff_context.linear_in.weight"), "img_ffin": g.w("ff.linear_in.weight"),
            "txt_ffout": g.w("ff_context.linear_out.weight"),
            "img_ffout": g.w("ff.linear_out.weight")}
        W = {k: weight(k, v, 0 if k.endswith("ffin") else None) for k, v in wts.items()}
        qkw = {"img": (g.w("attn.norm_q.weight"), g.w("attn.norm_k.weight")),
               "txt": (g.w("attn.norm_added_q.weight"), g.w("attn.norm_added_k.weight"))}
        X.write(bf(x_in))
        gstream = lambda part, op: f"{'txt' if part == 'txt' else f'r{R}_img'}_{op}"  # noqa: E731
        for part in ("txt", "img"):
            r0, n = parts[part]
            sh_msa, sc_msa, ga_msa, sh_mlp, sc_mlp, ga_mlp = mods[part]
            ew_run(f"r{R}_ln_{part}", rows(X, H, r0, n), None, pbuf(f"pln_{part}", [sh_msa, sc_msa]),
                   rows(Zn, H, r0, n), None)
            gemm(gstream(part, "qkv"), rows(Zn, H, r0, n), W[f"{part}_qkv"], rows(QKV, 3 * H, r0, n))
            q = rows(QKV, 3 * H, r0, n)
            pq = npu.buf(f"pqk_{part}", 5 * EL * 2)
            pq.write(np.concatenate([np.zeros(EL, bfloat16), qk_params(*qkw[part]),
                                     np.zeros(EL, bfloat16)]))
            ew_run(f"r{R}_qk_{part}", q, q, pq, q, q)
        fa.stream(f"r{R}_attn_dbl").run(QKV.bo, QKV.bo, QKV.bo, O.bo)
        for part in ("txt", "img"):
            r0, n = parts[part]
            sh_msa, sc_msa, ga_msa, sh_mlp, sc_mlp, ga_mlp = mods[part]
            gemm(gstream(part, "out"), rows(O, H, r0, n), W[f"{part}_out"], rows(AO, H, r0, n))
            x = rows(X, H, r0, n)
            ew_run(f"r{R}_res_{part}", x, rows(AO, H, r0, n),
                   pbuf(f"pres1_{part}", [ga_msa, sh_mlp, sc_mlp]), rows(Zn, H, r0, n), x)
            gemm(gstream(part, "ffin"), rows(Zn, H, r0, n), W[f"{part}_ffin"],
                 rows(FF, 6 * H, r0, n))
        for part in ("txt", "img"):
            r0, n = parts[part]
            sh_msa, sc_msa, ga_msa, sh_mlp, sc_mlp, ga_mlp = mods[part]
            gemm(gstream(part, "ffout"), rows(FF, 6 * H, r0, n), W[f"{part}_ffout"],
                 rows(AO, H, r0, n))
            x = rows(X, H, r0, n)
            ew_run(f"r{R}_res_{part}", x, rows(AO, H, r0, n),
                   pbuf(f"pres2_{part}", [ga_mlp, sh_msa, sc_msa]), rows(Zn, H, r0, n), x)
        ref = np.concatenate([g("out_encoder")[0], g("out_hidden")[0]])
        emu = np.concatenate([g("npu_encoder")[0], g("npu_hidden")[0]])
    else:
        x_in = g("in_hidden")[0]
        shift, scale, gate = np.split(g("mod").reshape(-1), 3)
        FU = B("FU", 11 * H)   # q|k|v, attention slot (64-of-128), MLP tiles
        W_in = weight("sgl_in", g.w("attn.to_qkv_mlp_proj.weight"), 3 * H)
        W_out = weight("sgl_out", g.w("attn.to_out.weight"))
        X.write(bf(x_in))
        ew_run(f"r{R}_ln_all", X.bo, None, pbuf("pln", [shift, scale]), Zn.bo, None)
        gemm(f"r{R}_sgl_in", Zn.bo, W_in, FU.bo)
        pq = npu.buf("pqk", 5 * EL * 2)
        pq.write(np.concatenate([np.zeros(EL, bfloat16),
                                 qk_params(g.w("attn.norm_q.weight"), g.w("attn.norm_k.weight")),
                                 np.zeros(EL, bfloat16)]))
        ew_run(f"r{R}_qk_sgl", FU.bo, FU.bo, pq, FU.bo, FU.bo)
        fa.stream(f"r{R}_attn_sgl").run(FU.bo, FU.bo, FU.bo, FU.bo)
        gemm(f"r{R}_sgl_out", FU.bo, W_out, AO.bo)
        ew_run(f"r{R}_res_all", X.bo, AO.bo, pbuf("pres", [gate, shift, scale]), Zn.bo, X.bo)
        ref, emu = g("out_hidden")[0], g("npu_hidden")[0]

    out = X.read(np.uint16).view(bfloat16).astype(np.float64).reshape(T, H)
    x0 = x_in.astype(np.float64)
    d_npu, d_ref, d_emu = out - x0, ref.astype(np.float64) - x0, emu.astype(np.float64) - x0
    rf = lambda a, b: float(np.linalg.norm(a - b) / np.linalg.norm(b))  # noqa: E731
    # rms of rounding a value of that size to bf16: ulp / sqrt(12), ulp = 2^(e - 7)
    ulp = np.exp2(np.floor(np.log2(np.maximum(np.abs(x0 + d_ref), 1e-30))) - 7)
    floor = float(np.sqrt((ulp ** 2 / 12).sum()) / np.linalg.norm(d_ref))
    res = {"finite": bool(np.isfinite(out).all()),
           "bf16_output_floor": floor,
           "update_vs_emulated": rf(d_npu, d_emu), "update_vs_diffusers": rf(d_npu, d_ref),
           "emulated_vs_diffusers": rf(d_emu, d_ref),
           "text_rows_vs_emulated": rf(d_npu[:L_TXT], d_emu[:L_TXT]),
           "image_rows_vs_emulated": rf(d_npu[L_TXT:], d_emu[L_TXT:])}
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kernels", required=True, help="export_dit_kernels.py --out dir")
    ap.add_argument("--goldens", required=True, help="capture_goldens.py output dir")
    ap.add_argument("--blocks", default="dbl0,sgl0")
    a = ap.parse_args()
    kdir, gdir = Path(a.kernels), Path(a.goldens)
    man = json.loads((gdir / "manifest.json").read_text(encoding="utf-8"))
    R = man["size"]
    T_img = (R // 16) ** 2
    ok = True
    for block in a.blocks.split(","):
        npu = Npu()
        sets = {"gemm": npu.kernel_set("gemm", kdir), "fa": npu.kernel_set("fa", kdir / "fa"),
                "ew": npu.kernel_set("ew", kdir / "ew")}
        kind = "double" if block.startswith("dbl") else "single"
        res = run_block_chain(npu, sets, Golden(gdir, block), kind, R, T_img)
        passed = res["finite"] and (res["update_vs_diffusers"]
                                    <= GATE_RATIO * res["emulated_vs_diffusers"])
        ok &= passed
        ms = sum(t for _, _, t in npu.log)
        print(f"{block} ({kind}, {R}^2): {len(npu.log)} dispatches, {ms:.1f} ms of NPU runs")
        for k, v in res.items():
            print(f"  {k:24s} {v if isinstance(v, bool) else f'{v:.3e}'}")
        by = {}
        for s, n, t in npu.log:
            by.setdefault(s, []).append(t)
        print("  per kernel set: " + ", ".join(f"{s} {sum(t):.1f} ms/{len(t)}" for s, t in by.items()))
        print(f"  {'PASS' if passed else 'FAIL'} (gate: update vs diffusers <= {GATE_RATIO} x "
              f"emulated vs diffusers)")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
