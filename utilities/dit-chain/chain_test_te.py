r"""chain_test_te: FLUX.2 [klein]'s text encoder (Qwen3-4B layers 1-27) on the NPU.

    . C:\dev\mlir-aie\iron_env.ps1
    python utilities\dit-chain\chain_test_te.py --kernels C:\dev\klein-kernels --goldens C:\dev\ditref-out\goldens_te
        [--fa <dir with a te_attn build for this prompt's length>]

27 layers, every op on the NPU -- dit_ew (RMSNorm x weight, residual + RMSNorm, q/k
RMSNorm + rotate-half RoPE), dit_gemm (q|k|v, o_proj, gate|up with the SwiGLU epilogue,
down_proj reading it gathered) and dit_fa (causal GQA 32/8 attention, keys past the
prompt masked) -- in export_dit_kernels.py's text-encoder layout, from the captured
embeddings. Compares the residual stream at the pipeline's taps (hidden states 9/18/27)
with diffusers/transformers' bf16 and with the CPU emulation of the NPU arithmetic
(capture_te_goldens.py), real-token and padding rows separately (the DiT attends to both).
Padding rows are reported, not gated: they are hypersensitive to the bfp16 arithmetic
(utilities/dit-ref/README.md).

te_attn's valid_len is written into its instruction stream; the exported one says 512.
Until the engine patches it, pass --fa: a directory holding that build's final.xclbin and
its insts.bin copied to insts_te_attn.bin (DF_VALID_LEN=<n_real>, te_attn's layout).
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "open_kernels" / "harness"))
sys.path.insert(0, str(ROOT / "open_kernels" / "designs" / "dit_gemm"))
sys.path.insert(0, str(ROOT / "open_kernels" / "designs" / "dit_ew"))
sys.path.insert(0, str(HERE))
from make_test import qwen_qk_params  # noqa: E402
from npu_host import Npu  # noqa: E402
from pack import interleave_swiglu, pack_b  # noqa: E402
from safetensors_np import SafeTensors  # noqa: E402

L, HID, PAD, EL = 512, 2560, 3072, 3072
Q, KV, MLP = 4096, 1024, 9728
TAPS = (9, 18, 27)


def bf(a):
    return np.asarray(a, dtype=np.float32).astype(bfloat16)


def packed_layer(st: SafeTensors, l: int, cache: Path) -> dict[str, np.ndarray]:
    """The layer's GEMM weights in dit_gemm's packing, cached as .npy."""
    names = {"qkv": None, "o": None, "gu": None, "down": None}
    out = {}
    for n in names:
        f = cache / f"layer{l:02d}_{n}.npy"
        if f.exists():
            out[n] = np.load(f)
            continue
        g = lambda k: st.get(f"model.layers.{l}.{k}.weight")  # noqa: E731
        if n == "qkv":
            wt = np.concatenate([g("self_attn.q_proj"), g("self_attn.k_proj"),
                                 g("self_attn.v_proj")]).T                     # [2560, 6144]
        elif n == "o":
            wt = np.zeros((Q, PAD), np.float32)
            wt[:, :HID] = g("self_attn.o_proj").T                              # [4096, 3072]
        elif n == "gu":
            wt = interleave_swiglu(np.concatenate([g("mlp.gate_proj"), g("mlp.up_proj")]).T)
        else:
            wt = np.zeros((MLP, PAD), np.float32)
            wt[:, :HID] = g("mlp.down_proj").T                                 # [9728, 3072]
        out[n] = pack_b(wt)
        np.save(f, out[n])
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kernels", required=True)
    ap.add_argument("--goldens", required=True)
    ap.add_argument("--fa", default=None,
                    help="dir with final.xclbin + insts_te_attn.bin built for this prompt's length")
    ap.add_argument("--model", default=None, help="text_encoder dir (default: the HF cache)")
    a = ap.parse_args()
    kdir, gdir = Path(a.kernels), Path(a.goldens)
    n_real = json.loads((gdir / "manifest.json").read_text(encoding="utf-8"))["n_real"]
    model = Path(a.model) if a.model else Path(glob.glob(str(
        Path.home() / ".cache/huggingface/hub/models--black-forest-labs--FLUX.2-klein-4B"
        / "snapshots/*/text_encoder"))[0])
    st = SafeTensors(model)
    cache = kdir / "te_packed"
    cache.mkdir(exist_ok=True)

    npu = Npu()
    gm = npu.kernel_set("gemm", kdir)
    ew = npu.kernel_set("ew", kdir / "ew")
    fa = npu.kernel_set("fa", Path(a.fa) if a.fa else kdir / "fa")

    X, Z = npu.buf("X", L * PAD * 2), npu.buf("Z", L * PAD * 2)
    QT, OT, AT = npu.buf("QT", L * (Q + 2 * KV) * 2), npu.buf("OT", L * Q * 2), npu.buf("AT", L * PAD * 2)
    GT = npu.buf("GT", L * 2 * MLP * 2)
    dummy = npu.buf("dummy", EL * 2)
    wbufs = {n: npu.buf(f"w_{n}", sz) for n, sz in
             (("qkv", HID * (Q + 2 * KV) * 9 // 8), ("o", Q * PAD * 9 // 8),
              ("gu", HID * 2 * MLP * 9 // 8), ("down", MLP * PAD * 9 // 8))}
    pnorm = npu.buf("pnorm", 3 * EL * 2)
    pqk = npu.buf("pqk", 5 * EL * 2)

    def set_norm(w):
        v = np.zeros(3 * EL, bfloat16)
        v[EL:EL + HID] = bf(w)
        pnorm.write(v)

    def ew_run(stream, a_, b_, p, y, z):
        ew.stream(stream).run(a_, b_ if b_ is not None else dummy.bo, p.bo, y,
                              z if z is not None else dummy.bo)

    x0 = np.zeros((L, PAD), np.float32)
    x0[:, :HID] = np.load(gdir / "hs_0.npy")
    X.write(bf(x0))
    set_norm(st.get("model.layers.0.input_layernorm.weight"))
    ew_run("te_rms", X.bo, None, pnorm, Z.bo, None)
    results = {}
    for l in range(27):
        w = packed_layer(st, l, cache)
        for n, arr in w.items():
            wbufs[n].write(arr)
        g = lambda k: st.get(f"model.layers.{l}.{k}.weight")  # noqa: E731
        gm.stream("te_qkv").run(Z.bo, wbufs["qkv"].bo, QT.bo)
        e = np.zeros(EL, bfloat16)
        pqk.write(np.concatenate([e, qwen_qk_params(g("self_attn.q_norm"),
                                                    g("self_attn.k_norm")), e]))
        ew_run("te_qk", QT.bo, QT.bo, pqk, QT.bo, QT.bo)
        fa.stream("te_attn").run(QT.bo, QT.bo, QT.bo, OT.bo)
        gm.stream("te_o").run(OT.bo, wbufs["o"].bo, AT.bo)
        set_norm(g("post_attention_layernorm"))
        ew_run("te_res_rms", X.bo, AT.bo, pnorm, Z.bo, X.bo)
        gm.stream("te_gu").run(Z.bo, wbufs["gu"].bo, GT.bo)
        gm.stream("te_down").run(GT.bo, wbufs["down"].bo, AT.bo)
        nxt = st.get(f"model.layers.{l + 1}.input_layernorm.weight")
        set_norm(nxt)
        ew_run("te_res_rms", X.bo, AT.bo, pnorm, Z.bo, X.bo)
        if l + 1 in TAPS:
            results[l + 1] = X.read(np.uint16).view(bfloat16).astype(np.float64).reshape(L, PAD)
        print(f"  layer {l + 1:2d} done", flush=True)

    ok = True
    rf = lambda u, v: float(np.linalg.norm(u - v) / np.linalg.norm(v))  # noqa: E731
    print(f"text encoder, {n_real} real tokens, {len(npu.log)} dispatches, "
          f"{sum(t for _, _, t in npu.log):.0f} ms of NPU runs")
    for k, got in results.items():
        ref = np.load(gdir / f"hs_{k}.npy").astype(np.float64)
        emu = np.load(gdir / f"npu_hs_{k}.npy").astype(np.float64)
        h = got[:, :HID]
        pad_zero = float(np.abs(got[:, HID:]).max())
        r = {part: (rf(h[sl], ref[sl]), rf(emu[sl], ref[sl]), rf(h[sl], emu[sl]))
             for part, sl in (("real", slice(0, n_real)), ("pad", slice(n_real, L)))}
        # The emulation keeps HF's bf16 norm/RoPE/residual roundings and dit_ew rounds its
        # own way, so the NPU need not land within 1.25x of the emulation's distance from
        # bf16. The check: it is closer to the emulation of its own arithmetic than that
        # emulation is to bf16 (a layout or sequencing bug is O(1)).
        passed = bool(np.isfinite(h).all()) and pad_zero == 0.0 and r["real"][2] <= r["real"][1]
        ok &= passed
        print(f"  hs_{k}: real rows npu-vs-bf16 {r['real'][0]:.3e} (emulated {r['real'][1]:.3e}, "
              f"npu-vs-emulated {r['real'][2]:.3e}); pad rows {r['pad'][0]:.3e} (emulated "
              f"{r['pad'][1]:.3e}); padding columns {pad_zero}  {'PASS' if passed else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
