r"""te_npu_embeds: klein prompt embeddings with the text encoder on the NPU's arithmetic.

    C:\dev\ditref-venv\Scripts\python.exe utilities\dit-ref\te_npu_embeds.py [--stages FULL] [--tag te-npu]

Writes <out>\klein_512_s4\prompt_embeds_<tag>.pt: the 8 study prompts encoded exactly as
the pipeline does (chat template, right-padded to 512, hidden states 9/18/27), with every
linear of Qwen3-4B's layers 1-27 as dit_gemm (klein_quant_study.EmuLinear bfp16-bf16acc)
and attention as dit_fa (fa_emul, causal + key padding). klein_quant_study's variant
"<tag>" then generates with them (DiT in bf16) to measure what the text encoder's NPU
arithmetic does to images. --stages picks fa_emul stages ("FULL", "FULL-exp_hw", ...);
--no-gemm keeps the linears exact.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import fa_emul  # noqa: E402
import klein_quant_study as kqs  # noqa: E402

N_REAL = {}


def parse_stages(spec: str):
    parts = spec.split("-")
    st = set(fa_emul.FULL if parts[0] == "FULL" else fa_emul.FIXED)
    for p in parts[1:]:
        st.discard(p)
    return frozenset(st)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", default="FULL")
    ap.add_argument("--no-gemm", action="store_true")
    ap.add_argument("--tag", default="te-npu")
    ap.add_argument("--smooth", action="store_true",
                    help="q_norm.w *= 2^k, k_norm.w /= 2^k per RoPE pair (q.k unchanged, exact in "
                         "bf16), balancing q and k's per-dim magnitudes before their bfp16 blocks")
    ap.add_argument("--fp32", action="store_true",
                    help="reference variant: the text encoder in plain fp32, no NPU arithmetic")
    ap.add_argument("--out", default=r"C:\dev\ditref-out")
    a = ap.parse_args()
    from diffusers import Flux2KleinPipeline
    from transformers import AttentionInterface

    stages = parse_stages(a.stages)

    def attn(module, q, k, v, attention_mask, scaling=None, dropout=0.0, **kw):
        rep = q.shape[1] // k.shape[1]
        k, v = k.repeat_interleave(rep, 1).float(), v.repeat_interleave(rep, 1).float()
        L = q.shape[2]
        cols = torch.arange(L)[None, :]
        mask = (cols > torch.arange(L)[:, None]) | (cols >= N_REAL["n"])
        o = fa_emul.dit_fa_attention(q.float(), k, v, stages, mask)
        return o.transpose(1, 2).to(q.dtype), None

    pipe = Flux2KleinPipeline.from_pretrained(
        kqs.MODEL, torch_dtype=torch.float32 if a.fp32 else torch.bfloat16,
        transformer=None, vae=None)
    te, tok = pipe.text_encoder, pipe.tokenizer
    te.model.layers = te.model.layers[:28]           # tap 27 stays the raw residual
    if a.smooth:
        for layer in te.model.layers[:27]:
            at = layer.self_attn
            wq, wk = at.q_norm.weight.data.float(), at.k_norm.weight.data.float()
            rq = torch.maximum(wq[:64].abs(), wq[64:].abs()).clamp_min(1e-6)
            rk = torch.maximum(wk[:64].abs(), wk[64:].abs()).clamp_min(1e-6)
            sc = torch.exp2(torch.round(0.5 * torch.log2(rk / rq)))
            sc = torch.cat([sc, sc])
            at.q_norm.weight.data = (wq * sc).to(at.q_norm.weight.dtype)
            at.k_norm.weight.data = (wk / sc).to(at.k_norm.weight.dtype)
    if not a.no_gemm and not a.fp32:
        for layer in te.model.layers[:27]:
            kqs.swap_linears_in(layer, "bfp16-bf16acc")
    if not a.fp32:
        AttentionInterface.register("npu_fa", attn)
        te.config._attn_implementation = "npu_fa"
    embeds = []
    for p in kqs.PROMPTS:
        text = tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False,
                                       add_generation_prompt=True, enable_thinking=False)
        enc = tok(text, return_tensors="pt", padding="max_length", truncation=True,
                  max_length=512)
        N_REAL["n"] = int(enc["attention_mask"].sum())
        with torch.inference_mode():
            hs = te(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"],
                    output_hidden_states=True, use_cache=False).hidden_states
        pe = torch.stack([hs[k] for k in (9, 18, 27)], 1).permute(0, 2, 1, 3).reshape(1, 512, -1)
        embeds.append(pe.to(torch.bfloat16))
        print(f"  {N_REAL['n']:3d} tokens: {p[:50]}", flush=True)
    dst = Path(a.out) / "klein_512_s4" / f"prompt_embeds_{a.tag}.pt"
    dst.parent.mkdir(parents=True, exist_ok=True)
    torch.save(embeds, dst)
    print(f"-> {dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
