r"""capture_te_goldens: FLUX.2 [klein]'s text encoder (Qwen3-4B, layers 1-27) for the NPU chain test.

    C:\dev\ditref-venv\Scripts\python.exe utilities\dit-ref\capture_te_goldens.py

Runs the klein pipeline's prompt encoding (chat template, right-padded to 512 tokens,
hidden states 9/18/27) for one prompt in bf16 on the CPU, and saves under
C:\dev\ditref-out\goldens_te\:

    input_ids, attention_mask, n_real        the tokens and how many are not padding
    hs_<k>                                   hidden_states[k], k = 0 (embeddings), 1, 9, 18, 27
    npu_hs_<k>                               the same with the NPU arithmetic: every linear
                                             of layers 0-26 as dit_gemm's bfp16 x bf16 GEMM
                                             (klein_quant_study.EmuLinear) and attention as
                                             dit_fa (fa_emul.dit_fa_attention, causal + the
                                             key-padding mask)
    prompt_embeds                            the pipeline's [512, 7680] output (taps 9/18/27)

Weights are not copied: the chain test reads them from the model's safetensors.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import fa_emul  # noqa: E402
import klein_quant_study as kqs  # noqa: E402

TAPS = (0, 1, 9, 18, 27)


N_REAL = {}


def fa_emul_attention(module, query, key, value, attention_mask, scaling=None, dropout=0.0,
                      **kwargs):
    """transformers attention interface: q [B, H, L, D], k/v [B, KVH, L, D] -> [B, L, H, D].
    The mask is built here -- causal plus the key padding of the right-padded prompt --
    not taken from `attention_mask`, whose form depends on what transformers makes for an
    implementation it does not know (with it, pad rows attended to pad keys)."""
    rep = query.shape[1] // key.shape[1]
    k = key.repeat_interleave(rep, dim=1)
    v = value.repeat_interleave(rep, dim=1)
    L = query.shape[2]
    cols = torch.arange(L)[None, :]
    mask = (cols > torch.arange(L)[:, None]) | (cols >= N_REAL["n"])
    o = fa_emul.dit_fa_attention(query.float(), k.float(), v.float(), fa_emul.FULL, mask)
    return o.transpose(1, 2).to(query.dtype), None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", type=int, default=0)
    ap.add_argument("--out", default=r"C:\dev\ditref-out")
    a = ap.parse_args()
    from diffusers import Flux2KleinPipeline
    from transformers import AttentionInterface

    out = Path(a.out) / "goldens_te"
    out.mkdir(parents=True, exist_ok=True)
    pipe = Flux2KleinPipeline.from_pretrained(kqs.MODEL, torch_dtype=torch.bfloat16,
                                              transformer=None, vae=None)
    te, tok = pipe.text_encoder, pipe.tokenizer
    prompt = kqs.PROMPTS[a.prompt]
    text = tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False,
                                   add_generation_prompt=True, enable_thinking=False)
    enc = tok(text, return_tensors="pt", padding="max_length", truncation=True, max_length=512)
    ids, mask = enc["input_ids"], enc["attention_mask"]
    n_real = int(mask.sum())
    np.save(out / "input_ids.npy", ids.numpy())
    np.save(out / "attention_mask.npy", mask.numpy())

    with torch.inference_mode():
        hs = te(input_ids=ids, attention_mask=mask, output_hidden_states=True,
                use_cache=False).hidden_states
    for k in TAPS:
        np.save(out / f"hs_{k}.npy", hs[k][0].float().numpy())
    pe = torch.stack([hs[k] for k in (9, 18, 27)], dim=1).permute(0, 2, 1, 3).reshape(1, 512, -1)
    np.save(out / "prompt_embeds.npy", pe[0].float().numpy())

    N_REAL["n"] = n_real
    AttentionInterface.register("fa_emul", fa_emul_attention)
    te.config._attn_implementation = "fa_emul"
    for layer in te.model.layers[:27]:
        kqs.swap_linears_in(layer, "bfp16-bf16acc")
    # keep 28 layers: with 27, hidden_states[27] would be the final-normed state, not the
    # raw residual the pipeline taps
    te.model.layers = te.model.layers[:28]
    with torch.inference_mode():
        hs = te(input_ids=ids, attention_mask=mask, output_hidden_states=True,
                use_cache=False).hidden_states
    for k in TAPS:
        np.save(out / f"npu_hs_{k}.npy", hs[k][0].float().numpy())
    (out / "manifest.json").write_text(json.dumps({"prompt": prompt, "n_real": n_real,
                                                   "taps": TAPS}, indent=1))
    print(f"text encoder goldens -> {out} (n_real {n_real})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
