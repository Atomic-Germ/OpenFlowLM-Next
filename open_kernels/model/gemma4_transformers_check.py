"""Hold replica_gemma4.py to transformers' Gemma 4 on a slice stitched from chosen layers (2026-10-16).

A 4-layer Gemma4ForCausalLM -- one sliding and one global layer with their own caches, then a sliding and a
global layer that share them -- is built in fp32 with the GGUF's own (dequantized) weights for real layers
0, 5, 24 and 29 of E4B, so the slice covers both head widths, both RoPE types, KV sharing and the per-layer
inputs. The embedding and the per-layer table are cut to the prompt's tokens (the replica reads the same
rows), so nothing near the whole model is ever in fp32. Both sides see the same numbers; what is compared is
the math.

    python gemma4_transformers_check.py <model.gguf> [--ids 2,105,...] [--layers 0,5,24,29]   (E2B: --layers 0,4,15,19)

The 12B (`--layers 0,5,6,11`) is transformers' own Gemma4UnifiedForCausalLM (5.10+).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import replica_gemma4 as R  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("gguf")
    ap.add_argument("--ids", default="2,105,2364,107,3689,563,506,5279,529,7001,236881,106,107")
    ap.add_argument("--layers", default="0,5,24,29")
    a = ap.parse_args()
    import torch
    from transformers import Gemma4ForCausalLM, Gemma4TextConfig

    g = R.Gguf(a.gguf)
    G = R.Geo.from_gguf(g)
    ids = [int(x) for x in a.ids.split(",")]
    layers = [int(x) for x in a.layers.split(",")]
    kinds = [G.kinds[l] for l in layers]
    # stitched layers that are KV-shared in the real model read the last earlier stitched layer of their kind
    first_real_shared = G.n_layers - len(G.sources)
    nsh = sum(1 for l in layers if l >= first_real_shared)
    first = len(layers) - nsh
    src = {layers[i]: layers[max(j for j in range(first) if kinds[j] == kinds[i])] for i in range(first, len(layers))}
    uniq = sorted(set(ids))
    remap = {t: i for i, t in enumerate(uniq)}
    small = [remap[t] for t in ids]
    # E2B lists one FFN width per layer, twice as wide in its KV-shared layers (use_double_wide_mlp)
    ff = g.md["gemma4.feed_forward_length"]
    ffs = [int(f) for f in ff] if isinstance(ff, (list, tuple)) else [int(ff)] * G.n_layers
    double = any(f == 2 * ffs[0] for f in ffs)

    kw = dict(
        vocab_size=len(uniq), hidden_size=G.hidden, intermediate_size=ffs[0],
        num_hidden_layers=len(layers), num_attention_heads=G.nh, num_key_value_heads=G.kvh, head_dim=G.hd,
        global_head_dim=G.ghd, hidden_activation="gelu_pytorch_tanh", rms_norm_eps=G.eps, sliding_window=G.window,
        layer_types=["sliding_attention" if k == "sliding" else "full_attention" for k in kinds],
        final_logit_softcapping=G.softcap or None, num_kv_shared_layers=nsh, attention_k_eq_v=G.k_eq_v,
        num_global_key_value_heads=G.gkvh or None,
        use_double_wide_mlp=double, tie_word_embeddings=True, pad_token_id=0,
        rope_parameters={"full_attention": {"rope_type": "proportional", "partial_rotary_factor": G.grot / G.ghd,
                                            "rope_theta": G.theta},
                         "sliding_attention": {"rope_type": "default", "rope_theta": G.theta_local}})
    if G.ple:
        cfg = Gemma4TextConfig(vocab_size_per_layer_input=len(uniq), hidden_size_per_layer_input=G.ple, **kw)
        cls = Gemma4ForCausalLM
    else:
        # the 12B is transformers' own gemma4_unified text model, which has no per-layer inputs at all
        from transformers import Gemma4UnifiedForCausalLM, Gemma4UnifiedTextConfig
        cfg, cls = Gemma4UnifiedTextConfig(**kw), Gemma4UnifiedForCausalLM
    cfg._attn_implementation = "eager"
    model = cls(cfg).float().eval()
    print(f"transformers {cls.__name__}")

    def put(name, arr):
        p = dict(model.named_parameters()).get(name)
        if p is None:
            p = dict(model.named_buffers())[name]
        t = torch.as_tensor(np.asarray(arr, np.float32).reshape(tuple(p.shape)))
        with torch.no_grad():
            p.copy_(t)

    emb_rows = g.get("token_embd.weight", rows=uniq)
    put("model.embed_tokens.weight", emb_rows)
    if G.ple:
        tok = g.get("per_layer_token_embd.weight", rows=uniq).reshape(len(uniq), -1, G.ple)[:, layers, :]
        put("model.embed_tokens_per_layer.weight", tok.reshape(len(uniq), -1))
        W = g.get("per_layer_model_proj.weight").reshape(-1, G.ple, G.hidden)[layers]
        put("model.per_layer_model_projection.weight", W.reshape(-1, G.hidden))
        put("model.per_layer_projection_norm.weight", g.get("per_layer_proj_norm.weight"))
    put("model.norm.weight", g.get("output_norm.weight"))
    for i, l in enumerate(layers):
        b, p = f"blk.{l}.", f"model.layers.{i}."
        m = {"input_layernorm.weight": "attn_norm.weight", "self_attn.q_proj.weight": "attn_q.weight",
             "self_attn.q_norm.weight": "attn_q_norm.weight", "self_attn.o_proj.weight": "attn_output.weight",
             "post_attention_layernorm.weight": "post_attention_norm.weight",
             "pre_feedforward_layernorm.weight": "ffn_norm.weight", "mlp.gate_proj.weight": "ffn_gate.weight",
             "mlp.up_proj.weight": "ffn_up.weight", "mlp.down_proj.weight": "ffn_down.weight",
             "post_feedforward_layernorm.weight": "post_ffw_norm.weight"}
        if G.ple:
            m.update({"per_layer_input_gate.weight": "inp_gate.weight", "per_layer_projection.weight": "proj.weight",
                      "post_per_layer_input_norm.weight": "post_norm.weight"})
        if g.has(b + "layer_output_scale.weight"):
            m["layer_scalar"] = "layer_output_scale.weight"
        if l not in src:
            m.update({"self_attn.k_proj.weight": "attn_k.weight", "self_attn.k_norm.weight": "attn_k_norm.weight"})
            if not (G.k_eq_v and kinds[i] == "full"):
                m["self_attn.v_proj.weight"] = "attn_v.weight"
        for hn, gn in m.items():
            put(p + hn, g.get(b + gn))

    states = {}
    hooks = [lyr.register_forward_hook(lambda mod, inp, outp, i=i: states.__setitem__(i, outp[0] if isinstance(outp, tuple) else outp))
             for i, lyr in enumerate(model.model.layers)]
    with torch.no_grad():
        out = model(input_ids=torch.tensor([small]), use_cache=False)
        last_hf = model.model(input_ids=torch.tensor([small]), use_cache=False).last_hidden_state[0].double().numpy()
    for hk in hooks:
        hk.remove()
    hf_logits = out.logits[0, -1].double().numpy()

    res, _ = R.forward(g, ids, G=G, want_logits=False, layers=layers, sources=src)
    hn = R.rms(res[-1][-1], g.get("output_norm.weight").reshape(-1), G.eps)
    rl = emb_rows @ hn
    if G.softcap:
        rl = G.softcap * np.tanh(rl / G.softcap)
    worst = 0.0
    print(f"slice layers {layers} (kinds {kinds}, shared {src}), {len(ids)} tokens")
    for i in range(len(layers)):
        b_ = states[i][0].double().numpy()
        rel = np.abs(res[i] - b_).max() / np.abs(b_).max()
        worst = max(worst, rel)
        print(f"  layer {layers[i]:>2}: max|replica - hf| / max|hf| = {rel:.2e}")
    last = R.rms(res[-1], g.get("output_norm.weight").reshape(-1), G.eps)
    rel = np.abs(last - last_hf).max() / np.abs(last_hf).max()
    worst = max(worst, rel)
    print(f"  final norm: {rel:.2e}")
    lrel = np.abs(rl - hf_logits).max() / np.abs(hf_logits).max()
    worst = max(worst, lrel)
    print(f"  logits over the prompt's {len(uniq)} tokens (soft-capped): {lrel:.2e}, argmax "
          f"replica {uniq[int(np.argmax(rl))]} hf {uniq[int(np.argmax(hf_logits))]}")
    ok = worst < 1e-4
    print("PASS" if ok else "FAIL", f"(worst {worst:.2e}; fp32 transformers against the fp64 replica)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
