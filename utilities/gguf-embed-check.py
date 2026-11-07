#!/usr/bin/env python3
"""What an embedding GGUF contains, and what the NpuEmbeddings packer would see.

    python utilities/gguf-embed-check.py model.gguf
    python utilities/gguf-embed-check.py model.gguf --vocab base/vocab.txt

The NpuEmbeddings packers read a HuggingFace checkpoint (F32 safetensors with
HF tensor names, config.json, vocab.txt, 1_Pooling/config.json). A GGUF holds
the same weights in llama.cpp's names and block layouts. `oflm add` installs
the GGUF; on first load the fork-owned adapter
(src/open_npue_adapter/npue_embedding.cpp) stages a checkpoint from it and
calls npue::prepare_model_auto() unchanged.

This tool answers, before an install, the three questions that decide whether
that will work:

  * is the architecture one the adapter maps (bert, nomic-bert)?
  * does every tensor the packer needs have an HF name?
  * does the GGUF carry a pooling mode, and is the reconstructed vocab.txt
    (used when the base repo cannot be reached) the same as the author's?

It parses the file with oflm-add's own GGUF reader, so it cannot drift from
what `oflm add` sees. Exit status is 0 when the file is packable, 1 when not.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "utilities" / "oflm-add"))

from oflm_add import parse_gguf_inventory  # noqa: E402

# Mirrors hf_embed_name() in src/open_npue_adapter/npue_embedding.cpp.
BERT_TOP = {
    "token_embd.weight": "embeddings.word_embeddings.weight",
    "position_embd.weight": "embeddings.position_embeddings.weight",
    "token_types.weight": "embeddings.token_type_embeddings.weight",
    "token_embd_norm.weight": "embeddings.LayerNorm.weight",
    "token_embd_norm.bias": "embeddings.LayerNorm.bias",
}
BERT_LAYER = {
    "attn_q.weight": "attention.self.query.weight",
    "attn_q.bias": "attention.self.query.bias",
    "attn_k.weight": "attention.self.key.weight",
    "attn_k.bias": "attention.self.key.bias",
    "attn_v.weight": "attention.self.value.weight",
    "attn_v.bias": "attention.self.value.bias",
    "attn_output.weight": "attention.output.dense.weight",
    "attn_output.bias": "attention.output.dense.bias",
    "attn_output_norm.weight": "attention.output.LayerNorm.weight",
    "attn_output_norm.bias": "attention.output.LayerNorm.bias",
    "ffn_up.weight": "intermediate.dense.weight",
    "ffn_up.bias": "intermediate.dense.bias",
    "ffn_down.weight": "output.dense.weight",
    "ffn_down.bias": "output.dense.bias",
    "layer_output_norm.weight": "output.LayerNorm.weight",
    "layer_output_norm.bias": "output.LayerNorm.bias",
}
NOMIC_TOP = {
    "token_embd.weight": "embeddings.word_embeddings.weight",
    "token_types.weight": "embeddings.token_type_embeddings.weight",
    "token_embd_norm.weight": "emb_ln.weight",
    "token_embd_norm.bias": "emb_ln.bias",
}
NOMIC_LAYER = {
    "attn_qkv.weight": "attn.Wqkv.weight",
    "attn_output.weight": "attn.out_proj.weight",
    "attn_output_norm.weight": "norm1.weight",
    "attn_output_norm.bias": "norm1.bias",
    "ffn_up.weight": "mlp.fc11.weight",
    "ffn_gate.weight": "mlp.fc12.weight",
    "ffn_down.weight": "mlp.fc2.weight",
    "layer_output_norm.weight": "norm2.weight",
    "layer_output_norm.bias": "norm2.bias",
}

# Tensors each packer actually reads (npue_pack.cpp). A missing one is a hard
# failure at pack time, so name it here rather than letting the install finish.
REQUIRED = {
    "bert": {"token_embd.weight", "position_embd.weight", "token_types.weight",
             "token_embd_norm.weight", "token_embd_norm.bias"}
    | {f"blk.{i}.{s}" for i in range(64) for s in BERT_LAYER},
    "nomic-bert": {"token_embd.weight", "token_types.weight",
                   "token_embd_norm.weight", "token_embd_norm.bias"}
    | {f"blk.{i}.{s}" for i in range(64) for s in NOMIC_LAYER},
}

GGML_TYPE_NAMES = {
    0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1",
    8: "Q8_0", 9: "Q8_1", 10: "Q2_K", 11: "Q3_K", 12: "Q4_K",
    13: "Q5_K", 14: "Q6_K", 15: "Q8_K", 16: "IQ2_XXS", 17: "IQ2_XS",
    18: "IQ3_XXS", 19: "IQ1_S", 20: "IQ4_NL", 21: "IQ3_S",
    22: "IQ2_S", 23: "IQ4_XS", 24: "I8", 25: "I16", 26: "I32",
    27: "I64", 28: "F64", 29: "IQ1_M", 30: "BF16",
}


def hf_name(arch: str, name: str) -> str | None:
    bert = arch == "bert"
    top, layer = (BERT_TOP, BERT_LAYER) if bert else (NOMIC_TOP, NOMIC_LAYER)
    if not name.startswith("blk."):
        return top.get(name)
    idx, _, suffix = name[4:].partition(".")
    if not idx.isdigit():
        return None
    mapped = layer.get(suffix)
    if mapped is None:
        return None
    return (f"encoder.layer.{idx}." if bert else f"encoder.layers.{idx}.") + mapped


def restore_wordpiece(t: str) -> str:
    """GGUF WordPiece token -> the piece a BERT vocab.txt holds (see the C++)."""
    if len(t) >= 3 and t[0] == "[" and t[-1] == "]":
        inner = t[1:-1]
        if inner in ("PAD", "UNK", "CLS", "SEP", "MASK") or inner.startswith("unused"):
            return t
    if t.startswith("\u2581"):
        return t[1:]
    return "##" + t


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("gguf", help="the embedding GGUF file")
    ap.add_argument("--vocab", help="the base repo's vocab.txt, to check the reconstruction")
    a = ap.parse_args()

    path = Path(a.gguf)
    inv = parse_gguf_inventory(path.read_bytes())
    if not inv:
        print(f"{path}: not a parseable GGUF", file=sys.stderr)
        return 1
    kv, tensors = inv
    arch = str(kv.get("general.architecture") or "")
    names = [t[0] if isinstance(t, (list, tuple)) else t for t in tensors]

    print(f"file           : {path}")
    print(f"architecture   : {arch or '(none)'}")
    if arch not in ("bert", "nomic-bert"):
        print("VERDICT: REFUSED -- the NpuEmbeddings adapter maps bert and "
              "nomic-bert only", file=sys.stderr)
        return 1

    def n(key, default="?"):
        v = kv.get(f"{arch}.{key}")
        return v if v is not None else default

    print(f"hidden         : {n('embedding_length')}")
    print(f"layers         : {n('block_count')}")
    print(f"heads          : {n('attention.head_count')}")
    print(f"feed_forward   : {n('feed_forward_length')}")
    print(f"context        : {n('context_length')}")
    print(f"pooling_type   : {n('pooling_type')} "
          f"({'mean' if n('pooling_type') == 1 else 'cls' if n('pooling_type') == 2 else 'UNSUPPORTED'})")
    print(f"tokenizer      : {kv.get('tokenizer.ggml.model') or '(none)'}")
    print(f"tensors        : {len(names)}")

    unmapped = [x for x in names if hf_name(arch, x) is None]
    if unmapped:
        print(f"not packed     : {', '.join(sorted(unmapped))}")

    # Required tensors, per layer count the file states.
    layers = int(n("block_count", 0) or 0)
    missing = []
    for i in range(layers):
        for suffix in (BERT_LAYER if arch == "bert" else NOMIC_LAYER):
            t = f"blk.{i}.{suffix}"
            if t not in names:
                # biases are present on every BERT layer and absent on nomic's
                # qkv/ffn operands by design; the packer zero-fills nomic's.
                if arch == "nomic-bert":
                    continue
                missing.append(t)
    if arch == "bert":
        for t in ("token_embd.weight", "position_embd.weight", "token_types.weight",
                  "token_embd_norm.weight", "token_embd_norm.bias"):
            if t not in names:
                missing.append(t)
    else:
        for t in ("token_embd.weight", "token_types.weight",
                  "token_embd_norm.weight", "token_embd_norm.bias"):
            if t not in names:
                missing.append(t)

    types = sorted({GGML_TYPE_NAMES.get(t[1], str(t[1])) for t in tensors
                    if isinstance(t, (list, tuple)) and len(t) > 1})
    print(f"tensor types   : {', '.join(types)}")

    rc = 0
    if missing:
        print(f"MISSING        : {', '.join(missing)} -- the packer will refuse "
              "this file", file=sys.stderr)
        rc = 1

    if a.vocab:
        ref = Path(a.vocab).read_text(encoding="utf-8").split("\n")
        if ref and ref[-1] == "":
            ref.pop()
        tokens = kv.get("tokenizer.ggml.tokens") or []
        got = [restore_wordpiece(t) for t in tokens]
        same = got == ref
        print(f"vocab.txt      : reconstructed {len(got)} pieces, base repo has "
              f"{len(ref)} -> {'IDENTICAL' if same else 'DIFFERENT'}")
        if not same:
            for i, (x, y) in enumerate(zip(got, ref)):
                if x != y:
                    print(f"  first diff at {i}: reconstructed {x!r}, base {y!r}")
                    break
            rc = 1

    if rc == 0:
        print("VERDICT: packable (the adapter stages a checkpoint from this file)")
    return rc


if __name__ == "__main__":
    sys.exit(main())
