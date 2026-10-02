r"""The fp64 reference over a whole chat prompt, layer-major, to the last position's logits.

A slice compare (make_decode.py + compare_decode.py) proves the kernels compute what the
reference computes, for a few layers and positions. It cannot say whether that is the
MODEL: a shared wrong assumption passes it. This runs the whole model on the CPU the way
replica_qwen35.py defines it -- every prompt token through layer 0, then layer 1, ..., each
layer's weights dequantized once -- and prints the reference's own next-token guesses,
then dumps what the engine can be compared against:

    <out>/ref_prompt_logits.bin        f32[vocab], the last prompt position
    <out>/ref_prompt_res{l}.bin        f32[hidden], layer l's output at the last position

    python open_kernels/model/replica_prompt.py --model-dir DIR "Explain what an NPU is in two sentences."
        [--layers N] [--out DIR] [--think]

The prompt is chat.py's Qwen form (think block closed unless --think), so
`open_qwen36_cli --ids <the printed ids> --max-tokens 1 --dump-logits <p>` gives the
engine's logits for the same position (<p>_p0.bin). Qwen3.5 dense only.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

from recipes.load import spec_from_model_dir  # noqa: E402
from q4nx import Q4NX  # noqa: E402
import replica_qwen35 as R35  # noqa: E402


def prompt_ids(tk, message: str, think: bool) -> list[int]:
    """chat.py's Qwen prompt: the user turn, the assistant opener, and a closed (or open) think block."""
    ids = tk.encode(f"<|im_start|>user\n{message}<|im_end|>\n<|im_start|>assistant\n", add_special_tokens=False).ids
    nl, nlnl = tk.encode("\n", add_special_tokens=False).ids, tk.encode("\n\n", add_special_tokens=False).ids
    th, eth = tk.token_to_id("<think>"), tk.token_to_id("</think>")
    return ids + ([th] + nl if think else [th] + nlnl + [eth] + nlnl)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("message")
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--layers", type=int, default=-1, help="stop after N layers (default: all)")
    ap.add_argument("--out", default=str(HERE / "out_prompt"))
    ap.add_argument("--think", action="store_true")
    a = ap.parse_args()

    from tokenizers import Tokenizer

    md, out = Path(a.model_dir), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    spec = spec_from_model_dir(md)
    if spec.family != "qwen35":
        sys.exit(f"replica_prompt: {spec.family} is not implemented (qwen35 only)")
    tk = Tokenizer.from_file(str(md / "tokenizer.json"))
    ids = prompt_ids(tk, a.message, a.think)
    print(f"prompt: {len(ids)} ids: {','.join(map(str, ids))}", flush=True)

    q = Q4NX(md / "model.q4nx")
    q.native_q8 = lambda n: False            # the q4_1 the packer writes, as make_decode's default for this size
    q.hidden = spec.hidden
    plain = q.matmul_w
    cache: dict = {}
    q.matmul_w = lambda name, o, i: cache[name] if name in cache else cache.setdefault(name, plain(name, o, i))

    T = len(ids)
    xs = [q.embed(t, spec.hidden).astype(np.float32) for t in ids]
    nl = spec.num_layers if a.layers < 0 else min(a.layers, spec.num_layers)
    t0 = time.time()
    for l in range(nl):
        cache.clear()
        conv = np.zeros((spec.conv_kernel - 1, spec.lin_qkv_dim))
        S = np.zeros((spec.lin_value_heads, spec.lin_value_dim, spec.lin_value_dim))
        K = np.zeros((0, spec.num_kv_heads, spec.head_dim))
        V = np.zeros((0, spec.num_kv_heads, spec.head_dim))
        for t in range(T):
            xs[t], conv, S, K, V = R35.layer_decode(q, spec, l, xs[t], conv, S, K, V, t)
            xs[t] = np.asarray(xs[t], np.float32)
        xs[-1].astype(np.float32).tofile(out / f"ref_prompt_res{l}.bin")
        print(f"  layer {l:2d} {spec.layer_types[l]:17s} |res| {np.linalg.norm(xs[-1]):10.3f}  ({time.time() - t0:.0f} s)",
              flush=True)
    _, logits = R35.final_logits(q, spec, xs[-1])
    logits = np.asarray(logits, np.float32)
    logits.tofile(out / "ref_prompt_logits.bin")
    top = np.argsort(-logits[:spec.real_vocab])[:10]
    print("top-10 next tokens:")
    for i in top:
        print(f"  {i:7d} {logits[i]:9.3f}  {tk.decode([int(i)])!r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
