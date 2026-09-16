"""Run LFM2 end to end in fp64 on the CPU, straight out of a `.q4nx` container.

This is the oracle the short-conv kernels get compared against once they exist, and it is
what establishes -- with no NPU and no toolchain -- that the geometry, the layer schedule,
the conv tap order and the container's tensor names are all read right: a model wired up
wrong answers with noise.

    python open_kernels/model/lfm2_forward.py --model-dir "%USERPROFILE%\\.flm\\models\\LFM2-1.2B-NPU2" -n 6

About 30-40 s per token on the 1.2B; every projection is dequantized on the fly and thrown
away, so it is memory-light and slow on purpose. `--layers` stops after N layers and prints
the residual's norm instead, which is the cheap way to localize a disagreement.

Traces: OPEN-FAMILY-LFM2, OPEN-SHORT-CONV-REF (specs/open-engine/spec.md).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
from q4nx import Q4NX, bf16_to_f32  # noqa: E402
from recipes.load import tokenizer_vocab  # noqa: E402
from recipes.spec import FULL, ModelSpec  # noqa: E402
import replica_dense as RD  # noqa: E402
import replica_lfm2 as RL  # noqa: E402

# LFM2 containers name the embedding table `token_embd`, not `embed_tokens`.
EMBED = "model.token_embd.weight"


def spec_of(model_dir: Path) -> ModelSpec:
    """The spec, without recipes.load.spec_from_model_dir: that one asks for the family's
    recipe to narrow the quant map, and lfm2 has no recipe yet (families.NOT_IMPLEMENTED)."""
    cfg = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    return ModelSpec.from_hf_config(cfg, real_vocab=tokenizer_vocab(model_dir / "tokenizer.json"))


def initial_state(spec):
    st = {}
    for l, t in enumerate(spec.layer_types):
        zero = np.zeros((0, spec.num_kv_heads, spec.head_dim))
        st[l] = (zero, zero.copy()) if t == FULL else np.zeros(RL.conv_state_shape(spec))
    return st


def step(m, spec, state, token: int, pos: int, tbl, layers: int | None = None):
    """One token through the stack. Returns the residual after the last layer run."""
    x = bf16_to_f32(tbl[token]).astype(np.float64)
    for l, t in enumerate(spec.layer_types[:layers] if layers else spec.layer_types):
        if t == FULL:
            K, V = state[l]
            x, K, V = RL.attn_decode(m, spec, l, x, K, V, pos)
            state[l] = (K, V)
        else:
            x, state[l] = RL.short_conv_decode(m, spec, l, x, state[l])
    return x


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--prompt", default="What is the capital of France?")
    ap.add_argument("--raw", action="store_true", help="take --prompt as already templated")
    ap.add_argument("-n", "--tokens", type=int, default=4)
    ap.add_argument("--layers", type=int, default=0, help="stop after N layers and print norms")
    a = ap.parse_args(argv)

    md = Path(a.model_dir)
    spec = spec_of(md)
    m = Q4NX(md / "model.q4nx")
    tbl = np.frombuffer(m.raw(EMBED), np.uint16).reshape(m.tensors[EMBED]["shape"])
    print(f"{spec.family} hidden {spec.hidden} layers {spec.num_layers} "
          f"({sum(t == FULL for t in spec.layer_types)} attention, "
          f"{spec.num_layers - sum(t == FULL for t in spec.layer_types)} short-conv, "
          f"{spec.conv_kernel} taps)")

    from tokenizers import Tokenizer
    tk = Tokenizer.from_file(str(md / "tokenizer.json"))
    text = a.prompt if a.raw else (
        f"<|startoftext|><|im_start|>user\n{a.prompt}<|im_end|>\n<|im_start|>assistant\n")
    ids = tk.encode(text, add_special_tokens=False).ids

    state = initial_state(spec)
    if a.layers:
        x = step(m, spec, state, ids[0], 0, tbl, a.layers)
        print(f"after {a.layers} layers: |x| = {np.linalg.norm(x):.6f}  mean {x.mean():.6e}")
        return 0

    out: list[int] = []
    for pos in range(len(ids) + a.tokens - 1):
        tok = ids[pos] if pos < len(ids) else out[-1]
        x = step(m, spec, state, tok, pos, tbl)
        if pos < len(ids) - 1:
            continue
        hn = RD.rms(x, spec.norm_eps) * m.bf16("model.norm.weight")
        logits = RD.lmhead_q4_logits(m, hn, spec)[:spec.real_vocab]
        out.append(int(np.argmax(logits)))
        top = [int(i) for i in np.argsort(-logits)[:5]]
        print(f"pos {pos}: {out[-1]} {tk.decode(out[-1:])!r}  top5 {top} "
              f"{[tk.decode([i]) for i in top]!r}".encode("ascii", "backslashreplace").decode(),
              flush=True)
        if len(out) >= a.tokens:
            break
    print("generated:", repr(tk.decode(out)).encode("ascii", "backslashreplace").decode())
    return 0


if __name__ == "__main__":
    sys.exit(main())
