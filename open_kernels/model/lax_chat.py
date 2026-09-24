r"""Chat with Qwen3.6-35B-A3B on the NPU: the whole 40-layer token is ONE lax runlist submit.

    python open_kernels/model/lax_chat.py --lax-l BUILD_L --lax-a BUILD_A --designs DESIGNS \
        [--model-dir DIR] [--workers 8] ["one prompt" ...]

Every weight -- the 40 layer pools and consts, the lm_head pool, the embedding table -- is packed
at startup from the model's own model.q4nx (model/lax_pack.py) and streamed over a pipe straight
into the harness's XRT buffers: nothing pre-packed is read, nothing is written, so it survives a
reboot. DESIGNS holds the ln / lm_head_q8 builds. The old path over pre-packed files is still
there: `--out DEC --embed EMB`, DEC a `make_decode.py --requant --layers 40` output (the per-layer
pools and consts), EMB the model's bf16 embedding table as a raw file. With prompts on the command line
it answers each in turn and exits; without, it reads prompts from stdin. The conversation is one
device session: the harness (`run_kernel -`) keeps the KV cache and the DeltaNet state, so every
turn only feeds its own new tokens. Greedy decoding, thinking off; a turn ends at <|im_end|> or
--max-new tokens.
"""
from __future__ import annotations

import time

T_START = time.time()

import argparse  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from lax_decode_cfg import HEAD, position, setup, setup_packed, with_fd  # noqa: E402
from lax_pack import Model, Npu, Stream  # noqa: E402

DEFAULT_MODEL_DIR = Path.home() / ".config" / "flm" / "models" / "Qwen3.6-35B-A3B-NPU2"
IM_START, IM_END, EOT = 248045, 248046, 248044
MAX_CTX = 4096                                   # the KV / ptab buffers' rows (make_decode --max-ctx)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lax-l", required=True)
    ap.add_argument("--lax-a", required=True)
    ap.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR), help="model.q4nx + tokenizer.json")
    ap.add_argument("--designs", default=str(HERE.parent / "designs"), help="where the ln / lm_head builds live")
    ap.add_argument("--workers", type=int, default=8, help="packing processes at startup")
    ap.add_argument("--out", default=None, help="old path: make_decode.py --requant --layers 40 output "
                    "(pre-packed pools and consts; needs --embed)")
    ap.add_argument("--embed", default=None, help="old path: bf16 embedding table [vocab, hidden], raw")
    ap.add_argument("--max-new", type=int, default=256)
    ap.add_argument("--vocab", type=int, default=248070)
    ap.add_argument("prompts", nargs="*")
    a = ap.parse_args()

    lax_l, lax_a = Path(a.lax_l).resolve(), Path(a.lax_a).resolve()
    if a.out:
        lines, ng, _ = setup(Path(a.out).resolve(), lax_l, lax_a, 40)
        emb = Path(a.embed).resolve()
        lines.append(f"buf embed {emb.stat().st_size} {emb}")
        stream = None
    else:
        # pack at startup: the workers fork here, before the tokenizer's threads or the harness
        model = Model(Path(a.model_dir))
        lines, ng, _, items = setup_packed(model, lax_l, lax_a, 40, Path(a.designs).resolve(), 0)
        stream = Stream(model, items, a.workers)
        lines = with_fd(lines, stream.rfd)
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(Path(a.model_dir) / "tokenizer.json"))
    print(f"[{'packing and loading' if stream else 'loading'} ~22 GB onto the NPU...]", file=sys.stderr, flush=True)
    npu = Npu(lines, stream)
    print(f"[ready in {time.time() - T_START:.1f} s since process start]", file=sys.stderr, flush=True)

    pos = 0
    prompts = iter(a.prompts) if a.prompts else (l.rstrip("\n") for l in sys.stdin)
    for text in prompts:
        if not text.strip():
            continue
        # Qwen3.6's template, thinking off: an empty think block opens the answer
        turn = (f"<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n")
        ids = tok.encode(turn).ids
        if pos:
            ids = tok.encode("\n").ids + ids         # after the previous turn's <|im_end|>
        if pos + len(ids) + a.max_new + 1 > MAX_CTX:
            print(f"[context full ({pos} of {MAX_CTX} positions)]", file=sys.stderr)
            break
        t1 = time.time()
        for i, t in enumerate(ids):                   # the prompt: no head until its last token
            npu.send(position(pos, ng, str(t)) + (HEAD + [f"greedy logits {a.vocab}"] if i == len(ids) - 1 else []))
            pos += 1
        nxt = int(npu.wait("greedy").split()[1])
        t2 = time.time()
        out, shown = [], ""
        while nxt not in (IM_END, EOT) and len(out) < a.max_new:
            out.append(nxt)
            s = tok.decode(out)
            print(s[len(shown):], end="", flush=True)
            shown = s
            npu.send(position(pos, ng, "last") + HEAD + [f"greedy logits {a.vocab}"])
            pos += 1
            nxt = int(npu.wait("greedy").split()[1])
        t3 = time.time()
        npu.send(position(pos, ng, "last"))       # the end-of-turn token enters the cache too
        pos += 1
        print(f"\n[prompt {len(ids)} tok in {t2 - t1:.2f} s, {len(out)} tok in {t3 - t2:.2f} s = "
              f"{len(out) / max(t3 - t2, 1e-9):.1f} tok/s, context {pos}]", file=sys.stderr, flush=True)
    npu.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
