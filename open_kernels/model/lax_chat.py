r"""Chat with Qwen3.6-35B-A3B on the NPU: the whole 40-layer token is ONE lax runlist submit.

    python open_kernels/model/lax_chat.py --out DEC --lax-l BUILD_L --lax-a BUILD_A --embed EMB \
        [--model-dir DIR] ["one prompt" ...]

DEC is a `make_decode.py --requant --layers 40` output (the per-layer pools and consts), EMB the
model's bf16 embedding table as a raw file (lax_decode_cfg.py). With prompts on the command line
it answers each in turn and exits; without, it reads prompts from stdin. The conversation is one
device session: the harness (`run_kernel -`) keeps the KV cache and the DeltaNet state, so every
turn only feeds its own new tokens. Greedy decoding, thinking off; a turn ends at <|im_end|> or
--max-new tokens.
"""
from __future__ import annotations

import argparse
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from lax_decode_cfg import HEAD, position, setup  # noqa: E402

DEFAULT_MODEL_DIR = Path.home() / ".config" / "flm" / "models" / "Qwen3.6-35B-A3B-NPU2"
IM_START, IM_END, EOT = 248045, 248046, 248044
HARNESS = HERE.parent / "harness" / "build" / "run_kernel"
MAX_CTX = 4096                                   # the KV / ptab buffers' rows (make_decode --max-ctx)


class Npu:
    """A `run_kernel -` session: write program lines, read what they print."""

    def __init__(self, lines: list[str]):
        self.p = subprocess.Popen([str(HARNESS), "-"], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  text=True, bufsize=1)
        # drain stdout on a thread: a long prompt is written before anything is read back, and
        # a full stdout pipe would otherwise block the harness while we block on its stdin
        self.q: queue.Queue = queue.Queue()
        threading.Thread(target=self._read, daemon=True).start()
        self.send(lines + ["tick"])
        self.wait("tick")

    def send(self, lines):
        self.p.stdin.write("\n".join(lines) + "\n")
        self.p.stdin.flush()

    def _read(self):
        for line in self.p.stdout:
            self.q.put(line)
        self.q.put(None)

    def wait(self, prefix: str) -> str:
        """The next output line starting with `prefix` (an ERROR line raises)."""
        while (line := self.q.get()) is not None:
            if line.startswith(prefix):
                return line
            if line.startswith(("ERROR", "RUN FAILED")):
                raise RuntimeError(line.strip())
        raise RuntimeError(f"harness exited (code {self.p.wait()})")

    def close(self):
        self.p.stdin.close()
        self.p.wait()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="make_decode.py --requant --layers 40 output")
    ap.add_argument("--lax-l", required=True)
    ap.add_argument("--lax-a", required=True)
    ap.add_argument("--embed", required=True, help="bf16 embedding table [vocab, hidden], raw")
    ap.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR), help="for tokenizer.json")
    ap.add_argument("--max-new", type=int, default=256)
    ap.add_argument("--vocab", type=int, default=248070)
    ap.add_argument("prompts", nargs="*")
    a = ap.parse_args()

    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(Path(a.model_dir) / "tokenizer.json"))
    lines, ng, _ = setup(Path(a.out).resolve(), Path(a.lax_l).resolve(), Path(a.lax_a).resolve(), 40)
    emb = Path(a.embed).resolve()
    t0 = time.time()
    print(f"[loading ~{sum(f.stat().st_size for f in Path(a.out).glob('consts_*.bin')) // 2**30 + 21} GB "
          f"onto the NPU...]", file=sys.stderr, flush=True)
    npu = Npu(lines + [f"buf embed {emb.stat().st_size} {emb}"])
    print(f"[ready in {time.time() - t0:.1f} s]", file=sys.stderr, flush=True)

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
