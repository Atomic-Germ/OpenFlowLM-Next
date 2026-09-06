"""Talk to the open Qwen3.6 engine without the FLM app: tokenize a user message
the way the app's chat path does, run open_qwen36_cli, detokenize as tokens
stream back.

    python src/open_qwen36/chat.py "Explain what an NPU is in two sentences." [--max-tokens 64]
        [--layers N] [--model DIR] [--kernels DIR] [--think]

The app (modeling_qwen3_6_moe.cpp) renders the chat template, then feeds
<think> \\n\\n </think> \\n\\n itself before sampling when thinking is off; this
mirrors that so the engine sees the same ids the app would give it.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from tokenizers import Tokenizer

HERE = Path(__file__).resolve().parent
def special_ids(tk):
    """<|im_start|>, <|im_end|>, <|endoftext|>, <think>, </think> by name; the newline ids by encoding."""
    ids = {t: tk.token_to_id(t) for t in ("<|im_start|>", "<|im_end|>", "<|endoftext|>", "<think>", "</think>")}
    missing = [t for t, i in ids.items() if i is None]
    if missing:
        raise SystemExit(f"tokenizer lacks {missing}")
    nl = tk.encode(chr(10), add_special_tokens=False).ids
    nlnl = tk.encode(chr(10) * 2, add_special_tokens=False).ids
    return ids["<|im_start|>"], ids["<|im_end|>"], ids["<|endoftext|>"], ids["<think>"], ids["</think>"], nl, nlnl


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("message")
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--layers", type=int, default=-1)
    ap.add_argument("--max-ctx", type=int, default=4096)
    ap.add_argument("--model", default=os.environ.get("FLM_MODEL_DIR", str(Path.home() / ".flm" / "models" / "Qwen3.6-35B-A3B-NPU2")))
    ap.add_argument("--kernels", default=str(HERE.parent / "xclbins" / "Qwen3.6-35B-A3B-NPU2" / "open_kernels"))
    ap.add_argument("--exe", default=str(HERE / "out" / "open_qwen36_cli.exe"))
    ap.add_argument("--think", action="store_true", help="let the model think (the app's enable_think)")
    ap.add_argument("--twice", action="store_true")
    a = ap.parse_args()

    tk = Tokenizer.from_file(str(Path(a.model) / "tokenizer.json"))

    def ids_of(*names):
        """The named tokens' ids; a family whose tokenizer lacks one would run past EOS."""
        got = {n: tk.token_to_id(n) for n in names}
        missing = [n for n, i in got.items() if i is None]
        if missing:
            raise SystemExit(f"tokenizer lacks {missing}")
        return [got[n] for n in names]

    if tk.token_to_id("<start_of_turn>") is not None:
        # Gemma: <bos><start_of_turn>user\n...<end_of_turn>\n<start_of_turn>model\n
        prompt = f"<bos><start_of_turn>user\n{a.message}<end_of_turn>\n<start_of_turn>model\n"
        ids = tk.encode(prompt, add_special_tokens=False).ids
        IM_END, EOT = ids_of("<end_of_turn>", "<eos>")
    elif tk.token_to_id("<|start_header_id|>") is not None:
        # Llama 3: <|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n...<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n
        prompt = (f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{a.message}<|eot_id|>"
                  f"<|start_header_id|>assistant<|end_header_id|>\n\n")
        ids = tk.encode(prompt, add_special_tokens=False).ids
        IM_END, EOT = ids_of("<|eot_id|>", "<|end_of_text|>")
    elif tk.token_to_id("<|startoftext|>") is not None and tk.token_to_id("<|extra_0|>") is not None:
        # HunYuan (chat_template.jinja): a user turn is <|startoftext|>...<|extra_0|>, the reply ends <|eos|>
        prompt = f"<|startoftext|>{a.message}<|extra_0|>"
        ids = tk.encode(prompt, add_special_tokens=False).ids
        IM_END, EOT = ids_of("<|eos|>", "<|endoftext|>")
    else:
        prompt = f"<|im_start|>user\n{a.message}<|im_end|>\n<|im_start|>assistant\n"
        IM_START, IM_END, EOT, THINK, END_THINK, NL, NLNL = special_ids(tk)
        ids = tk.encode(prompt, add_special_tokens=False).ids
        ids += [THINK] + NL if a.think else [THINK] + NLNL + [END_THINK] + NLNL
    print(f"prompt: {len(ids)} ids", file=sys.stderr)

    cmd = [a.exe, "--model", a.model, "--kernels", a.kernels, "--ids", ",".join(map(str, ids)),
           "--max-tokens", str(a.max_tokens), "--max-ctx", str(a.max_ctx)]
    if a.layers > 0:
        cmd += ["--layers", str(a.layers)]
    if a.twice:
        cmd.append("--twice")
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=sys.stderr, text=True, bufsize=1)
    out, text = [], ""
    for line in p.stdout:
        line = line.strip()
        if line.startswith("token "):
            t = int(line.split()[1])
            if t in (IM_END, EOT):
                print("\n[eos]", file=sys.stderr)
                break
            out.append(t)
            # decode incrementally so partial UTF-8 sequences resolve
            new = tk.decode(out)
            sys.stdout.write(new[len(text):])
            sys.stdout.flush()
            text = new
        elif line == "DONE":
            pass
    p.wait()
    print()
    print(f"{len(out)} tokens: {out}", file=sys.stderr)
    return p.returncode


if __name__ == "__main__":
    sys.exit(main())
