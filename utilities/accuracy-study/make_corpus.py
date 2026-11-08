r"""Token-id corpus for the prefill accuracy study: prose, code and chat prompts, 128-2048 tokens, fixed seed.

    python make_corpus.py --model <Granite dir> --out <dir> [--count 60]

Writes <out>/prompts.txt (one comma-separated id line a prompt) and <out>/meta.json (kind, source, length per prompt).
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
LENGTHS = [128] * 12 + [256] * 12 + [512] * 14 + [1024] * 12 + [2048] * 8


def sources() -> dict[str, list[Path]]:
    prose = [p for pat in ("docs/**/*.md", "specs/**/*.md", "README.md", "*.md", "open_kernels/**/*.md") for p in REPO.glob(pat)]
    code = [p for pat in ("src/**/*.cpp", "src/**/*.hpp", "open_kernels/**/*.py", "utilities/**/*.py") for p in REPO.glob(pat)]
    keep = lambda ps: sorted({p for p in ps if p.is_file() and p.stat().st_size > 6000})
    return {"prose": keep(prose), "code": keep(code)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--count", type=int, default=60)
    ap.add_argument("--seed", type=int, default=20261015)
    a = ap.parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(a.model))
    rng = random.Random(a.seed)
    src = sources()
    lengths = LENGTHS[:]
    rng.shuffle(lengths)
    lengths = lengths[: a.count]
    kinds = ["prose", "code", "chat"]
    asks = ["Summarize the text above in three sentences.", "What are the main points?", "Explain this to a new engineer.",
            "Find any mistakes in the text above.", "Rewrite the first paragraph more clearly."]
    out, meta = [], []
    for i, n in enumerate(lengths):
        kind = kinds[i % 3]
        pool = src["code" if kind == "code" else "prose"]
        path = rng.choice(pool)
        text = path.read_text(encoding="utf-8", errors="ignore")
        if kind == "chat":
            start = rng.randrange(0, max(1, len(text) - 4 * n))
            msg = [{"role": "user", "content": text[start:start + 5 * n] + "\n\n" + rng.choice(asks)}]
            ids = tok.encode(tok.apply_chat_template(msg, add_generation_prompt=True, tokenize=False), add_special_tokens=False)
            ids = ids[: n - 8] + ids[-8:] if len(ids) > n else ids
        else:
            ids = tok.encode(text, add_special_tokens=False)
            start = rng.randrange(0, max(1, len(ids) - n))
            ids = ids[start:start + n]
        if len(ids) < n // 2:
            continue
        out.append(ids)
        meta.append({"k": len(out) - 1, "kind": kind, "source": str(path.relative_to(REPO)), "tokens": len(ids)})
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "prompts.txt").write_text("\n".join(",".join(map(str, x)) for x in out) + "\n")
    (a.out / "meta.json").write_text(json.dumps(meta, indent=1))
    print(f"{len(out)} prompts, {sum(len(x) for x in out)} tokens -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
