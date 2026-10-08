r"""klein_tokens: a prompt's token ids for src/open_diffusion's CLI (the engine takes ids).

    python utilities\dit-chain\klein_tokens.py "a red fox in fresh snow" ids.npy [--bundle <model dir>]
    python utilities\dit-chain\klein_tokens.py --goldens specs\open-diffusion\tests\token_goldens.json

Qwen3's chat template (one user turn, enable_thinking=False) through the model's
tokenizer.json, int64, unpadded (the engine pads). Needs `tokenizers`.

--goldens writes OPEN-DIFFUSION-TOKENS' fixture: klein_quant_study's 8 prompts and the edge
cases below, each with the ids oflm's prompt path must produce (src/open_diffusion/prompt.cpp).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "open_kernels"))
import klein_pipeline as kp  # noqa: E402

EDGE_PROMPTS = [
    "Café crème on a zinc bar, 北京 street sign, \U0001F98A sticker",
    "a prompt that names <|endoftext|> inside it",     # the pad token, as prompt text
    "",
    " ".join(f"word{i}" for i in range(400)),          # past 512 once templated: truncated
]


def goldens(tok, out: Path) -> None:
    sys.path.insert(0, str(ROOT / "utilities" / "dit-ref"))
    from klein_quant_study import PROMPTS
    cases = []
    for p in PROMPTS + EDGE_PROMPTS:
        ids, n = kp.token_ids(tok, p)
        cases.append({"prompt": p, "ids": [int(i) for i in ids[:n]]})
    out.write_text(json.dumps({"max_tokens": kp.L_TXT, "cases": cases}, ensure_ascii=False, indent=1)
                   + "\n", encoding="utf-8")
    print(f"{len(cases)} cases -> {out} (longest {max(len(c['ids']) for c in cases)} ids)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("prompt", nargs="?")
    ap.add_argument("out", nargs="?")
    ap.add_argument("--bundle", default=str(Path.home() / (".oflm" if os.name == "nt" else ".config/oflm") / "models"
                                         / "FLUX.2-klein-4B-NPU2"),
                    help="the model directory (its tokenizer.json)")
    ap.add_argument("--goldens", default=None, help="write the OPEN-DIFFUSION-TOKENS fixture here")
    a = ap.parse_args()
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(Path(a.bundle) / "tokenizer.json"))
    if a.goldens:
        goldens(tok, Path(a.goldens))
        return 0
    if a.prompt is None or a.out is None:
        ap.error("a prompt and an output file, or --goldens")
    ids, n = kp.token_ids(tok, a.prompt)
    np.save(a.out, ids[:n])
    print(f"{n} tokens -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
