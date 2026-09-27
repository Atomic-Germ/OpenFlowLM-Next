r"""klein_tokens: a prompt's token ids for src/open_diffusion's CLI (the engine takes ids).

    python utilities\dit-chain\klein_tokens.py "a red fox in fresh snow" ids.npy [--bundle C:\dev\klein-bundle]

Qwen3's chat template (one user turn, enable_thinking=False) through the model's
tokenizer.json, int64, unpadded (the engine pads). Needs `tokenizers`.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "open_kernels"))
import klein_pipeline as kp  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("prompt")
    ap.add_argument("out")
    ap.add_argument("--bundle", default=r"C:\dev\klein-bundle")
    a = ap.parse_args()
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(Path(a.bundle) / "tokenizer.json"))
    ids, n = kp.token_ids(tok, a.prompt)
    np.save(a.out, ids[:n])
    print(f"{n} tokens -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
