"""Ground-truth oracle for the open embedding engine (E8 reference in flm-test).

Uses the dependency-free numpy reference (gemma3_reference) — no torch needed —
to emit reference embedding vectors for the flm-test embedding corpus, all
embedded with the task_query prompt that flm's server always uses.

Usage:
    reference_venv/bin/python src/open_embedding/tools/oracle.py \
        --out src/open_embedding/oracle/embeddings.json
"""

import argparse
import json
import sys
from pathlib import Path

from gemma3_reference import ReferenceModel

CORPUS = {
    "sample": "The embedding model should capture the meaning of this sentence.",
    "batch_hello": "Hello, world!",
    "batch_flm": "FastFlowLM is a local inference server.",
    "batch_fox": "The quick brown fox jumps over the lazy dog.",
    "fox": "The quick brown fox jumps over the lazy dog.",
    "cat": "cat",
    "kitten": "kitten",
    "car": "car",
    "ocean": "ocean",
    "sea": "sea",
    "desert": "desert",
    "long": (
        "The capital of France is Paris. It is a major global city "
        "with a rich history of art, culture and architecture. "
        "A fast brown fox jumps over the lazy dog near the Seine. "
        "Thousands of flowers and strawberry fields fill the countryside "
        "between the ocean and the desert plains."
    ),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    model = ReferenceModel()
    out = {}
    for name, text in CORPUS.items():
        vec = model.embed(text, "query")
        out[name] = {
            "text": text,
            "prompt": "task: search result | query: ",
            "dim": int(vec.shape[0]),
            "embedding": [round(float(x), 6) for x in vec],
        }
        print(f"{name:14s} dim={vec.shape[0]}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    sys.exit(main())