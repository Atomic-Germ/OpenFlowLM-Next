#!/usr/bin/env python3
"""Generate weights_manifest.json for the open embedding engine.

The engine loads gemma3 weights straight out of the safetensors files (no
copy/conversion): the manifest records the absolute file, raw byte offset and
shape of every fp32 tensor. Textures of the pipeline:

    model.safetensors          Gemma3TextModel body (embed_tokens, layers, norm)
    weights/2_Dense.safetensors  contrastive head Dense 768 -> 3072
    weights/3_Dense.safetensors  contrastive head Dense 3072 -> 768

Output goes into FLM's model directory, next to config.json, so the engine can
be pointed at the model dir just like the closed path.
"""

import argparse
import json
import struct
import sys
from pathlib import Path

HF_SNAP = Path(
    "/home/atomic-germ/.cache/huggingface/hub/models--google--embeddinggemma-300m/"
    "snapshots/57c266a740f537b4dc058e1b0cda161fd15afa75"
)
ORACLE_WEIGHTS = Path(__file__).resolve().parent.parent / "oracle" / "weights"


def safetensors_index(path: Path) -> dict[str, dict]:
    """Header with tensor offsets converted to ABSOLUTE file positions."""
    base = None
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n).decode())
        base = 8 + n
    out = {}
    for name, meta in hdr.items():
        if name == "__metadata__":
            continue
        out[name] = {"file": str(path), "offset": base + meta["data_offsets"][0], "shape": meta["shape"]}
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True, help="flm model dir to receive weights_manifest.json")
    parser.add_argument("--base", default=str(HF_SNAP / "model.safetensors"))
    parser.add_argument("--dense-dir", default=str(ORACLE_WEIGHTS))
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    if not (model_dir / "config.json").exists():
        parser.error(f"no config.json in {model_dir}")

    tensors = {}
    tensors.update(safetensors_index(Path(args.base)))
    for name in ("2_Dense", "3_Dense"):
        p = Path(args.dense_dir) / f"{name}.safetensors"
        if not p.exists():
            parser.error(f"missing head weights {p} (download via HF token from the gated repo)")
        for k, meta in safetensors_index(p).items():
            tensors[f"{name}.{k}"] = meta

    manifest = {
        "format": "flm-open-embedding-manifest-v1",
        "config": str(model_dir / "config.json"),
        "tokenizer": str(HF_SNAP / "tokenizer.json"),
        "tensors": tensors,
    }
    out = model_dir / "weights_manifest.json"
    out.write_text(json.dumps(manifest, indent=1))
    print(f"wrote {out} ({len(tensors)} tensors)")


if __name__ == "__main__":
    sys.exit(main())