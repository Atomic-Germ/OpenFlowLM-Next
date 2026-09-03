"""Reproducible builder for open (unquantized) causal-LM model repos.

This is the text-model counterpart to :mod:`q4nx.open_embedding` and the first
half of the Phase 0 deliverable in ``docs/plans/open_gemma3_text_plan.md``.

It assembles a complete, uploadable HuggingFace repo directory that the open
causal engine will load directly. Unlike the Q4NX converters there is no
quantization or tensor rearrangement: the source safetensors are copied
verbatim, which keeps the build reproducible and lossless.

Two details matter for the runtime and are recorded explicitly rather than
being left implicit:

* **per-tensor dtype** — the engine cannot assume fp32. Source weights are
  bf16, and storing them as bf16 is lossless: upcasting to fp32 would add size
  with no accuracy benefit, since there is no extra precision to recover.
* **tied embeddings** — Gemma3 ships no ``lm_head.weight``; the LM head reuses
  the embedding matrix. The manifest records that mapping so the engine does
  not have to infer it from config defaults.

The output is byte-reproducible: contents are copied verbatim, JSON is sorted,
and no timestamps or host paths are recorded.
"""

from __future__ import annotations

import json
import os
import shutil
import struct
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .open_embedding import sha256_file

# Files copied verbatim from the source model. Optional sidecars are included
# when present so the tokenizer is self-contained.
REQUIRED_BASE_FILES = ("config.json", "tokenizer.json", "tokenizer_config.json")
OPTIONAL_BASE_FILES = (
    "generation_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
)
SINGLE_SHARD = "model.safetensors"
SHARD_INDEX = "model.safetensors.index.json"

MANIFEST_NAME = "weights_manifest.json"
MANIFEST_FORMAT = "flm-open-causal-manifest-v1"
MODEL_INFO_ARTIFACT = "model_info_entry.json"

# Tensors that must never be block-quantized later; recorded for Phase 3.
EMBEDDING_NAMES = ("model.embed_tokens.weight", "lm_head.weight")


def safetensors_header(path: Path) -> Tuple[dict, int]:
    """Parse a safetensors header into {name: {"offset": abs, "shape", "dtype"}}.

    Layout: [u64 header_len][header JSON][tensor data]. `data_offsets` are
    relative to the start of the data section, so the absolute offset is
    `8 + header_len + data_offsets[0]`.
    """
    with path.open("rb") as stream:
        header_len = struct.unpack("<Q", stream.read(8))[0]
        header = json.loads(stream.read(header_len).decode("utf-8"))
    base = 8 + header_len
    out: Dict[str, dict] = {}
    for name, meta in header.items():
        if name == "__metadata__":
            continue
        start, _end = meta["data_offsets"]
        out[name] = {
            "offset": base + start,
            "shape": list(meta["shape"]),
            "dtype": meta["dtype"],
        }
    return out


def resolve_source(source: str) -> Path:
    """Resolve a local directory or an HF repo id to a local snapshot path."""
    if os.path.isdir(source):
        return Path(source)
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(repo_id=source))


def _shard_files(src: Path) -> List[Path]:
    """Return the shard files to copy, handling both single and sharded layouts."""
    index = src / SHARD_INDEX
    if index.is_file():
        weight_map = json.loads(index.read_text(encoding="utf-8"))["weight_map"]
        names = sorted(set(weight_map.values()))
        return [src / name for name in names]
    single = src / SINGLE_SHARD
    if single.is_file():
        return [single]
    raise FileNotFoundError(f"no {SINGLE_SHARD} or {SHARD_INDEX} in {src}")


def build_open_causal_repo(
    source: str,
    output_dir: str,
    npu_assets: Optional[str] = None,
    generate_manifest: bool = True,
) -> dict:
    """Build a distributable open causal-LM repo directory.

    Returns a dict describing what was produced, including the generated
    registry metadata under the "model_info" key.
    """
    src = resolve_source(source)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    produced: List[str] = []

    for name in REQUIRED_BASE_FILES:
        src_file = src / name
        if not src_file.is_file():
            raise FileNotFoundError(f"source model is missing {name}: {src_file}")
        shutil.copyfile(src_file, out / name)
        produced.append(name)

    for name in OPTIONAL_BASE_FILES:
        src_file = src / name
        if src_file.is_file():
            shutil.copyfile(src_file, out / name)
            produced.append(name)

    shards = _shard_files(src)
    tensors: Dict[str, dict] = {}
    for shard in shards:
        if not shard.is_file():
            raise FileNotFoundError(f"missing shard: {shard}")
        rel = shard.relative_to(src).as_posix()
        shutil.copyfile(shard, out / rel)
        if rel not in produced:
            produced.append(rel)
        for name, meta in safetensors_header(shard).items():
            tensors[name] = {
                "file": rel,
                "offset": meta["offset"],
                "shape": meta["shape"],
                "dtype": meta["dtype"],
            }

    if (src / SHARD_INDEX).is_file():
        shutil.copyfile(src / SHARD_INDEX, out / SHARD_INDEX)
        produced.append(SHARD_INDEX)

    if not tensors:
        raise RuntimeError(f"no tensors found in {src}")

    # Tied embeddings: Gemma3 ships no lm_head tensor and reuses the embedding.
    tied: Dict[str, str] = {}
    embed = next((n for n in EMBEDDING_NAMES[:1] if n in tensors), None)
    if embed and "lm_head.weight" not in tensors:
        tied["lm_head.weight"] = embed

    if npu_assets:
        npu_src = Path(npu_assets)
        if not npu_src.is_dir():
            raise NotADirectoryError(f"NPU asset dir not found: {npu_src}")
        npu_dst = out / "npu_matmul_f32"
        npu_dst.mkdir(parents=True, exist_ok=True)
        for asset in sorted(npu_src.iterdir()):
            if asset.is_file() and asset.suffix in (".xclbin", ".insts"):
                shutil.copyfile(asset, npu_dst / asset.name)
                produced.append(f"npu_matmul_f32/{asset.name}")

    if generate_manifest:
        manifest = {
            "format": MANIFEST_FORMAT,
            "config": "config.json",
            "tokenizer": "tokenizer.json",
            "tensors": {k: tensors[k] for k in sorted(tensors)},
        }
        if tied:
            manifest["tied"] = {k: tied[k] for k in sorted(tied)}
        (out / MANIFEST_NAME).write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
        produced.append(MANIFEST_NAME)

    model_info = [
        {
            "type": "file",
            "oid": sha256_file(out / rel),
            "size": (out / rel).stat().st_size,
            "path": rel,
        }
        for rel in sorted(produced)
    ]
    (out / MODEL_INFO_ARTIFACT).write_text(
        json.dumps(model_info, indent=1) + "\n", encoding="utf-8"
    )

    return {
        "output_dir": str(out),
        "source": str(src),
        "files": sorted(produced),
        "tensor_count": len(tensors),
        "tied": tied,
        "model_info": model_info,
    }
