"""Reproducible builder for open (unquantized) embedding model repos.

This is the open-model counterpart to the Q4NX converters. It assembles a
complete, uploadable HuggingFace repo directory for an embedding model that the
open CPU/NPU engine (`src/open_embedding`) can load directly.

Unlike the quantized families, embedding models need no tensor rearrangement or
quantization: the HF safetensors are used as-is. What this builder adds is the
packaging discipline the runtime needs:

* a fixed, predictable file layout;
* a portable `weights_manifest.json` using paths relative to the model dir
  (never absolute builder paths);
* optional NPU matmul assets in `npu_matmul_f32/`;
* a deterministic `model_info_entry.json` (path/size/oid per file) that can be
  pasted into `src/model_info.json` so `flm pull` actually downloads the files.

The output is byte-reproducible: file contents are copied verbatim, JSON is
emitted with sorted keys, and no timestamps or host paths are recorded.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import struct
from pathlib import Path
from typing import Dict, List, Optional

# Verbatim copies from the source model.
BASE_FILES = ("config.json", "tokenizer.json", "tokenizer_config.json")
BODY_FILE = "model.safetensors"
# Sentence-transformers dense heads shipped by the upstream embedding repos.
HEADS = ("2_Dense", "3_Dense")
MANIFEST_NAME = "weights_manifest.json"
MANIFEST_FORMAT = "flm-open-embedding-manifest-v1"
# Side artifact carrying the registry metadata (not part of the uploaded repo).
MODEL_INFO_ARTIFACT = "model_info_entry.json"


def sha256_file(path: Path) -> str:
    """Streaming SHA-256; this is also the git-lfs oid for the uploaded file."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safetensors_index(path: Path) -> Dict[str, dict]:
    """Parse a safetensors header into {name: {"offset": abs, "shape": [...]}}.

    Layout: [u64 header_len][header JSON][tensor data]. `data_offsets` in the
    header are relative to the start of the data section, so the absolute offset
    is `8 + header_len + data_offsets[0]`.
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
        out[name] = {"offset": base + start, "shape": list(meta["shape"])}
    return out


def resolve_source(source: str) -> Path:
    """Resolve a local directory or an HF repo id to a local snapshot path."""
    if os.path.isdir(source):
        return Path(source)
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(repo_id=source))


def _find_head(source_dir: Path, head: str) -> Path:
    """Locate a dense-head safetensors file in any known layout."""
    candidates = [
        source_dir / "weights" / f"{head}.safetensors",
        source_dir / head / "model.safetensors",
        source_dir / head / f"{head}.safetensors",
        source_dir / f"{head}.safetensors",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"missing dense head {head}; looked for "
        + ", ".join(str(c.relative_to(source_dir)) for c in candidates)
    )


def build_open_embedding_repo(
    source: str,
    output_dir: str,
    npu_assets: Optional[str] = None,
    generate_manifest: bool = True,
) -> dict:
    """Build a distributable open-embedding repo directory.

    Returns a dict describing what was produced, including the generated
    registry metadata under the "model_info" key.
    """
    src = resolve_source(source)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    produced: List[str] = []

    for name in BASE_FILES:
        src_file = src / name
        if not src_file.is_file():
            raise FileNotFoundError(f"source model is missing {name}: {src_file}")
        shutil.copyfile(src_file, out / name)
        produced.append(name)

    body_src = src / BODY_FILE
    if not body_src.is_file():
        raise FileNotFoundError(f"source model is missing {BODY_FILE}: {body_src}")
    shutil.copyfile(body_src, out / BODY_FILE)
    produced.append(BODY_FILE)

    head_files: Dict[str, Path] = {}
    for head in HEADS:
        head_src = _find_head(src, head)
        head_dst = out / head / "model.safetensors"
        head_dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(head_src, head_dst)
        head_files[head] = head_dst
        produced.append(f"{head}/model.safetensors")

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

    tensors: Dict[str, dict] = {}
    for name, meta in safetensors_index(out / BODY_FILE).items():
        tensors[name] = {"file": BODY_FILE, "offset": meta["offset"], "shape": meta["shape"]}
    for head, head_path in head_files.items():
        rel = head_path.relative_to(out).as_posix()
        for name, meta in safetensors_index(head_path).items():
            tensors[f"{head}.{name}"] = {
                "file": rel,
                "offset": meta["offset"],
                "shape": meta["shape"],
            }

    missing = [h for h in HEADS if f"{h}.linear.weight" not in tensors]
    if missing:
        raise RuntimeError(f"dense head tensors absent after packaging: {missing}")

    if generate_manifest:
        manifest = {
            "format": MANIFEST_FORMAT,
            "config": "config.json",
            "tokenizer": "tokenizer.json",
            "tensors": {k: tensors[k] for k in sorted(tensors)},
        }
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
        "model_info": model_info,
    }
