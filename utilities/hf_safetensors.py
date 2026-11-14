"""hf_safetensors: read individual tensors out of a Hugging Face safetensors checkpoint,
by byte range, without downloading the checkpoint.

Both verification tools need this: `check-mmproj.py` compares a vision projector against
the checkpoint it was converted from, and `dit-chain/gguf_weights.py` does the same for a
diffusion DiT GGUF. Fetching an 8 GB checkpoint to look at five tensors is not an option on
a normal connection, and both formats are self-describing, so a range read is enough.

    idx = index("black-forest-labs/FLUX.2-klein-4B")   # name -> spec
    vals, shape = tensor(idx["transformer_blocks.0.attn.to_q.weight"])
"""
from __future__ import annotations

import json
import struct
import urllib.request

__all__ = ["index", "tensor", "http_range", "fetch_all", "hf_url"]


def _headers(extra: dict | None = None) -> dict:
    """Request headers, with the HF token when there is one.

    A GATED checkpoint (meta-llama/*, black-forest-labs/FLUX.2-klein-9B) answers 401 to a
    bare range request while `hf_hub_download` succeeds from the same machine -- the token
    lives in the hub's cache, and urllib knows nothing about it.
    """
    h = dict(extra or {})
    try:
        from huggingface_hub import get_token

        tok = get_token()
        if tok:
            h["Authorization"] = f"Bearer {tok}"
    except Exception:
        pass
    return h


def http_range(url, start, end):
    """Bytes [start, end) of a URL."""
    req = urllib.request.Request(url, headers=_headers({"Range": f"bytes={start}-{end - 1}"}))
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.read()


def fetch_all(url):
    with urllib.request.urlopen(urllib.request.Request(url, headers=_headers()), timeout=120) as r:
        return r.read()


def hf_url(repo, filename):
    return f"https://huggingface.co/{repo}/resolve/main/{filename}"


def _decode(raw, dtype):
    n = len(raw) // (2 if dtype in ("BF16", "F16") else 4)
    if dtype == "F32":
        return list(struct.unpack(f"<{n}f", raw))
    if dtype == "F16":
        return list(struct.unpack(f"<{n}e", raw))
    if dtype == "BF16":
        return list(struct.unpack(f"<{n}f", b"".join(b"\x00\x00" + raw[i:i + 2] for i in range(0, len(raw), 2))))
    raise SystemExit(f"dtype {dtype} not handled")


def index(repo, prefix=""):
    """{tensor name: (url, data_base, dtype, (start, end), shape)} for a repo's shards.

    Reads only each shard's JSON header. The tensor's bytes are base+start..base+end.
    `prefix` is the checkpoint's subfolder when it is not at the repo root -- diffusers
    pipelines keep theirs in `transformer/`, `text_encoder/`, `vae/`.
    """
    try:
        idx = json.loads(fetch_all(hf_url(repo, prefix + "model.safetensors.index.json")))
        shards = sorted(set(idx["weight_map"].values()))
    except Exception:
        meta = json.loads(fetch_all(f"https://huggingface.co/api/models/{repo}"))
        p = prefix.replace("\\", "/").strip("/")
        shards = [s["rfilename"] for s in meta.get("siblings", [])
                  if s["rfilename"].endswith(".safetensors")
                  and (not p or s["rfilename"].startswith(p + "/"))]
        if p:   # the header lookup below wants names relative to the subfolder
            shards = [sh[len(p) + 1:] for sh in shards]
        if not shards:
            raise SystemExit(f"{repo}{('/' + p) if p else ''}: no safetensors shards")
        # Several shards and NO index is legal (FLUX.2-klein-9B ships two): each shard's own
        # header says which tensors it holds, so they are read and merged rather than refused.
    out = {}
    for shard in shards:
        url = hf_url(repo, prefix + shard)
        (hlen,) = struct.unpack("<Q", http_range(url, 0, 8))
        header = json.loads(http_range(url, 8, 8 + hlen))
        for name, info in header.items():
            if name == "__metadata__":
                continue
            out[name] = (url, 8 + hlen, info["dtype"], info["data_offsets"], info["shape"])
    return out


def tensor(spec):
    """(values, shape) for one tensor of an `index()` mapping."""
    url, base, dtype, (start, end), shape = spec
    return _decode(http_range(url, base + start, base + end), dtype), list(shape)
