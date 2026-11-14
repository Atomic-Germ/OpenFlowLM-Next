"""klein_checkpoint: the FLUX.2 checkpoint a klein GGUF was quantized from.

A klein GGUF carries only the DiT. The text encoder and the VAE have no GGUF form at all,
so a bundle built from a GGUF still needs the diffusers checkpoint -- and that is exactly
the repo the GGUF's README names:

    unsloth/FLUX.2-klein-4B-GGUF  ->  base_model: black-forest-labs/FLUX.2-klein-4B

which is the same climb the text models use to find a tokenizer and a config (a quant repo
names its base; the base has the files). `resolve` reads the frontmatter, falls back to the
repo's own name when a converter left `base_model` empty (leejet's does), and fetches only
the parts the bundle needs -- never the 8 GB transformer.

    ckpt = resolve("unsloth/FLUX.2-klein-4B-GGUF")     # -> Path with text_encoder/ vae/ ...
    ckpt = resolve("unsloth/FLUX.2-klein-4B-GGUF", cache=Path("~/ckpts"))
"""
from __future__ import annotations

import json
import os
import sys
import urllib.request
from pathlib import Path

# What export_bundle reads from the checkpoint. `transformer/` is deliberately absent: the
# DiT comes from the GGUF, and pulling 8 GB of bf16 to ignore it would be the whole point
# missed.
PARTS = ["text_encoder/*", "vae/*", "tokenizer/*", "LICENSE.md", "model_index.json"]

DEFAULT_CACHE = Path(os.environ.get("OFLM_DIT_CHECKPOINT_CACHE")
                     or Path.home() / ".cache" / "oflm" / "dit-checkpoints")


def _readme(repo: str) -> str:
    """The repo's README.md, or "" when it cannot be read."""
    try:
        from huggingface_hub import hf_hub_download

        return Path(hf_hub_download(repo_id=repo, filename="README.md")).read_text(errors="ignore")
    except Exception:
        pass
    try:
        with urllib.request.urlopen(f"https://huggingface.co/{repo}/resolve/main/README.md", timeout=60) as r:
            return r.read().decode("utf-8", "replace")
    except Exception:
        return ""


def base_model_from_readme(repo: str) -> str:
    """The `base_model` the frontmatter names, or "" -- a string or a YAML list."""
    text = _readme(repo)
    if not text.startswith("---"):
        return ""
    end = text.find("---", 3)
    if end == -1:
        return ""
    lines = text[3:end].splitlines()
    for i, line in enumerate(lines):
        key, _, rest = line.partition(":")
        if key.strip() != "base_model":
            continue
        rest = rest.strip()
        if rest.startswith("["):
            try:
                vals = json.loads(rest)
                return str(vals[0]) if vals else ""
            except ValueError:
                return ""
        if rest:
            return rest.strip().strip("'\"")
        # a YAML list: the first "- owner/name" after the key
        for follow in lines[i + 1:]:
            s = follow.strip()
            if s.startswith("- "):
                return s[2:].strip().strip("'\"")
            if s and not s.startswith("#"):
                break
    return ""


def guess_base(repo: str) -> str:
    """The repo's own name without the quant marker -- the fallback for an empty base_model."""
    owner, _, name = repo.partition("/")
    if not owner:
        return ""
    for suffix in ("-GGUF", "-gguf", "-GGML", "-ggml"):
        if name.endswith(suffix):
            return f"{owner}/{name[:-len(suffix)]}"
    return ""


def resolve(gguf_repo: str, cache: Path | None = None, quiet: bool = False) -> tuple[Path, str]:
    """(local checkpoint dir, the repo it came from) for a klein GGUF's repo."""
    cache = Path(cache).expanduser() if cache else DEFAULT_CACHE
    base = base_model_from_readme(gguf_repo) or guess_base(gguf_repo)
    if not base:
        raise SystemExit(f"{gguf_repo}: no base_model in its README and its name is not a quant "
                         "of a diffusers repo; pass the checkpoint explicitly")
    target = cache / base.replace("/", "--")
    have = all((target / p).exists() for p in ("text_encoder", "vae", "tokenizer"))
    if have:
        if not quiet:
            print(f"checkpoint: {base} (already at {target})", flush=True)
        return target, base
    if not quiet:
        print(f"checkpoint: {base} (the text encoder and VAE; the DiT comes from the GGUF)", flush=True)
    from huggingface_hub import snapshot_download

    target.mkdir(parents=True, exist_ok=True)
    snapshot_download(base, local_dir=str(target), allow_patterns=PARTS)
    return target, base
