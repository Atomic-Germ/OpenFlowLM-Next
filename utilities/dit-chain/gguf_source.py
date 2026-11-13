"""gguf_source: find and fetch a klein DiT GGUF from a repo (or a path).

Shared by `klein-from-gguf.py` (the local one-command build) and `q4nx-build
--open-diffusion` (the repo build), so the quant preference and the cache location have one
implementation.

    repo = "unsloth/FLUX.2-klein-4B-GGUF"
    name = pick(repo)            # flux-2-klein-4b-Q4_1.gguf
    path = ensure(repo, name)    # ~/.cache/oflm/dit-gguf/<repo>--<file>
"""
from __future__ import annotations

import os
from pathlib import Path

# The quants gguf_weights decodes, best first. Measured 2026-10-28 against the shipped bf16
# bundle (same prompt, seed, 4 steps, and the modulation convention detected):
#
#   * the checkpoint's bf16 packed by our code is BYTE-IDENTICAL to the shipped bundle;
#   * a Q8_0 DiT correlates 0.9946 with it (mean|d| 2.68/255) -- indistinguishable in
#     practice, and half the size;
#   * a Q4_1 DiT is 8.3% off on the block weights and is visibly grainy.
#
# (A grainy Q8_0 was a red herring for a day: the real cause was the adaLN tensor's halves
# being BFL-ordered, which gguf_weights now detects and corrects.)
QUANTS = ["Q8_0", "BF16", "F16", "Q5_1", "Q5_0", "Q4_1", "Q4_0"]

# The ones that visibly degrade klein, for the warning.
LOSSY = ("Q4_1", "Q4_0")

CACHE = Path(os.environ.get("OFLM_DIT_GGUF_CACHE") or Path.home() / ".cache" / "oflm" / "dit-gguf")


def _dit_ggufs(names) -> list[str]:
    return [f for f in names
            if f.lower().endswith(".gguf") and "mmproj" not in f.lower() and "mtp" not in f.lower()]


def pick(repo: str) -> str:
    """The DiT GGUF to take from a repo, Q4_1 first."""
    from huggingface_hub import list_repo_files

    files = list(list_repo_files(repo))
    dit = _dit_ggufs(files)
    for q in QUANTS:
        cand = sorted(f for f in dit if q.lower() in f.lower())
        if cand:
            return cand[0]
    raise SystemExit(f"{repo}: no DiT GGUF in {QUANTS} (have: {', '.join(sorted(files)) or 'none'})")


def ensure(repo: str, name: str) -> Path:
    """The GGUF on disk, downloading it once into the cache."""
    target = CACHE / f"{repo.replace('/', '--')}--{Path(name).name}"
    if target.is_file():
        print(f"DiT GGUF: {target} (cached)", flush=True)
        return target
    from huggingface_hub import hf_hub_download

    target.parent.mkdir(parents=True, exist_ok=True)
    print(f"DiT GGUF: {name} <- {repo}", flush=True)
    got = Path(hf_hub_download(repo_id=repo, filename=name, local_dir=str(target.parent)))
    if got != target:
        got.replace(target)
    return target


def is_gguf_repo(repo: str) -> bool:
    """True when a repo is a GGUF conversion (a DiT gguf and no diffusers checkpoint)."""
    from huggingface_hub import list_repo_files

    try:
        files = list(list_repo_files(repo))
    except Exception:
        return False
    return bool(_dit_ggufs(files)) and "model_index.json" not in files
