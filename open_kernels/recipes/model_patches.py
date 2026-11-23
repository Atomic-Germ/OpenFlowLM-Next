"""Per-base-model patches for fields a GGUF header cannot carry.

A quantized/finetuned repo inherits its BASE model's geometry, but two things a
GGUF never states reliably live outside the file:

  * ``rope_scaling`` -- llama.cpp bakes Llama-3's factor-32 wavelength ramp into
    ``rope_freqs.weight``; the header has no ``rope_scaling`` dict. Without the
    patch a GGUF-derived spec has a DIFFERENT ``kernel_hash`` from the shipped
    set and fails to link (the older workaround was to ignore the field in the
    hash via ``rope_ignore()``).
  * ``real_vocab`` -- the GGUF's ``vocab`` is the padded lm_head row count; the
    genuine tokenizer id count lives in ``tokenizer.json``. It is
    kernel-independent (a runtime logits mask), so supplying it here cannot
    change which xclbins serve the model.

The file is hand-curated but keyed by GENERIC BASE NAME, not by a repo slug or
geometry hash: every finetune and quant of a base inherits one entry. The key is
the same export label ``generic_model_name()`` / ``get_base_chain()[-1]``
produce (e.g. ``Llama-3.2-1B-Instruct-NPU2``), normalized by ``patch_key()``.

``utilities/gen-base-patch.py`` can script most of the file's creation by
crawling a repo's base chain; hand edits are preserved unless --refresh is
given.
"""
import json
from pathlib import Path

PATCHES_PATH = Path(__file__).resolve().parent / "patches.json"

# Fields a patch may supply. Anything else in an entry (e.g. ``_source``) is
# provenance and is never copied into a spec.
PATCH_FIELDS = ("rope_scaling", "real_vocab")


def patch_key(name: str) -> str:
    """Normalize a base/export name into a patches.json key.

    Both call sites already have the generic name in hand (the README base
    chain's last element, or ``generic_model_name()``). Strip a trailing quant
    suffix and normalize the ``-NPU2`` ending so a finetune and its quant hit
    the same entry regardless of which spelling the caller produced.
    """
    n = (name or "").strip()
    if not n:
        return ""
    if "/" in n:                       # meta-llama/Llama-3.2-1B-Instruct
        n = n.split("/")[-1]
    low = n.lower()
    for suffix in ("-gguf", "-q4nx", "-i1"):
        if low.endswith(suffix):
            n, low = n[:-len(suffix)], n[:-len(suffix)].lower()
    if not low.endswith("-npu2"):
        n = f"{n}-NPU2"
    return n


def load_patches(path: Path | None = None) -> dict:
    """The raw patches dict, keyed by normalized generic model name."""
    p = Path(path) if path else PATCHES_PATH
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if not isinstance(raw, dict):
        return {}
    return {patch_key(k): v for k, v in raw.items() if isinstance(v, dict)}


def base_patch(name: str, path: Path | None = None) -> dict:
    """The patch entry for a generic base name, or {} when there is none."""
    return load_patches(path).get(patch_key(name), {})


def apply_to_dict(spec: dict, name: str, path: Path | None = None) -> dict:
    """Fill patchable fields into a spec DICT (gen-kernel-spec's currency).

    Existing non-None values win -- the patch supplies what the GGUF/config
    cannot, it never overrides a value the model itself states.
    """
    patch = base_patch(name, path)
    for field in PATCH_FIELDS:
        if field in patch and patch[field] is not None and spec.get(field) is None:
            spec[field] = patch[field]
    return spec


def patch_fields(name: str, path: Path | None = None) -> dict:
    """Only the patchable fields for a base name (for dataclasses.replace)."""
    patch = base_patch(name, path)
    return {f: patch[f] for f in PATCH_FIELDS if patch.get(f) is not None}
