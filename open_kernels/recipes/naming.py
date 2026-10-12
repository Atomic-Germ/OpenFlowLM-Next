"""What a spec file is called, and what the kernels it builds get named.

`extra.model` in a spec is the export destination: `export_qwen36_kernels.py`
writes `src/xclbins/<extra.model>/open_kernels`, and `oflm add` links a converted
model's kernels by that same name. Two specs sharing a name means one silently
overwrites the other -- gemma3-12b once shipped as "Gemma3-4B-NPU2" and the 4B
sets were overwritten on every full export (#54).

So the name is DERIVED from the spec's file stem, not authored, and lives here
rather than in the generator and the test separately -- two copies of this rule
is the same bug with extra steps.
"""
from __future__ import annotations

import re

# Specs whose export name is not <family>-<size>: a distilled fine-tune, a MoE's
# parameter count, a vendor's own naming.
EXPECTED_MODEL = {
    "qwen35-9b": "Qwen3.5-9B-NPU2",
    "qwen35-27b": "Qwen3.8-27B-NPU2",
    "k2-horizon-3.7b": "K2-Horizon-3.7B-NPU2",
    "qwen36-35b-a3b": "Qwen3.6-35B-A3B-NPU2",
    "hy-mt2-7b": "Hy-MT2-7B-NPU2",
    "lfm2-1.2b": "LFM2-1.2B-NPU2",
    "lfm2-trans-2.6b": "LFM2-2.6B-Transcript-NPU2",
    "lfm2.5-it-1.2b": "LFM2.5-1.2B-NPU2",
    "lfm2.5-tk-1.2b": "LFM2.5-1.2B-Thinking-NPU2",
    "minicpm5-2b": "MiniCPM5-2B-NPU2",
    "phi4-mini-4b": "Phi4-mini-Instruct-NPU2",
    "llama31-8b": "Llama-3.1-8B-NPU2",
    "llama32-1b": "Llama-3.2-1B-NPU2",
    "llama32-3b": "Llama-3.2-3B-NPU2",
    "granite42-3b": "Granite-4.2-3B-NPU2",
    "qwen25-3b": "Qwen2.5-3B-Instruct-NPU2",
    "qwen25vl-3b": "Qwen2.5-VL-3B-Instruct-NPU2",
    "qwen3-4b": "Qwen3-4B-NPU2",
    "qwen3-8b": "Qwen3-8B-NPU2",
    "qwen3vl-it-4b": "Qwen3-VL-4B-Instruct-NPU2",
}

# Families whose export name carries a qualifier the stem does not: qwen3's
# thinking / instruct variants are separate containers, not the same one.
_VARIANT_MODEL = {
    "qwen3-it-4b": "Qwen3-4B-Instruct-2507-NPU2",
    "qwen3-tk-4b": "Qwen3-4B-Thinking-2507-NPU2",
}

_SAFE_DIR = re.compile(r"[A-Za-z0-9._-]+")


def expected_model(stem: str) -> str:
    """The `extra.model` a spec file must carry.

    qwen3-8b -> Qwen3-8B-NPU2: the family keeps its brand's spelling, the size
    is shouted. Raises on a stem with no size in it, which is what stops a
    matrix row that forgot its size from quietly exporting as a bare family.
    """
    if stem in EXPECTED_MODEL:
        return EXPECTED_MODEL[stem]
    if stem in _VARIANT_MODEL:
        return _VARIANT_MODEL[stem]
    family, _, size = stem.rpartition("-")
    if not family or not size:
        raise ValueError(f"{stem}: expected <family>-<size>")
    return f"{family}-{size.upper()}-NPU2".replace("gemma3", "Gemma3")


def is_safe_dir_name(name: str) -> bool:
    """Whether `name` is usable as a directory component under src/xclbins."""
    return bool(name) and "/" not in name and not name.startswith(".") and bool(_SAFE_DIR.fullmatch(name))