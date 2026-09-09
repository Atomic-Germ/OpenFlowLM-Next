"""The family recipes, by ModelSpec.family. Each module exposes the same
surface: recipe(spec, max_ctx), layout(spec, max_ctx), pack_plan(spec),
programs(spec), builds(spec), hf_config_check(spec), manifest_layout(spec,
max_ctx), KERNEL_SOURCES, GEN_KERNELS."""
from __future__ import annotations

from types import ModuleType

from .spec import ModelSpec


def family_module(name: str) -> ModuleType:
    if name == "qwen36moe":
        from . import qwen36moe
        return qwen36moe
    if name == "qwen35":
        from . import qwen35
        return qwen35
    if name in ("qwen3", "llama3", "gemma3", "hunyuan", "granite"):
        from . import dense
        return dense
    raise ValueError(f"no recipe for family {name!r} "
                     f"(have qwen36moe, qwen35, qwen3, llama3, gemma3, hunyuan, granite)")


def for_spec(spec: ModelSpec) -> ModuleType:
    return family_module(spec.family)


FAMILIES = ("qwen36moe", "qwen35", "qwen3", "llama3", "gemma3", "hunyuan", "granite")
