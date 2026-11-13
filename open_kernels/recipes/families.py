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
    if name == "lfm2":
        from . import lfm2
        return lfm2
    if name == "gemma4":
        from . import gemma4
        return gemma4
    if name == "gptoss":
        from . import gptoss
        return gptoss
    if name in ("qwen3", "llama3", "gemma3", "hunyuan", "granite", "phi3", "qwen2", "k2"):
        from . import dense
        return dense
    if name in NOT_IMPLEMENTED:
        raise NotImplementedError(f"the open kernels have no recipe for {name!r} yet: "
                                  f"{NOT_IMPLEMENTED[name]}")
    raise ValueError(f"no recipe for family {name!r} (have {', '.join(FAMILIES)})")


def for_spec(spec: ModelSpec) -> ModuleType:
    return family_module(spec.family)


FAMILIES = ("qwen36moe", "qwen35", "qwen3", "llama3", "gemma3", "hunyuan", "granite", "phi3", "qwen2",
            "lfm2", "k2", "gemma4", "gptoss")

# Families whose ModelSpec derives but whose kernels do not exist. Naming the gap here is
# the point: routing such a model to the nearest recipe would emit kernels that drop a whole
# stage of the layer and then report parity against a replica making the same mistake.
NOT_IMPLEMENTED: dict[str, str] = {}
