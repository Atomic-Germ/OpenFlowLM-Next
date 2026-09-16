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
    if name in ("qwen3", "llama3", "gemma3", "hunyuan", "granite", "phi3", "qwen2"):
        from . import dense
        return dense
    if name in NOT_IMPLEMENTED:
        raise NotImplementedError(f"the open kernels have no recipe for {name!r} yet: "
                                  f"{NOT_IMPLEMENTED[name]}")
    raise ValueError(f"no recipe for family {name!r} "
                     f"(have qwen36moe, qwen35, qwen3, llama3, gemma3, hunyuan, granite, phi3, qwen2)")


def for_spec(spec: ModelSpec) -> ModuleType:
    return family_module(spec.family)


FAMILIES = ("qwen36moe", "qwen35", "qwen3", "llama3", "gemma3", "hunyuan", "granite", "phi3", "qwen2")

# Families whose ModelSpec derives but whose kernels do not exist. Naming the gap here is
# the point: routing such a model to the nearest recipe would emit kernels that drop a whole
# stage of the layer and then report parity against a replica making the same mistake.
NOT_IMPLEMENTED = {
    "gptoss": "the learned per-head attention sink logit has no ATTN_SINK in attn.h; the "
              "experts want a clamped SwiGLU with gate and up interleaved down the rows "
              "rather than split in half; o_proj, the router and all three expert "
              "projections carry a bias the dense one does not cover; the MoE FFN has to "
              "compose with sliding-window layers, which no recipe does today; and the "
              "engine needs YaRN position tables. See .claude/plans/gptoss-attention-sinks.md",
    "lfm2": "ten of its sixteen layers replace attention with a short depthwise causal "
            "convolution (the short_conv layer type), and no designs/short_conv exists to "
            "run one. The fp64 reference and the element accounting are in "
            ".claude/plans/lfm2-short-conv.md; the attention half also needs its geometry "
            "(64, 32, 8, 64, True, False, False, False) validated in catalogue.py",
}
