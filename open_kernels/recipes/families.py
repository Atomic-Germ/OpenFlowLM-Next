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
    if name == "gptoss":
        # The MoE routing and the q/k/v bias exist; the rest does not. Routing this to the
        # dense or the MoE recipe would emit kernels that drop the sink and compute the wrong
        # FFN, and then report parity against a reference that had the same gaps.
        raise NotImplementedError(
            "no recipe for family 'gptoss' yet. On top of the MoE routing (designs/router, "
            "moe_chain, moe_experts, moe_combine, expert_fetch) and the q/k/v bias, GPT-OSS "
            "needs: the learned per-head attention SINK logit (attn.h has no ATTN_SINK); the "
            "experts' CLAMPED SwiGLU -- (up + 1) * gate * sigmoid(1.702 * gate), gate clipped "
            "above at 7 and up clipped both ways, with gate and up interleaved down the rows "
            "rather than split in half; a BIAS on o_proj, on the router and on all three "
            "expert projections, none of which the dense bias covers; an MoE FFN on "
            "sliding-window (dense_local) layers, which no recipe composes today; and YaRN "
            "position tables in the engine. See .claude/plans/gptoss-attention-sinks.md")
    raise ValueError(f"no recipe for family {name!r} "
                     f"(have qwen36moe, qwen35, qwen3, llama3, gemma3, hunyuan, granite, phi3, qwen2)")


def for_spec(spec: ModelSpec) -> ModuleType:
    return family_module(spec.family)


FAMILIES = ("qwen36moe", "qwen35", "qwen3", "llama3", "gemma3", "hunyuan", "granite", "phi3", "qwen2")
