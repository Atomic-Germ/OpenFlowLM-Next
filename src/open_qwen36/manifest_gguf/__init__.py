"""GGUF-direct twin manifest module (isolated).

Provides functions for loading, creating, and emitting the GGUF-direct twin
manifest that is embedded within the main Manifest struct's `gguf` field.

Functions
---------
load_gguf_twin : Load a GGUF twin manifest from JSON
make_gguf_twin_spec : Create a GGUF twin ModelSpec with quant="q4_1_f32"
emit_gguf_twin : Emit a GGUF twin manifest JSON fragment
gguf_quant_allowed : Check if a GGUF quant type is supported by pool pack ops
gguf_tensor_chunk_bytes : Get chunk bytes for a GGUF tensor type
"""

import json


def load_gguf_twin(j: dict, where: str = "manifest") -> dict:
    """Load a GGUF twin manifest from a JSON dict."""
    m = {}
    for key in ("manifest_version", "family", "spec_hash", "build_key",
                "max_ctx_default", "hidden", "vocab", "real_vocab",
                "chunk_bytes", "pool_bytes", "lmhead_pool_bytes",
                "kv_row", "ptab_row", "rotary_dim", "rope_theta",
                "rope_inv_freq", "rout_idx_off", "has_moe",
                "layers", "contexts", "kernels", "tail",
                "globals", "embed_tensor", "norm_tensor", "embed_scale",
                "norm_bytes", "lmhead_ops"):
        if key in j:
            m[key] = j[key]
    return m


def make_gguf_twin_spec(spec_family: str = "qwen2", spec_hidden: int = 2048,
                        spec_quant: str = "q4_1_f32") -> dict:
    """Create a GGUF twin specification dict."""
    return {
        "family": spec_family,
        "hidden": spec_hidden,
        "quant": spec_quant,
    }


def emit_gguf_twin(spec: dict, max_ctx: int = 4096, key: str = "") -> str:
    """Emit a GGUF-direct twin manifest JSON fragment.
    
    Mirrors the logic in export_qwen36_kernels.py:
      if gguf:
          m["gguf"] = manifest(replace(spec, quant="q4_1_f32"), max_ctx, key)
    """
    m = {
        "manifest_version": 1,
        "family": spec.get("family", ""),
        "spec": {
            "family": spec.get("family", ""),
            "hidden": spec.get("hidden", 2048),
            "quant": "q4_1_f32",
        },
        "spec_hash": "",
        "build_key": key if key else "",
        "max_ctx_default": max_ctx,
        "hf_config_check": {"model_type": "dense"},
        "hf_config_defaults": {},
        "layout": {
            "hidden": spec.get("hidden", 2048),
            "chunk_bytes": 6144,
            "pool_bytes": 0,
            "lmhead_pool_bytes": 0,
        },
        "layers": spec.get("layers", []),
        "contexts": {},
        "kernels": {},
        "layer_types": {},
        "tail": {},
        "globals": {},
        "embed_tensor": "token_embd.weight",
        "norm_tensor": "output_norm.weight",
        "embed_scale": 1.0,
        "norm_bytes": 0,
        "lmhead_ops": [],
    }
    return json.dumps(m, indent=1) + "\n"


# Supported GGUF quant types for the pool pack ops.
# Q4_0, Q4_1, Q8_0 are supported directly; Q4_K, Q6_K, Q8_K require q4nx-build.
_SupportedQQ = {"Q4_0", "Q4_1", "Q8_0"}


def gguf_quant_allowed(quant: str) -> bool:
    """Check if a GGUF quant type is supported by the pool pack ops."""
    return quant in _SupportedQQ


def gguf_tensor_chunk_bytes(quant: str) -> int:
    """Return the chunk size for a GGUF tensor type, matching the pool layout."""
    if quant in ("Q4_0", "Q4_1"):
        return 6144  # 32 rows x 256 k x f32 scales + min/nibble plane
    if quant == "Q8_0":
        return 9216  # q8 lm_head pool chunk: 32 rows x 256 k x f32 scales + int8 codes
    return 0  # unsupported / requires q4nx-build