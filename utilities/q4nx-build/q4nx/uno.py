"""A Uno diffusion LoRA (e.g. IFM/K2-Horizon-7B-Uno) as the q4_1 tensors (uno.q4nx) the open engine's L-row pass reads (OPEN-UNO-LORA)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from gguf import GGMLQuantizationType
from safetensors.torch import load_file, save_file

from .gguf_tensor import GGUFTensor

TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
NAMES = ("a_qkv", "a_o", "a_gu", "a_d", "b_q", "b_k", "b_v", "b_o", "b_g", "b_u", "b_d")


class _Packer:
    """The converter's q4nx packer without a GGUF behind it: only the tiling it reads."""

    def __init__(self, q4nx_config: dict):
        c = q4nx_config["q4nx_config"]
        self.row_block_size, self.col_block_size = c["row_block_size"], c["col_block_size"]
        self.parallel_size, self.keep_block_in_2D = c["parallel_size"], c["keep_block_in_2D"]

    def pack(self, w: np.ndarray) -> torch.Tensor:
        from . import model_converter
        Conv = getattr(model_converter, "__Q4NX_Converter")    # the packer is a method there (a mangled name)
        t = GGUFTensor("uno", (w.shape[1], w.shape[0]), np.ascontiguousarray(w, np.float32), GGMLQuantizationType.F32)
        d, m, q = t.unpack(GGMLQuantizationType.Q4_1)
        return Conv._pack_q4nx(self, d, m, q)


def projection_folds(config: dict) -> dict[str, float]:
    """Each target's factor that the base builder folded into its weights (Granite's multipliers); {} for none."""
    folded = config.get("q4nx_folded_multipliers")
    if not folded:
        return {}
    from .models.granite import fold_factors
    head_dim = int(config["head_dim"])

    def get(key, default):
        return default if folded.get(key) is None else float(folded[key])

    # the same defaults models/granite.py applies to a GGUF missing the key
    folds = fold_factors(get("attention_multiplier", head_dim ** -0.5), get("embedding_multiplier", 1.0),
                         get("residual_multiplier", 1.0), get("logits_scaling", 1.0), head_dim)
    # hd**-0.5 * sqrt(hd) is 1 + 2e-16 at some hd: that is no fold
    return {p: f for p in TARGETS for suffix, f in folds.items()
            if suffix.endswith(f".{p}.weight") and abs(f - 1.0) > 1e-9}


def derived(sd: dict, layer: int, scale: float, n_cores: int = 8,
            folds: dict[str, float] | None = None) -> dict[str, np.ndarray]:
    """The eleven GEMV-shaped matrices of one layer (fp32), named as NAMES."""
    folds = folds or {}

    def ab(p):
        mod = "self_attn" if p in ("q_proj", "k_proj", "v_proj", "o_proj") else "mlp"
        pre = f"model.layers.{layer}.{mod}.{p}"
        if f"{pre}.lora_A.weight" not in sd:
            pre = "base_model.model." + pre
        # the base's W was folded by f, so the update must be too: f (W + s B A) x
        b = sd[f"{pre}.lora_B.weight"].float().numpy() * (scale * folds.get(p, 1.0))
        return sd[f"{pre}.lora_A.weight"].float().numpy(), b

    A, B = {}, {}
    for p in TARGETS:
        A[p], B[p] = ab(p)
    r = A["q_proj"].shape[0]
    rows = n_cores * 64
    if 3 * r > rows or 2 * r > 256:
        raise ValueError(f"uno: rank {r} does not fit the padded layout ({rows} A rows, one 256-column B tile)")

    def pad_rows(*ms, at=None):
        """the matrices stacked from row 0, or each at its row in `at`, the rest zero"""
        out = np.zeros((rows, ms[0].shape[1]), np.float32)
        r0 = 0
        for i, m in enumerate(ms):
            r0 = at[i] if at else r0
            out[r0:r0 + m.shape[0]] = m
            r0 += m.shape[0]
        return out

    def tile(b, lo):
        t = np.zeros((b.shape[0], 256), np.float32)
        t[:, lo:lo + r] = b
        return t

    # z_v starts the second 256-wide window, which is what v's tile reads, at any rank
    return {"a_qkv": pad_rows(A["q_proj"], A["k_proj"], A["v_proj"], at=(0, r, 256)), "a_o": pad_rows(A["o_proj"]),
            "a_gu": pad_rows(A["gate_proj"], A["up_proj"]), "a_d": pad_rows(A["down_proj"]),
            "b_q": tile(B["q_proj"], 0), "b_k": tile(B["k_proj"], r), "b_v": tile(B["v_proj"], 0),
            "b_o": tile(B["o_proj"], 0), "b_g": tile(B["gate_proj"], 0), "b_u": tile(B["up_proj"], r),
            "b_d": tile(B["down_proj"], 0)}


def fetch_adapter(src: str) -> Path:
    """A local adapter dir as is; an HF repo id downloaded (only the adapter's two files)."""
    if Path(src).is_dir():
        return Path(src)
    from huggingface_hub import snapshot_download
    return Path(snapshot_download(repo_id=src, allow_patterns=["adapter_config.json", "adapter_model.safetensors"]))


def metadata(acfg: dict, scale: float, folds: dict[str, float], noise_high: int | None) -> dict[str, str]:
    meta = {"uno_scale": repr(scale), "uno_rank": str(acfg["r"]),
            "uno_base": str(acfg.get("base_model_name_or_path", "")), "uno_folds": json.dumps(folds, sort_keys=True)}
    if noise_high is not None:
        if noise_high < 2:
            raise ValueError(f"uno: noise ids are drawn from [1, {noise_high}), which is empty")
        meta["uno_noise_high"] = str(noise_high)
    return meta


def build(adapter_dir: Path, config: Path, out_dir: Path, q4nx_config: Path, noise_high: int | None = None) -> Path:
    acfg = json.loads((adapter_dir / "adapter_config.json").read_text())
    if acfg.get("use_rslora"):
        raise ValueError("uno: an rsLoRA adapter scales by alpha / sqrt(r); this builder folds alpha / r")
    scale = float(acfg["lora_alpha"]) / float(acfg["r"])
    cfg = json.loads(config.read_text())
    nl = cfg["num_hidden_layers"]
    folds = projection_folds(cfg)
    sd = load_file(str(adapter_dir / "adapter_model.safetensors"))
    packer = _Packer(json.loads(q4nx_config.read_text()))
    tensors = {}
    for l in range(nl):
        for name, w in derived(sd, l, scale, folds=folds).items():
            tensors[f"model.layers.{l}.uno.{name}.weight"] = packer.pack(w)
        print(f"\r[uno] layer {l + 1}/{nl}", end="", flush=True)
    print()
    out = out_dir / "uno.q4nx"
    meta = metadata(acfg, scale, folds, noise_high)
    save_file(tensors, str(out), metadata=meta)
    noise = f"[1, {noise_high})" if noise_high is not None else "the engine's default"
    print(f"[uno] wrote {out} ({out.stat().st_size} B, {len(tensors)} tensors, scale {scale:g}, "
          f"folds {folds or 'none'}, noise {noise})")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--adapter", required=True, help="the PEFT adapter dir (adapter_config.json + adapter_model.safetensors)")
    ap.add_argument("--config", required=True, help="the base model's config.json")
    ap.add_argument("--out", required=True, help="the model dir to write uno.q4nx into")
    ap.add_argument("--q4nx-config", default=str(Path(__file__).resolve().parents[1] / "configs" / "k2.json"))
    ap.add_argument("--noise-high", type=int, default=None,
                    help="exclusive upper bound of the noise ids the adapter was trained on (default: the engine's)")
    a = ap.parse_args(argv)
    build(fetch_adapter(a.adapter), Path(a.config), Path(a.out), Path(a.q4nx_config), a.noise_high)
    return 0


if __name__ == "__main__":
    sys.exit(main())
