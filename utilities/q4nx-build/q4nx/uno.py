"""K2-Horizon-7B-Uno's diffusion LoRA as the q4_1 tensors (uno.q4nx) the open engine's L-row pass reads (OPEN-UNO-LORA)."""
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


def derived(sd: dict, layer: int, scale: float, n_cores: int = 8) -> dict[str, np.ndarray]:
    """The eleven GEMV-shaped matrices of one layer (fp32), named as NAMES."""
    def ab(p):
        mod = "self_attn" if p in ("q_proj", "k_proj", "v_proj", "o_proj") else "mlp"
        pre = f"model.layers.{layer}.{mod}.{p}"
        if f"{pre}.lora_A.weight" not in sd:
            pre = "base_model.model." + pre
        return sd[f"{pre}.lora_A.weight"].float().numpy(), sd[f"{pre}.lora_B.weight"].float().numpy() * scale

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


def build(adapter_dir: Path, config: Path, out_dir: Path, q4nx_config: Path) -> Path:
    acfg = json.loads((adapter_dir / "adapter_config.json").read_text())
    if acfg.get("use_rslora"):
        raise ValueError("uno: an rsLoRA adapter scales by alpha / sqrt(r); this builder folds alpha / r")
    scale = float(acfg["lora_alpha"]) / float(acfg["r"])
    nl = json.loads(config.read_text())["num_hidden_layers"]
    sd = load_file(str(adapter_dir / "adapter_model.safetensors"))
    packer = _Packer(json.loads(q4nx_config.read_text()))
    tensors = {}
    for l in range(nl):
        for name, w in derived(sd, l, scale).items():
            tensors[f"model.layers.{l}.uno.{name}.weight"] = packer.pack(w)
        print(f"\r[uno] layer {l + 1}/{nl}", end="", flush=True)
    print()
    out = out_dir / "uno.q4nx"
    save_file(tensors, str(out), metadata={"uno_scale": repr(scale), "uno_rank": str(acfg["r"]),
                                            "uno_base": str(acfg.get("base_model_name_or_path", ""))})
    print(f"[uno] wrote {out} ({out.stat().st_size} B, {len(tensors)} tensors, scale {scale:g})")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--adapter", required=True, help="the PEFT adapter dir (adapter_config.json + adapter_model.safetensors)")
    ap.add_argument("--config", required=True, help="the base model's config.json")
    ap.add_argument("--out", required=True, help="the model dir to write uno.q4nx into")
    ap.add_argument("--q4nx-config", default=str(Path(__file__).resolve().parents[1] / "configs" / "k2.json"))
    a = ap.parse_args(argv)
    build(fetch_adapter(a.adapter), Path(a.config), Path(a.out), Path(a.q4nx_config))
    return 0


if __name__ == "__main__":
    sys.exit(main())
