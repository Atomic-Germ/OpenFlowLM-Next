"""Reproducible builder for the open Whisper engine's model repo (issue #72).

    q4nx-build --open-whisper -i openai/whisper-large-v3-turbo -o Whisper-V3-Turbo-OpenNPU2

It reads the HuggingFace checkpoint and writes the tensors in the shapes the open engine
consumes, so the engine does no rearranging beyond the DMA tiling of its GEMM operands:

* **Encoder GEMM operands are B = W^T, row-major [K, N], bf16.** Tiling for the NPU is
  deliberately NOT done here: it depends on the kernel set's tile tuple, and a container
  pre-tiled for one tuple is silently wrong for another (the b_layout_hash guard that
  NpuEmbeddings needs for exactly this). The engine tiles at load, from the tuple its
  kernel set records.
* **Fusions the engine's streams expect**: Q|K|V as one [1280, 3840] operand whose bias has
  a zero K third (k_proj has none); the four decoder layers' cross-attention K|V as one
  [1280, 10240] operand (k0|v0|k1|v1|...) with bias (0|bv0|0|bv1|...), computed once per
  encoded window; the conv stem as im2col operands with K ordered tap-major
  (K index = tap * C_in + channel), [3*128, 1280] and [3*1280, 1280].
* **Biases, LayerNorm parameters and position tables are f32** (the fp16 source converts
  exactly). Everything else is bf16, rounded to nearest-even from the fp16 source.
* **The decoder keeps transformers' names and [out, in] layout** in bf16; it runs on the
  host. `embed_tokens` doubles as the output head (tied). The cross-attention K/V
  projections are not duplicated: they live only in `dec.xkv`.

`tokenizer_config.json` gets `bos_token_id` and `eos_token_id` appended -- the host's
`setup_tokenizer` reads both and HuggingFace's file carries neither. These are the only two
fields FLM's shipped copy adds (diffed in NpuEmbeddings task 0179).

The weights file is named `model.open.safetensors`, not `model.q4nx`, so the closed
engine's reader can never be handed it. The output is byte-reproducible: tensors in sorted
order, sorted JSON, no timestamps or host paths.
"""

from __future__ import annotations

import json
import shutil
import struct
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from .open_causal import resolve_source, safetensors_header
from .open_embedding import sha256_file

WEIGHTS_NAME = "model.open.safetensors"
MANIFEST_NAME = "weights_manifest.json"
MANIFEST_FORMAT = "oflm-open-whisper-v1"
MODEL_INFO_ARTIFACT = "model_info_entry.json"

COPIED_FILES = ("config.json", "tokenizer.json")
OPTIONAL_FILES = ("generation_config.json", "preprocessor_config.json",
                  "special_tokens_map.json", "added_tokens.json")

# The only geometry the engine and its kernel set are built for. Anything else is refused
# here rather than producing a container no kernel set can serve.
GEOMETRY = {"d_model": 1280, "encoder_layers": 32, "decoder_layers": 4,
            "encoder_attention_heads": 20, "decoder_attention_heads": 20,
            "encoder_ffn_dim": 5120, "decoder_ffn_dim": 5120, "num_mel_bins": 128,
            "max_source_positions": 1500, "max_target_positions": 448, "vocab_size": 51866}

_NP_OF = {"F16": np.float16, "BF16": np.uint16, "F32": np.float32}


def bf16_bits(x: np.ndarray) -> np.ndarray:
    """fp32 -> bf16 bit patterns (uint16), round to nearest, ties to even."""
    u = np.ascontiguousarray(x, dtype=np.float32).view(np.uint32)
    return ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint16)


class _Reader:
    """Random access to the source safetensors without loading 1.6 GB at once."""

    def __init__(self, path: Path):
        self.path = path
        self.meta = safetensors_header(path)

    def __call__(self, name: str) -> np.ndarray:
        m = self.meta[name]
        dt = np.dtype(_NP_OF[m["dtype"]])
        count = int(np.prod(m["shape"])) if m["shape"] else 1
        a = np.fromfile(self.path, dtype=dt, count=count, offset=m["offset"]).reshape(m["shape"])
        if m["dtype"] == "BF16":
            a = (a.astype(np.uint32) << 16).view(np.float32)
        return a.astype(np.float32)


def conv_b(w: np.ndarray) -> np.ndarray:
    """Conv1d weight [C_out, C_in, 3] -> im2col operand [3*C_in, C_out], K = tap*C_in + c."""
    return np.ascontiguousarray(w.transpose(2, 1, 0).reshape(-1, w.shape[0]))


def whisper_tensors(read: _Reader, n_enc: int, n_dec: int, d: int) -> Dict[str, Tuple[str, np.ndarray]]:
    """Every output tensor as name -> (dtype, stored array): 'BF16' tensors are already
    bf16 bit patterns (uint16), 'F32' ones little-endian float32. Converting as each is
    read keeps the peak near one copy of the output (~1.6 GB), not a float32 one."""
    T: Dict[str, Tuple[str, np.ndarray]] = {}
    bf = lambda a: ("BF16", bf16_bits(a))                              # noqa: E731
    f32 = lambda a: ("F32", np.ascontiguousarray(a, dtype="<f4"))      # noqa: E731
    e, dcd = "model.encoder.", "model.decoder."

    T["enc.conv1.B"] = bf(conv_b(read(e + "conv1.weight")))
    T["enc.conv1.bias"] = f32(read(e + "conv1.bias"))
    T["enc.conv2.B"] = bf(conv_b(read(e + "conv2.weight")))
    T["enc.conv2.bias"] = f32(read(e + "conv2.bias"))
    T["enc.pos"] = f32(read(e + "embed_positions.weight"))
    T["enc.ln.w"] = f32(read(e + "layer_norm.weight"))
    T["enc.ln.b"] = f32(read(e + "layer_norm.bias"))
    zeros = np.zeros(d, np.float32)
    for i in range(n_enc):
        p, q = f"{e}layers.{i}.", f"enc.{i}."
        w = [read(p + f"self_attn.{n}_proj.weight") for n in "qkv"]
        T[q + "qkv.B"] = bf(np.concatenate(w).T)
        T[q + "qkv.bias"] = f32(np.concatenate([read(p + "self_attn.q_proj.bias"), zeros,
                                                read(p + "self_attn.v_proj.bias")]))
        T[q + "o.B"] = bf(read(p + "self_attn.out_proj.weight").T)
        T[q + "o.bias"] = f32(read(p + "self_attn.out_proj.bias"))
        T[q + "fc1.B"] = bf(read(p + "fc1.weight").T)
        T[q + "fc1.bias"] = f32(read(p + "fc1.bias"))
        T[q + "fc2.B"] = bf(read(p + "fc2.weight").T)
        T[q + "fc2.bias"] = f32(read(p + "fc2.bias"))
        T[q + "ln1.w"] = f32(read(p + "self_attn_layer_norm.weight"))
        T[q + "ln1.b"] = f32(read(p + "self_attn_layer_norm.bias"))
        T[q + "ln2.w"] = f32(read(p + "final_layer_norm.weight"))
        T[q + "ln2.b"] = f32(read(p + "final_layer_norm.bias"))

    xw, xb = [], []
    for l in range(n_dec):
        p = f"{dcd}layers.{l}.encoder_attn."
        xw += [read(p + "k_proj.weight"), read(p + "v_proj.weight")]
        xb += [zeros, read(p + "v_proj.bias")]
    T["dec.xkv.B"] = bf(np.concatenate(xw).T)
    T["dec.xkv.bias"] = f32(np.concatenate(xb))

    # Decoder: transformers' names, [out, in], minus the cross K/V that dec.xkv carries.
    for name in sorted(read.meta):
        if not name.startswith(dcd):
            continue
        if ".encoder_attn.k_proj." in name or ".encoder_attn.v_proj." in name:
            continue
        a = read(name)
        is_vector = a.ndim == 1 or name.endswith("embed_positions.weight")
        T[name.removeprefix("model.")] = f32(a) if is_vector else bf(a)
    return T


def write_safetensors(path: Path, tensors: Dict[str, Tuple[str, np.ndarray]]) -> None:
    """A minimal, deterministic safetensors writer (numpy has no bf16 dtype)."""
    header: Dict[str, dict] = {}
    offset = 0
    for name in sorted(tensors):
        dtype, a = tensors[name]
        header[name] = {"dtype": dtype, "shape": list(a.shape),
                        "data_offsets": [offset, offset + a.nbytes]}
        offset += a.nbytes
    hj = json.dumps(header, sort_keys=True, separators=(",", ":")).encode("utf-8")
    hj += b" " * (-len(hj) % 8)
    with path.open("wb") as f:
        f.write(struct.pack("<Q", len(hj)))
        f.write(hj)
        for name in sorted(tensors):
            f.write(np.ascontiguousarray(tensors[name][1]).tobytes())


def build_open_whisper_repo(source: str, output_dir: str, npu_assets: Optional[str] = None,
                            geometry: Optional[Dict[str, int]] = None) -> dict:
    """Build the repo. `geometry` overrides the required config values (tests only)."""
    if npu_assets:
        raise ValueError("--npu-assets is not used by --open-whisper: kernels are built by "
                         "open_kernels/export_whisper_kernels.py")
    src = resolve_source(source)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    cfg = json.loads((src / "config.json").read_text(encoding="utf-8"))
    wrong = {k: (cfg.get(k), v) for k, v in (geometry or GEOMETRY).items() if cfg.get(k) != v}
    if cfg.get("model_type") != "whisper" or wrong:
        raise ValueError(f"not whisper-large-v3-turbo's geometry (have, want): {wrong}")

    produced: List[str] = []
    for name in COPIED_FILES:
        shutil.copyfile(src / name, out / name)
        produced.append(name)
    for name in OPTIONAL_FILES:
        if (src / name).is_file():
            shutil.copyfile(src / name, out / name)
            produced.append(name)

    tok_cfg = json.loads((src / "tokenizer_config.json").read_text(encoding="utf-8"))
    tok_cfg.setdefault("bos_token_id", cfg["bos_token_id"])
    tok_cfg.setdefault("eos_token_id", [cfg["eos_token_id"]])
    (out / "tokenizer_config.json").write_text(
        json.dumps(tok_cfg, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    produced.append("tokenizer_config.json")

    weights = src / "model.safetensors"
    tensors = whisper_tensors(_Reader(weights), cfg["encoder_layers"], cfg["decoder_layers"],
                              cfg["d_model"])
    write_safetensors(out / WEIGHTS_NAME, tensors)
    produced.append(WEIGHTS_NAME)

    manifest = {
        "format": MANIFEST_FORMAT,
        "source": {"file": "model.safetensors", "sha256": sha256_file(weights)},
        "weights": WEIGHTS_NAME,
        "b_layout": "row-major [K, N] = W^T; tiled for the NPU by the engine at load",
        "conv_im2col_k_order": "tap-major: k = tap * C_in + channel, taps (t*s-1, t*s, t*s+1)",
        "fused": {"enc.{i}.qkv": "q|k|v, bias q|0|v",
                  "dec.xkv": "k0|v0|k1|v1|k2|v2|k3|v3, bias 0|bv0|0|bv1|..."},
        "tied": {"lm_head": "decoder.embed_tokens.weight"},
        "tensors": {n: {"dtype": tensors[n][0], "shape": list(tensors[n][1].shape)}
                    for n in sorted(tensors)},
    }
    (out / MANIFEST_NAME).write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n",
                                     encoding="utf-8")
    produced.append(MANIFEST_NAME)

    model_info = [{"type": "file", "oid": sha256_file(out / rel), "size": (out / rel).stat().st_size,
                   "path": rel} for rel in sorted(produced)]
    (out / MODEL_INFO_ARTIFACT).write_text(json.dumps(model_info, indent=1) + "\n", encoding="utf-8")
    return {"output_dir": str(out), "source": str(src), "files": sorted(produced),
            "tensor_count": len(tensors), "model_info": model_info}
