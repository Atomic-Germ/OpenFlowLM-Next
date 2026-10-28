"""gguf_weights: a FLUX.2 [klein] DiT GGUF, seen the way the bundle exporter wants it.

The export path (`utilities/dit-chain/export_bundle.py`) reads a checkpoint's
`transformer/` through `SafeTensors.get(name)`, which returns a weight in **HF layout**
(`[out, in]`, diffusers names). A GGUF of the same model has:

  * **BFL names** -- `double_blocks.0.img_attn.qkv`, `img_in`, `final_layer.linear` --
    where diffusers has `transformer_blocks.0.attn.to_q`, `x_embedder`, `proj_out`;
  * **ggml orientation** -- `[in, out]`, and llama.cpp/sd.cpp FUSE what diffusers keeps
    apart: one `qkv` per stream, one `linear1` for qkv+MLP in the single blocks.

So this module is a *view*: `GgufWeights.get(diffusers_name)` returns the HF-layout array,
splitting the fused tensors and transposing, which means `klein_pipeline.dit_weight_specs()`
and `fill_params()` take it exactly as they take a `SafeTensors`. Nothing about the packing,
the schedule or the kernels changes.

Quantization: F32, F16, BF16, Q4_0, Q4_1 and Q8_0 are decoded. The K-quants are refused by
name with the alternatives, because this is where a wrong decode would look like plausible
weights rather than an error -- and Q4_1 is what the rest of the tree runs anyway.

    python utilities/dit-chain/gguf_weights.py --check <diT.gguf>
        every weight the schedule needs resolves, builds, and has the right (K, N)

    python utilities/dit-chain/gguf_weights.py --check <diT.gguf> --safetensors <repo|dir>
        ... and the values agree with the checkpoint it was converted from
"""
from __future__ import annotations

import argparse
import json
import math
import mmap
import os
import struct
import sys
import urllib.request
from pathlib import Path

import numpy as np

TYPES = {0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 8: "Q8_0", 30: "BF16"}
BLOCK = {"F32": (1, 4), "F16": (1, 2), "BF16": (1, 2), "Q4_0": (32, 18), "Q4_1": (32, 20), "Q8_0": (32, 34)}


# --------------------------------------------------------------------------- header
class Gguf:
    """A GGUF's metadata, tensor table and data section (local file)."""

    def __init__(self, path):
        self.path = Path(path)
        self._f = open(self.path, "rb")
        size = os.fstat(self._f.fileno()).st_size
        self._mm = mmap.mmap(self._f.fileno(), 0, access=mmap.ACCESS_READ)
        if self._mm[:4] != b"GGUF":
            raise SystemExit(f"{path} is not a GGUF")
        self._parse(size)

    def _parse(self, size):
        mm = self._mm
        version, ntensors, nkv = struct.unpack_from("<IQQ", mm, 4)
        if version < 2:
            raise SystemExit(f"{self.path}: GGUF v{version} unsupported")
        p = 24

        def rd_str():
            nonlocal p
            (n,) = struct.unpack_from("<Q", mm, p)
            p += 8
            s = mm[p:p + n].decode("utf-8", "replace")
            p += n
            return s

        def rd_value(t):
            nonlocal p
            fmt = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f",
                   7: "<?", 10: "<Q", 11: "<q", 12: "<d"}
            if t == 8:
                return rd_str()
            if t == 9:
                et, n = struct.unpack_from("<IQ", mm, p)
                p += 12
                return [rd_value(et) for _ in range(n)]
            if t not in fmt:
                raise SystemExit(f"GGUF metadata type {t} not handled")
            v = struct.unpack_from(fmt[t], mm, p)[0]
            p += struct.calcsize(fmt[t])
            return v

        self.kv = {}
        for _ in range(nkv):
            k = rd_str()
            (t,) = struct.unpack_from("<I", mm, p)
            p += 4
            self.kv[k] = rd_value(t)
        self.tensors = {}
        for _ in range(ntensors):
            name = rd_str()
            (nd,) = struct.unpack_from("<I", mm, p)
            p += 4
            dims = list(struct.unpack_from("<" + "Q" * nd, mm, p))
            p += 8 * nd
            ty, off = struct.unpack_from("<IQ", mm, p)
            p += 12
            if ty not in TYPES:
                raise SystemExit(
                    f"{self.path}: {name} is {ty} ({'a K-quant' if ty in range(10, 16) else 'unsupported'}). "
                    "This reader decodes F32/F16/BF16/Q4_0/Q4_1/Q8_0 -- use a Q4_1 or BF16 GGUF "
                    "(Q4_1 is what the rest of the tree runs), or convert with q4nx-build.")
            self.tensors[name] = (TYPES[ty], dims, off)
        align = int(self.kv.get("general.alignment", 32))
        self.data_base = p + ((align - p % align) % align)

    def names(self):
        return set(self.tensors)

    def shape(self, name):
        """The tensor's row-major [rows, cols] shape (ggml stores fastest-first)."""
        return list(reversed(self.tensors[name][1]))

    def array(self, name):
        """The tensor decoded to f32 in **HF layout** (`[out, in]`, row-major).

        ggml stores a matmul weight as `[in, out]` with `in` varying fastest, which is the
        same bytes as an HF `[out, in]` array read row-major -- so the reshape is
        `dims[::-1]` and there is NO transpose to do. (Reshaping `dims` and transposing
        gives the right SHAPE with scrambled values: that is exactly how this read failed
        the first time, and why the numeric check exists.)
        """
        typ, dims, off = self.tensors[name]
        n = 1
        for d in dims:
            n *= d
        per, width = BLOCK[typ]
        raw = self._mm[self.data_base + off:self.data_base + off + n // per * width]
        return decode(np.frombuffer(raw, np.uint8), typ, n).reshape(dims[::-1])


def decode(buf, typ, n):
    if typ == "F32":
        return np.frombuffer(buf.tobytes(), "<f4", n).astype(np.float32)
    if typ == "F16":
        return np.frombuffer(buf.tobytes(), "<f2", n).astype(np.float32)
    if typ == "BF16":
        u = np.frombuffer(buf.tobytes(), "<u2").astype(np.uint32) << 16
        return u.view(np.float32).astype(np.float32)
    if typ == "Q8_0":
        b = buf.reshape(-1, 34)
        d = b[:, 0:2].copy().view("<f2").astype(np.float32)
        q = b[:, 2:34].copy().view(np.int8).astype(np.float32)
        return (d * q).reshape(-1)
    if typ == "Q4_0":
        b = buf.reshape(-1, 18)
        d = b[:, 0:2].copy().view("<f2").astype(np.float32)
        q = b[:, 2:18]
        lo = (q & 0x0F).astype(np.float32) - 8.0
        hi = (q >> 4).astype(np.float32) - 8.0
        out = np.empty((len(b), 32), np.float32)
        out[:, :16] = d * lo
        out[:, 16:] = d * hi
        return out.reshape(-1)
    if typ == "Q4_1":
        b = buf.reshape(-1, 20)
        d = b[:, 0:2].copy().view("<f2").astype(np.float32)
        m = b[:, 2:4].copy().view("<f2").astype(np.float32)
        q = b[:, 4:20]
        out = np.empty((len(b), 32), np.float32)
        out[:, :16] = d * (q & 0x0F) + m
        out[:, 16:] = d * (q >> 4) + m
        return out.reshape(-1)
    raise SystemExit(f"decode: {typ} not implemented")


# --------------------------------------------------------------------------- the map
def _split_fused(w, parts, axis=0):
    """One fused tensor -> `parts` equal slices along `axis`."""
    return np.split(w, parts, axis=axis)


def resolve(name):
    """(GGUF tensor, slice index or None, layout) for a diffusers weight name.

    layout is "T" (ggml [in, out] -> HF [out, in]), "1d" (as stored) or "Tf<k>"
    (transpose then take k-th of n equal row blocks, see FUSED below).
    """
    if name == "time_guidance_embed.timestep_embedder.linear_1.weight":
        return "time_in.in_layer.weight", None, "T"
    if name == "time_guidance_embed.timestep_embedder.linear_2.weight":
        return "time_in.out_layer.weight", None, "T"
    if name == "context_embedder.weight":
        return "txt_in.weight", None, "T"
    if name == "x_embedder.weight":
        return "img_in.weight", None, "T"
    if name == "proj_out.weight":
        return "final_layer.linear.weight", None, "T"
    if name == "double_stream_modulation_img.linear.weight":
        return "double_stream_modulation_img.lin.weight", None, "T"
    if name == "double_stream_modulation_txt.linear.weight":
        return "double_stream_modulation_txt.lin.weight", None, "T"
    if name == "single_stream_modulation.linear.weight":
        return "single_stream_modulation.lin.weight", None, "T"
    if name == "norm_out.linear.weight":
        return "final_layer.adaLN_modulation.1.weight", None, "T"

    if name.startswith("transformer_blocks."):
        rest = name[len("transformer_blocks."):]
        b, _, tail = rest.partition(".")
        p = f"double_blocks.{b}."
        qkv = {"attn.to_q.weight": (0, "img"), "attn.to_k.weight": (1, "img"), "attn.to_v.weight": (2, "img"),
               "attn.add_q_proj.weight": (0, "txt"), "attn.add_k_proj.weight": (1, "txt"),
               "attn.add_v_proj.weight": (2, "txt")}
        if tail in qkv:
            part, stream = qkv[tail]
            return p + f"{stream}_attn.qkv.weight", part, "Tf3"
        simple = {
            "attn.to_out.0.weight": "img_attn.proj.weight",
            "attn.to_add_out.weight": "txt_attn.proj.weight",
            "ff.linear_in.weight": "img_mlp.0.weight",
            "ff.linear_out.weight": "img_mlp.2.weight",
            "ff_context.linear_in.weight": "txt_mlp.0.weight",
            "ff_context.linear_out.weight": "txt_mlp.2.weight",
        }
        if tail in simple:
            return p + simple[tail], None, "T"
        norms = {
            "attn.norm_q.weight": "img_attn.norm.query_norm.scale",
            "attn.norm_k.weight": "img_attn.norm.key_norm.scale",
            "attn.norm_added_q.weight": "txt_attn.norm.query_norm.scale",
            "attn.norm_added_k.weight": "txt_attn.norm.key_norm.scale",
        }
        if tail in norms:
            return p + norms[tail], None, "1d"

    if name.startswith("single_transformer_blocks."):
        rest = name[len("single_transformer_blocks."):]
        j, _, tail = rest.partition(".")
        p = f"single_blocks.{j}."
        simple = {
            "attn.to_qkv_mlp_proj.weight": "linear1.weight",
            "attn.to_out.weight": "linear2.weight",
        }
        if tail in simple:
            return p + simple[tail], None, "T"
        norms = {"attn.norm_q.weight": "norm.query_norm.scale",
                 "attn.norm_k.weight": "norm.key_norm.scale"}
        if tail in norms:
            return p + norms[tail], None, "1d"
    return None


class GgufWeights:
    """`SafeTensors.get`-compatible view of a klein DiT GGUF (HF layout, f32)."""

    def __init__(self, path):
        self.gguf = Gguf(path)

    def get(self, name):
        r = resolve(name)
        if r is None:
            raise KeyError(f"gguf_weights: no mapping for {name}")
        tensor, part, layout = r
        w = self.gguf.array(tensor)          # already HF layout
        if layout in ("1d", "T"):
            return np.ascontiguousarray(w).astype(np.float32)
        if layout.startswith("Tf"):
            # a fused [q | k | v]: equal blocks along the OUTPUT axis
            n = int(layout[2:])
            return np.ascontiguousarray(np.split(w, n, axis=0)[part]).astype(np.float32)
        raise SystemExit(f"gguf_weights: unknown layout {layout}")


# --------------------------------------------------------------------------- checks
def check_shapes(gw):
    """Every weight the schedule needs resolves and builds at its (K, N)."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "open_kernels"))
    import klein_pipeline as kp

    failed = 0
    specs = kp.dit_weight_specs()
    for name, (K, N, build) in sorted(specs.items()):
        try:
            b = build(gw.get)
        except Exception as e:                    # noqa: BLE001 - report, do not stop
            print(f"  FAIL  {name:22} {type(e).__name__}: {e}")
            failed += 1
            continue
        ok = b.shape == (K, N)
        print(f"  {'ok  ' if ok else 'FAIL'}  {name:22} {b.shape}" + ("" if ok else f" != {(K, N)}"))
        failed += 0 if ok else 1
    print(f"{len(specs)} weights, {failed} failed")
    return failed


def _lookup(ckpt):
    """name -> HF-layout f32, from a local checkpoint dir or an HF repo (byte ranges).

    The ranges matter: the DiT is 8 GB in bf16 and a handful of tensors is all this needs.
    A repo id is used when the path does not exist locally.
    """
    path = Path(ckpt)
    if path.is_dir():
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import generate as G
        st = G.SafeTensors(path)
        return lambda n: np.asarray(st.get(n), np.float32)
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import hf_safetensors as st

    idx = st.index(str(ckpt), prefix="transformer/")

    def get(n):
        vals, shape = st.tensor(idx[n])
        return np.asarray(vals, np.float32).reshape(shape)

    return get


def check_values(gw, ckpt):
    """A few weights against the checkpoint the GGUF was converted from.

    Small tensors on purpose (a range read): the transposes, the fused-qkv split and a
    1-D norm all get covered without pulling the 8 GB checkpoint down.
    """
    get = _lookup(ckpt)
    pairs = [
        "x_embedder.weight",
        "proj_out.weight",
        "time_guidance_embed.timestep_embedder.linear_1.weight",
        "transformer_blocks.0.attn.to_q.weight",       # fused qkv, slice 0
        "transformer_blocks.0.attn.add_k_proj.weight", # fused qkv, slice 1 of the txt stream
        "transformer_blocks.0.attn.norm_q.weight",     # 1-D, BFL ".scale"
    ]
    failed = 0
    for name in pairs:
        a = gw.get(name)
        try:
            b = get(name)
        except Exception as e:                    # noqa: BLE001
            print(f"  skip  {name:44} ({e})")
            continue
        if a.shape != b.shape:
            print(f"  FAIL  {name:44} {a.shape} vs {b.shape}")
            failed += 1
            continue
        # A BF16 source must agree EXACTLY; a quantized one cannot. Q4_1 has 16 levels a
        # block, so its worst element in a tensor is about the tensor's own rms -- the
        # useful measures are the mean error and the correlation.
        a64, b64 = a.astype(np.float64).ravel(), b.astype(np.float64).ravel()
        rms = float(np.sqrt(np.mean(b64 ** 2))) or 1.0
        mean = float(np.mean(np.abs(a64 - b64))) / rms
        cos = float(np.dot(a64, b64) / ((np.linalg.norm(a64) * np.linalg.norm(b64)) or 1.0))
        exact = np.array_equal(a, b)
        ok = exact or (cos > 0.98 and mean < 0.2)
        print(f"  {'ok  ' if ok else 'FAIL'}  {name:44} "
              + ("exact" if exact else f"cos={cos:.5f} mean|d|/rms={mean:.2e}"))
        failed += 0 if ok else 1
    return failed


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("gguf", help="the klein DiT GGUF")
    ap.add_argument("--check", action="store_true", help="every schedule weight resolves and has the right shape")
    ap.add_argument("--safetensors", help="the transformer/ checkpoint directory (or an HF repo id) to compare values against")
    a = ap.parse_args(argv)
    if not a.check and not a.safetensors:
        ap.error("--check and/or --safetensors is required")

    gw = GgufWeights(a.gguf)
    print(f"{a.gguf}: {gw.gguf.kv.get('general.architecture')} / {len(gw.gguf.names())} tensors")
    failed = 0
    if a.check:
        failed += check_shapes(gw)
    if a.safetensors:
        failed += check_values(gw, a.safetensors)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
