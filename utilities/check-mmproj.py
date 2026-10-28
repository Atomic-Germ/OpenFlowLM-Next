#!/usr/bin/env python3
"""check-mmproj: a llama.cpp `mmproj` GGUF against the model's own safetensors.

The vision tower's GGUF reader (src/open_qwen36/vision/vit.cpp) interprets llama.cpp's
conventions: a matmul weight is [in, out] (its fastest dim is the input), the patch
embed is one [P, P, C, hidden] conv per temporal slice, and the patch vector is
(C, T, P_h, P_w). Getting any of those backwards produces plausible-looking numbers,
not a crash -- so this checks the mapping against the checkpoint the projector was
converted FROM.

No big downloads: both formats are self-describing, so the script reads their headers
and fetches only the tensors it compares, by byte range.

    python utilities/check-mmproj.py \
        --mmproj https://huggingface.co/<repo>/resolve/main/<mmproj>.gguf \
        --safetensors Qwen/Qwen3-VL-4B-Instruct

Exit status is 1 when a tensor disagrees, so it can gate a change.
"""
from __future__ import annotations

import argparse
import json
import struct
import sys
import urllib.request

GGUF_TYPES = {0: ("F32", 4), 1: ("F16", 2), 30: ("BF16", 2)}


def http_range(url, start, end):
    """Bytes [start, end) of a URL, or the whole prefix when end is None."""
    req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{'' if end is None else end - 1}"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.read()


def fetch_all(url):
    with urllib.request.urlopen(url, timeout=120) as r:
        return r.read()


def hf_url(repo, filename):
    return f"https://huggingface.co/{repo}/resolve/main/{filename}"


# ----------------------------------------------------------------- GGUF header
def gguf_header(url):
    """(kv, tensors{name: (type, dims, offset)}, data_base) from a GGUF's header."""
    head = fetch_all(url) if not url.startswith("http") else http_range(url, 0, 4 << 20)
    if head[:4] != b"GGUF":
        raise SystemExit(f"{url} is not a GGUF")
    p = 4
    version, ntensors, nkv = struct.unpack_from("<IQQ", head, p)
    p += 20
    if version < 2:
        raise SystemExit(f"GGUF v{version} unsupported")

    def rd_str():
        nonlocal p
        (n,) = struct.unpack_from("<Q", head, p)
        p += 8
        s = head[p:p + n].decode("utf-8", "replace")
        p += n
        return s

    def rd_value(t):
        nonlocal p
        if t == 4:      # u32
            v = struct.unpack_from("<I", head, p)[0]; p += 4; return v
        if t == 5:      # i32
            v = struct.unpack_from("<i", head, p)[0]; p += 4; return v
        if t == 6:      # f32
            v = struct.unpack_from("<f", head, p)[0]; p += 4; return v
        if t == 7:      # bool
            v = head[p] != 0; p += 1; return v
        if t == 8:      # string
            return rd_str()
        if t in (0, 1, 2, 3):
            fmt = {0: "<B", 1: "<b", 2: "<H", 3: "<h"}[t]
            v = struct.unpack_from(fmt, head, p)[0]; p += struct.calcsize(fmt); return v
        if t == 10:     # u64
            v = struct.unpack_from("<Q", head, p)[0]; p += 8; return v
        if t == 11:     # i64
            v = struct.unpack_from("<q", head, p)[0]; p += 8; return v
        if t == 12:     # f64
            v = struct.unpack_from("<d", head, p)[0]; p += 8; return v
        if t == 9:      # array: element type + count, then the values
            et, n = struct.unpack_from("<IQ", head, p)
            p += 12
            return [rd_value(et) for _ in range(n)]
        raise SystemExit(f"GGUF metadata type {t} not handled")

    kv = {}
    for _ in range(nkv):
        k = rd_str()
        (t,) = struct.unpack_from("<I", head, p)
        p += 4
        kv[k] = rd_value(t)
    tensors = {}
    for _ in range(ntensors):
        name = rd_str()
        (nd,) = struct.unpack_from("<I", head, p)
        p += 4
        dims = list(struct.unpack_from("<" + "Q" * nd, head, p))
        p += 8 * nd
        ty, off = struct.unpack_from("<IQ", head, p)
        p += 12
        tensors[name] = (ty, dims, off)
    align = int(kv.get("general.alignment", 32))
    data_base = p + ((align - p % align) % align)
    return kv, tensors, data_base


def gguf_tensor(url, tensors, data_base, name):
    """The tensor decoded to a float list, row-major in the GGUF's own fastest-first order."""
    ty, dims, off = tensors[name]
    if ty not in GGUF_TYPES:
        raise SystemExit(f"{name}: type {ty} not handled by this check")
    label, width = GGUF_TYPES[ty]
    n = 1
    for d in dims:
        n *= d
    raw = http_range(url, data_base + off, data_base + off + n * width)
    if label == "F32":
        return list(struct.unpack(f"<{n}f", raw))
    if label == "F16":
        return list(struct.unpack(f"<{n}e", raw))
    # BF16: widen by shifting into the top half of an f32
    return list(struct.unpack(f"<{n}f", b"".join(bytes((0, 0)) + raw[i:i + 2] for i in range(0, len(raw), 2))))


# ----------------------------------------------------------------- safetensors
def safetensors_index(repo):
    """tensor name -> (shard, dtype, [start, end]), reading the index (or the one shard)."""
    try:
        idx = json.loads(fetch_all(hf_url(repo, "model.safetensors.index.json")))
        shards = sorted(set(idx["weight_map"].values()))
        mapping = idx["weight_map"]
    except Exception:
        idx = None
        mapping = None
        shards = None
    if shards is None:
        # one shard: the file list has exactly one .safetensors
        meta = json.loads(fetch_all(f"https://huggingface.co/api/models/{repo}"))
        shards = [s["rfilename"] for s in meta.get("siblings", []) if s["rfilename"].endswith(".safetensors")]
        if len(shards) != 1:
            raise SystemExit(f"{repo}: {len(shards)} safetensors shards and no index; name the shard explicitly")
        mapping = {}
    out = {}
    for shard in shards:
        url = hf_url(repo, shard)
        (hlen,) = struct.unpack("<Q", http_range(url, 0, 8))
        header = json.loads(http_range(url, 8, 8 + hlen))
        for name, info in header.items():
            if name == "__metadata__":
                continue
            out[name] = (url, 8 + hlen, info["dtype"], info["data_offsets"], info["shape"])
    return out


def safetensor(spec):
    url, base, dtype, (start, end), shape = spec
    raw = http_range(url, base + start, base + end)
    n = len(raw) // (2 if dtype in ("BF16", "F16") else 4)
    if dtype == "F32":
        vals = list(struct.unpack(f"<{n}f", raw))
    elif dtype == "F16":
        vals = list(struct.unpack(f"<{n}e", raw))
    elif dtype == "BF16":
        vals = list(struct.unpack(f"<{n}f", b"".join(b"\x00\x00" + raw[i:i + 2] for i in range(0, len(raw), 2))))
    else:
        raise SystemExit(f"dtype {dtype} not handled")
    return vals, list(shape)


def max_err(a, b):
    """The largest difference, relative to the reference's RMS.

    NOT per-element relative error: a tower weight near zero has no meaningful relative
    error (1e-8 against 1e-8 is 100%), and reporting it hides that every element agrees
    to the last f16 bit. RMS-relative is the same thing as a cosine check, scaled.
    """
    n = 0
    ss = 0.0
    worst = 0.0
    for x, y in zip(a, b):
        n += 1
        ss += y * y
        d = abs(x - y)
        if d > worst:
            worst = d
    rms = (ss / n) ** 0.5 if n else 0.0
    return worst / rms if rms else 0.0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--mmproj", required=True, help="the mmproj GGUF (URL or local path)")
    ap.add_argument("--safetensors", required=True, help="the HF repo the projector was converted from")
    ap.add_argument("--tol", type=float, default=2e-2,
                    help="worst allowed max|diff| / rms (f16/bf16 rounding is ~1e-3)")
    a = ap.parse_args(argv)

    kv, tensors, data_base = gguf_header(a.mmproj)
    st = safetensors_index(a.safetensors)
    hidden = kv.get("clip.vision.embedding_length")
    patch = kv.get("clip.vision.patch_size")
    print(f"mmproj: {kv.get('general.architecture')} / {kv.get('clip.projector_type')}, "
          f"hidden {hidden}, patch {patch}, blocks {kv.get('clip.vision.block_count')}")

    failed = 0
    # (projector tensor, HF tensor, how the C++ reads it). The two towers differ in almost
    # every name: Qwen3-VL fuses qkv and has a 2-layer GELU MLP; Qwen2.5-VL keeps q/k/v
    # apart with a SwiGLU MLP and a 2-D position table the tower ignores.
    proj = kv.get("clip.projector_type")
    if proj == "qwen2.5vl_merger":
        checks = [
            ("v.patch_embd.weight", "patch_embed.proj.weight", "patch conv, slice 0"),
            ("v.patch_embd.weight.1", "patch_embed.proj.weight", "patch conv, slice 1"),
            # Qwen2.5-VL's checkpoint keeps ONE fused qkv and llama.cpp splits it; the
            # third index says which slice to compare against.
            ("v.blk.0.attn_q.weight", "blocks.0.attn.qkv.weight", "q_proj (1/3 of qkv)", 0),
            ("v.blk.0.attn_k.weight", "blocks.0.attn.qkv.weight", "k_proj (2/3 of qkv)", 1),
            ("v.blk.0.attn_v.weight", "blocks.0.attn.qkv.weight", "v_proj (3/3 of qkv)", 2),
            ("v.blk.0.attn_out.weight", "blocks.0.attn.proj.weight", "o_proj"),
            ("v.blk.0.ffn_gate.weight", "blocks.0.mlp.gate_proj.weight", "gate"),
            ("v.blk.0.ffn_up.weight", "blocks.0.mlp.up_proj.weight", "up"),
            ("v.blk.0.ffn_down.weight", "blocks.0.mlp.down_proj.weight", "down"),
            ("mm.0.weight", "merger.mlp.0.weight", "merger fc1"),
            ("mm.2.weight", "merger.mlp.2.weight", "merger fc2"),
        ]
    else:
        checks = [
            ("v.patch_embd.weight", "patch_embed.proj.weight", "patch conv, slice 0"),
            ("v.patch_embd.weight.1", "patch_embed.proj.weight", "patch conv, slice 1"),
            ("v.blk.0.attn_qkv.weight", "model.visual.blocks.0.attn.qkv.weight", "fused qkv, [in,out] -> [out,in]"),
            ("v.blk.0.attn_out.weight", "blocks.0.attn.proj.weight", "o_proj"),
            ("v.blk.0.ffn_up.weight", "model.visual.blocks.0.mlp.linear_fc1.weight", "fc1"),
            ("mm.0.weight", "model.visual.merger.linear_fc1.weight", "merger fc1"),
            ("mm.2.weight", "model.visual.merger.linear_fc2.weight", "merger fc2"),
            ("v.deepstack.5.fc1.weight", "model.visual.deepstack_merger_list.0.linear_fc1.weight", "deepstack fc1"),
            ("v.position_embd.weight", "model.visual.pos_embed.weight", "position table"),
        ]
    def resolve(name, third=None):
        """The checkpoint's tensor for a name, under either tower prefix.

        Qwen3-VL wraps it (`model.visual.`), Qwen2.5-VL does not (`visual.`) -- and both
        are written differently again by each converter, which is the point of reading the
        checkpoint rather than trusting a name.
        """
        for cand in (name, "model." + name, "visual." + name, "model.visual." + name):
            if cand in st:
                vals, shape = safetensor(st[cand])
                if third is not None:
                    rows = shape[0] // 3
                    if rows * 3 != shape[0]:
                        raise SystemExit(f"{cand} is not a 3-way fused qkv")
                    vals = vals[third * rows * shape[1]:(third + 1) * rows * shape[1]]
                    shape = [rows, shape[1]]
                return vals, shape
        return None, None

    for entry in checks:
        gname, hname, what = entry[0], entry[1], entry[2]
        third = entry[3] if len(entry) > 3 else None
        if gname not in tensors:
            print(f"  skip  {gname} (not in this projector)")
            continue
        svals, sshape = resolve(hname, third)
        if svals is None:
            print(f"  skip  {gname} ({hname} not in the checkpoint)")
            continue
        gvals = gguf_tensor(a.mmproj, tensors, data_base, gname)
        ty, dims, _ = tensors[gname]
        # both sides are row-major in their own layout; compare the raw storage after
        # the mapping this reader performs
        H = hidden
        P = patch
        if gname.startswith("v.patch_embd.weight"):
            # ggml [P, P, C, H], fastest = patch width; HF Conv3d [H, C, T, P, P]
            slice_idx = 0 if gname == "v.patch_embd.weight" else 1
            T = sshape[2]
            gw, gh, gc, go = dims
            want = []
            for h in range(go):
                for c in range(gc):
                    for ph in range(gh):
                        for pw in range(gw):
                            want.append(gvals[pw + gw * (ph + gh * (c + gc * h))])
            got = [svals[((h * gc + c) * T + slice_idx) * P * P + ph * P + pw]
                   for h in range(go) for c in range(gc) for ph in range(P) for pw in range(P)]
        elif gname.startswith("v.position_embd"):
            gw, gn = dims        # [hidden, positions]
            want = [gvals[h + gw * p] for p in range(gn) for h in range(gw)]
            got = svals
        else:
            gin, gout = dims     # ggml [in, out]
            want = [gvals[i + gin * o] for o in range(gout) for i in range(gin)]
            got = svals
        if len(want) != len(got):
            print(f"  FAIL  {what:28} {gname}: {len(want)} values vs {len(got)} in {hname}")
            failed += 1
            continue
        rel = max_err(want, got)
        ok = rel <= a.tol
        print(f"  {'ok  ' if ok else 'FAIL'}  {what:28} {gname:34} max|d|/rms={rel:.2e}")
        if not ok:
            failed += 1
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
