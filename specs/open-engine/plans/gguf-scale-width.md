# Plan: how wide the GGUF pool's scale planes need to be

**Status:** evidence landed, decision open.
**Spec impact:** four new requirements proposed (`OPEN-GGUF-READ`,
`OPEN-GGUF-PACK`, `OPEN-GGUF-EMBED`, `OPEN-GGUF-CONFIG`). No existing
requirement changes. `OPEN-PACK-PLAN`'s frozen q4_1 bytes stay byte-equal;
`OPEN-PACK-Q4-0` keeps its meaning and gains a disambiguating name.

## Why

The GGUF-direct path gives a model its own pool layout — `spec.quant ==
"q4_1_f32"`, a 6144-byte q4 chunk and a 9216-byte q8 chunk whose `d`/`m` planes
hold the GGUF fp16 block scales widened exactly to f32. The stated reason is
fidelity: the file loses nothing, so a GGUF model is bit-identical to its
`.q4nx`-converted twin.

The cost is a second build of every design. On this branch that is `dx_f32`,
`lm_head_q4_f32`, five `gemv_q4s32_*` translation units, the `GEMV_Q4_PREFIX` /
`GEMV_Q4_WRAP` symbol machinery threaded through `designs/dense/gen_kernels.py`,
a nested `gguf` manifest the engine swaps in wholesale, a Gemma 3 12B stack
workaround (the f32 `dx_f32` main tile runs 768 B over a 6 KiB stack at hidden
15360), and an open numerics bug — `open-gguf-direct/SKILL.md` records
`lm_head_q8` f32-scale at maxrel ~5e-3 against the q4nx build's 4.5e-6, and
calls it the thing blocking the MoE family.

**But the kernel does not read f32 scales.** `load_scale` in
`designs/gemv_q4/gemv_q4.h:136-143` narrows them to bf16 on the way in, and its
own comment says so:

> f32 plane -> bf16 lanes via the native RNE convert […] the same result a q4nx
> chunk's pre-narrowed bf16 scales give.

fp16 → f32 is exact. So the kernel's RNE(f32 → bf16) and a host
RNE(fp16 → bf16) round the same real value into the same format. Same bits. The
f32 plane defers a rounding; it does not avoid one, and the "bit-identical to
its converted twin" property survives narrowing at pack time.

## What this change adds

`open_kernels/gguf_scale_width.py` — a self-test that checks it three ways, no
hardware required:

1. **The narrowing is one rounding, not two.** Exhaustively over all 63488
   finite fp16 bit patterns, the kernel's integer RNE applied to the widened
   f32 equals a host `astype(bfloat16)` of the fp16. 63488/63488.
2. **The two pools carry the same information.** For the same GGUF tensor,
   `q4_1_pack.pack_q4_1_pool` (5120 B, bf16) and `gguf_pool.pack_q4_pool`
   (6144 B, f32) hold **byte-identical codes**, the bf16 planes are exactly the
   kernel's narrowing of the f32 planes, and both dequantize to the same
   values under the kernel's own arithmetic. Q4_0 and Q4_1, rs = 2 and rs = 4.
3. **The q8 lm_head chunk too** — 9216 B f32 against 8704 B bf16, same codes,
   same values. That is the chunk the open numerics bug lives in.

`open_kernels/q4_1_pack.py` — Q4_0 support. `main` has had this file all along;
`PROVENANCE.md` calls it "the packer seed", and `designs/gemv_q4`,
`designs/gemm_q4_prefill` and `designs/moe_batch` already build their fixtures
from it. It was Q4_1-only. Both block widths now go through it, told apart by
the block width rather than by an argument, so no caller changes.

`open_kernels/gguf_pool.py` — `q4_chunk_bytes` wrote a zero min plane for Q4_0.
`pack_q4_pool` has always written `-8*d`; the single-chunk helper had not, and
nothing called it. Fixed, and the two are now checked against each other.

## The evidence that is not in this repo

`ggml-xdna` (github.com/Cyronius/ggml-xdna) ported `q4_1_pack.py` to C++ as
`hybrid/q4_pack.{h,cpp}` — both block types, `Q4_CHUNK_BYTES = 5120`, the same
nibble law `p = (r/16)*4096 + k*16 + (r%16)`. `hybrid/test-q4-pack.cpp` holds
the two byte-identical **on real GGUF tensors**, and its README records the
hardware run: Qwen3-1.7B weights read from a GGUF, packed that way, dispatched
through the stock kernels, correct inside the kernel's error budget at 1.44
TOPS.

So the bf16 path is not a proposal. It is packed, byte-checked against the
Python seed, and run on an NPU.

## What the decision would delete

If `std_perm_gguf` targets the 5120-byte chunk instead:

- the double export of every spec;
- `dx_f32`, `lm_head_q4_f32`, the five `gemv_q4s32_*` TUs, `GEMV_Q4_PREFIX`;
- the Gemma 3 12B 5 KiB-stack workaround;
- the `lm_head_q8` f32 numerics bug, and the MoE blocker with it;
- the landmine `open-gguf-direct/SKILL.md` already flags — that
  `designs/layer_x/gen_kernels.py` still emits unprefixed names, so the MoE f32
  phase fails with undefined `gemv_q4s32_*` exactly as the dense ones did;
- `Manifest::gguf` and the engine's whole-manifest view swap; a GGUF model's
  manifest becomes plain `q4_1`;
- 20% of the q4 pool's device bytes (6144 → 5120) and 6% of the q8 lm_head's
  (9216 → 8704). On a 22 GB model that is roughly 4 GB, on a box where 21 GB of
  NPU buffers is already the difference between fitting and paging
  (`q4nx_file.hpp`, `drop_pages`).

The `std_perm_gguf` edit itself is small — the code region is byte-identical
between the layouts, so it is `f32_to_bf16` on the two plane writes and the code
base moving from 2048 to 1024.

**One thing needs care.** The Q8_0 / Q4_K / Q6_K path derives its nibbles from
the fp16 `d`/`m` it just computed (`std::lround((v[i] - mf) * inv)`). Against
bf16 `d`/`m` the codes must be re-derived, and should use `main`'s existing law
in `requant_q4_1_chunks` — `m` rounded toward −inf, `d = (max − m)/15` rounded
toward +inf, so `[m, m+15d]` covers `[min, max]`. The round-to-nearest version
here has no such guarantee. That leaves one requant law in the tree instead of
two.

## Requirements this plan declares

None land with this change; they land with the behavior. Listed so the shape is
agreed before the code moves.

### `OPEN-GGUF-READ`: the engine reads a GGUF container
**Applies to:** OpenFlowLM-Next
**Verification:** test

GGUF v2/v3: typed KV metadata, tensor info, and tensor offsets relative to the
**aligned** data section (v3 fixed the alignment at 32 and dropped the field).

### `OPEN-GGUF-PACK`: GGUF blocks into the q4_1 pool chunk
**Applies to:** OpenFlowLM-Next
**Verification:** test

Q4_0 (with `m = −8d`) and Q4_1 pack byte-exact into the 5120-byte chunk;
Q8_0 / Q4_K / Q6_K go through `requant_q4_1_chunks`' law; every other type is
refused **by name**, pointing at `q4nx-build`.

### `OPEN-GGUF-EMBED`: the embedding row dequantizes on the host
**Applies to:** OpenFlowLM-Next
**Verification:** test

One row as f32 from F32/F16/BF16/Q4_0/Q4_1/Q8_0/Q4_K/Q6_K, against llama.cpp's
own loops.

### `OPEN-GGUF-CONFIG`: where a GGUF model's config comes from
**Applies to:** OpenFlowLM-Next
**Verification:** manual

`config.json` wins when the model dir has one. Without one, the checked fields
derive from GGUF KV. Fields GGUF cannot supply — Granite's folded multipliers —
are named and **refused rather than guessed**.

## A naming fix to fold in

`Q4_0` means two byte layouts in this tree. GGUF's nibbles are offset binary
(0 means −8); the `.q4nx` containers `q4nx-build` writes hold the same values as
two's complement (0 means 0, 8 means −8). Reading one as the other shifts every
weight by `8d`, silently, because both are valid. `OPEN-PACK-Q4-0` covers the
q4nx layout; give the GGUF one a distinct name in the refusal messages and in
`OPEN-GGUF-PACK` before the two paths meet.

## How to run it

```
python open_kernels/q4_1_pack.py          # Q4_1 and Q4_0 -> 5120 B bf16 chunks
python open_kernels/gguf_pool.py          # Q4_1, Q4_0, Q8_0 -> 6144/9216 B f32 chunks
cd open_kernels && python gguf_scale_width.py    # the two, against each other
```

All three pass as of 2026-09-20 (numpy 2.4.6, ml_dtypes 0.6.0).
