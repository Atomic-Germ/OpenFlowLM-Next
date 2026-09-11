# Plan: the `.q4nx` chunk formats -- which ones we read, and refusing the rest

Two axes: the block encoding inside a 5120-byte chunk (q4_1 vs q4_0), still
proposed; and the chunk size itself (q4_1 5120 vs q8 8704), **done 2026-09-06**
and written up under "q8 sources" at the end.


**Status:** proposed, not implemented.
**Spec impact:** one new requirement, `OPEN-QUANT-FORMAT`. No existing
requirement changes; `OPEN-PACK-PLAN`'s frozen q4_1 bytes must stay byte-equal.

## Why

OFLM ships two block encodings inside the same 5120-byte chunk, and the engine
only understands one of them. Measured 2026-09-06 by range-reading chunk 0 of
`model.q4nx` for every model Atomic-Germ publishes
(`.claude/plans/open-kernel-model-priority.md` has the full census):

| | `d` | `m` | value | models |
|---|---|---|---|---|
| **q4_1** | positive | real per-block minimum (`m ~= -7.4 d`) | `nib * d + m` | 60 of 68 containers |
| **q4_0** | **signed** | **identically zero** | `(nib - 8) * d` | LFM2 (all 6), Qwen2.5-3B-Instruct, Qwen2.5-3B-Coder-Instruct |

Evidence for the q4_0 reading, on Qwen2.5-3B-Instruct-NPU2 and LFM2-1.2B-NPU2,
`model.layers.N.mlp.down_proj.weight` chunk 0: all 256 mins exactly zero, `d`
of both signs, and 94% of blocks touch both nibble 0 and nibble 15 with mean
nibble ~7.0 — a symmetric 16-level code centred at 8, i.e. GGUF Q4_0
(`d = max / -8`, `q = round(x/d) + 8`).

Today `Q4nxFile` accepts any 5120-byte-chunk container and `pools.cpp` copies
the chunks verbatim, so a q4_0 model **loads and produces garbage**: with
`m = 0` the q4_1 formula gives every weight in a block the sign of `d`. Nothing
fails, nothing warns. That is the part worth fixing regardless of when the
q4_0 architectures land.

There is also a second axis the census turned up — chunk *geometry*. Gemma3-1B
and 270M use 1280-byte chunks (2048 values), gpt-oss and Whisper 2560 (4096
values), gemma4 several sizes for its embedding tensors. Those are already
refused (`q4nx_file.cpp:80`), which is correct; this plan only makes the
refusal message say what was found rather than guessing "OFLM 1.0.3 / Q4_K?".

## Why it is cheap

No kernel change. `(nib - 8) * d` is exactly `nib * d + m` with `m = -8 * d`,
and `-8 * d` is exact in bf16 — multiplying by 8 shifts the exponent and leaves
the mantissa alone. The packer already rewrites nothing but chunk *order*, so
substituting the min array as each chunk is copied into the pool is enough.
`gemv_q4.h` keeps computing `nib * d + m` and never learns there was a second
format.

## New requirement

### OPEN-QUANT-FORMAT: the engine reads q4_1 and q4_0 chunks and refuses the rest
**Applies to:** openflowlm-next (`src/open_qwen36/q4nx_file.cpp`, `pools.cpp`,
`open_kernels/model/q4nx.py`, `open_kernels/recipes/pack.py`)
**Test category:** unit

The container reader shall classify a `.q4nx` file's quantized block form at
open, from the file itself (nothing in the header records it), and the packer
shall emit pool chunks whose `nib * d + m` reading equals the container's
intended values for both forms.

- **q4_1** — some min is non-zero: chunks are copied verbatim, as today.
- **q4_0** — every min in the sampled tensor is zero *and* some `d` is
  negative: each pool chunk's min array is written as `bf16(-8 * d)` while the
  chunk is copied.
- **Ambiguous** — every min zero and every `d` non-negative: refused, naming
  the tensor sampled. (A real q4_1 tensor cannot have negative `d`, since
  `d = (max - min)/15`; a real q4_0 tensor of a non-degenerate weight matrix
  will have both signs. All-zero mins with all-positive `d` is neither, and
  guessing would silently corrupt weights.)
- **Any other chunk geometry** — refused at open, naming the byte count found
  and the tensor it came from.

**Acceptance criteria:**
- A synthetic q4_0 chunk (signed bf16 `d`, zero mins, random nibbles) packed
  through `recipes/pack.py` dequantizes, under the pool reading
  `nib * d + m`, to exactly `(nib - 8) * d` computed in f32 from the same
  bytes — element for element, no tolerance.
- `bf16(-8 * d) == -8 * d` for every bf16 `d` with `|d|` in
  `[2^-120, 2^120]` (a sweep over exponents and a random mantissa sample):
  the substitution introduces no rounding error.
- The frozen q4_1 pools of `OPEN-PACK-PLAN` are byte-identical before and
  after the change (`tests/test_pack_plan.py` still passes unmodified).
- A container whose mins are all zero and whose `d` are all >= 0 is refused,
  the error naming the sampled tensor.
- A container with 1280-byte quant chunks is refused, the error naming `1280`.
- `model/q4nx.py`'s `dq_tile` on a synthetic q4_0 tensor equals the same
  `(nib - 8) * d` reference (the replica must agree with the device path, or
  the fp64 oracle would score a correct model as broken).

## Changes

| file | change |
|---|---|
| `src/open_qwen36/q4nx_file.hpp/.cpp` | add `enum class Quant { Q4_1, Q4_0 }` and `Quant quant() const`. Detect at open: pick the largest non-`lm_head` `I8` tensor, scan the `d`/`m` arrays of its first N chunks (N ~= 4096, bounded so a 22 GB file costs a few MB of page faults), classify per the rules above, refuse the ambiguous case. Replace the "OFLM 1.0.3 / Q4_K?" guess in the chunk-size refusal with the byte count and tensor name. |
| `src/open_qwen36/pools.cpp` | add `static void q4_0_mins(uint8_t* dst, size_t nchunks)`: for each 5120-byte chunk, read `d[256]` at +0 and write `bf16(-8 * d)` over the 512 bytes at +512. Call it after the copy loops of `std_perm`, `expert_stripes` and `expert_down` when `m.quant() == Q4_0 && ch == 5120`. `put` and `conv_transpose` move bf16/small tensors and are untouched. |
| `open_kernels/model/q4nx.py` | same detection on the Python reader; `dq_chunks_q4_1` gains a `q4_0` branch (or a sibling `dq_chunks_q4_0`) so the replica and `dense_probe.py` read these containers correctly. |
| `open_kernels/recipes/pack.py` | the NumPy packer applies the same min substitution, so `test_pack_plan.py`-style byte comparisons hold for both forms. |
| `specs/open-engine/spec.md` | add `OPEN-QUANT-FORMAT` above `OPEN-PACK-PLAN`. |
| `specs/open-engine/tests/test_quant_format.py` | new: the six acceptance criteria above. |
| `src/open_qwen36/q4nx_test.cpp` (new) + `CMakeLists.txt`, `build.cmd` | a C++ unit on a synthetic 3-tensor container written to a temp file: q4_1 classified, q4_0 classified, ambiguous refused, 1280-byte chunks refused. Follows `manifest_test.cpp` (no XRT). |

Roughly 120 lines of code plus the tests.

## Order (TDD, both requirement halves are `unit`)

1. `test_quant_format.py` — write all six criteria, confirm they fail for the
   right reason (no `Quant`, no q4_0 branch), not on an exception stub.
2. `model/q4nx.py` detection + dequant; the Python criteria go green.
3. `recipes/pack.py` substitution; the pack criteria go green;
   `test_pack_plan.py` must still pass untouched.
4. `q4nx_file.cpp` + `pools.cpp` + `q4nx_test.cpp`; the C++ criteria go green.
5. Add the requirement to `spec.md`, move this plan to `archive/`.

## What this does NOT do

**No end-to-end validation.** Every q4_0 container belongs to a family the open
kernels cannot run for architectural reasons (LFM2 short-conv/delta-net,
Qwen2.5 dense + a vision tower — rows 15, 17, 21 of the priority plan). So the
device path stays unexercised until one of those recipes exists, and the
acceptance criteria above are all synthetic. That is the honest limit of this
change: it makes the packer *correct by construction* for q4_0 and makes a
misread *impossible to do silently*, and the first family to need it will be
the first real test.

If the goal is only to stop silent corruption, steps 1 and 4's refusal path are
enough on their own and can land alone.

## Alternatives considered

- **Refuse q4_0 outright.** Smaller, but throws away a correct 30-line packer
  change we already have the evidence for, and the refusal is most of the work.
- **Detect per chunk instead of per file.** Would catch a mixed container, but
  a q4_1 chunk whose 256 mins happen to be zero would be silently rewritten.
  Per-file with the `d`-sign cross-check is stricter.
- **Carry the format in the manifest.** The manifest describes the *kernels*,
  not the model file; two models sharing one kernel set can differ in
  container format (row 17 has one of each). The container has to say.


---

# Done 2026-09-06: q8 sources

**Status:** implemented. Requirement: the "q8 sources" paragraph of
`OPEN-PACK-PLAN` in `specs/open-engine/spec.md`.

## What was wrong

The chunk format was treated as a property of the FILE. `Q4nxFile` picked the
first non-`lm_head` I8 tensor, read its last shape dimension, and refused the
whole container if it was not 5120; `model/q4nx.py` did the same. That is only
true of the stock models. Five of the six most-downloaded Qwen3.6-35B
fine-tunes (Darwin, Grug, BigBang 1.0, Ornith 1.5, Aquila-mini) and
Atomic-Germ's own `Qwen3.6-35B-A3B-NPU2` mirror pack **251 of 733 tensors at
q8** -- attention, linear-attention and shared-expert projections -- and keep
only the routed experts, the bulk of the bytes, at q4_1. A Qwen3.5 dense
container stores `ssm_out_proj` and alpha / beta at q8. All of them were
refused on sight, with a message that guessed "OFLM 1.0.3 / Q4_K?" -- wrong:
Q4_K is 4736 bytes.

## What it does now

The chunk size is read per tensor (`Q4nxFile::chunk_bytes(name)`,
`Q4NX.chunk_bytes_of(name)`), nothing is refused at open, and the three chunk
ops -- `std_perm`, `expert_stripes`, `expert_down` -- dequantize a q8 chunk and
re-quantize it to q4_1 on the way into the pool. A q8 chunk and a q4_1 chunk
hold the same 32-row x 256-column tile, so **no chunk index law changes**: the
permutation is applied to the re-quantized chunks exactly as it is to file
chunks. Nothing above the packer knows -- no manifest field, no plan entry, no
recipe branch, no kernel change -- and an arbitrary q8 / q4_1 mix works.

Two consequences worth stating:

* **There is one mechanism, not two.** The `requant_q4_1` *op* that
  `qwen35.py`'s plan carried is gone; `ssm_out_proj` is now a plain `std_perm`,
  the same entry the 35B uses for its q4_1 copy of the same tensor. The
  arithmetic (`pools::requant_q4_1_chunks`, `recipes.pack.requant_q4_1`)
  stayed, as the shared function both the transparent path and the unit tests
  call.
* **A chunk size that is neither is refused where the tensor is read**, naming
  it, the byte count, and what the count probably is (8704 = q8, 4736 = Q4_K,
  1280 / 2560 = a smaller chunk geometry).

## The cost, and how it is reported

Re-quantizing q8 to q4_1 is the same 4-bit cost the stock 35B already pays on
those projections -- on the 9B's `ssm_out_proj` it measured a weight RMS
relative error of 0.0805 and a logits correlation of 0.999682 on a 4-layer
slice, with the same argmax and top-5 (`.claude/plans/q-qwen35-handoff.md`,
R3). Because that is three orders of magnitude coarser than the 1e-5 the
kernels are held to, the fp64 replica reads a q8 tensor as the **re-quantized**
values by default, so an acceptance number measures the kernels;
`make_decode.py --q8-weights` (`Q4NX.requant_q8 = False`) reads the container's
own q8 values, and that correlation is reported separately as the quality
number.

A main-core q8 GEMV remains the follow-up if a real evaluation disagrees; it
would be a kernel change, and nothing above the packer would have to move.
