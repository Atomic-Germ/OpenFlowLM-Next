# Plan: read Q4_K containers, and teach the converter to write them

**Status:** IMPLEMENTED 2026-09-08, merged into `spec.md` as `OPEN-QUANT-Q4K`.
Steps 1-5 done and green (13 unit tests either side of the NumPy / C++ line); step 6
done for the small model through the standalone engine, still open for `oflm serve` /
`oflm-test` and the 27B's native-Q4_K source. Unblocks Phases 1-4 of
`.claude/plans/open-engine-both-quants-and-distribution.md`, and corrects that
plan on two points (see below).

**Spec impact:** one new requirement, `OPEN-QUANT-Q4K`. `OPEN-PACK-PLAN` gains a
paragraph -- Q4_K joins q8 as a source form the three q4 chunk ops accept. No
existing requirement changes behaviour, and the frozen q4_1 pool bytes must stay
byte-equal. **No kernel change, no new kernel point, no build-key movement.**

## What changed since that plan was written

Two things, and both make this smaller than it looked.

**The 4736-byte layout is no longer unknown.** The both-quants plan says "32
rows x 144 = 4608 -- 128 B short of 4736, and I don't yet know what they are.
Pinning that down is step 1." It is pinned. ROCm merged `_pack_q4k` into
`OFLM_Q4NX_Converter` main -- the upstream of `utilities/q4nx-build` -- and its
docstring gives `q4k_block_t` from the OFLM 1.0.3 decoding kernels'
`model_spec.h` field for field. The missing 128 bytes are two bf16 per-row
super-block values, `S` and `M`, 64 bytes each. There is no llama.cpp
`block_q4_K` in the file at all: the 12 packed 6-bit bytes are expanded to 8 + 8
plain uint8, and the fp16 `d` / `dmin` are re-fitted to bf16.

**The kernel does not have to learn anything.** That plan's Phase 3 assumed a
Q4_K variant of the chunk loader in `gemv_q4.h`, with the 16 KB program memory
as the risk. It is not needed. Q4_K's effective per-group scale and min have
*exactly* the granularity the pool's q4_1 chunk already has -- one pair per
(row, 32-column group), 8 groups x 32 rows per chunk, the same 256-entry index
-- so a Q4_K chunk becomes a q4_1 pool chunk with two multiplies per entry and
one byte de-interleave. That puts this on the seam the q8 work already built
(`OPEN-PACK-PLAN`, "q8 sources"): the packer is the only place that knows.

## The format, concretely

One `q4k_block_t` per 32-row x 256-column chunk, everything column-major over
the chunk. 4736 B = 256 + 256 + 4096 + 64 + 64:

| field | bytes | index | note |
|---|---|---|---|
| `scales` uint8[8][32] | 0 : 256 | `g*32 + r` | per (32-column group, row) |
| `mins` uint8[8][32] | 256 : 512 | `g*32 + r` | same |
| `qs` uint4[256][16] | 512 : 4608 | byte `k*16 + r/2` | even row = low nibble |
| `S` bf16[32] | 4608 : 4672 | `r` | one super-block per chunk |
| `M` bf16[32] | 4672 : 4736 | `r` | **stored negated** |

Value: `w[r][k] = S[r] * scales[g*32+r] * q + M[r] * mins[g*32+r]`, `g = k/32`.
GGUF's Q4_K subtracts its min; the OFLM kernel adds both accumulators, so the
converter negates `M` on the way out and the reading above is a plain add.

The pool's q4_1 chunk is `d[256] bf16 | m[256] bf16 | 4096 nibble bytes`, `d`
and `m` indexed `g*32 + r`, nibble `(r/16)*4096 + k*16 + (r%16)`, read as
`nib*d + m` (`designs/gemv_q4/gemv_q4.h:12-18`). So the transcode is:

- `d[i] = bf16(S[i%32] * scales[i])` and `m[i] = bf16(M[i%32] * mins[i])` for
  `i < 256` -- the scale index is already identical, no permutation;
- nibbles: `q4_1[h*2048 + k*8 + j] = q4k[k*16 + h*8 + j]` for `h` in {0,1},
  `j < 8`. Q4_K keeps a column's 32 rows in 16 contiguous bytes; q4_1 splits
  rows 0-15 and 16-31 into two 2048-byte planes. In NumPy,
  `qs.reshape(256, 2, 8).transpose(1, 0, 2)`. The nibble values are unchanged:
  both formats hold unsigned uint4 against a (scale, min) pair, and both pair
  rows `(2b, 2b+1)` into a byte with the even row in the low nibble.

### What it costs

One thing, and it should be measured rather than argued: `S` is bf16 and
`scales` is uint8, so the exact product carries about 16 significant bits and
the pool holds 8. Collapsing it rounds each group's scale by at most a half-ulp
of bf16 -- `2^-8`, 0.39% relative -- correlated across that group's 32 weights.

For scale, against something already measured: the q8 -> q4_1 re-quantization
the packer does today costs 8% weight RMS and still scored 0.999682 logits
correlation on a 4-layer slice. 0.4% is twenty times smaller and should sit well
above the 0.99999 bar every family is held to -- but "should" is why the
acceptance criteria name a number against a Q4_K-faithful fp64 reference.

If that number ever disappoints, the fallback is the original Phase 3: a Q4_K
chunk loader in `gemv_q4.h` keeping `S` and `scales` separate. It buys back 8
bits of scale precision and costs a kernel design against 16 KB of program
memory. Not first.

## New requirement

### OPEN-QUANT-Q4K: the packers read Q4_K chunks
**Applies to:** openflowlm-next (`open_kernels/model/q4nx.py`,
`open_kernels/recipes/pack.py`, `src/open_qwen36/pools.cpp`,
`utilities/q4nx-build/q4nx/{gguf_tensor,model_converter}.py`)
**Test category:** unit (the transcode, both packers, the reference) + manual
(the hardware run: needs the NPU and a Q4_K container)

A tensor stored in 4736-byte Q4_K chunks shall be accepted by the three q4 chunk
ops (`std_perm`, `expert_stripes`, `expert_down`) and converted to the pool's
q4_1 chunk on the way in, exactly as a q8 source is. A Q4_K chunk and a q4_1
chunk hold the same 32-row x 256-column tile, so no chunk index law, plan,
manifest, kernel or build key changes. `q8_perm` continues to demand q8: a
kernel set built to stream a projection at q8 is not satisfied by Q4_K.

`utilities/q4nx-build` shall additionally be able to *write* Q4_K, because OFLM
1.0.3+ requires it for the 35B MoE projections -- packing those as q4_1 gives
infinite `////` decoding or SIGSEGV in the closed runtime, so this is not a
preference.

**Acceptance criteria:**
- A synthetic Q4_K chunk (random bf16 `S` / `M`, random uint8 scales and mins,
  random nibbles) transcoded to q4_1 and read back as `nib*d + m` equals
  `S*scales*q + M*mins` computed in f64 from the same bytes, every value within
  `2^-8 * (|S*scales*q| + |M*mins|)` and no more.
- The nibble de-interleave is exact: for every `(r, k)`, the uint4 read out of
  the q4_1 chunk at nibble `(r/16)*4096 + k*16 + (r%16)` equals the one read out
  of the Q4_K chunk at byte `k*16 + r/2`, nibble `r%2`.
- NumPy and C++ agree: both build the same synthetic Q4_K chunks, transcode, and
  assert the same FNV-1a (`tests/test_quant_q4k.py`,
  `src/open_qwen36/pools_test.cpp`).
- A Q4_K tensor put through `std_perm` lands at the same pool chunk positions a
  q4_1 tensor of the same shape does: the permutation is applied to transcoded
  chunks, exactly as it is to file chunks.
- `q8_perm` over a Q4_K tensor is refused naming the tensor and both byte
  counts. A chunk width that is none of 5120 / 8704 / 4736 is still refused,
  naming the width and what it probably is.
- The frozen q4_1 pools of `OPEN-PACK-PLAN` are byte-identical before and after
  (`tests/test_pack_plan.py` passes unmodified).
- `model/q4nx.py`'s `dq_tile` on a Q4_K tensor reads the transcoded q4_1 the NPU
  actually holds by default -- the convention q8 already follows, so a slice
  comparison measures the kernels -- and the container's own Q4_K values under
  `requant=False`, which is the transcode's own quality number.
- Writer and reader agree: a tensor packed by `_pack_q4k` from known float input
  and read back by `dq_chunks_q4_k` recovers the input to the Q4_K grid.
- **Manual:** a Q4_K container built by `q4nx-build` for a validated shape scores
  >= 0.99999 logits correlation with the same argmax and top-5 against the fp64
  replica reading the container's own Q4_K values, then passes
  `utilities/oflm-test --llm` through `oflm serve`.

## Changes

| file | change |
|---|---|
| `open_kernels/model/q4nx.py` | `CHUNK_Q4K = 4736`; `dq_chunks_q4_k` (the f64 reference dequant); `q4k_to_q4_1` (the transcode); a `dq_tile` branch; drop 4736 from `_refuse`'s guess list. |
| `open_kernels/recipes/pack.py` | share `q4k_to_q4_1`; `q4_chunks_of` accepts 4736 alongside 8704, batched the same way; `_chunk_guess` drops 4736. |
| `src/open_qwen36/pools.hpp/.cpp` | `Q4K_CHUNK = 4736`; `q4k_to_q4_1_chunks` beside `requant_q4_1_chunks`; `q4_source` accepts it; `chunk_guess` drops 4736. |
| `src/open_qwen36/pools_test.cpp` | the C++ half of the agreement test. |
| `utilities/q4nx-build/q4nx/gguf_tensor.py` | port `_refit_one_side` and `unpack_q4_k` from ROCm main; route `Q4_K` to the native path instead of dequantize-then-requantize (upstream measures that at 0.240 bits of ENOB and 0.375 bpw worse, with most of the damage in the tail). |
| `utilities/q4nx-build/q4nx/model_converter.py` | port `_pack_q4k`; add `Q4_K` to `default_tensor_type` and `get_ggml_type`; dispatch in `_pack`. Bring the two fixes that ride with it: `_requantize_to` on 3D expert weights, and the tied-embedding fix for Qwen3.5 4B. |
| `utilities/q4nx-build/configs/*.json` | `default_tensor_type: Q4_K` where OFLM 1.0.3+ requires it (35B MoE projections first); leave everything else where it is. |
| `specs/open-engine/spec.md` | `OPEN-QUANT-Q4K`; a Q4_K sentence in `OPEN-PACK-PLAN`'s "q8 sources" paragraph, which becomes "source forms". |
| `specs/open-engine/tests/test_quant_q4k.py` | new: the unit criteria above. |
| `src/open_qwen36/README.md` | both formats, one paragraph near the top -- after it is true. |

Roughly 150 lines of engine and packer code, plus the converter port: four
functions, a targeted merge. Our copy is Atomic-Germ's fork plus local work and
has diverged from ROCm main across the whole file -- do not re-sync it.

## Order (TDD)

1. `tests/test_quant_q4k.py` -- write the unit criteria, confirm they fail for
   the right reason (no `dq_chunks_q4_k`, no 4736 branch), not on a stub.
2. `model/q4nx.py`: reference dequant + transcode. The Python criteria go green.
3. `recipes/pack.py`: accept the source. `test_pack_plan.py` must still pass
   untouched.
4. `pools.cpp` + `pools_test.cpp`. The FNV-1a agreement goes green.
5. Converter port; the writer/reader round trip goes green.
6. Build real Q4_K containers and validate (below).
7. Spec, README, archive this plan.

## Validating without a shipped Q4_K container

There isn't one anywhere yet. Every container this repo has read is q4_1, q4_0,
q8 or a smaller geometry -- 31 local models scanned plus the 68-container HF
census in `.claude/plans/open-kernel-model-priority.md`; zero at 4736.
Atomic-Germ said on ROCm issue #21 that he would merge the Q4_K converter "and
then start updating some models on my huggingface", so shipped ones are coming.
They are not here.

So step 6 builds them, and the source is already on disk:

- **Fast loop: Qwen3.5-0.8B.** `qwen3.5-0.8b-condB-q8.gguf` (812 MB, in
  `.lmstudio`), converted with `default_tensor_type: Q4_K`. Shape row 8, already
  validated, 18.5 tok/s here. Exercises the re-quantize-into-Q4_K path.
- **The real one: Qwen3.6-27B-A2.8B.** `qwen36-27b-a2.8b-mtp-Q4KM.gguf` (16.5 GB)
  sits next to the q4_1 `model.q4nx` that was built *from that same GGUF* and is
  the validated 27B MoE. Converting it again at Q4_K gives an A/B on identical
  weights: same source, two containers, one kernel set, and the only difference
  is the format. That is the strongest check available, and it needs no
  download. Budget ~17 GB of free disk for the second container.

The honest limit: our reader is checked against our writer, both derived from
the same upstream source. If OFLM 1.0.4 moved the struct again, this catches
nothing. What it does guarantee is that a mismatch shows up as a **refusal
naming the width**, never as silently wrong weights -- the reader dispatches on
chunk width and knows exactly three.

## Not in scope

- A Q4_K-native GEMV (the precision fallback above).
- GGUF as a runtime source. It shares this dequant and gets cheaper once this
  lands, but it is a separate path -- issue #14.
- q4_0 containers (`container-formats.md`, still proposed) and the 1280 / 2560
  geometries. Different problems that happen to live at the same seam.
- Phases 5-8 of the both-quants plan: family-keyed kernel sets in `oflm-add`,
  release distribution, `oflm-test` coverage of the dense families. This plan is
  the reader and the writer, nothing else.
