# Plan: Qwen3.8-27B (Qwen3.5 dense at hidden 5120) on the open kernels

**Status:** done 2026-10-01 (Opus 5.5); merged into OPEN-FAMILY-QWEN35 and the new
OPEN-CONVERT-QWEN35-VHEADS. The user asked to skip review. Two things the plan did not
foresee: the 3-element x activations in a 2-deep fifo (now `prep_stream`), and the published
container's scrambled DeltaNet value heads -- a q4nx-build bug no kernel compare can see, found
against the HF source with `model/container_vs_hf.py` and fixed in the converter. The block
prefill route's 37 kernels were built and verified afterwards (OPEN-PREFILL-BATCH step 7,
#148): 7.8x faster prefill on a 300-token prompt, the same greedy output.
**Spec impact:** OPEN-FAMILY-QWEN35 gains a fifth size, and its glue criteria change for
widths beyond what the four published sizes exercise. **Modified requirement:**
OPEN-FAMILY-QWEN35 (acceptance criteria for the glue walk, the side-fill count, the norm
helper and the down projection; a 27B result). **New requirement:** none. Catalogue points
enter on the hardware pass, as before. Every change is gated on the geometry, so the four
published sizes and the 35B compile to the same `insts.bin` and stamps-only `final.xclbin`
(checked with `--check` against `origin/main` builds of the 9B q4_1, the 9B mixed, the 4B, the
0.8B and the 35B).

## The model

`Atomic-Germ/Qwen3.8-27B-NPU2` (20.1 GB `model.q4nx`, `model_type qwen3_5`): hidden 5120, 64
layers (48 linear / 16 full), FFN 17408, attention 24/4 at head dim 256 (partial RoPE 64,
gated), DeltaNet 16 key heads / **48 value heads** of 128, conv 4, vocab 248320, untied head.

## What the recipe refuses today, and the fix for each

Measured by running `qwen35.recipe` on the container's config (and bypassing each refusal to
see the next one):

| # | wall | at the 27B | fix |
|---|---|---|---|
| 1 | main core L1: the FFN down GEMV's activation table | K = 17408 -> 39 168 B of table; 69 888 B of 61 440 at `PER_CALL 1` | **split the down projection along K** at a 4 KB element boundary: K 8192 + 9216, both already-validated GEMV widths. Band law (`gemv_q4.h`): pool chunk `c` of a band covers k-tile `c/2`, so the first 64 chunks of every K=17408 band ARE a K=8192 band and the next 72 a K=9216 one -- the split is two DMA taps over the existing pool, no repack. Each half runs the ordinary `gy` GEMV into its own act region (`out2`, `out2b`); the table is sized for the wider half (20 736 B) and the core fits at `PER_CALL 1` (51 456 B). |
| 2 | norm helper core L1 (Tile(0,3); the recipe does not check it) | five 10 KB inputs + one output + stack = 67 584 B of 65 536 | **split norm helper**: the residual adds stream half by half (`ln_add2` / `ln_add3`: y_i = x_i + a_i [+ b_i]), the sum goes to DDR, and the norm reads it back through the existing `ln_nr`. Never more than three elements held: 46 KB. Stage 3 also absorbs the second down half (`xres = res + out2 + out2b`). |
| 3 | the standalone final `ln` (same element maths) | 6 x 10 KB + stack | the same split in `designs/ln/ln.py` for N > 4096: `[x_i a_i] -> y_i` twice, then `[y0 y1 w] -> xn`. |
| 4 | glue side channel | 14 fills (three xn halves) of 13 | **half-outer walk**: for each xn half, copy it once and run BOTH accumulators' tiles for that half; per half the side carries xn, alpha's rows, beta's rows = 3 fills, so 3 x 3 + 2 = 11. |
| 5 | 48 value heads vs dn_glue's 32-lane accumulator | `ab_lanes` 64, `AB_ELEMS` 160 vs 80 walked tiles | a **64-lane alpha / beta tile** (`glue_ab_w.cc`: 32 rows x 64 bf16 per 4 KB element, two 32-lane accumulators), acc buffers sized to the lanes; `kGrp` = 3 is already the knob's arithmetic (`DNGLUE_NHEAD=48`). |
| 6 | catalogue points | `ln 5120`, `lm_head_q8 K 5120`, `gemv_q4 K 5120`, `deltanet heads 48`, `attn (256, 24, 4, 64, ...)` | built with `OPEN_KERNELS_UNVALIDATED=1`, entered after the slice passes. `gemv_q4 K 17408` is never asked for -- the split asks 8192 and 9216. |

Gating: new behaviour switches on a recipe field that is false / zero / absent for every
existing geometry (`Ffn.DOWN_SPLIT`, `Linear.AB_LANES > 32` and the fill count, a norm-helper
budget check), and new `Layout` fields are emitted only when non-zero, so no existing
manifest, plan or build key moves for a reason other than the source hash.

Not changed: the engine (the manifest already carries every buffer size), the packer (the
split is DMA taps), q8 `linear_out` (hidden 5120 is not in `MIXED_CORE_FITS`, so the recipe
narrows it to q4_1 as it does the 4B's). The block prefill route is attempted; if its GEMM
shapes do not build, the 27B gets the sequential set only (`qwen35.gemm_route` returning None)
and that is recorded.

## Verification

- Unit (`specs/open-engine/tests/test_qwen35.py`, verification `test`): the 27B config
  derives; its recipe composes with the unvalidated override and refuses without it by the
  catalogue points only; the down split's taps cover each band's chunks exactly once and the
  two halves' K are element-aligned; side fills = 11; the norm helper's L1 sum is under budget;
  every published size's recipe output is unchanged (frozen layouts, existing tests).
- Regression: `--check` of the 9B (q4_1 and mixed), 4B, 0.8B and 35B builds against
  `origin/main`'s.
- Hardware (manual, OPEN-FAMILY-QWEN35's procedure): standalone glue compare at 48 heads;
  8-layer slice (six linear, two full), 3 greedy tokens, against `replica_qwen35.py`;
  the engine CLI bit-identical to the harness; all 64 layers through `chat.py`; tok/s.
- Speed expectation: ~17.5 GB of weights per token at the 9B's ~31 GB/s -> ~550-600 ms/token.
