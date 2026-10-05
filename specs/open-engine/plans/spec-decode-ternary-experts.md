# Qwen3.6-A3B on the NPU: ternary / 2-bit experts + DFlash speculative decoding

2026-10-05. Status: **step 1 under way** (branch `feat/spec-decode`). Supersedes the iGPU
evaluation that was in this file earlier. §0 records what the existing designs and
measurements said once step 1 opened, and amends §1b and §2 where they were wrong.

## Direction (decided)

- **Target and drafter both on the NPU**, on the open engine (`src/open_qwen36`). This repo is
  the home repo.
- **Two expert formats**, ternary and 2-bit, for routed-expert gate/up. down_proj stays q4_1.
  Everything else stays at today's precision.
- **Models:** Josh's pruned Qwen3.6-27B-A2.8B, plus the popular 40-layer Qwen3.6-35B-A3B
  variants.
- **Drafter:** DFlash v1 first, then fine-tune it, and look at transplanting it into DFlash 2
  (§6).
- **No measurement phase; build directly.** Correctness checks stay in, as part of building.

## Expectations

Today the open engine decodes the 35B at ~140 ms/token (~7-8 tok/s through `oflm serve`).
Of that, 95 ms is the layer chain and 26 ms is the routed experts (spec, OPEN-QUANT-Q8 result).
Single-token decode on the NPU is **latency-bound, not bandwidth-bound**: it reads ~1.6 GB per
token at ~12 GB/s. That favours speculation. Verifying B tokens through the GEMM block route
amortises the 95 ms that decode pays one token at a time.

The cost that grows with B is routed experts. Verification streams the union of the experts
the block's tokens route to (uniform-routing bound: 57/layer at B=8 and 102 at B=16, against 8
for one token). That is exactly where ternary/2-bit experts pay: routed-expert bytes drop ~36%.

Rough estimate with q4 experts:

- verify(16) ≈ 2x a decode step
- τ ≈ 4-6
- result: **~2-2.5x, i.e. 15-20 tok/s**

With 2-bit/ternary experts the estimate rises to roughly 20-25 tok/s. These are NPU numbers. They
won't reach the 44 tok/s iGPU build. The fair bar is "the NPU at ~3x today's decode".

Corrections to the earlier version of this file:

- **DFlash v1 training is public**, through SpecForge (sgl-project), NVIDIA NeMo AutoModel
  and Red Hat `speculators` v0.5.0.
- **DFlash 2 training isn't.** I found an inference model class (`DFlash2DraftModel`), but no
  trainer.

## 0. What the code says, before step 1 (2026-10-05)

Three readings of the existing designs and results change step 1's shape. Nothing here
changes the direction; it changes which kernel the verify route is built out of.

### 0a. There is no T = 8/16 instruction stream to add

§1b said "today's streams are at T = 256". `designs/gemm_q4_prefill` compiles T **in**, and
its array mapping is the reason: the eight columns each take a `tile_n`-wide slice of T
(`assert t % (tile_n * n_aie_cols) == 0`, `tile_n` 32), so the whole array covers 256 token
columns at once and the weight is broadcast to all of them. At T = 16 seven of eight columns
would idle, and T = 16 is not even a legal build (`tile_n` must stay a multiple of the mmul's
8). The block route's GEMM is the wrong kernel for a small block, by construction.

The right one already exists. `designs/gemm_q4` is `gemv_q4`'s dataflow with an M dimension:
each core streams its own contiguous slice of the weight bands once and keeps M activation
tables, emitting M band accumulators per band. That is **band-parallel** -- work splits over N,
not over T -- so it is indifferent to how small B is, and the weight DMA per band is unchanged
from the GEMV the decode path already runs. Its docstring says what it was written to measure:
"how much compute headroom sits under the weight stream". That is exactly the verify route's
question, and the answer was never taken to a shipped kernel. Step 1's projection work is
promoting it: runtime band law (`N` / `K` out of the instruction stream, as `lm_head_q4` and
the 2026-09-13 context collapse already do), M = 8 / 16, and a recipe entry. B = 16 needs its
blocked dataflow (`GEMM_BLOCKED=1`): plain mode tops out at M = 8 and blocked at 24, L1 being
the ceiling, so 16 fits with room and does not have to be split into two passes over the same
weights.

`moe_batch` needs nothing: it is already the batched expert kernel at any multiple of
`nt = 8`, and the verify route uses it as prefill does.

### 0b. The term the Expectations section leaves out: hardware context changes

A context change costs **2.47 ms** into a GEMM context, **2.49** into the attention context and
**2.93** into the expert context, measured three ways and flat across 32x the streamed bytes
(`plans/archive/gemm-context-collapse.md`, gate 1). It is array reconfiguration, so it does not
scale with B at all.

That makes the route's own structure the first-order term of a verify pass:

| verify(B) built as | contexts entered per pass | cost before any arithmetic |
|---|---|---|
| the block route's stages (GEMM + attention + experts) | 2 per linear layer, 4 per full | **266 ms** (5.41 x 30 + 10.38 x 10) |
| one context per layer, as `lx` / `ax` decode | 22 per pass (the `lx` <-> `ax` change) | **~55 ms** |

266 ms is 1.9 decode steps spent on nothing. At tau = 5 that alone takes the predicted 2-2.5x
down to about 1.25x, and no host schedule hides it: `OFLM_OPEN_SUBMIT_AHEAD=2`, which queues
across a context change, **hangs the array** (OPEN-DECODE-PIPELINE).

So the binding constraint on the verify route is not a kernel's throughput, it is
**how many contexts one verify pass enters.** The small-B projections, the batched experts and
the block attention products want to be ONE xclbin with an instruction stream per stage --
`ax`'s own `part` trick, where one core program carries both halves of a layer and the stream
picks which runs. The constraint on that is program memory: 16 KB a core, and `lx` has already
linked at 16,480 B and been refused (OPEN-QUANT-Q8). It is plausible rather than certain,
because `gemm_q4_prefill`'s fused q4_1 dequant and `moe_batch`'s are already the same function
(`gqd_scale` is `mb_scale`'s twin, 2026-09-20), so the union of those two cores is much less
than the sum.

Worth noting what the same collapse is worth to prefill, where it is pure profit and needs no
speculation to pay off: **266 ms off every 256-token block**, against the 3.8 s the block costs
today.

### 0c. The expectation arithmetic, with measured bandwidth

The numbers in Expectations were estimates. Two measured rates make them checkable:

- decode reads ~1.6 GB a token in ~140 ms: **11.4 GB/s**
- `mb_s256` streams 256 expert visits -- 1.966 MB each at q4_1, 503 MB -- in 14.06 ms:
  **35.8 GB/s**, 3.1x what decode achieves

The batched expert kernel is the reason verify(16) can be cheap. At the uniform-routing bound,
102 experts a layer x 40 layers x 1.966 MB = **8.0 GB**, which at `mb`'s rate is **224 ms**,
not the 700 ms decode's own rate would imply. Adding the dense projections (the same ~0.97 GB
as one token, since B does not change the weight bytes) and one-context-per-layer switching:

| | experts | dense + host | contexts | total | vs a 140 ms decode step |
|---|---|---|---|---|---|
| verify(16), one context per layer | 224 ms | ~60 ms | ~55 ms | **~340 ms** | 2.4x |
| verify(16), the block route's contexts | 224 | ~60 | 266 | ~550 | 3.9x |

The first line is the plan's "verify(16) ~ 2x a decode step" and it survives contact with the
measurements. The second line is what step 1 produces if it is built on the block route. That
is the whole finding.

## 1. Shared building blocks

### 1a. Expert formats (OPEN-QUANT-T2, OPEN-QUANT-Q2)

On the NPU, **ternary packed at 2 bits costs the same bytes as 2-bit affine.** The only
difference is the dequant map:

- ternary: `{-1, 0, +1} × s`
- 2-bit affine: `q × s + m`, which has strictly more levels

So both formats share one packing and one kernel template, built as separate kernel sets
(compile-time dequant switch). They must not both live in one binary, because program memory
is tight: the all-q8 35B `lx0` has 80 B left. Base-3 packing (1.6 bpw) would make ternary
smaller, but at a decode cost; it's a later option.

**Proposed native layout:**
- 2-bit codes, laid out in the same 8k × 8 tile order the q4_1 nibbles use, so the
  `moe_batch` B-operand path (mask + convert) carries over
- a per-row-group scale (plus a min, for Q2)
- group size chosen per format: 64 or 128 for ternary, 32 or 64 for Q2. That works out to
  roughly 2.1-2.5 bpw.

**Sources.** Both come from GGUF, following the OPEN-QUANT-Q4K precedent of transcoding exactly
into the native format:
- **Q2:** `llama-quantize --imatrix` with `--tensor-type` overrides puts gate/up at **Q2_K**.
  The packer transcodes Q2_K's two-level scales into the native layout exactly.
- **Ternary:**
  - **post-training:** TQ2_0 (or our own absmean + calibration pass) → native T2
  - **healed checkpoint:** the safetensors pack directly

**Kernels touched:**
- `mx`, the per-token MoE inside `lx`, used by decode
- `mb_*`, the token-batched experts used by verify and prefill

Both get the 2-bit unpack and a dequant switch.

**Toolchain:**
- `q4nx-build` gets `--expert-quant {t2,q2}`
- the recipe's quant map gets `t2` / `q2` roles for `expert_gate_up`, with `spec_hash`
  covering them
- the validated-set rules stay as they are: new points enter after a hardware compare

### 1b. Verify route (OPEN-SPEC-VERIFY)

A small-T variant of the OPEN-PREFILL-BATCH block route that runs B ∈ {8, 16} tokens and
returns **logits at every position**. Prefill skips the lm_head.

New pieces:
- **The small-B projections.** NOT a T = 8/16 build of the block route's GEMM -- that design
  splits T across the array's eight columns and compiles T in (§0a). `designs/gemm_q4`,
  `gemv_q4`'s band-parallel dataflow with an M dimension, promoted to a shipped kernel with a
  runtime band law at M = 8 / 16. `moe_batch` needs nothing: it already handles any multiple of
  `nt = 8`.
- **One hardware context for the pass.** The projections, the experts and the attention
  products as instruction streams over one core program (`ax`'s `part`), because the block
  route's three contexts cost 266 ms a pass whatever B is (§0b). Program memory is the risk.
- **lm_head over B positions.** A block GEMM over N = 248320, plus top-k on the device so 16 MB
  of logits never reach the host.
- **Feature taps.** The residual at the drafter's target layers for every accepted position:
  layers `[1,6,11,16,22,27,32,37]` for the stock 35B.
- **Rollback to the accepted prefix k:**
  - full-attention KV: rewind the position pointer
  - conv state: keep the last 3 inputs per position
  - **GatedDeltaNet recurrent state** (32×128×128 per layer): keep the block-start state and
    the per-token (k, v, g, β) the verify already computed, then replay the update for the k
    accepted tokens. The recurrence already runs in the block route's host stage, so the
    replay is ≤16 small rank-1 updates per layer, not a re-run of the projections. Every
    orchestration wait stays PASSIVE: CPU spin throttles the NPU (`core.cpp:94`).

### 1c. DFlash v1 drafter on the NPU (OPEN-SPEC-DFLASH)

`z-lab/Qwen3.6-35B-A3B-DFlash` config:

- 6 Qwen3 dense layers: 5 sliding-window (4096) + 1 full
- hidden 2048; 32 heads × hd 128; 8 KV heads; ff 6144; qk-norm
- block 16, mask id 248077
- 8 target taps fused by an fc (16384 → 2048)
- ~0.39B parameters; **uses the target's embedding and lm_head**, so the drafter has no copy
  of its own

Per cycle:
1. **Inject context.** For the newly accepted tokens: fuse the taps (fc + norm), then project
   each drafter layer's K/V for them and append to the drafter's KV cache. After prefill,
   do the same for the whole prompt at T = 256.
2. **Draft.** Embed `[last token, mask × 15]`, run 6 layers with **non-causal attention within
   the block** over (context KV + block), then apply the shared lm_head to 15 positions and
   take top-1 (or sample).

Fit with what exists:
- hidden 2048, K 2048/4096/6144 and `ln` 2048 are all validated points
- the 32×128 / 8 KV attention geometry is the Qwen3-4B tuple

New pieces:
- non-causal block attention over [injected KV + block]
- the fc fuse
- T = 16 GEMM streams
- sharing the target's lm_head buffer from the drafter's context

Drafter weights start at q4_1, with a bf16 reference run on the host to confirm acceptance
holds. Kernel family: `dflash-qwen3-h2048` (one family xclbin, shared by every A3B variant).

### 1d. Orchestration (OPEN-SPEC-LOSSLESS)

`prefill → [draft → verify → accept/rollback → inject]*`, serial on one device.

- **Temperature 0:** greedy acceptance.
- **Temperature > 0:** standard rejection sampling against the drafter's distribution, so the
  output distribution equals the target's.
- **Lossless means:** at temperature 0, spec output equals non-spec decode except at logit
  near-ties. That is the standard the prefill route was held to; the verify path's bf16 GEMMs
  are not bit-identical to sequential decode.
- Exposed as `oflm serve … --draft <drafter>` / an auto-link from `oflm-add`.

## 2. Build order

| Step | Deliverable | Why this order |
|---|---|---|
| 1 | Verify route (1b) on the **stock 35B, q4_1**, with a B=1 self-check against sequential decode. In order: the host-stage rollback (unit, no hardware), then the one-context `designs/gemm_q4` promotion (§0a, §0b), then the block lm_head and the taps | Nothing waits on quant or training, and the rollback waits on no kernel at all |
| 2 | DFlash v1 drafter on the NPU (1c), drafts compared to the HF reference drafter fed the same features | Off-the-shelf drafter for this exact target |
| 3 | Orchestration (1d), greedy then sampling; `oflm serve` wiring; `oflm-test --llm --tools` with speculation on | First end-to-end speedup |
| 4 | T2 + Q2 formats (1a): converter, packers, `mx` + `mb` variants, two kernel sets; quality via logits corr against a fake-quant replica, then `oflm-test --llm --tools` | Formats plug into a working spec loop |
| 5 | Pruned 27B on the open engine through the manifest path (it ran only as the `make_27b.py` harness chain). Per-layer shapes match the 35B, so it should link to the **same family xclbins** with its own manifest | Required for its drafter |
| 6 | Drafter fine-tunes (§5): ternary/2-bit targets, the pruned model, top variants | Training runs off-box |
| 7 | DFlash 2 transplant (§6) | Research; v1 must work first |
| 8 | Healing pass for ternary, if step 4's tool-use check fails at post-training precision | Driven by step 4's result |

After each model is packed: a skill file (per AGENTS.md), a `q4nx-build` builder, and
`oflm-add` linking `model → target family + drafter family + drafter weights`.

## 3. Variants

| | Kernel family | Drafter |
|---|---|---|
| Stock 35B-A3B | `qwen36moe` ×{q4_1, t2, q2} | z-lab v1, as is |
| 40-layer fine-tunes (Ornith 1.0/1.5, Darwin-36B-Opus, Grug, BigBang, Aquila-mini — already spec-identical here) | same | z-lab v1; fine-tune only for the ones whose acceptance drops |
| Pruned 27B-A2.8B (30 layers, interval 3) | same family, own manifest | **Own drafter**: remap the 8 taps onto the surviving layers, then fine-tune (needs the pruning map) |

Each expert format ×{t2, q2} is a separate container per model. The drafter is shared across
formats, with an optional quant-specific fine-tune.

## 4. Requirements to add to `specs/open-engine/spec.md`

| ID | Verification | Notes |
|---|---|---|
| OPEN-QUANT-T2 | test + manual | Packer round-trip / TQ2_0 transcode exact (unit); hardware compare |
| OPEN-QUANT-Q2 | test + manual | Q2_K → native transcode exact (unit, like `test_quant_q4k.py`); hardware compare |
| OPEN-SPEC-VERIFY | test + manual | **Added to `spec.md` 2026-10-05, unit part DONE:** `deltanet_rollback` + `DeltaTape`, bit-exact against k single-token blocks at four k. Hardware procedure written, not run |
| OPEN-SPEC-DFLASH | manual | Draft tokens vs HF reference given identical features |
| OPEN-SPEC-LOSSLESS | test + manual | **Unit:** rejection sampler with a fixed RNG gives the target distribution. Hardware: greedy equality modulo near-ties |
| OPEN-SPEC-PACKAGING | manual | `oflm-add` links the drafter; serve picks it up |

This file IS that move (2026-10-05); `.claude/plans/qwen36-a3b-ternary-dflash-plan.md` is a
stale copy.

## 5. Fine-tuning DFlash v1

Use SpecForge's DFlash trainer in offline mode: precomputed target features, so only the
drafter needs to fit on the training GPU.

- **Quantized targets.** Ternary/2-bit gate/up shift the hidden states at the taps. Dump
  features from a fake-quant HF model that reproduces our exact dequant (same group sizes and
  rounding), on target-generated responses, and fine-tune from z-lab's weights. This is cheap,
  since it starts from a working drafter.
- **Pruned 27B.** v1 reads `target_layer_ids` 1, 6, 11, 16, 22, 27, 32, 37 of 40:
  1. map each tap to the surviving layer it became; where a tap layer was removed, use the
     nearest survivor
  2. keep fc's column blocks in the same order
  3. fine-tune

  `num_target_layers` becomes 30. Taps landing near cut points will need the most retraining.
- **Variants.** Fine-tune only where acceptance on that variant's own chat outputs falls well
  below the stock model's.

Fine-tuning needs GPU compute, which this box doesn't have; see decision 3.

## 6. DFlash 1 → DFlash 2

What changes, from the two released configs:

| | v1 (Qwen3.6-35B-A3B) | v2 (Qwen3.8-27B) |
|---|---|---|
| Block | 16 | **8** |
| Layers | 6 (5 SWA + 1 full) | 5 (all SWA) |
| Taps | 8 | 5 |
| Backbone | Qwen3 dense, block non-causal | + **two-tap dynamic convolution** (`conv_kernel_size 2`, `conv_group_size 16`) |
| Output | per-position argmax | **top-16 candidates per position + low-rank selector** (`selector_rank 256`) choosing one coherent path |
| Vocab / tokenizer | 248320 | 248320 (same family) |

**Why v2 suits this target specifically.** Block 8 at higher acceptance per position halves
the experts a verify has to stream (57 vs 102 per layer, uniform bound). On an
expert-union-limited verifier, that matters more than raw draft quality. The NPU cost of
v2's additions is small:
- the conv is two taps along the block dimension, a vector op
- the selector works on the top-16 the lm_head already produces

**Option A: transplant v1 → v2 at width 2048 (recommended).** v2 is the same Qwen3 dense block
at the same width, so:
1. copy v1's decoder layers and fc. Keep 8 taps: the config controls the tap count, and
   there's no reason to drop to 5
2. add the dynamic-conv generator initialised to the identity tap (`[0, 1]`), so the
   transplanted model computes exactly v1 at step 0
3. add a fresh selector
4. train the selector with the backbone frozen, then unfreeze and train at block 8

Risk: **there is no public v2 trainer.** We'd write the training loop: the v1 objective
(SpecForge's) plus the selector's path loss, reconstructed from the v2 model code and paper.
That is the real cost of v2.

**Option B: transplant Qwen3.8-27B's v2 drafter.** Widths are 5120 vs 2048, so every matrix
would have to be projected. Only the tokenizer is shared. Not worth it.

**Option C: v1 at block 8.** It costs nothing, and it gives the comparison that tells us
whether v2's gain comes from its architecture or just from the shorter block. Do this before
committing to Option A.

## 7. Decisions needed

1. **The pruning map:** which 30 of the original 40 layers the 27B kept. Needed for the tap
   remap.
2. **Variant list:** the 40-layer set already spec-identical here (above), or top-N by HF
   downloads?
3. **Training compute:** cloud GPUs for the drafter fine-tunes, the v2 transplant and any heal.
   What's available or budgeted?
4. **Ternary source:** start from llama.cpp TQ2_0 (quick, naive rounding), or write our own
   calibrated ternarizer (absmean + error feedback) from the start?
