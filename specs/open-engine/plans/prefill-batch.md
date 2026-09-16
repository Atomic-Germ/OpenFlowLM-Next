# Plan: batched prefill -- the 35B first, then the dense families

**Status:** stage 1 implemented and measured on the 35B (2026-09-10, branch `prefill/35b-block`): `OPEN-PREFILL-BATCH` is in `spec.md` with its result paragraph. Stage 2 (`OPEN-MOE-BATCH`) landed 2026-09-12 (the kernel, the recipe, the driver; `OPEN-MOE-BATCH` is in `spec.md` with its result paragraph, 40.0 -> 24.0 s on the 1020-token prompt); the dense section is open.
**Detail:** `.claude/plans/prefill-batch-35b.md` (the stage costs, the expert
kernel design and its alternatives, the dense section). This file is the spec
impact.

## Spec impact

**New requirements**

| ID | what it specifies | category |
|---|---|---|
| `OPEN-PREFILL-BATCH` | a kernel set may carry a block prefill route per layer type (`gemm_block`: `kind` dense / linear / full, block 256, the GEMM steps for that kind against shape-keyed contexts, the const offsets the host stages read); the engine takes the route whenever the set carries it (`OFLM_OPEN_GEMM_BLOCK=0` disables), stays sequential for prompts with an image and below the measured crossover, runs attention only for the real tokens of a padded block, and leaves the KV rows and DeltaNet state exactly as the sequential path would; on the 35B the DeltaNet recurrence, attention, router and combine run on the host over the block, the shared expert is repacked to the band law at load, and the routed experts run per token (stage 1) or through the token-batched expert kernel (stage 2) | unit for the recipe emission, the shared-expert repack and the host stages against the numpy replica; manual for the hardware agreement (argmax and top-5 vs the sequential prefill on the 1005-token prompt, the family's near-tie exemption) and the recorded TTFT |
| `OPEN-MOE-BATCH` (stage 2) | the token-batched expert kernel (`designs/moe_batch`): eight token slots per expert visit, one expert per column, the four rows splitting its output rows; a kernel set carries one xclbin and one instruction stream per dispatch length (256 / 128 / 32 / 8 slots, `gemm_block.moe_batch`), and the route gathers each expert's tokens eight at a time, runs the shortest stream that holds the experts still owed tokens with the slots patched to them (`moebatch`), and scatters the outputs with the router weights until every token is served; the kernel agrees with the fp64 reference from the same pool bytes at rel_fro <= 5e-3, and the per-layer expert time on a 256-token block is bounded (the number recorded when it lands). `OFLM_OPEN_MOE_BATCH=0` runs mx per token instead | unit for the band offsets (from the packer's permutations), the recipe emission and the manifest schema; manual for the harness agreement / timing and the full-model check |

**Modified requirements**

- `OPEN-MANIFEST`: `gemm_block` gains `kind`, per-kind steps, the const
  offsets, `attn_kernel`, `act`, `sandwich`; the dense form is #39's five
  steps under `kind: dense`.
- `OPEN-BUILD-CACHE`: the key covers `designs/gemm_q4_prefill/*` and
  `npu_offload/gemm_rtp/npue.py` for every family that emits a route.
- `OPEN-FAMILY-QWEN36MOE` and the five dense family requirements: the result
  paragraph gains the route's TTFT and agreement numbers. No behaviour change
  to the sequential path.

**Removed:** none. `open_kernels/model/install_gemm_prefill_kernels.py` is
deleted when the dense section lands (superseded by the recipe; not a
requirement).

**Explicitly out of scope:** Qwen3.5 (`qwen35`: the same DeltaNet host stage
applies, but its dense FFN and its own layouts are a follow-up once the 35B
route exists); the mixed q8/q4 35B fine-tunes.

## Acceptance criteria, sketched

- 35B manifest: `linear_attention` carries `kind: linear` with steps
  `gemm_n12288_k2048` (qkv|z) and `gemm_n2048_k4096` (out); `full_attention`
  carries `kind: full` with `gemm_n9216_k2048` (q|k|v|gate) and
  `gemm_n2048_k4096` (o); both name the shared-expert contexts
  `gemm_n512_k2048` and `gemm_n2048_k512`; every N and K a multiple of 256.
- The shared-expert repack, applied to a stripe-law fixture, equals the
  band-law packing of the same tensor.
- The host DeltaNet, attention, router and combine stages on a 4-layer slice
  agree with `replica.py` (corr and max-error bounds recorded when they land),
  and the state BO after a block equals the sequential path's within bf16.
- Hardware: `--gemm-block` vs plain on the 1005-token prompt agrees on argmax
  and top-5 except documented near ties; `flm-test --llm` passes through
  `flm serve` with the route on; TTFT before and after recorded.
- Dense families: as the earlier sketch (five fixture specs, shape-keyed
  contexts, Gemma's `dxB_local` and sandwich flags, q8 emits no route,
  Granite equals the hand-built set).

## Order

1. 35B stage 1: recipe emission and tests; the host stages against the
   replica; the driver; the 4-layer slice on hardware; the full model.
2. 35B stage 2: the expert kernel experiment, gated on its per-layer bound.
3. The dense families (section 4 of the detail plan), Granite through the
   recipe last.

Merge `spec.md` for `OPEN-PREFILL-BATCH` when step 1 lands; `OPEN-MOE-BATCH`
with step 2; archive this plan after step 3.
