# Speculative decoding for Qwen3.6-35B-A3B on the NPU

2026-10-05. Status: **PR 1 under way** (branch `feat/spec-decode`).

This is one of three plans that came out of the original "ternary experts + DFlash" plan:

| plan | covers |
|---|---|
| **this file** | the verify pass, the drafter on the NPU, and wiring them into `oflm serve` (PRs 1-5) |
| [`expert-quant-2bit.md`](expert-quant-2bit.md) | 2-bit and ternary routed experts, built on PR #161's t2 format (PR 6) |
| [`dflash-drafters.md`](dflash-drafters.md) | the pruned 27B, drafter fine-tunes, DFlash 2 (PRs 7-8) |

## The idea

Decode on the NPU spends most of each step waiting rather than computing. A small draft
model guesses the next 8-16 tokens, the big model checks all of them in one pass (the
**verify pass**), and we keep the guesses up to the first wrong one. If checking 16 tokens
costs about 2.4 decode steps and ~5 guesses are accepted on average, decode gets about 2x
faster.

- Target and drafter both run on the NPU, on the open engine (`src/open_qwen36`).
- Drafter: z-lab's DFlash v1 for Qwen3.6-35B-A3B, used as is.
- Output must be **lossless**: at temperature 0, the same tokens plain decode would give,
  except where the top two logits are a near-tie.

## What the existing code says

Three findings from reading the designs, each backed by a measurement already in the repo.

**1. The prefill GEMM can't be shrunk to 16 tokens.** `designs/gemm_q4_prefill` splits the
tokens across the array's eight columns, 32 each, and the token count is compiled in. At 16
tokens, seven columns would sit idle, and the build isn't even legal. The right kernel
already exists as an experiment: `designs/gemm_q4` splits the work over the weight rows
instead, so it doesn't care how few tokens there are. In its blocked mode it takes up to 24
tokens, so 16 fits in one pass. `moe_batch`, the batched expert kernel, already handles 8 or
16 tokens.

**2. Switching kernel sets is the biggest cost, and it doesn't shrink with fewer tokens.**
Each switch between hardware contexts costs 2.5-2.9 ms
(`plans/archive/gemm-context-collapse.md`). Built from the prefill route's pieces, a verify
pass switches 100 times, which is 266 ms (about two decode steps) before any arithmetic.
Plain decode switches 22 times a step, between the two layer kernels `lx` and `ax`.

**3. The expert kernel is fast enough to make verify cheap.** `mb_s256` reads expert
weights at 35.8 GB/s, about 3x the rate decode manages. Checking 16 tokens touches up to 102
experts per layer, about 8 GB in total, which takes 224 ms at that rate.

Putting it together, for one verify pass of 16 tokens:

| | experts | everything else | switching | total | in decode steps |
|---|---|---|---|---|---|
| switching kept to ~0.4 ms each | 224 ms | ~60 ms | ~40 ms | **~325 ms** | **2.3** |
| switching as it is today | 224 ms | ~60 ms | 266 ms | ~550 ms | 3.9 |

The first row gives the ~2x speedup; the second gives about 1.25x. **Cutting the switch cost
decides whether this is worth doing.**

## Cutting the switch cost

There are two known ways to do it, both already working somewhere in the tree:

- **One merged image.** Put both layer kernels in one binary as separate instruction
  streams, so a step never switches. PR #161 does this for dense models
  (OPEN-DECODE-ONE-CONTEXT-DENSE, `dux.py`). The limit is each core's 16 KB of program
  memory, and the 35B's `lx` has already overflowed it once (16,480 B). So it may not fit
  for the MoE model.
- **One ELF, reconfigured by register writes.** Each kernel set keeps its own program, and
  switching rewrites the array's configuration from the instruction stream. On the
  diffusion side (PR #140) that costs 0.3-0.7 ms instead of ~2.1, and a whole image came
  out byte-identical. Program memory doesn't matter here. The cost is more complex
  packaging (`compose_elf.py`) and a longer load.

Try the merged image first, because it's simpler and #161 has built it. Fall back to the
ELF if program memory refuses. Either way this is **PR 2**, and it speeds up plain decode by
itself (22 switches a step become ~0-9 ms, on the order of 30% of a ~105 ms step), with no
drafter at all.

## The pieces

**Verify pass (OPEN-SPEC-VERIFY).** Run B = 8 or 16 tokens through every layer and return
the logits at every position.
- Projections: `designs/gemm_q4`, made into a shipped kernel, with the weight shape read at
  run time and B = 8 / 16.
- Experts: `moe_batch`, unchanged.
- lm_head over all B positions in one pass over its weights, with top-k on the NPU so the
  16 MB of logits never reach the host.
- Feature taps: the hidden state at the drafter's 8 target layers
  (`[1, 6, 11, 16, 22, 27, 32, 37]`) for every accepted position.
- Rollback to the accepted prefix k. **Done (PR 1).** Attention's KV cache only needs its
  position moved back. The DeltaNet state can't be run backwards, so we keep the state the
  block started from plus the per-token values the pass already computed, and replay k
  tokens (`DeltaTape`, `deltanet_rollback` in `block_host.cpp`). It matches k tokens of
  plain decode bit for bit.

**Drafter on the NPU (OPEN-SPEC-DFLASH).** `z-lab/Qwen3.6-35B-A3B-DFlash`: 6 small dense
layers at width 2048, ~0.39B parameters, and it reuses the target's embedding and lm_head.
Each cycle it takes in the newly accepted tokens' features, then drafts 15 tokens in one
pass with attention that sees the whole block at once. What's new: that attention, the
feature fuse (16384 -> 2048), and sharing the target's lm_head. Weights start at q4_1, with a
bf16 run on the host to check that acceptance holds.

**Wiring (OPEN-SPEC-LOSSLESS, OPEN-SPEC-PACKAGING).** `prefill -> [draft -> verify ->
accept/rollback -> feed back]`, one device, every wait passive (CPU spinning slows the NPU).
Greedy acceptance at temperature 0; standard rejection sampling above it, so the output
distribution is the target's. Exposed as `oflm serve ... --draft <drafter>`, and `oflm-add`
links a model to its drafter.

## PRs

| PR | contents | waits on |
|---|---|---|
| **1** | this plan; the rollback and its test; OPEN-SPEC-VERIFY in `spec.md` | rebase onto #161 (both edit `deltanet_block`) |
| **2** | one context for the 35B's decode (OPEN-DECODE-ONE-CONTEXT-MOE): merged image, else ELF | #161 (same `lx` / `ax` files) |
| **3** | the verify pass on hardware: `gemm_q4` at B = 8/16, block lm_head, taps | 2 |
| **4** | the drafter on the NPU | 3, for the taps |
| **5** | wiring + `oflm serve` + `oflm-test --llm --tools` with speculation on: the first real speedup number | 3, 4 |

Before PR 2's design is settled, measure two things:
1. whether `lx` and `ax` fit one merged image for the 35B (build it; the linker reports the
   size);
2. if not, what an ELF reconfiguration costs for `lx` / `ax` specifically. The diffusion
   numbers grow with configuration size, and these are larger kernels.

## Requirements

| ID | verification | status |
|---|---|---|
| OPEN-SPEC-VERIFY | test + manual | in `spec.md`; unit part passing; hardware not run |
| OPEN-DECODE-ONE-CONTEXT-MOE | test + manual | new, PR 2. The MoE twin of #161's OPEN-DECODE-ONE-CONTEXT-DENSE |
| OPEN-SPEC-DFLASH | manual | new, PR 4. Draft tokens match the HF reference given the same features |
| OPEN-SPEC-LOSSLESS | test + manual | new, PR 5. Unit: the rejection sampler with a fixed seed gives the target distribution. Hardware: greedy output equal except near-ties |
| OPEN-SPEC-PACKAGING | manual | new, PR 5. `oflm-add` links the drafter; serve finds it |
