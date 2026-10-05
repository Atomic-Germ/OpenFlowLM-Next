# Speculative decoding for Qwen3.6-35B-A3B on the NPU

2026-10-05. Status: **PR 1 open as #169** (branch `feat/spec-decode`, on `main`).

This is one of three plans that came out of the original "ternary experts + DFlash" plan:

| plan | covers |
|---|---|
| **this file** | the verify pass, the drafter on the NPU, and wiring them into `oflm serve` (PRs 1-5) |
| [`expert-quant-2bit.md`](expert-quant-2bit.md) | 2-bit and ternary routed experts, built on PR #161's t2 format (PR 6) |
| [`dflash-drafters.md`](dflash-drafters.md) | the pruned 27B, drafter fine-tunes, DFlash 2 (PRs 7-8) |

## The idea

Decode on the NPU spends most of each step reading weights, one token at a time. A small
draft model guesses the next 8-16 tokens, the big model checks all of them in one pass (the
**verify pass**), and we keep the guesses up to the first wrong one. The weights are read
once for the whole block, so the check costs far less than decoding the same tokens one by
one.

- Target and drafter both run on the NPU, on the open engine (`src/open_qwen36`).
- Drafter: z-lab's DFlash v1 for Qwen3.6-35B-A3B, used as is.
- Output must be **lossless**: at temperature 0, the same tokens plain decode would give,
  except where the top two logits are a near-tie.

## Expected gain (measured where possible, 2026-10-05)

**The baseline.** On a q4_1 kernel set, #116 runs the decode layer loop in one hardware
context: **69-70 ms a token, 14.4 tok/s** (this machine, two prompts, 199 tokens each). The
official `Atomic-Germ/Qwen3.6-35B-A3B-NPU2` container stores attention and DeltaNet at q8,
which turns off both that and the block prefill route: **156-167 ms, ~6.2 tok/s.** The
verify pass is built on q4_1 kernels, so the comparison below is against 14.4 tok/s.

**The expert set a block touches, measured** (`utilities/spec-decode/route_union.py`, the
same two prompts, both kernel sets agree):

| block | distinct experts per layer | uniform-routing estimate |
|---|---|---|
| 8 tokens | 34-36 | 57 |
| 16 tokens | 53-56 | 102 |

Consecutive tokens reuse experts, so a verify pass reads 40-48% less expert weight than
first assumed.

**Estimated verify pass and speedup.** Experts at the batched kernel's measured ~33 GB/s
at these slot counts. Not yet measured: the other projections (~35 ms assumed), reconfiguring
inside one ELF (~40 ms a pass), the drafter (~12 ms), and how many guesses are accepted
(~4 plus the one bonus token at B = 8, ~5.5 plus one at B = 16).

| | experts | verify pass | tokens kept | tok/s | vs 14.4 tok/s |
|---|---|---|---|---|---|
| B = 8, q4 experts | 81 ms | ~185 ms | ~5 | ~27 | **~1.9x** |
| B = 16, q4 experts | 129 ms | ~240 ms | ~6.5 | ~27 | ~1.9x |
| B = 8, 2-bit experts | 52 ms | ~155 ms | ~5 | ~32 | **~2.25x** |
| B = 8, switching NOT fixed | 81 ms | ~410 ms | ~5 | ~12 | **slower than today** |

Fixing the switch cost inside the verify pass is still mandatory. B = 8 and B = 16 come out
even; acceptance on real chat decides between them.

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
(`plans/archive/gemm-context-collapse.md`). Built from the prefill route's pieces (GEMM,
attention, experts: three separate binaries), a verify pass switches 100 times, which is
266 ms before any arithmetic. #116 fixed this for plain decode only; the verify pass uses
different kernels and still has the problem.

**3. The expert kernel is fast enough to make verify cheap.** `mb_s256` reads expert
weights at 35.8 GB/s, faster than decode's own weight reads. Checking 16 tokens touches up to
102 experts per layer, about 8 GB in total, which takes 224 ms at that rate; 8 tokens touch up
to 57, about 4.5 GB and 125 ms. The experts are most of a verify pass's cost, which is why the
table above favours B = 8 and 2-bit experts.

## Cutting the switch cost inside the verify pass

#116 merged the two decode layer kernels into one binary, and its main cores are now at
16,256 of 16,384 B of program memory. There's no room to add the verify pass's batched
kernels to that binary.

The way that fits is the one the diffusion engine uses (PR #140): **one ELF, reconfigured by
register writes.** Each kernel set keeps its own program, and a switch rewrites the array's
configuration from the instruction stream. On this machine that costs 0.3-0.7 ms instead of
~2.5, and a whole image came out byte-identical
(`utilities/reconfig-probe/README.md` on `feat/one-context`). Program memory doesn't matter
here. The costs are more complex packaging (`compose_elf.py`) and a longer load.

That is **PR 2**. It also takes ~266 -> ~40 ms off every 256-token prefill block, which is
worth having on its own.

Measure first: what one register-write reconfiguration costs for the GEMM, expert and
attention sets. The diffusion numbers grow with configuration size, so they don't carry
over directly.

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
| **1** (#169) | this plan; the rollback and its test; OPEN-SPEC-VERIFY in `spec.md` | nothing |
| **2** | the verify pass's kernels (GEMM, experts, attention) in one ELF, switched by register writes. Also speeds up prefill | nothing |
| **3** | the verify pass on hardware: measure the real expert set per block first, then `gemm_q4` at B = 8/16, block lm_head, taps | 2 |
| **4** | the drafter on the NPU | 3, for the taps |
| **5** | wiring + `oflm serve` + `oflm-test --llm --tools` with speculation on: the first real speedup number | 3, 4 |

None of PRs 1-5 depend on #161. Only the 2-bit expert work (PR 6) does.

## Requirements

| ID | verification | status |
|---|---|---|
| OPEN-SPEC-VERIFY | test + manual | in `spec.md`; unit part passing; hardware not run |
| OPEN-SPEC-ONE-CONTEXT | test + manual | new, PR 2. The verify pass's (and the block prefill's) kernel sets in one ELF; bit-exact against the xclbin route |
| OPEN-SPEC-DFLASH | manual | new, PR 4. Draft tokens match the HF reference given the same features |
| OPEN-SPEC-LOSSLESS | test + manual | new, PR 5. Unit: the rejection sampler with a fixed seed gives the target distribution. Hardware: greedy output equal except near-ties |
| OPEN-SPEC-PACKAGING | manual | new, PR 5. `oflm-add` links the drafter; serve finds it |
