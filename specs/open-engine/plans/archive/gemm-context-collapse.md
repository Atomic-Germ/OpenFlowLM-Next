# Plan: one GEMM xclbin for every K — collapse the block route's three GEMM contexts

**Status:** COMPLETE 2026-09-13, archived. The durable change is folded into
`spec.md` under `OPEN-PREFILL-BATCH` ("Result 2026-09-13 (one GEMM context for
the whole route)"), which is the current statement; this file is the working
record of how it got there. Branch `prefill/gemm-contexts`.

**End-to-end, the number that mattered:** the GEMM stage went **1489 -> 1096 ms
a block, a 393 ms saving**, and prefill on 2582 tokens **43.1 -> 38.9 s**, with
the identical eight-token greedy continuation. That is *larger* than the 273 ms
this plan predicted from the per-switch cost, because a switch costs ~3.37 ms
inside a block against the 2.47 the isolated probe reports. Every projection
whose context change was removed dropped 2.75-3.39 ms per dispatch; the two that
still follow the expert dispatch were unchanged (-0.17, +0.06), which is the
mechanism showing up kernel by kernel.

**Spec impact:** one modified requirement, `OPEN-PREFILL-BATCH`. Nothing new,
nothing removed, no manifest schema change.
**Detail:** `.claude/plans/gemm-context-collapse.md` in the worktree (the byte
diffs, the cost model, the fallbacks). This file is the spec impact.

## Why

Changing hardware context on the NPU costs 2.47 ms into a GEMM context and
2.93 into the expert kernel's (measured, three runs), and the 35B's block prefill
route does it 210 times per 256-token block: 539 ms of a ~3.8 s block. That is the largest removable item left. Three of the route's contexts are the
same GEMM built three times, once per K (2048, 4096, 512), because K is the
trip count of a loop inside the core program. Making it a runtime value leaves
one GEMM context and removes 110 of those 210 changes.

## The evidence this rests on

**K reaches the hardware as a single immediate in one instruction.** Diffing the
shipped kernel set `k35v6` with the export script's own `xclbin_equivalent`
(`open_kernels/export_qwen36_kernels.py:113`), no hardware needed:

- `gemm_n1024_k2048` vs `gemm_n12288_k2048` vs `gemm_n9216_k2048`: identical
  apart from build stamps. N is already instruction-stream-only.
- `gemm_n1024_k2048` vs `gemm_n2048_k4096`: same file size, **32 bytes differ**
  outside the stamps — one byte per AIE core, at the same offset inside each
  core's 4704-byte program image (2327, 7031, 11735, … step 4704). The byte is
  `0x1d` against `0x3d`: `NBG-1` (7 and 15) in bits [7:2] of a loop-setup field,
  where `NBG = K // 256` is the `range_(NBG)` bound in
  `designs/gemm_q4_prefill/gemm_q4_prefill.py`'s `core_fn`.

**The mechanism that removes it already ships here.** `pretiled_array` with
`rtp=True` (`npu_offload/gemm_rtp/gemm_pretiled.py:433-446, 554-560, 713-724`)
puts both loop bounds in a two-int32 per-core buffer written by the instruction
stream and ordered by a `WorkerRuntimeBarrier`. `designs/attn_block/attn_gemm.py:44`
turns it on, and the receipt is in the same kernel set: `ag_s256`, `ag_s2048`
and `ag_pv2048` — spanning K 256→2048 and N 256→2048 — are byte-equivalent
xclbins. One context already carries 32 instruction streams that way.

## Gate 1 result: the switch cost is per-context, not per-kernel (2026-09-13)

`--bench 8` on `k35v7`, three runs, with the interleaved-baseline probe
(`3b86aac4`) — each row's own baseline taken in the same loop as its probe, and
the delta quoted on minima. The `switch` column, all three runs:

| entered | work alone | run 1 | run 2 | run 3 |
|---|---|---|---|---|
| same context (`gemm_n1024_k2048`) | 0.82 ms | +0.075 | +0.048 | +0.024 |
| GEMM (`gemm_n2048_k4096`) | 2.93 | +2.359 | +2.464 | +2.470 |
| GEMM (`gemm_n2048_k512`) | 0.56 | +2.503 | +2.465 | +2.507 |
| attention (`ag_s4096`) | 1.76 | +2.480 | +2.476 | +2.506 |
| attention (`ag_pv4096`) | 1.69 | +2.511 | +2.493 | +2.465 |
| experts (`mb_s8`) | 0.57 | +2.931 | +2.943 | +2.931 |
| experts (`mb_s32`) | 1.87 | +3.001 | +2.897 | +3.136 |
| experts (`mb_s128`) | 7.14 | +2.928 | +2.886 | +2.912 |
| experts (`mb_s256`) | 14.06 | +2.887 | +2.915 | +2.944 |
| `mx_linear` | 0.49 | +0.594 | +0.598 | +0.598 |

**It does not scale.** The `mb_s*` ladder spans 0.57 to 14.1 ms of work and 32x
the streamed bytes, and its switch cost is 2.82–3.14 ms across all of it with no
trend — `mb_s8` and `mb_s256` are within 0.05 ms of each other. So the 110
GEMM-to-GEMM switches this change removes are **not** the cheap ones; the worry
that prompted this gate does not hold.

What the cost does depend on is **which context is entered**: experts 2.93,
GEMM 2.47, attention 2.49, `mx` 0.60. Every kernel sharing a context costs the
same to switch into, which points at the array reconfiguration rather than the
work. `mx_linear`'s 0.60 is the one genuine outlier and it is reproducible to
0.004 ms — but `mx` is not in the block route when the batched expert kernel is
on, so it does not enter the arithmetic.

Spread across the three runs is at worst 0.24 ms (`mb_s32`, whose run-3 own
min/median gap of 1.950/2.617 shows a single noisy rep) and typically under
0.1 ms, with 13–22 % background CPU throughout. The same-context control sits at
0.02–0.08 ms in all three runs.

**The arithmetic, with the measured costs** (attributing each change to the
context entered — GEMM 2.47, experts 2.93, attention 2.49):

| | today | after |
|---|---|---|
| linear layer (30) | 12.86 ms | 5.41 ms |
| full layer (10) | 15.35 ms | 10.38 ms |
| block | **539 ms** | **266 ms** |

**Saving 273 ms**, against the ~270 ms this plan estimated before the
re-measurement. The estimate held.

## Gate 2 result and the correctness gate (2026-09-13)

The 35B's five GEMM shapes built from the final design (`b508af8c`), compared
pairwise with `xclbin_equivalent`, then run against the fp64 reference and
against the pre-change kernel on the same vectors:

```
                    xclbin    insts   rel_fro   vs the pre-change kernel
gemm_n1024_k2048    203231    12560   2.173e-3  bit-exact
gemm_n12288_k2048   203231   133648   2.251e-3  bit-exact
gemm_n9216_k2048    203231   100624   2.222e-3  bit-exact
gemm_n2048_k4096    203231    23568   2.182e-3  bit-exact
gemm_n2048_k512     203231    23568   2.204e-3  bit-exact

all 10 xclbin pairs: SAME (71-74 bytes differ, all build stamps)
```

Bit-exact is the strong form and it holds: the hoist changes where K comes
from, not the arithmetic, so the outputs should match to the byte and they do.
The `rel_fro` figures are the same bf16 rounding the design always had (its
docstring records 2.19e-3 for the original).

**The K=512 anomaly is explained and gone.** It used to build to 195039 B - 192
bytes per core smaller than the others, diverging broadly rather than by one
immediate, and it was the open risk on this plan. A compile-time trip count of 2
got a different loop form out of the compiler; with the bound in a register
every K gets the same one.

**Program memory: the runtime bound costs 64 bytes a core.** The shared image is
203231 B against the old 201183, i.e. +2048 B over 32 cores. That is 1.4 % of the
~4.7 KB each core's program occupies, against a 16 KB budget - the objection a
reviewer would raise first, and it is small. (An earlier variant that also took
T from the buffer came out 2048 B *smaller*; that variant deadlocked and is not
what shipped, so the honest number is +64 a core, not -64.)

**Where K went is visible in the streams.** Diffing `gemm_n2048_k512` against
`gemm_n2048_k4096` - same length - the largest groups of differing words are 32
reading `2` against `16`, one per core at the head of the stream, plus the
activation-tap and weight-tap sizes. K is data now, not code.

## The deadlock the first attempt hit, and why the fix is not a barrier

`08433fb6` followed `gemm_pretiled.py`'s `rtp=True` path exactly, including its
`WorkerRuntimeBarrier`, and **hung on the first dispatch** (state 8, 7 s
timeout). Bisected on hardware, one variant per hypothesis:

| variant | result |
|---|---|
| no buffer, no barrier (rebuild of the pre-change design) | runs, 3.37 ms - and byte-equivalent to the shipped `k35v6` build, so the toolchain is faithful |
| RTP buffer written, bounds still compiled in, no barrier | runs, 3.83 ms |
| barrier only, no buffer | **hangs** |
| buffer + barrier | **hangs** |

So the barrier is the deadlock and the RTP buffer is innocent. The reason:
`WorkerRuntimeBarrier.set()` is a build-time operation, and the runtime releases
the barrier once per dispatch - but this worker body runs once per weight
row-block group, four times for a 1024-row projection. From the second group on
the core waits for a release that never comes. `attn_block` survives the same
idiom because its body runs exactly once per dispatch (`n_tiles_per_core` is 64
there, all inside one acquire).

**The fix needs no barrier at all.** Acquire the first weight band *before*
reading the band count: that acquire can only complete once the runtime has
issued the fill, which it does after the rtp writes, so the dataflow orders the
read. One peeled iteration. Verified in the generated MLIR - the A-fifo
`AcquireGreaterEqual` precedes the `memref.load` of the RTP slot, so the
compiler did not hoist it.

The cost is that `T_TILES` goes back to compile-time, since the peel needs a
bound before the first acquire. Every set this repo emits is T=256, so one
xclbin still serves every N and K, which is the whole point. If T ever varies,
that is a second context per T, not per K.

Nothing else in `gemm_q4_prefill` depends on K in the static design: the
accumulator is `[64, 32]` fp32 whatever K is, the weight fifo element is one
10240-byte band (that is what the 256-K band law buys), the taps live in the
runtime sequence, and `b_slice_tiles` — the only other place K shapes a type —
is under `b_reuse`, which `recommend_b_reuse` refuses unconditionally at eight
columns.

## Spec impact

**Modified requirements**

- `OPEN-PREFILL-BATCH`:
  - `spec.md:1240` — *"Hardware contexts are shared: one GEMM xclbin per K (the
    core program does not depend on N)"* becomes *"one GEMM xclbin for the whole
    route (the core program depends on neither N nor K: both are runtime loop
    bounds written by the instruction stream)"*.
  - `spec.md:1253`, the acceptance criterion — contexts `gemm_k2048`,
    `gemm_k4096`, `mx` become `gemm`, `mx`. The rest of the criterion (the
    per-kind steps, the `gemm_x_k{K}` / `gemm_y_n{N}` globals, the q8 refusal)
    is unchanged: the globals stay keyed by K and N, only the context name
    collapses.
  - A result paragraph recording the measured per-block switching cost before
    and after, once the gates below run.

**New:** none. **Removed:** none.

**Not modified, and why:** `OPEN-MANIFEST` — `contexts` is already a
name→path map and every kernel already carries a `context` string, so the
schema does not change, only the emitted content. `OPEN-BUILD-CACHE` already
covers `designs/gemm_q4_prefill/*` (`spec.md:278`).

**No driver change.** `Core::Core`'s wanted-kernel list already names every
`gemm_n*` kernel through `gemm_block.program` and `shared_program`
(`src/open_qwen36/core.cpp:102-116`), and `Core::context` keys off whatever
string the manifest gives it (`:161-170`). No new kernel kind, so no
`kerns_.at` throw at the first dispatch.

## Acceptance criteria, sketched

- The 35B emission names one GEMM context: `m["contexts"]` has `gemm` and no
  `gemm_k*`, and every `gemm_n{N}_k{K}` kernel carries `{"context": "gemm",
  "insts": ..., "build": ...}` (`tests/test_prefill_batch.py:81-94`). The
  `gemm_x_k{K}` = K·T·2 and `gemm_y_n{N}` = N·T·4 globals are unchanged.
- The parser and the fixture agree: `manifest_test.cpp:119-125` —
  `contexts.size()` 10 → 8, `files().size()` 61 → 59, and the assertion that
  `gemm_n9216_k2048`, `gemm_n1024_k2048` and `gemm_n2048_k512` all resolve to
  the one `gemm` context.
- ~~**The no-hardware gate:**~~ **met.** All five GEMM builds produce
  `final.xclbin` files that `xclbin_equivalent` reports as identical apart from
  build stamps (2026-09-13, `b508af8c`).
- ~~**Correctness:**~~ **met.** Every shape passes the fp64 reference at
  rel_fro ≤ 2.25e-3 and is bit-exact against the pre-change kernel.
- Correctness on hardware: each shape through the harness at
  `rel_fro <= 5e-3`, as `OPEN-PREFILL-BATCH`'s procedure step 2 already
  requires; the 1020-token run's greedy continuation unchanged.
- The point, on hardware: in `--bench`'s context-switch probe — which now takes
  its own interleaved baseline (`3b86aac4`) — every `gemm_n*` row reads within
  ~0.15 ms of its own-context baseline on minima, the way `gemm_n1024_k2048`
  (same context as the alternator) already does at +0.101 / +0.098 across the
  two runs on disk.

## The gate, honestly

**The original gate — under 200 ms of context switching a block — is not
reachable, and not by a margin that more work would close. Measured, the floor
is 266 ms.**

Today: 30 linear layers run `k2048 → k4096 → k2048 → k512 → mb` and back to the
next layer's `k2048`, five changes each; 10 full-attention layers insert the
attention context and make six. 210 changes, **539 ms** at the gate-1 costs.

After: linear becomes `g, g, g, g, mb` — two changes; full becomes
`g, ag, g, g, g, mb` — four. **100 changes, 266 ms.**

Eighty of those hundred are the GEMM↔expert alternation, and every layer forces
it: the experts cannot run before the projections that feed them. Removing it
would mean `gemm_q4_prefill` and `moe_batch` in one xclbin, and that is blocked
twice:

- **L1.** The q4 GEMM uses ~60 KB of the 64 KB per core (weight fifo 20480 B,
  activation 8192, output 16384, nibble scratch 4096, bf16 scratch 8192, stack
  4096). `moe_batch` uses ~45 KB (weight fifo 20480, activation 8192, output
  4096, the up/gate accumulators 8192, stack 4096). They do not both fit, and
  the fifos cannot alias — an ObjectFifo's buffers are statically allocated.
- **Topology.** The GEMM broadcasts weights per *row* down an L3→L2→L1 chain and
  joins C at the shim; `moe_batch` splits a four-row weight element per *column*
  at `Tile(c,1)` and joins there. One core program cannot have both wirings.

Those eighty are also the *expensive* ones — the expert context costs 2.93 ms to
enter against the GEMM's 2.47 — so what is left is 234 ms of expert switching and
50 ms of attention switching, and neither has a cheap fix.

**The gate should be restated as under 300 ms**, which this change meets at
266 ms. The remaining lever is not fewer contexts but fewer blocks: the switch
cost is per block, so a larger T would divide it per token. T is compiled in
(see the deadlock section), so that would be a second context per T rather than
per K, and T=512 also doubles the weight refetch. Its own measurement, not this
plan.

## What a reviewer will ask — all three now answered

1. ~~**The "2.8 ms flat" switch cost is inherited and in question.**~~
   **Closed.** `Core::bench_dispatch` measured every kernel's baseline in one
   phase and every probe minutes later, so drift landed in the deltas — the two
   older runs disagree by 5.6 ms on `mb_s256`'s mean, one of them reading a
   switch as free. `3b86aac4` rewrites that probe to alternate a baseline rep
   with a probe rep inside one loop and quote minima and medians, after the
   shape of `open_kernels/designs/moe_batch/stream_probe.py`. Three runs now
   agree to 0.1 ms typical, 0.24 worst.

   **`spec.md` and `.claude/plans/prefill-gap.md` both assert "2.8 ms flat,
   independent of the kernel's size".** That claim survives in substance — it is
   2.47–2.93 depending on the context entered, and flat within a context — but
   the numbers there were produced by the faulty probe and the "flat" was luck
   rather than evidence. Worth re-stating as per-context when those documents
   are next touched; not a correction that changes any conclusion.

   **The other five probes in `bench_dispatch` still have the phase-drift
   fault**, including the ones behind `prefill-gap.md`'s host-churn model. Read
   them with that in mind.

2. ~~**The `ag` context switch has never been measured.**~~ **Closed.**
   `43bd6e52` put the attention streams in the bench and they come in at
   +2.48/+2.51 — essentially the GEMM's own cost, which is what the plan had
   assumed. 50 ms of the residual, confirmed rather than guessed.

3. ~~**Why is the K=512 xclbin 192 bytes per core smaller?**~~ **Closed by gate
   2.** At `NBG = 2` the compiler emitted a different loop form for the
   compile-time bound; with the bound in a register it emits the same one for
   every K.

**New, and the one thing a reviewer should still push on:** the peel relies on
the compiler not hoisting the RTP load above the ObjectFifo acquire. It does not
today — checked in the generated MLIR, the `AcquireGreaterEqual` precedes the
`memref.load` — but nothing in the source *forces* that, and a future mlir-aie
could reorder it. A build that did would hang immediately and loudly rather than
compute quietly wrong answers, which is the good failure mode, but it is worth a
line in the design file if this ever moves toolchain versions.

## Order

1. ~~**The switch cost, re-measured with the interleaved baseline**~~ **DONE
   2026-09-13, passed** — see "Gate 1 result". The cost does not scale; the
   saving is 273 ms.
2. ~~**The five-build equivalence check**~~ **DONE, passed**, and the correctness
   gate with it: five shapes, bit-exact against the pre-change kernel
   (`b508af8c`).
3. **Next, and all that is left:** the recipe (`recipes/qwen36moe.py:988` —
   `ctx = "gemm"` instead of `f"gemm_k{K}"`), `test_prefill_batch.py:83-94`,
   `manifest_test.cpp:119-125` (contexts 10 → 8, files 61 → 59), the fixture
   regeneration, and the `spec.md` edit.
4. Then the block line on the 1020-token prompt with `OFLM_OPEN_DISPATCH_LOG=1`
   to confirm the 273 ms lands where the arithmetic says.
5. Merge the result paragraph into `spec.md` and archive this plan.

**Fallback:** none needed. The K=512 padding route is no longer relevant.
