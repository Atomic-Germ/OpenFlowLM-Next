# Plan: one GEMM xclbin for every K — collapse the block route's three GEMM contexts

**Status:** designed 2026-09-13, branch `prefill/gemm-contexts` (worktree
`C:/code/openflowlm-ctx`, based on `prefill/35b-block` at `052f364e`).
**Gate 2 PASSED 2026-09-13** (`08433fb6`, see "Gate 2 result" below): the design
change is in, all five of the 35B's GEMM shapes build to byte-equivalent
xclbins, and the K=512 anomaly is explained and gone. Gate 1 — the switch cost,
re-measured — is still owed and still decides whether the saving is the size
this plan claims. Nothing has run on hardware; the harness correctness gate is
owed too.
**Spec impact:** one modified requirement, `OPEN-PREFILL-BATCH`. Nothing new,
nothing removed, no manifest schema change.
**Detail:** `.claude/plans/gemm-context-collapse.md` in the worktree (the byte
diffs, the cost model, the fallbacks). This file is the spec impact.

## Why

Changing hardware context on the NPU costs about 2.5 ms between two GEMM
contexts — the best-measured case, and the only one this change removes — and
the 35B's block prefill route does it 210 times per 256-token block, roughly
550 ms of a 4.0 s block. That is the largest removable item left. Three of the route's contexts are the
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

## Gate 2 result (2026-09-13, WSL only, no NPU)

All five of the 35B's GEMM shapes built from the runtime-bound design
(`08433fb6`), then compared pairwise with `xclbin_equivalent`:

```
gemm_n1024_k2048    199135 B xclbin    13328 B insts
gemm_n12288_k2048   199135 B          134416 B
gemm_n9216_k2048    199135 B          101392 B
gemm_n2048_k4096    199135 B           24336 B
gemm_n2048_k512     199135 B           24336 B

all 10 pairs: SAME (71-77 bytes differ, all build stamps)
```

**The K=512 anomaly is explained and gone.** It used to build to 195039 B — 192
bytes per core smaller than the others, diverging broadly rather than by one
immediate, and it was the open risk on this plan. A compile-time trip count of 2
evidently got a different loop form out of the compiler; with the bound in a
register every K gets the same one, and K=512 now lands on the same 199135 B
image as the rest. (That common image is 2048 B — 64 per core — *smaller* than
the old K=2048 one, so nothing was added to pay for this.)

**Where K went is visible in the streams.** Diffing `gemm_n2048_k512` against
`gemm_n2048_k4096` — same length, 190 differing words of 6084 — the largest
groups are 32 words reading `2` against `16`, one per core, 12 words apart at
the head of the stream, plus 64 words of B-tap size and 32 of weight-tap stride.
K is data now, not code. All five instruction streams are distinct and each is
2304 B longer than its predecessor: the 32 RTP writes and 32 barrier sets.

**What this does not prove:** that the kernel still computes the right answer.
That is the harness gate (`rel_fro <= 5e-3` per shape) and it needs the NPU.

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
  build stamps (2026-09-13, `08433fb6`).
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
reachable, and not by a margin that more work would close.**

Today: 30 linear layers run `k2048 → k4096 → k2048 → k512 → mb` and back to the
next layer's `k2048`, five changes each; 10 full-attention layers insert the
attention context and make six. 210 changes, ~550 ms.

After: linear becomes `g, g, g, g, mb` — two changes; full becomes
`g, ag, g, g, g, mb` — four. **100 changes, ~282 ms.**

**The saving and the residual do not rest on the same evidence, and should not
be trusted equally.** All 110 removed changes are GEMM-to-GEMM boundaries, and
those are the two best-reproduced numbers in the tree: `gemm_n2048_k4096` at
+2.486 (k35v5) and +2.415 (k35v6) on minima, `gemm_n2048_k512` at +2.525 and
+2.468, agreeing within 3 % across two runs a day apart. **110 × ~2.47 ms
≈ 270 ms saved**, and that figure is solid.

The 282 ms *residual* is not. Eighty of the remaining hundred changes are
GEMM↔expert, where the two runs give `mb_s256` +3.777 and +0.930 on minima
(+4.791 and −1.361 on means — a negative switch cost, which is impossible), and
twenty are GEMM↔attention, which has never been measured at all. Taking the mb
switch at 2.9 ms puts the residual at 282; at k35v5's 3.8 it is ~354. **So the
residual is 280–350 ms and gate 1 below decides which.**

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

So the floor with three NPU programs in the route is the residual above, 280–350
ms. **The gate should be restated as under 350 ms** — under 300 only if gate 1
puts the GEMM↔expert switch near 2.9 rather than 3.8. What is not in question is
that this change removes ~270 ms. The remaining lever is not fewer contexts but fewer
blocks: with K *and* T as runtime bounds one xclbin serves every (K, T), so T
could rise above 256 without adding a context, and the switching cost is per
block. T=512 doubles the weight refetch, so it needs its own measurement and is
not part of this plan.

## What a reviewer will ask, and what is not settled

1. **The "2.8 ms flat" switch cost is inherited from 2026-09-12 and is now in
   question.** `Core::bench_dispatch` measured every kernel's baseline in one
   phase and every probe in later phases, so drift in what else the box was
   doing landed straight in the deltas. Across the two runs on disk the same
   kernel pair gives `mb_s256` +4.791 (k35v5) and −1.361 (k35v6) on means; a
   negative switch cost is impossible, and k35v6's baseline phase is the corrupt
   one (`gemm_n12288_k2048` alone reads 11.77 ms there against 8.99 in k35v5,
   and k35v6's own later phases record `mb_s256` at 16.26 ms against the 18.25
   its baseline phase claimed). The same-context control, `gemm_n1024_k2048`,
   sits at +0.101 and +0.098 — sound, which is what says the fault is in the
   baselines and not in the probe.

   Commit `3b86aac4` on this branch rewrites that probe to alternate a baseline
   rep with a probe rep inside one loop and report minima and medians instead of
   means, after the shape of `open_kernels/designs/moe_batch/stream_probe.py`.
   The other five probes still have the phase-drift fault.

   **`spec.md` and `.claude/plans/prefill-gap.md` both assert the flat 2.8 ms.**
   If the re-measurement moves it they are stale in place and should be marked
   so — not corrected before the number exists.

2. **The `ag` context switch has never been measured.** `bench_dispatch` never
   put the attention streams in its job list, so the ~2.5 ms assumed for `g→ag`
   and `ag→g` is an assumption carrying roughly 50–100 ms of the residual.
   Commit `43bd6e52` adds those two streams to the bench (built clean, host test
   suite passes; not yet run on the NPU), so the next `--bench 6` closes it.

3. ~~**Why is the K=512 xclbin 192 bytes per core smaller?**~~ **Closed by gate
   2.** At `NBG = 2` the compiler emitted a different loop form for the
   compile-time bound; with the bound in a register it emits the same one for
   every K, and K=512 now builds to the same image as the rest.

## Order

1. **The switch cost, re-measured with the interleaved baseline** (NPU, **still
   owed**). `--bench 8` on `k35v6ws` — the current expert kernel, ~1.15x faster
   than `k35v6`'s and byte-identical in its streams — with `3b86aac4` built in.
   Is the cost flat at ~2.8 ms or does it scale with what the kernel streams?
   Both the ~270 ms saving and the 280–350 ms residual rest on it, and the
   measurement underneath them will not currently support either. **If it
   scales, this plan needs rewriting before anyone builds further** — and
   `spec.md` and `prefill-gap.md` need their flat-2.8 claims marked stale.
2. ~~**The five-build equivalence check**~~ (WSL, no NPU). **DONE 2026-09-13,
   passed** — see "Gate 2 result" above. `08433fb6`.
3. Recipe (`recipes/qwen36moe.py:988`: one `gemm` context), the three test
   files, the fixture regeneration, the spec edit.
4. Hardware: the harness per shape, then `--bench 6` again for the after number,
   then the block line on the 1020-token prompt with `OFLM_OPEN_DISPATCH_LOG=1`.
5. Merge the result paragraph into `spec.md` and archive this plan.

**Fallback if step 1 fails.** Serve the K=512 shared-expert down projection from
the K=2048 program instead: no repack and no kernel change — zeroing activation
rows 512..2047 makes six filler bands harmless whatever bytes they hold, so the
weight tap repeats the two real bands four times. It removes one change per
layer (~100 ms) against ~52 ms of extra FLOPs on that dispatch. Worth doing only
if the main route does not land; the main route gives the same 100 ms for free.
