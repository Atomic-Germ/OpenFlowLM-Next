# Plan: one GEMM xclbin for every K — collapse the block route's three GEMM contexts

**Status:** designed 2026-09-13, branch `prefill/gemm-contexts` (worktree
`C:/code/openflowlm-ctx`, based on `prefill/35b-block` at `052f364e`). Not
built. The no-hardware equivalence gate below has not been run; it is queued
behind another agent's WSL builds.
**Spec impact:** one modified requirement, `OPEN-PREFILL-BATCH`. Nothing new,
nothing removed, no manifest schema change.
**Detail:** `.claude/plans/gemm-context-collapse.md` in the worktree (the byte
diffs, the cost model, the fallbacks). This file is the spec impact.

## Why

Changing hardware context on the NPU costs 2.4–2.9 ms and the 35B's block
prefill route does it 210 times per 256-token block — about 550 ms of a 4.0 s
block, the largest removable item left. Three of the route's contexts are the
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
- **The no-hardware gate:** all five GEMM builds produce `final.xclbin` files
  that `xclbin_equivalent` reports as identical apart from build stamps. If any
  pair does not, a K-dependence remains and the change stops.
- Correctness on hardware: each shape through the harness at
  `rel_fro <= 5e-3`, as `OPEN-PREFILL-BATCH`'s procedure step 2 already
  requires; the 1020-token run's greedy continuation unchanged.
- The point, on hardware: in `--bench`'s context-switch probe every `gemm_n*`
  row reads within ~0.15 ms of its alone time, the way `gemm_n1024_k2048`
  (same context as the alternator) already does at +0.09 ms.

## The gate, honestly

**The original gate — under 200 ms of context switching a block — is not
reachable, and not by a margin that more work would close.**

Today: 30 linear layers run `k2048 → k4096 → k2048 → k512 → mb` and back to the
next layer's `k2048`, five changes each; 10 full-attention layers insert the
attention context and make six. 210 changes, ~550 ms.

After: linear becomes `g, g, g, g, mb` — two changes; full becomes
`g, ag, g, g, g, mb` — four. **100 changes, ~282 ms.**

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

So the floor with three NPU programs in the route is ~282 ms. **The gate should
be restated as under 300 ms**, which this change meets with ~270 ms of headroom
removed rather than 350. The remaining lever is not fewer contexts but fewer
blocks: with K *and* T as runtime bounds one xclbin serves every (K, T), so T
could rise above 256 without adding a context, and the switching cost is per
block. T=512 doubles the weight refetch, so it needs its own measurement and is
not part of this plan.

## What a reviewer will ask, and what is not settled

1. **Why is the K=512 xclbin 192 bytes per core smaller, and diverging broadly
   rather than by one byte?** `gemm_n2048_k512` is 195039 B against 201183, and
   unlike the K=2048/K=4096 pair the difference is not a single immediate. At
   `NBG = 2` the compiler evidently emits a different loop form. This is
   unexplained, and it is the single most likely reason the no-hardware gate
   fails. It is also the reason that gate runs before any other work.
2. **The `ag` context switch has never been measured.** `Core::bench_dispatch`
   never put the attention streams in its job list, so the ~2.5 ms assumed for
   `g→ag` and `ag→g` is an assumption carrying roughly 100 ms of the 282 ms
   estimate. Commit `43bd6e52` on this branch adds those two streams to the
   bench (built clean, host test suite passes; not yet run on the NPU), so the
   next `--bench 6` closes this.
3. **2.4–2.9 ms per switch, not a flat 2.8.** In `bench_k35v6.log`'s
   context-switch probe `mb_s256` costs +0.93 ms and `mx_linear` +0.45, which do
   not fit a flat model. The 282 ms figure could plausibly be 250 or 310.

## Order

1. **The no-hardware gate first.** Hoist the loop bounds to runtime parameters
   in `designs/gemm_q4_prefill/gemm_q4_prefill.py`, build all five GEMM sets in
   WSL, and check every `final.xclbin` against every other with
   `xclbin_equivalent`. Nothing else is worth doing until that passes. It needs
   no NPU — only the toolchain — and it retires risk 1 above.
2. Recipe (`recipes/qwen36moe.py:988`: one `gemm` context), the three test
   files, the fixture regeneration, the spec edit.
3. Hardware: the harness per shape, then `--bench 6` for the switch cost, then
   the block line on the 1020-token prompt with `OFLM_OPEN_DISPATCH_LOG=1`.
4. Merge the result paragraph into `spec.md` and archive this plan.

**Fallback if step 1 fails.** Serve the K=512 shared-expert down projection from
the K=2048 program instead: no repack and no kernel change — zeroing activation
rows 512..2047 makes six filler bands harmless whatever bytes they hold, so the
weight tap repeats the two real bands four times. It removes one change per
layer (~100 ms) against ~52 ms of extra FLOPs on that dispatch. Worth doing only
if the main route does not land; the main route gives the same 100 ms for free.
