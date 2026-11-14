# Phase 25 — architecture comparison with the whole-array batched prefill

**Scope: analysis only. No kernel was changed, no runtime was changed, nothing was
optimised.** Every claim below is traced to a line of code or to a committed measurement,
never to a filename.

## Verdict first

| question | answer |
|---|---|
| Classification of `gemm_q4_prefill` against the upstream whole-array batched GEMM | **B — derived, materially modified** |
| Is there a materially different upstream mechanism attacking a ≥ 20 % bottleneck? | **No** |
| Does the host submit eight independent runs for the eight array columns? | **No.** One `xrt::run` per whole-array launch |
| Is the "factor 8" a real submission inefficiency? | **No.** It is the paired four-product fold, and it is the frozen correctness mechanism |
| Is there another ≥ 5–10 % actionable component in the residue? | **Not from the residue's own contents.** One adjacent candidate does clear 10 %: the host GEMM tiling (8.2 % wall) |
| Decision per the rule | **STOP GEMM architecture research. Proceed to Phase 26.** |

The Phase 24 report's own explanation of the 1024 dispatches per block was **wrong** and
is corrected here and in [`E2E_PROFILE.md`](E2E_PROFILE.md).

---

## 1. What upstream material actually exists in this tree

PR #39 / issue #32 are **not reachable from this tree**. What is in the tree is the
architectural reference the design is derived from, and it is unambiguous about its own
provenance:

- [`npu_offload/gemm_rtp/gemm_pretiled.py`](../../npu_offload/gemm_rtp/gemm_pretiled.py)
  — header: "NpuEmbeddings -- M5: whole-array bf16 GEMM consuming PRE-TILED B", itself
  "derived from mlir-aie programming_examples/.../whole_array/whole_array.py".
- [`open_kernels/designs/gemm_q4_prefill/gemm_q4_prefill.py`](../../open_kernels/designs/gemm_q4_prefill/gemm_q4_prefill.py)
  — module docstring: "Architecturally mapping (see the task brief / NpuEmbeddings
  experiments/m5-pretiled-gemm/gemm_pretiled.py for the validated whole-array bf16 GEMM
  this is **structurally derived from**)".

So the comparison below is against `gemm_pretiled.py`, which is the same file the design
names. Anything attributed to "upstream" means that file. The classification is made by
reading both dataflows, not by matching names — and the names differ completely
(`A_l3l2_fifos` / `B_l2l1_fifos` / `rtp` vs `A_l3l2_fifos` / `B_l3l2_fifos` / `rtp_bufs`),
which is precisely why filename matching would have been worthless.

## 2. Dataflow comparison

Legend: **same** = the mechanism is carried over unchanged; **modified** = the idea is
upstream's but the implementation differs; **new** = no upstream counterpart.

| dimension | upstream `gemm_pretiled.py` | this design `gemm_q4_prefill.py` | class |
|---|---|---|---|
| **Activation source / layout** | runtime operand `A`, row-major `[M,K]`, **not** pre-tiled; B is pre-tiled `[K,N]` in `k,n` order via `tile_b()` | activation is operand **B**: pre-tiled offline by the host (`host::tile_x`, `core.cpp:2321`) into the reference's exact `k,n` tile order | **same idea, roles swapped** |
| **Weight source** | operand B, plain bf16 in DRAM | operand **A**: q4_1 pool, fetched at the pool's native band granularity (10240 B = 64 rows × 256 K) | **new datapath** |
| **Dequantisation location** | none — weights are already bf16 | **on the core**, immediately before each of the 4 k=64 matmul sub-steps a band covers (`gemm_q4_dequant.h`; two passes, scalar integer gather then vectorised bf16) | **new** |
| **Dequantised tile lifetime** | n/a | one band-wide `[64,64]` scratch reused in place, and only for the k-step being computed — it is *not* retained across bands | **new** |
| **A / B tile geometry** | `m=64, k=64, n=48` (`PRESETS`), `n_aie_rows=4`, `n_aie_cols=4` by default | `M_TILE=64, K_TILE=64, TILE_N=32`, `N_AIE_ROWS=4, N_AIE_COLS=8` | **modified** (n 48→32, cols 4→8) |
| **Core / column ownership** | `Worker.grid(4, n_aie_cols)`, A indexed by row, B by column | identical ownership: `Worker.grid(4, 8)`, `A_l2l1_fifos[row]`, `B_l2l1_fifos[col]` | **same** |
| **Whole-array launch** | one `Program` + one `Runtime.sequence` → one dispatch covers all columns | identical: one `Program(dev, rt, workers=...)`, one `sequence` | **same** |
| **Accumulator lifetime** | `out_c.acquire(1)` **before** the K loop, `zero()` once, K-reduction accumulates into that one C tile, `release()` after | `elem_out = out_c.acquire(1)` before the band loop, `zero(elem_out)` once, all bands' sub-steps accumulate into it, `release()` after | **same** |
| **Output accumulation** | inside the core, over every `K//k` step of the C tile | identical over all `NBG` bands × `kys` sub-steps of one T_TILE | **same** |
| **Weight reuse across T-tiles** | A refetched per row block; `b_reuse` exists for B (the weight) and **refuses to build at 8 columns** | weight refetched per T-tile (docstring admits it: "a real inefficiency above T=256, reported honestly rather than hidden"); `b_reuse` attempted for the *activation* and **also refuses at 8 columns** (module constant `recommend_b_reuse` → `False`) | **same limitation, opposite operand** |
| **Host XRT submissions** | one `xrt::run` per GEMM | one `xrt::run` per `Step` per projection per input component — **four** per projection, because of the paired fold | **modified** |
| **AIE ops in the core body** | `zero`, `matmul`, optional `narrow` / `gelu` epilogue | `zero`, `matmul`, plus 4 hand-written dequant entry points (`gqd_entry_k0..k3`), inside an unrolled Python loop over `kys` | **new** |
| **Task-group structure** | `TaskGroup` per ping-pong half; `tg.finish()` after every group; optional `tg_depth` pipelining (T61-2); `tb_max_n_rows` groups per half | `sequence()` issues A/B fills and C drains with `b_tap()`/`C_cons_hs`; `T_TILES` and the band count arrive as **RTP** values | **modified** |
| **RTP mechanism** | `rtp=True`: two `int32` per core + `WorkerRuntimeBarrier` | `rtp_bufs` two `int32` per core, **but no barrier** — and the file documents why: the barrier deadlocks here because the core body runs once per weight row-block group, not once per dispatch | **modified, with a measured reason** |
| **Transpose / tiling location** | host builds B with `npue.tile_b()` once per model | host `tile_x()` **per dispatch** (`core.cpp:2321`), and the output is transposed back **per dispatch** (`host::transpose`, `core.cpp:2356`) | **modified — and this is the cost** |
| **RMSNorm location** | n/a (a standalone GEMM) | host, `host::rmsnorm_rows`, **before** the projection, per layer per block (`core.cpp:3216`, `3708`, `3770`) | **new (outside the kernel)** |
| **SwiGLU location** | n/a | host, `ffn_block` (`core.cpp:3173-3179`) / `ffn_staged` (`3641-3648`), between up|gate and down | **new (outside the kernel)** |
| **Per-layer sequence** | n/a (one GEMM) | 5 projections per layer: `gemm_n16384_k5120` (qkv+z fused) → `gemm_n5120_k6144` (out) → `gemm_n34816_k5120` (up+gate fused) → `gemm_n5120_k17408` (down), each as hi then low component; `full` layers also run attention on `ag_*` | **new composition** |
| **Block size / tail** | `M` is a compile-time shape | `t = 256` compiled in; a request is ⌈T/256⌉ blocks, the last padded with a repeated in-range id; the kernel's T is never recompiled | **modified** |

**Why B and not A.** A would mean "direct continuation": same arithmetic, same operands,
same submission structure. That fails on four counts, each of which changes the dataflow
rather than a constant: (1) the weight is a Q4 pool **dequantised on the core** — upstream
has no dequantiser at all; (2) activation and weight **swap the A/B roles**; (3) the host
**re-tiles and re-transposes per dispatch** instead of once per model; (4) a projection is
dispatched **four times** for the paired fold instead of once. Any one of those is a
material modification; together they are a different datapath that reuses upstream's
fifos-and-grid skeleton.

**Why not C.** C would mean the core mechanism is different. It is not: the same
`Worker.grid(4, cols)`, the same C-join across four rows, the same one-accumulator-per-
output-tile discipline, the same `repeat_count`/`b_reuse` analysis reaching the same
negative conclusion at 8 columns, and the same RTP shape-hoisting idea. The skeleton is
upstream's, and the design says so itself.

---

## 3. The critical question: what is the factor 8?

**Short answer: there is no factor 8 in the submission path. The host does not submit
eight runs. The 1024 dispatches per block are 64 layers × 2 projections × 4 products.**

### The five quantities, measured or traced separately

For **one 256-token block** (one 64-layer pass):

| quantity | count | evidence |
|---|---|---|
| host `xrt::run` submissions | **1024** | dispatch table: 8192 at 2048 tokens = 8 blocks × 1024 (`e2e-profile-v2.json`) |
| context / design switches | **0 in steady state** for a block | the 5 high shapes share the single `gemm` context; the 5 `_low` shapes each own a context, so the switch cost is paid once per **layer** at the boundary between a projection's high and low dispatch — see below for why that is an artifact, not a requirement |
| whole-array launches | **1024** | one `xrt::run` → one `MLIR_AIE` kernel → one `sequence` |
| per-column launches | **0** | `Worker.grid(N_AIE_ROWS=4, N_AIE_COLS=8)` over one `Program`; no per-column `xrt::run` exists anywhere in the host path |
| AIE task-group starts | **1024 × (task groups in `sequence`)** | the `sequence` issues its fills/drains in the `TaskGroup` structure; the *design* is started once per launch, the groups are inside it |

### The decomposition of 1024, from the dispatch table

```
1024 per block = 64 layers × 2 projections per layer × 8
                 8 = 2 weight components × 2 input components
```

Measured at 2048 tokens, per kernel name over 8 blocks:

| kernel | calls | calls / (64 layers × 8 blocks) |
|---|---|---|
| `gemm_n34816_k5120` (up+gate high) | 1024 | 2 |
| `gemm_n34816_k5120_low` (up+gate low) | 1024 | 2 |
| `gemm_n5120_k17408` (down high) | 1024 | 2 |
| `gemm_n5120_k17408_low` (down low) | 1024 | 2 |
| `gemm_n16384_k5120` (qkv+z high) | 768 | 2 × 48/64 layers |
| `gemm_n16384_k5120_low` | 768 | 2 |
| `gemm_n5120_k6144` (out high) | 1024 | 2 |
| `gemm_n5120_k6144_low` | 1024 | 2 |
| `gemm_n14336_k5120` (q+k+v+gate high) | 256 | 2 × 16/64 layers |
| `gemm_n14336_k5120_low` | 256 | 2 |
| `ln`, `lm` | 1 each | the tail |

So: **2 calls per projection per layer per block**, and the paired fold issues two
`dispatch()` invocations per kernel (`x_hi` then `x_lo`, from
[`prefill_precision.hpp`](../../src/open_qwen36/prefill_precision.hpp)), each of which is
one `xrt::run`. `2 × 2 = 4` launches per projection. Eight is `4 × 2` where the trailing
`2` is the high/low **kernel** pair.

**There is no ×8 for columns anywhere.** Eight columns cost one launch, which is exactly
what upstream's whole-array design is for, and this design inherits it faithfully.

### Would upstream avoid it? No — upstream has no counterpart at all

Upstream issues **one** launch per GEMM because it computes `A@B` once. The four launches
here are the frozen **correctness** mechanism: `(W_hi X_hi + W_hi X_lo) + (W_lo X_hi +
W_lo X_lo)`. Phase 23 measured that removing any part of it breaks the 0.9999 gate —
`weight_high` (drop the weight residual) failed 301/513 positions, `three_term` (drop
`W_lo X_lo`) failed 1/513, and the failure is structural rather than localised. Upstream
has nothing to avoid, because upstream does not have the precision requirement.

### The one real inefficiency this exposes, measured not assumed

The dispatch table shows the **low** kernel of each projection is consistently *slower*
than its high twin, at identical shape and identical instruction stream:

| pair (2048 tokens, 1024 calls each) | high mean | low mean | low/high |
|---|---|---|---|
| `n34816_k5120` | 42.737 ms | 45.592 ms | ×1.067 |
| `n5120_k17408` | 21.525 ms | 23.039 ms | ×1.070 |
| `n5120_k6144` | 8.969 ms | 9.624 ms | ×1.073 |
| `n16384_k5120` | 21.246 ms | 22.551 ms | ×1.061 |
| `n14336_k5120` | 18.587 ms | 19.928 ms | ×1.072 |

The instruction streams are **byte-identical** (`insts.bin` sha256 identical, 177680 B
both). The only difference is that the five `_low` shapes each live in their **own
`hw_context`**, while the five high shapes share one:

```
contexts: gemm -> 5 kernels: n5120_k6144, n5120_k17408, n14336_k5120, n16384_k5120, n34816_k5120
          gemm_n16384_k5120_low -> 1 kernel
          gemm_n5120_k6144_low  -> 1 kernel
          gemm_n34816_k5120_low -> 1 kernel
          gemm_n5120_k17408_low -> 1 kernel
          gemm_n14336_k5120_low -> 1 kernel
```

That ordering is an **artifact of packaging**, not of arithmetic: the high dispatches come
first for every projection, and then each falls into its own context. An alternative
schedule — all high products, then all low products — would pay the high/low context
crossing far less often. It is worth noting that `Core::context(name)` is already a cached
map, so **the context object is not recreated**; only the submission order changes.

Total measurable prize: the low products sum to 103556.5 ms of 198619.2 ms, and a 6.7 %
reduction on them is **≈ 6.9 ms per block, ≈ 6920 ms at 2048 tokens ≈ 2.4 % of wall**
(≈ 3.5 % of GEMM time). This is a *packaging/ordering* observation, not an
architecture change, and it is recorded here rather than implemented, per instruction.

---

## 4. The Phase 24 residue, resolved

At 2048 tokens, wall = 294310.0 ms. Two remainders were outstanding.

### 4a. `outside_engine` = 9877.2 ms (3.4 % of wall) — **fully explained, not actionable**

It is **constant** at every length: 10315.0 / 9956.6 / 9936.2 / 9877.2 ms. It is the
process, not the request: model load, xclbin registration and the 11 `xrt::hw_context`
creations in `Core::load_kernel`/`Core::context`, and teardown. It cannot be reduced by
changing prefill. At 128 tokens it is 23 % of wall; at 2048 it is 3.4 %. **Nothing to do.**

### 4b. `unattributed_engine` = 22847.0 ms (7.8 % of wall) — **bounded, and not one thing**

It grows linearly at a near-constant **2856 ms per 256-token block**
(2926.7 / 2887.0 / 2865.4 / 2855.9), so it is per-block work, not fixed setup. What it
**can and cannot** contain is decidable from the accounting model alone, without more
instrumentation:

- It **cannot** be host GEMM tiling, transpose, entry norm, residual/norm/router, FFN
  activation, BO sync, KV state, or the LM head. Each of those has its own exclusive
  bucket above, and each is separately 0.0–8.2 % of wall.
- It **cannot** be the attention GEMMs or the DeltaNet conv/rule work: `dn_conv`,
  `dn_rule` and `attn` are all inside `mid`, and `mid` is inside `part1`.
- It **cannot** be the projection GEMMs' device time: that is `part0` and it is counted,
  including the blocked half.
- It **cannot** be `shared` or `state`, which were taken back out of the residue by name.
- What is left is host work in `step_gemm_prompt`/`step_gemm_block`/`block_layer_*` that
  is clocked by **no** `StepTiming` field: the embedding rows, the residual adds and
  memcpys between stages, `BlockScratch::fit` reallocations, the `std::vector` scratch
  bookkeeping around each projection, and the per-dispatch setup/teardown in
  `start_run`/`wait_run` not attributed to `sync`.

The decisive point for the plan: **every named component that could plausibly be inside it
already has a measured bucket that is small.** Splitting 7.8 % further would need new
timers, and the largest single thing it could be is bounded by the small sizes of the
terms that are individually measured. There is no evidence for another ≥ 5–10 %
actionable component hiding in it, and three independent measurements (Phase 21, Phase 24
twice) agree on its magnitude. It is recorded as a bounded unknown, not a target.

**The one adjacent term that does clear 10 %** is not in the residue at all — it is
`gemm_host_tiling` at **8.2 % of wall** (24124.2 ms). See §5.

---

## 5. Measured economics — Amdahl ceilings at 2048 tokens

Wall-time based: **wall = 294310.0 ms**, and the new wall is `wall − removed`. These are
ceilings for prioritisation, not claims.

| # | optimisation | removed | new wall | speed-up | % of wall |
|---|---|---|---|---|---|
| 1 | eliminate **all** host GEMM tiling + transpose | 31534.9 | 262775.1 | **1.120×** | 10.71 % |
| 1a | eliminate host GEMM tiling only | 24124.2 | 270185.8 | 1.089× | 8.20 % |
| 1b | eliminate host GEMM transpose only | 7410.7 | 286899.3 | 1.026× | 2.52 % |
| 2 | eliminate the "factor 8" submissions | **0.0** | 294310.0 | **1.000×** | 0.00 % |
| 3 | ideal **1.5×** GEMM | 66206.4 | 228103.6 | **1.290×** | 22.50 % |
| 4 | ideal **2.0×** GEMM | 99309.6 | 195000.4 | **1.509×** | 33.74 % |
| 3b | ideal 1.5× GEMM **and** no host tiling/transpose | 97741.3 | 196568.7 | 1.497× | 33.21 % |

Reading:

- **(2) is zero**, because the factor 8 does not exist. Any plan premised on removing it
  would deliver nothing. This is the single most valuable outcome of Phase 25.
- **(1)** — deleting *every* host tile and transpose, which is not physically possible
  because the kernel requires that layout — buys at most **1.12×**. The host side around
  the GEMM is real but it cannot carry the project.
- **(3)/(4)** — only the GEMM itself is worth more than 1.29×. The GEMM is already the
  frozen paired arithmetic on a whole-array launch; upstream offers no mechanism that
  changes that (§2), and Phase 22 closed the within-datapath levers (weight-stationary
  bounded at 1.08×, `tile_n=64` L1 overflow, `b_reuse` unbuildable at 8 columns,
  layer-major no gain, K slicing refetch conserved, submit-ahead already on).
- Therefore **no GEMM architecture research remains that clears the plan's ≥ 20 % bar**,
  and the 2.4 % ordering prize in §3 is the only cheap item on the table.

---

## 6. Decision

The rule: *if A or B and there is no materially different upstream mechanism attacking a
≥ 20 % bottleneck → STOP GEMM architecture research and proceed to Phase 26.*

- Classification is **B**.
- The comparison found **no** upstream mechanism this design lacks. The two mechanisms
  upstream does have that this design does not use — `b_reuse` and the `WorkerRuntimeBarrier`
  RTP ordering — are both **already tested and rejected here on hardware**, with the
  reason recorded in the source: `b_reuse` cannot build at 8 columns (`repeat_count` is
  unavailable for shim tiles, and the mem-tile BD pool caps at 4–6), and the barrier
  **deadlocks** because the core body runs once per weight row-block group rather than once
  per dispatch.
- The remaining ≥ 20 % target is the GEMM, whose arithmetic is frozen by the correctness
  gate and whose datapath is already upstream's.

**STOP GEMM architecture research. Proceed to Phase 26.** No new POC is requested, because
the rule's "C + mechanism" branch — the only branch that would ask for one — does not
apply.

Recorded but **not implemented**, for a later cheap pass if the project wants ~2.4 %:
grouping the high/low dispatches so the high/low context crossing is paid per layer
rather than per projection (§3).

---

## 7. Recommended Phase 26 architecture

Phase 26 is **correctness**, not performance: production batched prefill whose resulting
state equals the sequential route's. The minimum design, from what Phase 25 established:

### Start granularity: block 32

The measured economics do not justify anything larger as a starting point. A 256-token
block is already what the kernel compiles for, so block 32 is a *scheduling* choice made
inside the existing compiled T, not a new kernel. Start at 32 to keep the number of
inter-block boundaries small enough to localise the first divergence, and because
attention is only ~4.6 % of wall at 2048 — the state-equivalence problem, not throughput,
is what block size should be tuned against.

### The four things that must match sequential state exactly

Only these carry state across a block boundary; everything else is a pure function of the
block's own input rows.

1. **DeltaNet recurrent state `S`** — per head, `head_dim × head_dim`, in `state_[l]` at
   `state_s_off`, `s_rows` rows, written back with `st.sync(TO_DEVICE)` after the block
   (`core.cpp:3733`). It chains strictly in position order.
2. **DeltaNet conv state** — the last `conv_kernel − 1` rows of each conv channel, stored
   in the same `state_` BO; `conv_dtype` is `float32` for this set. Also position-ordered.
3. **KV cache** — `state_row` bytes per position per `full` layer, `read_back` before the
   window and `sync(TO_DEVICE)` after (`core.cpp:3757`, `3803`).
4. **The final hidden row** — only its last real row feeds `ln` + `lm` for the next-token
   logits; the padding rows are already invariant (hardware-proven on 22 real vs 234
   padding columns).

### The two invariants to gate on

- **Batched ≈ sequential**: same prompt, block-at-a-time vs token-at-a-time, must agree
  on all four state artefacts above — not merely on the logits, because a state
  divergence that has not yet reached the head is the failure mode this phase exists to
  catch.
- **Padding invariance**: a block whose tail is padded with a repeated in-range id must
  produce the same real rows as the same tokens without padding. The current route
  already relies on this; Phase 26 must prove it per state artefact rather than assume it.

### Deliberately excluded from Phase 26

- Any change to `gemm_q4_prefill` or its arithmetic (`paired` stays frozen).
- Any of the Phase 22 levers (all closed with evidence).
- Any further role/sublayer/token-position tuning (Phase 23 closed as STOP).
- The ordering micro-optimisation in §3 — note it, do not do it here.

## Artefacts

- [`E2E_PROFILE.md`](E2E_PROFILE.md) — the corrected Phase 24 report (denominators named,
  double count removed, factor-8 claim withdrawn).
- [`e2e-profile-v1.json`](e2e-profile-v1.json) — the original measurement, unchanged.
- [`e2e-profile-v2.json`](e2e-profile-v2.json) — the corrected allocation of the same run,
  with `checks` per length. Recompute with `e2e_profile.py --from-json`; no hardware.
