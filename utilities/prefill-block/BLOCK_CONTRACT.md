# The block contract of the current prefill kernels (Phase 26.0)

**Determined from source and from the exported manifest, before any runtime change.**
Nothing here is a proposal; every line is what the code does today.

Sources: [`gemm_q4_prefill.py`](../../open_kernels/designs/gemm_q4_prefill/gemm_q4_prefill.py),
[`core.cpp`](../../src/open_qwen36/core.cpp), [`block_host.cpp`](../../src/open_qwen36/block_host.cpp),
`Models/qwen38-27b/production-v2/kernels-model/manifest.json` (manifest v4).

## The contract

| property | value | where it comes from |
|---|---|---|
| **physical token tile** | **256** | `manifest.layer_types.*.gemm_block.t == 256`; `gemm_q4_prefill.py` `T` is a `CompileTime`, and `core.cpp` throws `"T=… exceeds the kernel set's compiled …"` above it |
| **minimum legal logical T** | **1**, as a *logical* count, but see the constraint row below | `gemm_dispatch` accepts `T < gb.t` and pads (`core.cpp:2691-2698`); the projections therefore run at any logical T |
| **maximum logical T per dispatch** | **256** | same throw in `gemm_dispatch` |
| **the binding constraint** | **physical T must be exactly 256** | `attention_npu` throws unless `M == ab.m`, and `M = (nh/kvh) · T = 6 · T`, `ab.m = 1536` ⟹ **T = 256 exactly** (`core.cpp:3282-3285`). This is not a preference, it is the only legal value for a `full` layer. |
| **tail mechanism** | the caller pads the block to 256 with a **repeated in-range id** and passes `t_real` | `cli.cpp:407-408`: `blk.resize(GT, blk.empty() ? 0 : blk.back())`; `pos_ += t_real` so padding never advances position |
| **padding mechanism** | the padded rows are *computed*, not skipped. Every host stage runs over all 256; the kernels are told the real count only where it changes a result | `deltanet_block` uses `R = g.t_real`; `softmax_chunk` uses `pos[r]`; `tile_rows_as_b` zero-fills `r >= k_real` |
| **valid-row mechanism** | `t_real` plus per-row `pos[]`. There is **no** valid-row bitmask in the GEMMs; they compute all 256 columns unconditionally | `core.cpp`, `block_host.cpp` |
| **activation layout** | bf16, pre-tiled by the host into the reference's `k,n` tile order, `[K,T]` → tiles of `(64,32)` | `host::tile_x` / `tile_rows_as_b`; `gemm_q4_prefill.py` `B_l1_ty = (k_tile, n)` |
| **output layout** | fp32 `[N_WEIGHT, T]` **on the device**, read back and transposed to `[T, N]` on the host | `gemm_dispatch` reads `N * gb.t * 4`; `host::transpose` in `gemm()` and `staged_tail()` |
| **are inactive rows guaranteed ignored?** | **Only where it matters, and by three different mechanisms** — see below | — |

## The three masking mechanisms, precisely

This is the part a bug would hide in, so it is stated exactly.

1. **Attention causality is exact per row.** `softmax_chunk` computes
   `valid = pos[r] >= c0 ? min(L, pos[r] - c0 + 1) : 0`, and fills `pr[valid..L)` with
   zero. Row `r` of query head `r / T` at token `r % T` therefore attends to
   `[0, pos[r]]` and to nothing after it. **A padded row is masked out of every real
   row's window, because a real row's `pos` is below it.**

2. **DeltaNet advances the recurrence over `t_real` rows only.** `deltanet_block` sets
   `R = g.t_real` and computes
   `std::fill(og, og + g.T * vw, 0.f)` first, so the output rows `>= t_real` are zero and
   the state transition stops at the last real token. `block_layer_linear` passes
   `t_real` from the caller. **Padding cannot reach `S` or the conv history.**

3. **Padding rows are still *computed* through the projections and the FFN**, including
   the residual (`block_layer_linear` does `res[i] = xres[i] + out[i]` over `i < T·hid`).
   `xres` is a host `[T, hidden]` array and the residual is fp32, so a padded row's
   garbage input can only affect that row. The only cross-row operations are attention
   (masked by (1)) and the DeltaNet recurrence (bounded by (2)).

Consequence for the oracle: **a padded block must produce byte-identical KV rows, `S`
and conv history to a sequential run.** The padded rows' own hidden states are not
meaningful and are excluded from comparison; the state artefacts and the logits are not.

## Position handling

- Absolute position is `pos_`, advanced by `t_real` only.
- RoPE is applied from `g.pos0 = pos0 + row0` in `host::attention_prep`, so row `i` of a
  block starting at `P` uses position `P + i`. This is exactly what 26.6 requires.
- The KV write is `st.sync(TO_DEVICE, t_real * row, pos0 * row)` — `t_real` rows starting
  at byte offset `pos0 * row`. **A padded row is not written to the cache.**
- The conv history is `(taps − 1) · nch · 4` bytes at offset 0 of the linear state BO
  (`3 · 10240 · 4 = 122880`), and `S` begins at `state_s_off = 122880` for
  `48 · 130 · 128 · 4 = 3194880` bytes, total `3317760` = the declared `state.bytes`.

## Logical block sizes Phase 26 will use

`T = 32` is **not adopted as a physical geometry**. The plan's own escape clause applies:
the design physically uses 256 and can execute `active_t = 32, physical_t = 256` through
the padding path above. Phase 26 therefore keeps `physical T = 256` and varies only the
logical block size.

The logical block sizes tested are:

```
B = 32    (the plan's suggested starting granularity, as a logical count)
       and the physical boundary 256, plus the tail rule
```

At `logical B = 32` a single 256-token physical block covers it with 224 padding columns.
At `logical B = 256` there is no padding. Both are the same physical dispatch.

**No new GEMM geometry is authorised, and none is needed.**

### Consequence worth recording before it is discovered as a failure

Because a `full` layer forces physical `T = 256`, and because a logical block smaller than
256 is padded, **splitting a prompt into chunks smaller than 256 tokens costs one full
256-column dispatch each**. That is a throughput property, explicitly out of scope here
(26.14). It must *not* be "fixed" in Phase 26 by introducing a second geometry or a second
hardware context.

## What this means for the sequential oracle

Path A must use the production per-token route so that the reference is the real
arithmetic, not a replica. In this kernel set the per-token `lx`/`ax` programs exist
(`layer_types.*.program`), so `core.step()` is available and is the reference.

The two paths will **not** be bit-identical, and must not be required to be: the batched
projection accumulates over 256 columns where the per-token route accumulates over 1.
The plan's rule applies — use the existing numerical acceptance, and do **not** invent a
weaker tolerance merely because accumulation order changed. The comparison therefore
reports cosine, max abs, max relative, worst index and PASS/FAIL per tensor against the
same gate the repository already uses for prefill (`cosine ≥ 0.9999`).

## The one thing this contract does not establish

Whether the *padded rows'* presence changes a real row through the shared GEMM. It cannot
by construction (each output column is an independent dot product over K, and K is fully
real — there is no masked K), but that is an argument, not a measurement. Stage 1 and
Stage 2 measure it: a 32-token logical block (224 padding columns) must agree with a
256-token block over the same first 32 tokens, on the same basis as against sequential.
