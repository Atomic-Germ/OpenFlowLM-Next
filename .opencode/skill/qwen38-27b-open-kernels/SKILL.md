---
name: qwen38-27b-open-kernels
description: Where the Qwen3.8-27B stands on the open path - the four walls already cleared, the one still open (of_lni at depth 5), the theories that were wrong, and the closed-kernel version gate. Use when picking the 27B work back up, when a dense width overflows a shim tile, or before proposing any depth/fifo change.
---

# Qwen3.8-27B on the open path

A 27B dense model: hidden 5120, 64 layers, intermediate 17408, **48 value heads
over 16 key heads** (3:1, where every other Qwen3.5 dense model is 2:1). That
last fact is the source of most of the difficulty, and it is real architecture,
not a quirk of this model.

**The artifact works.** `oflm pack --prune-ffn 12288` produces a correct
16.7 GB container — 64 layers, all 192 FFN tensors 12288 wide, embed bf16, MTP
block dropped, `oflm_pruned_ffn` in the config with the retained fraction, and
an honest README. See `imatrix-ffn-pruning`.

**It does not run on the closed kernels on this build**, and cannot. See
"Version gate" below. The open path is the only path.

## Four walls cleared, one open

### 1. The q4 GEMV fold — FIXED
`gemv_q4_pool_group_rt` is `static inline`, so `gemv_q4_gy` and `gemv_q4_gms`
each carry their own band walk. The fold into one `gemv_q4_gyms` was gated on
`bool(Q8)`; the folded entry picks its destination at runtime and is correct
with or without a q8 body, so every all-q4_1 dense width was excluded. Now
`xcommon.FOLD = NEED_Q4_GY and NEED_Q4_GMS`. Unblocked **2560** and cleared the
guard that was watching it.

### 2. `AB_ELEMS` was not a tile count — FIXED
It was `hidden * ab_lanes * 2 // ELEM`, which reduces to `HID/64` only because
`ab_lanes` has always been 32, where the 32 cancels against the fixed tile
width and the `*2` counts the alpha/beta accumulators being *walked*. At 48
heads it rounds to 64 lanes, the cancellation breaks, and the count doubles to
160 against an xn holding 80 tiles — with `SIDE_BYTES` claiming 1368 KB for it.
Now `ab_tiles(spec) = hidden // AB_ROWS`, decoupled from lane width. Identical
for all four shipped widths.

### 3. The x channel was sized for the models that existed — FIXED
`of_x` was `depth=2`, true of every built model and a limit for none. The x
channel is **scratch**: `prep` copies each element into the table and
`role_gemv_bands` reads only the table, so an element is dead the moment its
`prep` returns. `prep_bands` and `ffn_body` now acquire in chunks of
`XE_CHUNK = 2` and the depth is the chunk, not the model. Two details: the
element index must stay **global** (`dense_prep` derives its block range as
`64*i`), and the fill side is unchanged — the fifo hands elements out in order
across chunk acquires. This also removed a latent over-release in `ffn_body`.

Also cleared: `shim_fills` 13 → 14 (a comparison against a recorded
observation, not an enforced limit — `Pipeline(3)` throttles regardless), and
attn's 24-head gated partial-RoPE tuple.

**All four shipped widths still build, and the 9B's total is unchanged at
23680 B of TU text** — its og has 3 elements and now takes two chunk acquires,
so that number is the check that the change is only a reordering.

### 4. OPEN: `of_lni` — the shim tile is 12288 B over data memory

```
stack 0x1800                 6144
lno (depth 1)               10240
lni (depth 5 -> 6 buffers)  61440
                            77824   vs 65536
```

Two independent reasons depth 5 is load-bearing, both established by reading
`ln.h` and `ln_xn.cc` and confirmed by `--trace`:

- **The operands.** `ln.h:11` says *"xn needs the whole y for its statistics"*,
  and `ln_xn.cc` bears it out: `ln_inv` accumulates `mean((x+a)²)` over both
  halves, then loops again to emit `xn`, re-reading `x` and `a`. All five of
  `[x0 x1 w a0 a1]` are live across both passes. This is the norm's
  definition, not an implementation choice — categorically unlike the x
  channel, and the reason chunking is free there and not here.
- **The pipeline.** The trace shows acquires emitted *before* fills: the shim
  runtime resolves the program symbolically and matches them afterwards, so an
  `acquire(5)` is satisfied by five fills anywhere earlier in program order.
  `ln_body` consumes 13 elements per layer against the shim's 8 fills worth 11;
  the 2-element difference is the next layer's `a0 a1`. **The depth is a
  cross-layer buffer.**

So depth must reach 3, and both jobs have to be resolved by something other
than fifo depth. Arithmetic, all verified: depth 4 → 67584 (over by 2048),
depth 3 → 57344 (fits, 8192 free). `lno` cannot absorb it — it is strictly
serialized (7 acquires, every one `n=1`, at depth 1), and removing it entirely
still leaves 67584.

**The direction.** Unfuse the residual add, which takes the window to
`[x0 x1 w]` = 3 elements. `ln_nr32.cc` already exists and is exactly that
(`t = norm(x)*w`, residual "added by the next ln call"). The add has a home and
needs no new machinery — stages 2 and 3 (`lx.py:332-336`, `343-348`) already
read their operands from DDR, and `A_OUT` is written at stage 6 and re-read at
stage 7, so both operands are live when the add must happen. Cost is one extra
`HID*2` read per norm, ~1.31 MB of extra DDR traffic per token at 64 layers:
bandwidth, not memory, and no numerics change.

**Gate it on width**, following the precedent in `ln.py:40`
(`if N <= 2048: ln_fn ... else: ln_y/ln_xn`), so the shipped family is
bit-identical and only 5120 takes the new path. Prefer the computed form over
a width constant — the tile fits while `2*ELN*(depth+1) + STACK <= 65536` — so
the condition explains itself and moves when the stack size does.

Unverified and worth checking before writing code: the trace says depth is the
cross-layer carry, so unfusing one stage may not be sufficient. Both stages use
the 5-window, and the overlap has to survive.

## Version gate, and what their repo is for

The shipped `/opt/openflowlm` is `OFLM v0.1.0`, forked from FastFlowLM 1.0.1.
Their `AMD-FastFlow-Closed/src/model_list.json` has a real, correctly annotated
entry for this model: family `qwen3.8-mtp`, `parameter_size: 27B`, `size: 27e9`,
`footprint: 18.1`, `flm_min_version: 1.0.7` — the **highest** in their entire
list (next is 1.0.5; Qwen3.5 is on 1.0.3). So AMD support for 27B exists and
postdates our fork, and the closed `proj_weights` buffer is simply the older
sizing. Their container is full-width, not pruned; the 17.1 GB one that started
this is not what they publish.

Their tree is a **leak surface, not a reference**: `proj_weights` and
`qwen3_8mtp_npu.hpp` are closed there too and will not open. Useful when free,
never authoritative. `modeling_qwen3_8mtp.cpp` is real evidence that their 27B
is the MTP variant with speculative decoding — which is why keeping the `.nextn`
block matters to them and not to us.

## Theories that were wrong

Held here because they cost the most time.

- **"The 4B violates `sum(AB_TILES) == AB_ELEMS` and builds."** It does not —
  40 == 40. Only 5120 broke it, and because of `ab_lanes`.
- **"`lanes = 32` satisfies 5120, so the arithmetic can't be worked."** Wrong
  question. `ab_lanes` should not be in the formula at all.
- **"Unfusing needs a new elementwise add in the dataflow; neither exists."**
  Wrong. Inferred from `ln_body`'s signature instead of tracing what feeds it.
- **"There is a half-empty x element to reclaim."** No — 5120×2 = 10240 B is
  2.5 elements, so it needs 3 whole ones.
- **"The AMD registry is wrong about the 27B (`parameter_size: 9B`)."** That
  copy-paste is in **your local** `~/.config/oflm/model_list.json`. Their
  shipped list has no `qwen3.8` family at all; theirs says 27B.
- **"Hidden 5120 built."** The tool prints a per-TU table even when the design
  does not. Read the header.

Every one of these came from reasoning about a formula or a file without
reading its consumer. `glue_ab_tile` settled wall 2 in one step after three
layers of inference got it wrong. `utilities/kernel-size.py` and
`utilities/why-no-fifo.py` exist because of that, and `--trace` settled the
accounting that reading could not.

## Re-measured 2026-10-01 on the imx12288 container (pruned, 12.92 GB)

The container is `hidden 5120 / intermediate 12288 / 64 layers / 24 heads over 4
kv / head_dim 256 / vocab 248320`, `oflm_pruned_ffn 17408 -> 12288`,
`activation_mass_retained 0.8849`. Both files in one snapshot dir:
`Qwen3.8-27B-Q8_0.gguf` and `Qwen3.8-27B.imatrix.gguf`.

**A spec is NOT the barrier.** Derived from the container
(`cat_family='qwen3.5', size='9b'` — the quant map resolves through
`qwen3.5_9b.json`; passing no size refuses) and `recipes.qwen35` **accepts** it.
The attention tuple `(256, 24, 4, 64, ...)` is already in the validated set:
`partial_rotary_factor 0.25 x head_dim 256` gives rotary 64, and
`attn_output_gate: True` gives the gate. The build then dies exactly where it
always has:

    aie.tile (0,3) buffers exceeded available memory
      stack 6144 + lno 10240 + 6 x lni_cons_buff 10240 = 77824 B  vs 65536

**The prune does not help, and cannot.** `lni` scales with **hidden** (5120),
not with the FFN width. Narrowing 17408 -> 12288 shrinks a different buffer.

| lni depth | bytes | |
|---|---|---|
| 6 | 77824 | over by 12288 |
| 5 | 67584 | over by 2048 |
| **4** | **57344** | **fits, 8192 spare** |
| 3 | 47104 | fits, 18432 spare |

Removing `lno` entirely still leaves depth 6 over by 2048 B, so the operand
count itself has to come down. **Depth 4 is the target: 6 -> 4.** This is the
same wall as before, unchanged in kind; `why-no-fifo.py --trace` showed all five
norm operands live for `mean((x+a)^2)` plus a two-element cross-layer carry, so
the fix is to stop holding every operand at once, which is what #122's
"streamed residual RMSNorm at width 5120" is.

### Unresolved: the spec's quant does not match the container

Deriving the spec prints:

    open_kernels: warning: q8 (8-bit) weights are not implemented yet for
    qwen35 hidden 5120 -- the kernel does not fit in program memory.
    Falling back to q4_1.

so the spec comes out `quant=q4_1`, but the container stores **q8 (8704-byte
chunks)** for `linear_attn.ssm_alpha_proj`, `ssm_beta_proj` and `ssm_out_proj`
in every layer. Kernels built from this spec would read 5120-byte q4_1 chunks
where the container holds q8 — silent corruption, not an error. Two ways out,
and this is a decision, not a detail:

- make q8 at hidden 5120 work, so the spec states the container's real format
  (#122 claims to have validated the q8 head at K5120, all 248320 logits), or
- repack the container with those three roles at q4_1 so it matches the spec.

Also note `recipes/specs/` is gitignored and wiped by `gen_catalogue_specs.py`,
so a hand-written spec needs a `src/model_list.json` entry to survive.
