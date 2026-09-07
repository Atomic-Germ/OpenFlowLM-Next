# Plan: the Qwen3.5 sizes that did not run — the 9B's glue core and the 16-head DeltaNet

**Status:** designed 2026-09-06 (Fable) from the hardware findings in
`.claude/plans/q-hw-results.md` §1 and §3. Not implemented.
**Spec impact:** no new requirement. OPEN-FAMILY-QWEN35's procedure gains the 9B and the
2B / 0.8B as sizes; three catalogue points enter the validated sets on success
(`lm_head_q8 K=4096`, `deltanet heads=16`, `attn (256, 8, 2, 64, True, True, False)`).
Both changes are gated on `DENSE` / the spec, so the shipped 27B's object code cannot move
(the `--check` byte-identity run is the regression test, as before).

## Where the family stands

| size | state 2026-09-06 |
|---|---|
| 4B (HID 2560) | **PASS** — first of the family; engine bit-identical to the harness; 6.7 tok/s |
| 9B (HID 4096) | `lx` does not build: the DeltaNet **glue core** (Tile(2,3)) needs 68 096 B of 65 536 |
| 2B (HID 2048, 16 value heads) / 0.8B (HID 1024, 16) | build, then **hang** on the first `lx` dispatch: `dn_glue.h` hardcodes 32 heads |

## A. The 9B: re-stream the norm output per half instead of holding it

**The wall.** The glue core keeps a private copy of the layer-entry norm output,
`xnb = bf16[HID]`, because the alpha / beta weight tiles arrive on the same `side` fifo and
a fifo element cannot be held across later acquire / release pairs. Everything else on
that core is HID-independent and totals 59 904 B; `xnb` may be at most 5 632 B, i.e.
HID ≤ 2816. The 4B (5 120 B) fits with 512 B to spare; the 9B (8 192 B) does not.

**The fix** (the hardware log's sketch, adopted): keep `xnb` at one 4 KB element and walk
the projection in halves.

1. `designs/dn_glue/glue_ab_e.cc` — `glue_ab_tile` with the accumulator reset as an
   argument (`first`) instead of inferred from `tile == 0`, and the tile index relative to
   the half. `glue_ab.cc` is in the shipped 27B xclbin and is not touched.
2. `lx.py` `glue_body`, DENSE branch only: for each of the two accumulators, for each half
   `h` in `range(XN_ELEMS)` (a Python loop, 2 for the 9B, 1 for HID ≤ 2048): acquire one
   side element, copy it into the 4 KB `xnb`, release, then run that half's
   `AB_ELEMS / XN_ELEMS` weight tiles with `first = (h == 0 and tile == 0)`. At HID 2560
   (XN_ELEMS 2, the second element 1 KB used) the same loop applies and `xnb` shrinks from
   5 120 to 4 096 B — the 4B must re-pass byte-for-byte on logits after this change, which
   is the regression test for the DENSE path.
3. `lx.py` `dense_sequence` `tg_s`: the two side fills become the interleaved sequence
   `[xn half 0, alpha tiles of half 0, xn half 1, alpha tiles of half 1, xn half 0, beta
   tiles of half 0, …]` with taps into `C_SIDE`'s alpha / beta sub-regions. The consts
   layout does not change, only the fill order.

**The one number to check before building:** the shim fill budget
(`catalogue.LIMITS["shim_fills"] = 13`). `tg_s` goes from 2 fills to `2 + 2·XN_ELEMS·2`
= 10 at XN_ELEMS 2 if each half's tiles are one fill; the recipe must count them and
refuse over budget rather than let IRON fail late. If it does not fit, the fallback is a
second `side`-class fifo for the xn halves — a design change, to be decided then, not
built speculatively.

Not adopted: shrinking the glue worker's `stack_size` (0x1800 → 0x800 would fit, but a
stack overflow on that core corrupts `qk` silently and nobody has measured the real use).

## B. The 2B / 0.8B: `kNHead` becomes a knob

`dn_glue.h` carries `static constexpr unsigned kNHead = 32;`, used by `glue_ab_tile`'s
32-lane accumulator (`kV = 32`), `glue_small`'s head loop, and the `f32[NHEAD]` buffers
`acc_a`, `acc_b`, `decay`, `beta`. The 2B / 0.8B have 16 linear value heads
(`ssm_a[16]`, conv over 6144 channels, qkv 6144 = 2·16·128 + 16·128). The build succeeds
because nothing reaches the constant; the dispatch hangs because the glue emits and the
main cores consume records for 32 heads that do not exist.

**The fix**, exactly as `DNX_ROWS` was done (`q-qwen35-handoff.md`, R1): `#ifndef
DNGLUE_NHEAD` defaulting to 32; `xcommon.DN_FLAGS`-style flags on the five `glue_*` TUs
passed **only when the value differs from the default** (so the 27B's compile commands
stay byte-identical — the `DNX_PAD` lesson); the accumulator width stays 32 lanes with the
upper 16 unused when `NHEAD = 16` (padding, not a narrower vector type, keeps
`glue_ab_tile`'s loads aligned); the `f32[NHEAD]` buffers and `glue_small`'s loop sized
from the knob; the alpha / beta W element (`kAbRows` rows × `NHEAD`) keeps its 4 KB size
with 64 rows × 32 lanes and the packer's `transpose` writes `[hid, 32]` with zero columns
16..31 for a 16-head model (the `put` cap is unchanged). `dn_glue.py`'s standalone test
gets a 16-head case; `recipes/qwen36moe.py`'s `Linear` geometry (`NHEAD`, `AB_ELEMS`,
`HEADS_PER_TILE`, the record count) already derives from the spec — verify each against
the 2B and 0.8B by hand before building, since the hang means at least one main-core
consumer expects 32 records.

The attention half of those two models, `(256, 8, 2, 64, gate, qk-norm)`, is a second
first-time point; it is validated by the same slice compare once `lx` completes.

## Order

1. B first — it is smaller, its regression test (the 27B `--check`) is cheap, and the
   2B/0.8B exports already exist so the compare runs minutes after the build.
2. A — build the 9B `lx`, then the family procedure; on PASS add `lm_head_q8 K=4096`.
3. Both sizes' Result lines under OPEN-FAMILY-QWEN35; the 4B re-run after A as its
   regression.

Work is code-only until the builds; the same worktree-then-merge routine as the q8 work
if the NPU is busy.
