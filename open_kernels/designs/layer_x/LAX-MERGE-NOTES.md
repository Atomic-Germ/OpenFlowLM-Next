# lax: the merged lx+ax xclbin — measured constraints and required architecture

The 35B (`qwen36-35b-a3b`) mixes linear-attention (`lx`) and full-attention (`ax`)
layers. An `xrt::runlist` is bound to ONE `hw_context` (= one xclbin UUID), so the
per-token runlist can hold all 40 layer runs only if ONE xclbin carries both layer
types (`run_kernel.cpp`: "that is why the 35B's two layer types have to live in one
xclbin before 40 layers can be one submit").

This note records what the build itself proved about that merge, so the dead ends are
not re-tried. Measurements are from the checked-in 35B spec,
`OPEN_KERNELS_SPEC=recipes/specs/qwen36-35b-a3b.json MOE_ONDEVICE_ROUTE=1`.

## What lives where (the model that decides the merge)

`build_emitter/insts.elf` has a single `.ctrltext` section of 0x10b3c = 68412 B,
exactly `insts.bin`'s size. So

* the **tile programs are in the xclbin** (the PDI), and
* **`insts.elf` / `insts.bin` is the host control text** (the DMA task sequence).

Two layer types therefore differ in *both*: the xclbin must hold both main-core
programs, and the control text selects which tiles run. "only the instruction stream
differs" is true of the control text, **not** of the xclbin's per-tile programs.

## Measured budgets (npu2 = 8 cols x 6 rows; row 0 shim, row 1 memtile, rows 2-5 core)

| item | value |
|---|---|
| core program limit | 16384 B |
| `lx` main core (c,2) | **16272 B** (112 B free) |
| `ln`/router core (0,3) | 10192 B |
| glue core (2,3) | 13744 B |
| emitter core (c,4) | 1264 B |
| shim MM2S | 2 per ShimNOCTile; 8 cols -> 16 total |
| `lx` shim fills | 13 (lni + w0..7 + x + side + gact + pin) |
| `ax` shim fills | 14 (lx's 13 + `ain`) |

Per-core `w` element counts for the 35B (call_bytes = 10240, `n_groups(K)=K/256`):

* `lx` = qkv|z 24x8 = 192, out 4x16 = 64, DeltaNet 4 heads x 3 = 12, MoE 219 -> **487**.
* `ax` = q|gate|k|v 18x8 = 144, out 4x16 = 64, MoE 219 -> **427**.
* `x` (broadcast) is identical for both: xn 1 + og 2 + xm 1 + rout/cfg 2 + h `NX` 9 = **15**
  (the emitters take 14 -- they skip the shared slot's `h`).

## Approaches the build ruled out

1. **Runtime dispatch / drain inside the main core.** Adding a runtime-bounded
   `acquire/release` drain loop to `main_body` overflows program memory on every main
   core:

   ```
   ld.lld: error: section '.text' will not fit in region 'program': overflowed by 224 bytes
   aiecc: core main_core_0_2: its code exceeds the tile's program memory.
   ```

   112 B is not enough for any per-core branch/drain. **The main cores' programs cannot
   grow.**

2. **Per-layer-type `w` streams** (`of_w_lx[c]` + `of_w_ax[c]`, each produced on
   `shim(c,0)`): 2 MM2S x 8 columns = 16 channels for `w` alone, exhausting the whole
   shim budget before `lni`/`x`/`side`/`gact`/`pin`/`ain`. Impossible.

3. **Run both bodies on one padded shared stream, discarding the inactive `y`.** The
   two bodies consume different `w` counts (487 vs 427); reconciling them needs an
   in-core drain -> back to (1). Impossible.

## Required architecture

Keep **both main-core programs byte-identical to today**, one set per layer type, and
make the stream *fan-out* runtime-selected so the non-selected set is **never fed** and
its cores simply block on an empty fifo (idle, zero extra code):

* tiles: `lx` main `(c,2)`, `ax` main `(c,5)`; one shared `ln`/router core `(0,3)`
  (both types run `X.ln_router_body`); `lx` post `(1,3)`, glue `(2,3)`; `ax` attn
  `ACORES=4` at `(4,3)..(7,3)`; the 8 emitters `(c,4)` shared (identical body).
* one `shim(c,0)` MM2S per column feeds `of_w_lx[c]` **or** `of_w_ax[c]`, chosen by the
  run's control text (the `ironutil.configure_only_fill` / ONDV-descriptor pattern:
  configure once, start the selected target per run), so `w` stays at 8 channels. The
  alternative is a memtile `(c,1)` fan-out with only the selected output started.
* `x` likewise split per set (or dispatched), since a shared broadcast `x` would make
  the non-selected core consume the active layer's elements.
* budget: shim MM2S = `w` 8 + `lni` 1 + `x` 1 + `side`/`gact`/`pin` 3 + `ain` 1 = **14 <= 16**;
  shim S2MM = `lno` 1 + `y` 8 + `gout`/`pout` 1 + `aout` 1 + `og` 3 = **15 <= 16**
  (per-shim placement still to be checked column by column).

### The 112 B is not recoverable by re-optimising

Rebuilding `lx` with the main-core TUs at `-O2` gives 16256 B (16 B smaller); `-Oz`
fails the aiecc pipeline outright. There is no meaningful headroom to buy, so the main
programs must be used exactly as they are.

### Dispatch primitive

`aie.iron` `ObjectFifo.split()` / `forward()` place an `ObjectFifoLink` on a memtile
(`AnyMemTile` by default) and expose per-sub-fifo `repeat_counts`. That is the natural
place to make the `w` / `x` fan-out runtime-selected: one `shim(c,0)` MM2S -> memtile
`(c,1)` -> `of_w_lx[c]` or `of_w_ax[c]`, with the memtile's forward for the inactive
set never started (repeat count 0) by the run's control text. Whether `repeat_count`
is a control-text field (patchable per run, like the ONDV descriptors) or a static
attribute is the next thing to confirm.

`lx.py` / `ax.py` already return `(workers, rt_args, flows, sequence)` when built with
`pieces=True`; `lax.py` is the consumer of that scaffolding. The missing piece is the
runtime stream dispatch, not the tile or channel layout.
