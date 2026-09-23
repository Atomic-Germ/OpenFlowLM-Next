# lax: the merged lx+ax xclbin — measured constraints and the resolution

The 35B (`qwen36-35b-a3b`) is Qwen3-Next's 3:1 hybrid (30 linear-attention + 10
full-attention layers). An `xrt::runlist` is bound to ONE `hw_context` (= one xclbin
UUID), so a per-token runlist of all 40 layers needs both layer types in one xclbin
(`run_kernel.cpp`: "that is why the 35B's two layer types have to live in one xclbin
before 40 layers can be one submit").

This note records what the build proved. Measurements use the checked-in 35B spec,
`OPEN_KERNELS_SPEC=recipes/specs/qwen36-35b-a3b.json MOE_ONDEVICE_ROUTE=1`.

## What lives where

`build_emitter/insts.elf` has a single `.ctrltext` of 0x10b3c = 68412 B, exactly
`insts.bin`'s size. So the **tile programs are in the xclbin** (the PDI) and
**`insts.elf`/`insts.bin` is the host control text**. Two layer types therefore differ
in both: the xclbin must hold both main programs unless they *are* one program.

## Measured budgets (npu2: 8 cols x 6 rows; row 0 shim, row 1 memtile, rows 2-5 core)

| item | value |
|---|---|
| core program limit | 16384 B |
| `lx` main core | **16272 B** (112 B free) |
| `ax` main core | 12064 B |
| `ln`/router | 10192 B, glue 13744 B, emitter 1264 B |
| shim DMA | **2 MM2S + 2 S2MM per ShimNOCTile**, 8 cols -> 16 each |

## Approaches the build ruled out

1. **Dispatch/drain inside the main core.** A runtime-bounded acquire/release drain loop
   overflows program memory on every main core:
   `ld.lld: error: section '.text' will not fit in region 'program': overflowed by 224 bytes`.
2. **Re-optimising for headroom.** `-O2` gives 16256 B; `-Oz` fails the aiecc pipeline;
   GEMV-only `-Oz`/`-O2` change nothing. The size is the IRON/lock scaffolding.
3. **Per-layer-type `w` streams.** 2 MM2S x 8 columns = 16 for `w` alone, exhausting the
   whole shim budget.
4. **Running both bodies on one padded shared stream.** They consume different `w`
   counts, and reconciling them needs an in-core drain -> back to (1).

## The resolution: the two main bodies are one body

On the 35B **every projection is q4_1** (`R.q8` is empty), so the `role` argument of the
GEMV is inert: `linear`, `linear_out` and `attn` resolve to the same kernel and the same
group/band law. The two main bodies then differ in exactly two places:

    lx: role_gemv_bands(..., QKV_PC + Z_PC = 24, HID); dn_body(...); (out, 4)
    ax: role_gemv_bands(..., 2*Q_PC + 2*KV_PC = 18, HID);              (out, 4)

So **one main body -- lx's, byte for byte -- serves both layer types**; only the *host
sequence* differs. For a full-attention layer the sequence fills 24 pre-MoE bands (18
real q|gate|k|v + 6 dummy) and, because the body always runs the DeltaNet step, feeds and
sinks that step inside `state`, which a full-attention layer does not use.

`lax.py` is that design: it has the unified main set (row 2), the shared `ln`/router
core (0,3), the linear `post`/`glue` helpers (1,3)/(2,3), the 4 full-attention cores
(3,3)..(6,3), and the 8 emitters (row 4). The `CompileTime kind` (0 linear / 1 full)
selects the host sequence.

## Result (built, not device-run)

* `LAX_KIND=0` and `LAX_KIND=1` both build (`build_lax_l`, `build_lax_a`).
* Main-core program **16272 B, unchanged**; the attention cores are 14000-14320 B.
* Shim budget: **14 MM2S fills, 15 S2MM drains**, at most 2 per shim.
* The two xclbins are **semantically identical**: same size, the per-tile ELFs
  byte-identical, and the only differing bytes are the `UniqueID`/`TimeStamp`/
  `XclBinUUID` metadata. Either can host both control texts.
* The `w{c}` MM2S channels are identical for both kinds (`[1,0,1,1,1,1,0,0]`), so one
  `qmap_lax.bin` (the per-column queue registers, `0x1D214`/`0x1D21C`) serves both runs;
  `run_lax.cfg` registers `lxf` + `axf` against one `xclbin X` and submits them in one
  `runlist`.

What remains is device verification on `accel0` (numerics, and the ax stream's
`attnpos` per token), which is outside the non-device build step.

## Local run artifacts (gitignored, like every other design `.cfg`/`.bin`)

`qmap_lax.bin` (32 B, little-endian u32 per column) -- the merged design's `w{c}`
queue registers, identical for both kinds:

```
1cd2 0100 14d2 0100 1cd2 0100 1cd2 0100 1cd2 0100 1cd2 0100 14d2 0100 14d2 0100
= 0x1D21C 0x1D214 0x1D21C 0x1D21C 0x1D21C 0x1D21C 0x1D214 0x1D214
```

`run_lax.cfg` -- one context, two control texts, one submit:

```
device
xclbin X build_lax_l/final.xclbin
kernel  lxf X build_lax_l/insts.elf
kernel  axf X build_lax_a/insts.elf
buf pool 536870912
buf xres 8192
buf consts 11882496
buf kv 8388608
buf act 190464
buf ptab 4194304
buf state 2342912
buf cfg 4096
buf qmap 32
load qmap qmap_lax.bin
poolbase cfg 0 pool
copy cfg 8 qmap 0 32
runlist rl
runlist_add rl lxf pool xres consts kv act ptab state cfg
runlist_add rl axf pool xres consts kv act ptab state cfg
runlist_exec rl
```
