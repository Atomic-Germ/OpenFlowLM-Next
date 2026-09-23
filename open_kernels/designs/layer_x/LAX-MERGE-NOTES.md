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

`check_lax_merge.py <build_lax_l> <build_lax_a>` re-derives all of the above from the
builds (same-size xclbins differing only in metadata, byte-identical per-tile
programs, main program == lx's 16272 B, <=2 DMA channels per shim and <=16 total, and
identical `w` channels) and exits non-zero if any of it regresses.

## Device status (2026-09-23): the fused path does not complete

After the merged build I tried to device-verify. `lx0` (the shipped first half,
classic flow) and `expert_fetch/build_live` (the one-emitter packet probe) both
complete -- `state 4`, 3.4 ms and 1.1 ms -- but **every fused whole-layer build
hangs**: `build_emitter`, a fresh rebuild of `lx.py` from the current sources, all
`build_ondv*` variants, and `lax` itself, via both the ELF and the classic flow.
This is not contention: it reproduces after `modprobe -r amdxdna` with no
`accel0` holder, and on a machine where `lx0`/the probe complete immediately
afterwards. `dmesg` shows the failure directly:

```
[21675] amdxdna 0000:c6:00.1: AIE2_TDR_WORK: Device isn't making progress... Count 6 timeout 15
[21738] amdxdna 0000:c6:00.1: AMD-Vi: Event logged [IO_PAGE_FAULT domain=0x0002 address=0x7fb5f2700000 flags=0x0007]
[21781] amdxdna 0000:c6:00.1: AIE2_DUMP_CTX: Firmware timeout state capture: ... JOB[0]: op: 0x14 msg: 0x1d000001
```

The FAULT addresses are BO bases, consistent with the fused design's routed
descriptors being retargeted at addresses the IOMMU has not mapped, which would
stall the shim transfer, block the main core on `w`, and fail the run as a TDR (the
fault lines interleave with the peer's jobs on this shared box, so that attribution
is the design's, not proven line by line). The probe's own header names exactly this
outcome -- "a timeout means it was never pushed, which is the failure the fused
design hits today" -- so this is the pre-existing ONDV blocker, not something `lax`
introduced. The standalone
`build_emitter` no longer reproduces the `state 4` the pause reason records (its
control text is byte-identical to a fresh rebuild, and the two xclbins differ only
in metadata), so either the array/driver no longer supports the packet push or the
recorded run predates a regression. Device verification of `lax` (numerics, and the
40-layer token) is blocked on that until it is resolved.

### Where the fused hang is (bisect, 2026-09-23)

Same session, same box, three results with the current sources:

| build | flow | result |
|---|---|---|
| `lx0` (shipped first half) | classic | `state 4`, 3.4 ms |
| `expert_fetch/build_live` (1 emitter) | ELF | `state 4`, out=1 |
| `lx.py` with `ONDV_SKIP_MOE=1` | ELF | **`state 4`, 4.65 ms** |
| `lx.py` (full, incl. the MoE) | ELF/classic | TDR, no completion |

The skipped-MoE build still has all 8 emitters, the 8 packet `PacketFlow`s and the 8
pinned per-column `ctrlw` buffers -- it only omits `moe_sequence(..., ondv=(cfg,))`. So
the emitters and the pre-MoE path are fine and **the fault is inside the ONDV MoE
sequence** (the routed-expert retarget+push), which is also exactly what the one-emitter
probe exercises and passes. `lx.py` now carries an env-gated `ONDV_DONE_ACQ=1` hook that
pairs every packet BD's `release(pktdone)` with an `acquire` in the emitter, matching the
probe; it did not fix the hang (tested, though every attempt that session also hit
peer contention). With the hook unset the build is unchanged: `insts.elf` is byte-
identical to `build_emitter` and every core program keeps its size.

### The fault is the packet push, and only the packet push

`ONDV_HOST_PUSH=1` (the host enqueues the routed slots' placeholder descriptors, as the
non-fused path would) plus `ONDV_NO_EMITTERS=1` completes in a clean window:
**`state 4`, zero contention** (`build_lx_hp`). So the ONDV MoE sequence, the main
body, the pre-MoE path and the pinned-descriptor layout are all correct; what fails is
the emitters' TileControl push, which the one-emitter probe (`build_live`) performs
successfully. Two candidate causes were tested and did **not** fix it:

* `ONDV_DONE_ACQ=1` -- pair every packet BD's `release(pktdone)` with an `acquire` in
  the emitter (the probe's pattern). No change.
* `ONDV_BDS=13,14,15` -- move the pinned descriptors off BDs 8/9/10 in case the `w`
  channel's own pipeline descriptors collide with them (both `xcommon.ONDV_BD_*` and
  `ondv_ctrl.h`'s `kOndvBd*`, now `#ifndef`-overridable). No change.

All three hooks are env-gated and unset by default. One pre-existing discrepancy worth
knowing: `build_emitter`'s row-3 helper cores (ln/router, post, glue) are 32 B larger
than a fresh build of the same sources (`0x27d0` vs `0x27b0` etc.), so the recorded
state-4 build is not exactly reproducible from the tree as it stands.

Remaining leads for the push: the `cfg` element the emitters read (`cfg[0..1]` pool
base, `cfg[2+col]` queue) versus the probe's `poolbase`-written `cfg`; whether the 9-BD
packet chain makes the shim's task queue overflow (the probe sends a single 28-B BD); and
whether the `Pipeline.configure` descriptors are still live when the packet arrives.

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
