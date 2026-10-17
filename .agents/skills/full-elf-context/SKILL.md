---
name: full-elf-context
description: Run several mlir-aie designs (kernel sets) in ONE XDNA2 hardware context by assembling their built streams into a full ELF (aiecc --get-full-elf --expand-load-pdis, aiebu-asm), switching between them with register-write reconfiguration (0.3-0.7 ms instead of ~2.1 ms per xclbin context switch). Use when a pipeline alternates between several xclbins and the switches cost time, when building or debugging open_kernels/compose_elf.py or the diffusion engine's ELFs, when a full-ELF run hangs with ERT_CMD_STATE_TIMEOUT (especially while another process uses the NPU), when XRT kernel creation is slow, or when a baked-in runtime value (an RTP write) has to vary per run.
---

# One hardware context for several designs

Worked example: FLUX.2 klein (six kernel sets, 822 switches per image). The switch cost
went from ~2.1 ms to 0.3-0.7 ms: 5.2 → 3.7 s at 512², 13.5 → 12.0 s at 1024², pixels
identical.
- Code: `open_kernels/compose_elf.py` (assembler), `src/open_diffusion/engine.cpp`
  (replay).
- Probes and numbers: `utilities/reconfig-probe/` (README), and
  `utilities/dit-chain/switch_probe.py`.
- Plan: `specs/open-diffusion/archive/one-context.md`.

## What a switch costs (measure before building anything)

`switch_probe.py` times each op three ways: queued, waited, and alternating with an op of
another set.
- Host round trip: 0.005-0.16 ms.
- xclbin context switch: 2.0-2.4 ms, whatever the op or the set.

That is configuration loading. The options inside one full ELF:

| mode (aiecc flag) | cost per switch | notes |
|---|---:|---|
| `load_pdi` (`--get-full-elf`) | ~2.0 ms | no gain; the firmware skips a repeat of the loaded PDI |
| register writes (`--expand-load-pdis`) | 0.30 / 0.51 / 0.70 ms (gemm / ew / fa) | scales with the set's config size; every configure pays it |
| control packets (`--load-pdi-to-ctrl-pkt`) | — | won't build if the design uses every shim DMA channel |

## The shape that works

- **Assemble, don't compile one module.** aiecc's `AIEAssignBufferAddresses` walks the
  whole device once per tile (`AIEAssignBuffers.cpp:155/572`). Its per-core split clones
  the whole module once per core (`Actions.h:159`). So one module with every stream costs
  cores × sequences: 67 streams took 671 s and ~10 GB.
  - Instead run one configuration build: `main` with a configure-only sequence per set,
    plus each device with ONE stream. That gives every PDI and cfg code under one
    numbering.
  - Then one small build per stream, in parallel. A stream's control code and its set's
    PDI are byte-identical to the monolithic build's.
  - Write `full_elf_config.json` yourself and run `aiebu-asm -t aie2_config -j <json> -o
    <elf>`; that is all aiecc's last step does. 138 streams: 7 min cold, seconds cached.
- **Kernels:** a device's own sequence is kernel `<device>:<sequence>`, and it does NOT
  configure. Run it with the set not loaded and it hangs until timeout. A configure-only
  kernel is `aie.runtime_sequence @cfg_S() { aiex.configure @S { } }` in `main`.
- **Issue a configure only on a set change.** Reconfiguring a set on top of itself
  (ew then ew) corrupted outputs about 5% of the time.
- **Host side** (C++, XRT 2.21):
  - `xrt::elf(path)` → `xrt::hw_context(device, elf, {{"priority", 0x180}}, shared)`;
  - `xrt::ext::kernel(ctx, "S:stream")`;
  - `xrt::ext::bo(device, size)`, which is zero-copy (`group_id` on an ext kernel
    crashes);
  - pyxrt has `elf`, `ext.kernel`, `ext.bo`, `runlist` and `run.wait2`, but not
    `get_ctrl_scratchpad_bo`.

## Surviving another process on the NPU (the part that bites)

With register writes, the configuration is one the firmware doesn't know. If another
context takes the NPU mid-image, ours comes back without it: `ERT_CMD_STATE_TIMEOUT`,
usually taking the other process down too. A pure `load_pdi` full-ELF context failed the
same way. The xclbin path survives, so this is specific to multi-PDI full-ELF contexts.
What made it survive (`utilities/reconfig-probe/contention_trial.ps1`: 8 of 8 hung at
normal priority across four variants; 6 of 6 passed with all three below):
1. **Each set's configure and its ops as one `xrt::runlist`** (a stretch).
2. **Every configure starts from a real reset.** A cfg's first op is a `load_pdi` of an
   empty device, and the firmware skips it when it names the PDI already loaded. So build
   two variants of each cfg that name the two empty devices (`compose_elf.cfg_variants`,
   a u16 at byte 2 of that op) and alternate them.
3. **QoS priority 0x180** (amdxdna "high"; normal is 0x200). With it, the other context
   runs only between our stretches; without it, both hang.

Not tested: a contender that is itself high or realtime priority.

## Kernel creation cost

Every `xrt::ext::kernel` walks all of the ELF's control code, ~2 ms per MB per kernel.
Threads don't help; XRT serializes it. So:
- one ELF per resolution, holding only what its schedule runs;
- split a per-run variant into a small head plus a shared tail, never whole copies
  (below);
- create the kernels on a thread while the weights load. It fully overlaps (1024²:
  load 5.8 s, the same as six xclbins).

## A value that varies per run (an RTP write)

Instruction words can't be patched inside an ELF. For te_attn's valid_len:
- diff a probe build to find the words: they are the value field (byte 16) of one
  `write32` per core;
- all of them come before the stream's first DMA op, and none carries an argument patch;
- so split the TXN: a head of just those writes, one per value (784 B each, 512 values),
  plus the rest as a shared tail, run head then tail in the same stretch.

The TXN format (`compose_elf.txn_ops`):
- a 16-byte header: 6 geometry bytes, 2 spare, u32 op count, u32 bytes;
- ops back to back: write32 24 B, maskwrite/maskpoll 28 B, load_pdi 16 B;
- blockwrite: its size at +12; custom ops (≥128): their size at +4.
- The parse must end exactly at the header's byte count.
