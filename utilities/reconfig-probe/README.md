# reconfig-probe: what a kernel-set change costs on XDNA2, and how to make it cheap

Probes behind `specs/open-diffusion/archive/one-context.md`. klein's pipeline changes
kernel set 822 times per image. These tools measure where that time goes and test the
one-context alternative.

| tool | measures |
|---|---|
| `../dit-chain/switch_probe.py` | real ops on their six xclbin contexts: queued, waited (host round trip) and alternating between sets |
| `loadpdi_probe.py` | the three in-stream reconfiguration modes on mlir-aie's one-core `reconfigure_loadpdi` pair |
| `compose_probe.py` | real sets composed into one full ELF: switch cost per mode, and configure-on-change |
| `fullelf_generate.py` | a whole klein image through one full-ELF context (compiled as one module, the slow way), compared byte for byte with `generate.py` |
| `preempt_probe.py` | real ops of an installed v2 ELF in engine-style stretches, queued or as runlists, outputs checked every iteration |
| `cold_probe.py` | real ops of an installed v2 ELF right after their set's configure vs run again: is an op slower cold? (`--scale` fills activations with N(0, σ)) |
| `contention_trial.ps1` | one engine trial while `switch_probe.py` hammers the NPU from another process: image correct? contender alive? |

All of them need the IRON environment and turbo:

```
. C:\dev\mlir-aie\iron_env.ps1
xrt-smi configure --pmode turbo
```

## Results (2026-09-29, HX 370, turbo, CPU load 5-20%)

| | cost |
|---|---:|
| host round trip, one context | 0.005-0.16 ms (the Windows timer at 1 ms changes nothing) |
| a switch between two xclbin contexts | 2.0-2.4 ms, whatever the op or the set |
| `load_pdi` in one full ELF, 1-core design (1.6 KB PDI) | 81 µs |
| `load_pdi` in one full ELF, klein's sets (120-440 KB PDIs) | ~2.0 ms: the switch cost is PDI loading |
| `--expand-load-pdis` (register writes) into gemm / ew / fa | 0.30 / 0.51 / 0.70 ms |
| `--load-pdi-to-ctrl-pkt` | does not build: the overlay needs a shim DMA channel klein's designs use |

- With register writes the configuration is paid on **every** configured run, and a
  repeated set is not skipped. So the host issues a configure-only kernel
  (`main:cfg_<set>`) only on a set change, then the stream's own device sequence
  (`<set>:<stream>`), which does not configure.
- A device's own sequence run without its set loaded hangs until the timeout
  (`ERT_CMD_STATE_TIMEOUT`, cleanly recovered).
- Whole image, 512², text skipped (`fullelf_generate.py --ctx-ref`): 831 dispatches + 632
  configurations. 3.51-3.54 s against `generate.py`'s 4.53 s on the same inputs; latents
  and PNG byte-identical, 3 runs.

## Another process on the NPU

`contention_trial.ps1`, klein at 512² and 1024², with a six-xclbin-context contender:

| engine | trials | outcome |
|---|---:|---|
| queued runs, normal priority | 2 | hung (`ERT_CMD_STATE_TIMEOUT`), usually taking the contender down too |
| runlists, normal priority | 1 | hung |
| runlists + alternating resets, normal priority | 4 | hung |
| `load_pdi` configures (firmware-known PDIs), normal priority | 1 | hung |
| runlists + alternating resets, priority 0x100 (3) or 0x180 (3) | 6 (24 images, 3 at 1024²) | byte-identical, contender alive, images 10-25% slower while it ran |
| the xclbin path (`generate.py`) | 1 | byte-identical |

`preempt_probe.py` passes under contention at normal priority: it idles between
iterations, which gives the scheduler harmless places to switch. The engine keeps its
queue full, so it doesn't get those.

## After a configure (2026-09-30, quiet, `cold_probe.py`, 1024²)

An op runs no slower right after its set's configure: `attn_sgl` took 31.9 ms first and
31.6 ms again, `sgl_out` 21.6 / 21.6, `qk_sgl` 3.16 / 3.16, `res_all` 2.05 / 2.06. The
configures cost 0.8 (fa), 0.4 (gemm) and 0.6 ms (ew). With `--scale`, attention slows on
wide synthetic spreads (32.4 ms at σ = 1, 38.5 at σ = 4, 41.6 at σ = 16): dit_fa's lazy
rescale. klein's real scores rarely trigger it (`../dit-chain/attn_rescales.py`), so this
doesn't explain the engine's slower attention; see `specs/open-diffusion/archive/phase7-speed.md`.

## Kernel creation

`xrt::ext::kernel(ctx, name)` walks all of the ELF's control code:

| ELF | control code | per kernel |
|---|---:|---:|
| 144 instances of 7 KB | ~1 MB | 2.2 ms |
| 13 instances, full size | ~1 MB | 1.3 ms |
| 144 instances, full size | 15 MB | 17 ms |
| + 512 whole te_attn copies | 36 MB | 56 ms |

Creating kernels on 8 threads is no faster. On a thread next to file reading, it
overlaps completely.

## Why one composed module builds slowly

The 67-stream module took 671 s and ~10 GB in aiecc, 9.5 min of it before any core
compiled.
- `AIEAssignBufferAddresses` walks the whole device once per tile, so it visits every op
  of every runtime sequence each time.
- The per-core split clones the whole module once per core.

Hence the exporter assembles the ELF from per-set and per-stream builds instead
(`open_kernels/compose_elf.py`).

## Traps

- aiecc merges adjacent `aiex.configure` of one device, so their BDs accumulate. A
  sequence that does not free its tasks runs out of shim BDs (16) once inlined many times.
- A cfg's first op loads an empty device's PDI, and the firmware skips it if that PDI is
  the one loaded. Alternate two variants naming different empties, or configures don't
  reset.
- Configuring a set on top of itself (ew, cfg_ew, ew) corrupted `qk` outputs in ~5% of
  iterations. Configure only on a set change.
- The runtime-sequence probes need unique SSA names per inlined `aiex.run`.
- pyxrt 2.21 has `elf`, `ext.kernel`, `ext.bo` and `run.wait2`, but no
  `run.get_ctrl_scratchpad_bo`. The C++ API has it.
