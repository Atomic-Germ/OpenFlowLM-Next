# reconfig-probe: what a kernel-set change costs on XDNA2, and how to make it cheap

Probes behind `specs/open-diffusion/plans/one-context.md`. klein's pipeline changes
kernel set 822 times per image. These tools measure where that time goes and test the
one-context alternative.

| tool | measures |
|---|---|
| `../dit-chain/switch_probe.py` | real ops on their six xclbin contexts: queued, waited (host round trip) and alternating between sets |
| `loadpdi_probe.py` | the three in-stream reconfiguration modes on mlir-aie's one-core `reconfigure_loadpdi` pair |
| `compose_probe.py` | real sets composed into one full ELF: switch cost per mode, and configure-on-change |
| `fullelf_generate.py` | a whole klein image through one full-ELF context, compared byte for byte with `generate.py` |

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
- The runtime-sequence probes need unique SSA names per inlined `aiex.run`.
- pyxrt 2.21 has `elf`, `ext.kernel`, `ext.bo` and `run.wait2`, but no
  `run.get_ctrl_scratchpad_bo`. The C++ API has it.
