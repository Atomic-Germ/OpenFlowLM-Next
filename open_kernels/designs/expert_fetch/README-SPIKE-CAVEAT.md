# expert_fetch spikes — read this before trusting their recorded results

These `.mlir` spikes are the project's original proof of the on-device expert-routing
primitive (a control packet that retargets and enqueues a shim DMA descriptor with no host
round-trip). Their headers record PASS results, and those results were used — by the runlist
goal (`mtusoiy1-cfdhqr`) and by anything else that cites them — as the authority for claims
such as *"a shim's own DMA cannot reach its own TileControl"* and *"the legal shape is a
core-side packet-stamped BD plus a matching `controller_id`"*.

**On the current box those spikes cannot be re-run.** Every spike is a hand-written register
program — `aiex.npu.blockwrite`, `aiex.npu.write32`, `aiex.npu.maskwrite32`,
`aiex.npu.address_patch`, and the six-operand `aiex.npu.sync` — and every one of them fails on
dispatch with `qds_device::wait() unexpected command state`, while every design in this repo
that uses the high-level `aiex.dma_configure_task_for` / `dma_start_task` / `dma_await_task`
ops runs fine on the same device, same harness, seconds apart (the shipped `lx0` completes in
~3.6 ms). So the spikes' recorded PASSes belong to an earlier XRT/driver combination and their
claims are **unverified on this environment**.

Two consequences worth knowing:

1. Re-derive anything that depends on a spike with an **IRON probe using the high-level ops**
   (`designs/expert_fetch/ondv_flow_probe.py` and `ondv_flow_cross_probe.py` already build the
   shim→TileControl route, and `ondv_core_packet_probe.py` builds the core-side packet BD and
   its emitted `packet_dest<shim, South:0>` + `packet_dest<shim, TileControl:0>`); note that
   those probes only show the route is emitted — they have nothing to observe, so they cannot
   tell "applied" from "dropped".
2. If a spike must be revived, rebuild it for `aie.device(npu2)` and run it through
   `open_kernels/harness/run_kernel` (classic `kernelx` flow), but first check the
   `aiex.npu.*`/`sync` sequence against the current driver — that is the part that stopped
   working, not the mechanism.

Related, measured elsewhere: `unexpected command state` is also a **device-contention**
signature (it reproduces on the known-good `lx0` while another process holds `accel0`), which
is why `run_kernel.cpp` now has `HARNESS_RETRY_CONTENTION`.

Full measurements and the surviving conclusions:
`benchmarks/RESULTS-ondv-fused-layer-descriptor-milestone-2026-09-22.md` (goal
`mtusoiy1-cfdhqr`), sections "A discrepancy that must be settled", "Why the spikes cannot be
re-verified here" and "The spike re-run failure is a device-target artifact".
