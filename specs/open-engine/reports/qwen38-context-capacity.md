# XDNA2 hardware-context capacity on NPU Strix

## Question

The Qwen3.8 paired-prefill control declares eleven hardware contexts. Probes
failed on the ninth of them, which was recorded as a possible eight-context
limit. This investigation asks whether that is a firmware resource limitation,
a driver/resource-solver defect, a leak, or transient runtime state, and
whether it depends on partition width.

**Answer: there is no stable eight-context limit, and the ceiling is not fixed.**
Within one boot the same measurement returned **16** and later **8**. Sixteen
identical or distinct full-width contexts were held repeatedly; the seventeenth
failed with firmware `NOAVAIL` while the driver advertises `hwctx_limit = 16`.
Later, with the device reporting no held contexts, only **8** could be created,
and that persisted across both installed runtimes and after repeated
create/destroy and child-process churn. Destroying a context returns capacity
in the healthy state; process exit leaks nothing. The remaining explanation is
a persistent driver/firmware-side state that is not visible to
`aie-partitions`, not a proven firmware maximum and not a simple count cap.

Nothing here changes firmware, driver parameters, model precision or recipes.

## Method

Tool: `utilities/q4-gemm/context_probe.py --experiment diagnose`. One pyxrt
process per arm creates and destroys contexts explicitly; the kernel harness
can only create, so destroy/reclaim and churn need pyxrt.

- Artifacts: the unchanged, verified `production-v2/kernels-model` contexts.
  All eleven are width 8 at start column 0. A width experiment needs a
  validated narrower artifact, which does not exist in this tree.
- Runtimes: both installed stacks, recorded from `/proc/self/maps`:
  `/opt` (`libxrt_coreutil.so.2.26.0`) and `/usr/lib`
  (`libxrt_coreutil.so.2.25.00`). `xrt-base` is 2.26, `libxrt2` is 1:2.25.0-4,
  the tools report 2.26.0.
- Driver/firmware: amdxdna 2.26.0_20260817, NPU firmware 1.1.2.64, Strix
  aie2p, 8 columns, 32 MSI-X vectors.
- Each arm is wrapped in the shared NPU lock and records its kernel-log window
  (decoded management status) and
  `xrt-smi examine --report aie-partitions --verbose`.

## Evidence matrix

Measured 2026-11-15, ~26 h after boot (no reboot between the 16 and 8 results).

### Phase A — healthy: ceiling 16

| Experiment | Target | Max held | Next create | FW status | Capacity returned |
|---|---:|---:|---|---|---|
| 11 distinct xclbins | 16 | 16 | n/a | — | n/a |
| 11 distinct xclbins | 17 | 16 | EINVAL | 0x2000003 | n/a |
| single asset, repeated | 17 / 18 / 20 / 32 | 16 | EINVAL | 0x2000003 | n/a |
| identical contexts | 17 | 16 | EINVAL | 0x2000003 | n/a |
| 8 held, destroy 1, create 1 | 9 | 8 → replacement created | — | — | **yes** |
| 64 churn cycles, then capacity | 81 | 16 | EINVAL at 17th | 0x2000003 | no loss |
| 4 held by another process | — | 12 more creatable | — | — | after exit, returned |

Both runtimes behaved identically in this phase.

### Phase B — degraded: ceiling 8

After a 12-process end-to-end benchmark campaign (256-token prefill/decode
runs), the same measurement returned 8 and stayed there:

| Experiment | Result |
|---|---|
| capacity, `ln` artifact | 8 |
| capacity, `gemm_n34816_k5120_low` artifact | 8 |
| capacity, `ag_s256` artifact | 8 |
| 11 xclbins registered in one process, then capacity | 8 |
| 200 sequential create/destroy in one process | 0 failures; capacity still 8 |
| 4 child processes each creating 6 contexts and exiting without explicit destroy | capacity still 8 |
| capacity after 5 s and after 60 s idle | still 8 |
| the same measurement on the 2.25 and 2.26 runtimes | 8 on both |
| `xrt-smi examine --report aie-partitions` throughout | “No hardware contexts running on device” |

The 11-context model now fails to load at 4 layers and at 64 layers; the
256-token end-to-end comparison completed before the degradation and is the
only end-to-end result available.

## Distinguishing the hypotheses

| Hypothesis | Result | Evidence |
|---|---|---|
| Firmware-reported unavailability | **Supported** | `NOAVAIL` (`0x2000003`) on create-context opcode 0x2 at the refusal |
| Firmware-enforced maximum of 8 | **Rejected** | 16 held in the same boot, both runtimes, distinct and identical assets |
| Firmware-enforced maximum of 16 | **Not proven** | 16 is an observed ceiling and matches the advertised `hwctx_limit`, but Phase B drops to 8 with no reboot; the firmware reply is a generic unavailability |
| Driver Resource Solver defect | **Not supported, not excluded** | the solver hands every context the same full-width partition and returns capacity on destroy; refusals come from the firmware reply. In Phase B the solver still hands out partitions while firmware refuses at 8 |
| Leak | **Not supported as a monotonic leak** | 200 create/destroy in one process and 4 child-process exits left capacity unchanged; churn in Phase A did not reduce it |
| Stale firmware state | **Best remaining explanation** | a persistent, non-monotonic reduction from 16 to 8 within one boot, invisible to `aie-partitions`, that does not clear with idle time, explicit destroys or process exit |
| Partition-width dependence | **Not measured** | every checked artifact is width 8; no validated narrower xclbin |
| Shared cap across processes | **Supported** | 4 held elsewhere + 12 created = 16 total in Phase A |

The driver path: every context requests 8 columns at column 0, the solver marks
that partition shared and returns the same one, and `aie2_create_context`
forwards to firmware. The logged failure is the firmware reply to opcode 0x2;
the solver's `Allocate AIE resource failed` line is a consequence of that
reply, not a separate solver rejection.

## What remains to establish root cause

The decisive experiment is capturing the firmware's create-context **response
payload** for a refused request. The driver logs only the status word and
discards `create_ctx_resp`. A tracing build of amdxdna, or a tracepoint on
`aie_send_mgmt_msg_wait`, would show whether firmware is refusing on context
slots, a memory region, or a queue resource — and would settle whether 16 is a
firmware maximum or a driver-advertised policy.

Also outstanding, in order of value:

1. Reproduce the 16 → 8 transition deliberately: run the 12-process benchmark
   campaign from a fresh boot, measuring capacity before, between and after
   each run, to identify the operation that changes it.
2. Repeat after a reboot, which is the only confirmed recovery path observed.
3. Repeat with a validated narrower-partition xclbin, once one exists.
4. Repeat capacity under contention with in-flight work, not idle holders.

## Reproduction

```bash
source utilities/env.sh
OPENBLAS_NUM_THREADS=1 ironvenv/bin/python utilities/q4-gemm/context_probe.py \
  --experiment diagnose \
  --kernels Models/qwen38-27b/production-v2/kernels-model \
  --out Models/qwen38-27b/pr3-gemm/context-final-2 \
  --count 17 --repeats 2 --churn 64
```

`result.json` holds per-arm events, the libraries actually loaded, the
kernel-log window and the partition report; `matrix.md` is the summary. Add
`--narrow-xclbin PATH` for the width arm; it is refused unless the artifact's
declared width is narrower than the device partition.
