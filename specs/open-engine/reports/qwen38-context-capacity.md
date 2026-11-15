# XDNA2 hardware-context capacity on NPU Strix

## Question

The Qwen3.8 paired-prefill control declares eleven hardware contexts. Probes
failed on the ninth of them, which was recorded as a possible eight-context
limit. This investigation asks whether that is a firmware resource limitation,
a driver/resource-solver defect, a leak, or transient runtime state, and
whether it depends on partition width.

**Current result: fresh-boot Phase A completes without a capacity decrease.**
All 12 sequential benchmark processes exit successfully and are followed by
16 available contexts; all three initial probe-only controls also hold 16.
Actual XRT 2.25 library mappings are verified throughout. No triggering
operation or minimal reproducer is identified by this run. Observer effects,
longer idle periods, concurrency and error paths remain unexcluded.

Historical observations include 16 and 8 available full-width contexts. A
baseline-only observation also held 16 again before the operator reboot.
Consequently, reboot has not been proven necessary for recovery. The operator
then rebooted specifically to establish the fresh-boot Phase A precondition;
this is not a controlled recovery test from a known degraded state. No driver
reload or power-management setting change is performed by this investigation.

The earlier benchmark and the later low-capacity measurement are not tightly
bracketed: the campaign ends at **04:49:07 UTC**, while the first later failure
is at **13:00:12 UTC**. No capacity observation locates the change within this
eight-hour interval. The benchmark is a candidate sequence, not a proven cause.
Neither a leak, firmware defect, Resource Solver defect nor a runtime-PM
trigger has been established.

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
- The shared lock excludes cooperating project diagnostics, not arbitrary
  system clients. The transition recorder uses journal cursors, preserves all
  kernel entries (including PM/reset events), and captures partition reports
  while contexts are held and after explicit release. Missing diagnostics are
  reported as unavailable, not as an absence of errors.

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

The observations below follow a 12-process benchmark campaign, with an
eight-hour unobserved interval. They do not locate the operation at which
capacity changed:

| Experiment | Result |
|---|---|
| capacity, `ln` artifact | 8 |
| capacity, `gemm_n34816_k5120_low` artifact | 8 |
| capacity, `ag_s256` artifact | 8 |
| 11 xclbins registered in one process, then capacity | 8 |
| 200 sequential create/destroy in one process | 0 failures; capacity still 8 |
| 4 child processes each creating 6 contexts and exiting without explicit destroy | capacity still 8 |
| repeated capacity checks separated by idle | reported as 8; no controlled PM attribution |
| the same measurement on the 2.25 and 2.26 runtimes | 8 on both |
| `xrt-smi examine --report aie-partitions` throughout | “No hardware contexts running on device” |

The retained model attempts fail in the low-capacity observation window. This
does not describe a permanent present state: a later baseline-only observation
holds 16 contexts again. The 256-token comparison is retained separately and
does not complete PR #3 acceptance.

### Pre-reboot baseline-only observation — 16, aged boot

Boot ID: `2335c9c7-064a-4ee0-8b13-0d941c6e4812`, approximately 35 hours old.
Actual mappings include the core, coreutil and xdna plugin from `/usr/lib`, all
2.25.00. Three consecutive probes each hold 16 contexts, release them and exit
zero; no errors appear in their cursor-bounded kernel-log windows. The live
report records PID 380000 with context IDs 886–901 for the first probe. These
are report-visible driver IDs, not assumed firmware IDs or MSI-X assignments.
The release report shows no active contexts. Runtime PM is `suspended`/D3hot
before the first probe and `active`/D0 afterward; this is an observation, not
proof that resume caused recovery.

Evidence: `Models/qwen38-27b/pr3-gemm/transition-current-observation/`.
The Phase A preflight is separately retained at `transition-preflight/`; it
returns `BLOCKED_FRESH_BOOT` without creating contexts or launching a benchmark.

### Fresh-boot Phase A — completed without reproducing degradation

Measured 2026-11-15, 14:00:26–14:11:46 UTC on the same Ryzen AI 9 365 / Strix
host and driver described above. Boot ID:
`9285622f-257b-4e83-87b0-494393f16623`; start uptime **174.58 seconds**.
Total recorder elapsed time is **679.85 seconds**, including pinned-input
verification and probes. This is one sequential replay, not repeated proof of
system-wide runtime stability or an accepted performance campaign.
The probes request at most 16 contexts and stop on the first creation failure;
success demonstrates **at least 16 available**, not a maximum of 16. They do
not intentionally issue a seventeenth over-capacity request in Phase A.

The prepared historical process order and artifacts are unchanged, while the
recorder deliberately binds installed XRT 2.25 rather than the historical
2.26-linked runtime. All 192 pinned inputs pass their checks. Three no-work
capacity probes precede the first benchmark. The full 64-layer CLI then runs
256 prompt tokens and generates 16 tokens in each of the following processes:

| Process | Arm | PID | Exit | Capacity before → after |
|---|---|---:|---:|---|
| 00 | after | 14015 | 0 | 16 → 16 |
| 01 | before | 15159 | 0 | 16 → 16 |
| 02 | before | 15331 | 0 | 16 → 16 |
| 03 | after | 15697 | 0 | 16 → 16 |
| 04 | after | 15885 | 0 | 16 → 16 |
| 05 | before | 16079 | 0 | 16 → 16 |
| 06 | before | 16689 | 0 | 16 → 16 |
| 07 | after | 16927 | 0 | 16 → 16 |
| 08 | after | 17131 | 0 | 16 → 16 |
| 09 | before | 17354 | 0 | 16 → 16 |
| 10 | before | 18445 | 0 | 16 → 16 |
| 11 | after | 18635 | 0 | 16 → 16 |

All 12 processes have verified XRT 2.25 mappings and identical 16-token greedy
trajectories. This does **not** substitute for residual, full-state or independent
oracle gates. All 27 cursor-bounded kernel-log windows (15 probes and 12 actions)
are available, with zero parsed amdxdna errors. Each probe retains its live
partition report, post-release report, lifecycle and actual library mappings.

Raw evidence: `Models/qwen38-27b/pr3-gemm/transition-fresh-01/`.
The result's SHA-256 is
`0a0d6a2584f001529fbb38c0987d0ef76cdc24bac16e34b84bf2bf0eee4bcdf0`.
The checked-in GEMM report JSON contains the process matrix, timestamps and
identities for all 36 raw process log/lifecycle files. Recorder exit is zero,
status `COMPLETED_NO_DECREASE`, with no trigger assigned.

A separate cleanup/observer follow-up refuses to start at 905.79 seconds uptime
because its fresh-start window is 900 seconds; none of those actions run. This
does not invalidate the completed Phase A sequence, which started fresh.

At 14:14:19 UTC the device is naturally suspended/D3hot without a PM setting
write. A separately labelled baseline-only observation holds 16 contexts in
three probes on wake, with no parsed amdxdna errors. Evidence is retained at
`transition-fresh-idle-observation-01/`. One natural suspend/wake does not
reproduce the loss; it does not rule out other PM sequences or longer idle.

## Distinguishing the hypotheses

| Hypothesis | Result | Evidence |
|---|---|---|
| Firmware-reported unavailability | **Supported** | `NOAVAIL` (`0x2000003`) on create-context opcode 0x2 at the refusal |
| Firmware-enforced maximum of 8 | **Rejected** | 16 held in the same boot, both runtimes, distinct and identical assets |
| Firmware-enforced maximum of 16 | **Not proven** | 16 is an observed ceiling and matches the advertised `hwctx_limit`, but Phase B drops to 8 with no reboot; the firmware reply is a generic unavailability |
| Driver Resource Solver defect | **Not supported, not excluded** | the solver hands every context the same full-width partition and returns capacity on destroy; refusals come from the firmware reply. In Phase B the solver still hands out partitions while firmware refuses at 8 |
| Leak or context/resource retention | **Unresolved** | selected create/destroy and exit controls show no observed loss; they do not cover every dispatch, error, cleanup or concurrent-client path |
| Firmware restart/state issue | **Unresolved** | no tightly bracketed transition or validated restart correlation; `NOAVAIL` alone is insufficient |
| Runtime PM effects | **Unresolved** | automatic suspend/resume is observable but not correlated with the original transition |
| Partition-width dependence | **Not measured** | every checked artifact is width 8; no validated narrower xclbin |
| Shared cap across processes | **Supported** | 4 held elsewhere + 12 created = 16 total in Phase A |

The driver path: every context requests 8 columns at column 0, the solver marks
that partition shared and returns the same one, and `aie2_create_context`
forwards to firmware. The logged failure is the firmware reply to opcode 0x2;
the solver's `Allocate AIE resource failed` line is a consequence of that
reply, not a separate solver rejection.

## What remains to establish root cause

1. Phase A is completed once with no decrease, so there is no smaller failing
   bracket to replay yet. Compare a matched run with less intrusive capacity
   sampling and longer natural idle intervals, retaining existing kernel/PM
   logs. The inserted probes and observers alter wake/idle scheduling; agreement
   across three probe-only controls does not prove that they cannot mask a
   trigger. Stop on the first loss. A future bracket identifies an action plus
   its following probe, not yet the exact internal operation.
2. Compare creation-only, dispatch through the existing harness/CLI, explicit
   destruction and normal process exit with pinned inputs and the same runtime.
   Use separate fresh-boot runs for sequential/concurrent workloads and
   controlled abnormal termination; the latter is not enabled automatically.
3. Preserve evidence before recovery tests. Compare idle and complete exit,
   then an explicitly authorized driver reload, then reboot if necessary.
   **No recovery operation has been proven necessary or sufficient.**
4. Use existing driver dynamic debug, firmware logging and kernel PM tracing
   around the minimal bracket before considering a diagnostic driver change.
   A failed `create_ctx_resp` has no documented resource-exhaustion reason
   beyond status. Undefined fields on failure must not be decoded as an
   exhausted-resource identity.

The loaded and on-disk driver version is
`2.26.0_20260817,2b71d02b641561a40ab2db1c26a69620794f4e9f`, srcversion
`D9A90D693FFE30975B52EBB`. Installed DKMS source files for the allocation path,
response header, solver and debugfs match that exact revision, not the newer
development checkout. On success it logs firmware context ID, PASID and MSI-X
assignment through existing debug messages and receives command-queue counts
and addresses. On failure it returns after the management status check. Existing
debugfs exposes firmware log level and log-to-dmesg controls. The tracing and
debugfs mounts are root-only here; no permission or logging setting is changed.

A driver issue is justified if a minimal pinned sequence shows unreclaimed
resources, inconsistent accounting or a restart/PM failure. A separate
OpenFlowLM cleanup fix is justified if application object lifetime is proven
to retain live references. Neither conclusion is established yet; context
consolidation is not a demonstrated remedy. PR #3 remains Draft.

## Reproduction

```bash
source utilities/env.sh
OPENBLAS_NUM_THREADS=1 ironvenv/bin/python utilities/q4-gemm/context_probe.py \
  --experiment diagnose \
  --kernels Models/qwen38-27b/production-v2/kernels-model \
  --out Models/qwen38-27b/pr3-gemm/context-final-2 \
  --count 17 --repeats 2 --churn 64
```

For the deterministic transition experiment, use the existing probe's new
preparation and transition modes (no replacement harness):

```bash
ironvenv/bin/python utilities/q4-gemm/context_probe.py --experiment prepare \
  --kernels Models/qwen38-27b/production-v2/kernels-model \
  --campaign Models/qwen38-27b/pr3-gemm/e2e-256/result.json \
  --out Models/qwen38-27b/pr3-gemm/transition-sequence

# Immediately after an operator reboot; choose a NEW output directory per run.
ironvenv/bin/python utilities/q4-gemm/context_probe.py --experiment transition \
  --kernels Models/qwen38-27b/production-v2/kernels-model \
  --sequence Models/qwen38-27b/pr3-gemm/transition-sequence/plan.json \
  --out Models/qwen38-27b/pr3-gemm/transition-fresh
```

Preparation freezes the retained campaign's argv/order and input identities;
the reproduction intentionally pins installed XRT 2.25, unlike the historical
2.26-linked benchmark. Every foreground process has PID, exit status, loaded
libraries, stdout/stderr and wall/monotonic times recorded. Each action is
bracketed by capacity probes and PM snapshots. Input changes, runtime mismatch,
failed setup and partial results stop the sequence. A process watchdog is a
labelled forced termination, never a graceful-exit control.

The runner refuses boots older than 900 s, unavailable/partial runtime evidence,
or a baseline below 16. Three no-work probe repetitions precede all actions.
`--baseline-only` permits an aged-boot observation but executes no actions and
never calls it Phase A reproduction. Plans can also use `create-destroy`,
`process-exit`, `observer-control` and bounded `idle` actions to isolate the
observer and cleanup. The runner does not implement a reset, reload, reboot,
power-state write or deliberate abnormal termination. Optional root-only
diagnostics and recovery remain operator-controlled.
