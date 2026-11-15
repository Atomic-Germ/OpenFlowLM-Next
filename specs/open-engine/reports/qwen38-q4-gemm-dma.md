# Qwen3.8-27B: bounded GEMM DMA batches (PR #3 draft)

## Acceptance state

**DRAFT — implemented and isolated-kernel validated, not production accepted.**
The production default remains one DMA visit per TaskGroup. No recipe opts in,
no installed artifact or model registration changes, and no end-to-end speedup
is claimed. The required full-model gates are blocked by context creation in
the retained control as well as the candidate on this machine.

Baseline: commit `836072db`. The review branch is `perf/npu-q4-gemm`.
The measured candidate is identified by its source SHA-256 snapshot, not a
fabricated measurement commit. The containing review commit records the draft.
[Structured measurements](qwen38-q4-gemm-dma.json) bind source, fixture, build,
kernel and output identities. Raw evidence is under
`Models/qwen38-27b/pr3-gemm/` (local, not shipping artifacts).

## Architecture and hypothesis

The previous profile attributes 25,020.1 ms of one 256-token block to GEMM
dispatch accounting; that is a historical single observation, not a new
baseline. The device sequence submits one output visit (256 weight rows and
one token tile per array column), waits for its drains, then submits the next.
This exposes avoidable DMA queue gaps without requiring a new GEMM datapath.

The candidate submits up to four consecutive visits before awaiting and
reclaiming their tasks. `dma_schedule.py` preserves row-group-major,
token-tile-minor order, and emits no padded visit in a partial final batch.
All drains still complete before the group's descriptors are freed. The
compiler remains responsible for the actual 16-descriptor-per-shim limit.
The allowed build-time `GQP_DMA_BATCH` values are 1, 2 and 4; reuse-mode
activation streams refuse batching. The default is 1.

This is **device DMA pipelining**, not a change to host dispatch scheduling:

- No runtime C++ changes, new dispatches or submit-ahead policy.
- No weight, quantization, packed pool, buffer ABI or manifest schema changes.
- No activation-preparation, dequantization, FP32 accumulation or reduction
  ordering changes; all four paired-input/weight products remain present.
- No FIFO-depth, physical-block-size, routing-threshold or DeltaNet changes.
- Core programs are byte-identical across the matched DMA variants.

The exporter clears ambient `GQP_DMA_BATCH` so a shell experiment cannot leak
into an ordinary recipe build or bypass its content key. A future promotion
must name the value explicitly in the recipe environment, on both high and low
companions. Existing private replay builders copy/hash the new planner beside
their design. Recipe source globs already include it; a cache-coverage test
requires it.

## Machine and method

Measured 2026-11-15 on AMD Ryzen AI 9 365, 91 GiB RAM, Strix XDNA2/aie2p
(6x8 topology), Linux 7.0.0-43-generic, glibc 2.43, GCC 15.2.0.
XRT 2.26.0 (`e9db9ab15f10173f8d2fc93ff92ab4c7eb09d2e6`), amdxdna
2.26.0_20260817, firmware 1.1.2.64. NPU power mode is externally observed
`default`; CPU governor is `powersave`. No power configuration is changed.
Desktop/services remain active and host tests/compilation overlap part of the
isolated campaign. Sampled one-minute load ranges from 2.10 to 3.30. Per-pair
load, memory and available ACPI temperatures are recorded; ACPI readings are
not NPU die temperatures. The shared oracle lock serializes cooperating NPU
jobs; no other NPU client is observed in the context investigation.

Each row below has **five alternating pairs**. Each arm starts a fresh harness
process, excludes one warm-up dispatch and measures ten dispatches. The table
uses the median of the five per-process medians; ranges are the minimum and
maximum of those medians. Every pair improves for every listed component.
Timings are host submit-to-completion, including host wait overhead, **not
device-only execution time**. They exclude context creation and BO sync costs.
These are isolated-kernel observations under the stated load, not evidence of
end-to-end model acceptance.

The JSON also reports effective workload TFLOPS as `2*N*K*T / elapsed`, using
this same host-observed window. Separate H2D/D2H transfer costs, device-active
cycles and NPU idle time are **NOT MEASURED**; unchanged buffer sizes are not
a substitute for those measurements.

Fixtures use seed 26026 and 256 distinct BF16 token activations. The independent
FP64 reference decodes the packed pool bytes. The high reference uses exact
dequantized weights, retaining the existing weight-rounding budget; the low
reference uses their BF16 residual. Both retain `rel_fro <= 5e-3`. Comparisons
also require finite, complete, byte-identical outputs across both arms and all
repetitions. Completion count/state checks reject partial or failed runs.

## Isolated measurements

Units are milliseconds, median (range). Before/after means DMA batch 1/4 on
the same design, inputs, arithmetic and harness.

| Projection N x K | Component | Before | After | Change |
|---|---|---:|---:|---:|
| 14336 x 5120 | high | 17.426 (17.329–17.664) | 16.919 (16.886–17.025) | −2.91% |
| 14336 x 5120 | low | 18.713 (18.619–19.155) | 18.099 (17.994–18.406) | −3.28% |
| 16384 x 5120 | high | 19.871 (19.810–20.012) | 19.366 (19.185–19.426) | −2.54% |
| 16384 x 5120 | low | 21.334 (21.249–21.950) | 20.631 (20.554–21.351) | −3.29% |
| 34816 x 5120 | high | 42.446 (41.308–44.221) | 40.829 (40.519–41.378) | −3.81% |
| 34816 x 5120 | low | 43.740 (43.674–43.766) | 42.315 (42.294–42.354) | −3.26% |
| 5120 x 17408 | high | 19.864 (19.861–19.874) | 19.671 (19.659–19.851) | −0.97% |
| 5120 x 17408 | low | 21.255 (21.251–21.329) | 21.076 (21.042–21.133) | −0.84% |
| 5120 x 6144 | high | 7.261 (7.254–7.292) | 7.040 (7.030–7.054) | −3.03% |
| 5120 x 6144 | low | 7.774 (7.736–7.979) | 7.547 (7.512–7.566) | −2.91% |

All ten cases pass numerical and exact-output gates. High-component rel_fro
ranges from 0.001662 to 0.001730; low-component rel_fro from 1.00e-7 to 2.13e-7.
No tolerance is weakened. There are 1,100 successful dispatches in this campaign.

Twenty matched builds cover five production projection shapes and both weight
components. All 32 core ELFs match byte-for-byte between batch 1/4 for each
shape/component. Maximum program size is 4,784 bytes high / 5,568 bytes low,
under 16,384. Maximum compute-buffer end is 61,448 bytes, under 65,536;
the design reserves a 4,096-byte stack. Static buffers/FIFO depths do not grow.
Total driver-owned allocation is **NOT MEASURED**.

An additional N=768/K=512/T=512 case has three row groups and two token tiles:
six visits, grouped as four plus two. Both high and low pass the FP64 and
byte-identity gates over five alternating pairs. This tests an odd row-group
count, crossing token-tile boundaries, activation refill ordering and a partial
final batch on hardware. It does not establish a new supported model resolution
or context length. Default batch-1 compatibility is also checked: the
N=14336/K=5120/T=256 instruction stream exactly matches the pre-edit baseline
(`032ec3405135f5d08e3406b7cd50e26292adc3a4947399b228932602ee111ea4`).

## Full-model blocker and remaining gates

The current retained manifest declares eleven contexts. On this machine each
artifact creates a context successfully alone; keeping them together succeeds
for the first eight and fails on the ninth with
`DRM_IOCTL_AMDXDNA_CREATE_HWCTX ... err=-22: Invalid argument`.
The fresh current-branch CLI and the retained accepted CLI both fail with the
**untouched** retained export. The attempted 64-layer, 256-token candidate
benchmark fails during initialization, before inference. The binaries bind
the same installed XRT library; the fresh CLI's SHA-256 is
`c8c1ec5eebebc6e7e1d28114393870d386a6eeb545b3277dac651b0ff1aa66ad`.

This establishes a current combined-context blocker in the control, not a
numerical failure of the candidate. Repeating one unchanged full-width xclbin
(`ln` or `gemm_n34816_k5120_low`) also succeeds eight times and fails on the
ninth. The kernel log reports firmware status `0x2000003`
(`AIE2_STATUS_MGMT_ERT_NOAVAIL`) for the create-context command. All retained
artifacts request eight columns at start column zero, so they share one
full-device partition; the Strix driver's advertised 16-context telemetry
limit is not the limiting resource here. The accepted PR #2 binary, with the
same build ID and installed XRT library, now fails in the same way. This is a
persistent runtime/firmware capacity issue, not a GEMM-specific incompatibility.

`context_probe.py` retains individual/combined commands, logs and artifact
hashes. No driver reset, context-coalescing change, precision fallback or
manifest relabelling is used to bypass the gate. No NPU client is active, but
passwordless administrative access is unavailable, so a module reload or PCI
reset was not attempted. A valid recovery requires an administrator to reload
the NPU driver or reboot, followed by a fresh unchanged-control smoke test;
if the nine-context control still fails, the runtime/firmware must be fixed or
the model runtime must be redesigned in a separate PR. The older six-context
export is not a valid substitute because it omits the paired low-weight
companions, and host-attention flags do not avoid the eager GEMM context loads.

A subsequent user-reported availability check was rerun against the unchanged
export. The combined probe still fails on the ninth context and the four-layer
smoke still exits before inference with the same firmware `NOAVAIL` status.
The probe now returns its recorded result and exits nonzero on any failed
context, preventing a recorded failure from being mistaken for recovery.

Review probes reproduce the same ninth-context failure for the untouched
export and both staged batch-1/batch-4 arms; all eleven contexts pass when
created individually in every arm. No competing hardware context is visible.
The process file-descriptor limit is 524,288 and locked memory is unlimited;
these observations do not establish the driver's root cause. Read-only sysfs
inspection exposes no module parameter to adjust on this host. Evidence is
retained in `context-probe-review*` and `context-review.log`.

| Required end-to-end metric/gate | Baseline | Candidate |
|---|---|---|
| Prefill / TTFT, 256 / 512 / 1024 tokens | NOT MEASURED | NOT MEASURED |
| Decode, short / approximately-4K context | NOT MEASURED | NOT MEASURED |
| Peak process RSS / total NPU memory | NOT MEASURED | NOT MEASURED |
| Full 64-layer residuals/logits/state, 16-token greedy continuation | NOT MEASURED | NOT MEASURED |
| 95/96/97, 255/256/257, 511/512/513 model boundaries | NOT MEASURED | NOT MEASURED |
| Production CLI, server repeated-request isolation | NOT MEASURED | NOT MEASURED |

Native Windows validation is **NOT MEASURED**. The recipe remains unchanged and
the PR remains draft until the control initializes and all required model,
performance and integration gates can run. The next action is an administrative
runtime recovery followed by the unchanged-control smoke test, then this same
controlled ablation—not PR #4, a full-ELF rewrite, or promotion of an isolated
speedup.

## Host validation and reproduction

- Full portable suite: **2,731 passed, 172 skipped**; skips are not passes.
- Latest recovery-recheck focused suite: **25 passed**; the unchanged control
  remains blocked, so no new end-to-end result is claimed.
- Fresh standalone CLI and Docker harness builds pass.
- Docker standalone CTest: **5/5 passed**.
- Direct settings scan/regeneration matches the committed registry.
- Settings scratch-merge wrapper: **PASS** on the clean committed branch merged
  with actual upstream `9b1409d3`; the generated registry matches. Because
  `origin` is a fork, a per-command fetch-URL override points the helper at the
  `upstream` URL. Repository configuration and the working tree are unchanged.
- Whitespace/diff checks pass. No new runtime settings or product flags exist.

[Build, fixture, staging and comparison commands](../../../utilities/q4-gemm/README.md)
describe the workflow. The local evidence includes `sets/`, `fixtures/`,
`shapes/`, `tail-*`, `model-256/`, `context-probe-recorded/`, `pytest-final.log`
and `ctest.log`. `build.json` records compiler dependencies and per-core budgets;
the checked-in JSON retains artifact identities and measured summaries.
No precompiled binaries or weights are committed.

`utilities/q4-gemm/verify.py --local-evidence` audits the projection summary
against retained source hashes, compiler dependencies, binaries, per-core
budgets, inputs, output hashes and dispatch logs. It recomputes the paired
medians, ranges, win counts, percentages and effective throughput, and requires
all 32 core images to match between arms. The review audit passes for **1,955
files and 1,100 dispatch records**. Without `--local-evidence`, it checks the
portable JSON artifact only and reports local evidence as **NOT CHECKED**.

## Upstream review checklist

- [x] Opt-in only; recipe defaults stay unchanged and ambient tuning is cleared.
- [x] Behavioural tests reject incomplete evidence, weakened gates, mismatched
  companions, inconsistent arithmetic and unvalidated promotion claims.
- [x] Numerical projection fixtures, resource budgets, exact core/output
  identities, DMA tail ordering and paired measurements are recorded.
- [x] Diagnose the combined-context failure: firmware `NOAVAIL` on the ninth
  identical full-width context, reproduced with the accepted control binary.
- [ ] Recover the runtime with an administrator driver reload/reboot and
  establish a runnable unchanged control without changing precision or context
  architecture.
- [ ] Run independent per-layer residual gates and the full 64-layer logit/state
  and 16-token greedy continuation comparison.
- [ ] Run all nine model-boundary cases and prefill-to-decode state checks.
- [ ] Measure end-to-end prefill/TTFT at 256/512/1024 tokens, short and
  approximately-4K decode, transfer costs and host/NPU memory with repeated
  matched controls; reject a candidate without a credible end-to-end gain.
- [ ] Validate production CLI and repeated server request/reset isolation.
- [ ] Validate/export the target artifacts through the standard pipeline and
  complete platform-specific checks before considering any recipe promotion.

These open gates are acceptance work, not permission to enable the experiment.

## Candidate selection and rejected paths

An initial N=14336/K=5120/T=256 comparison with the same retained input files
tests batch 2 and batch 4 independently against batch 1: five alternating pairs,
one warm-up plus ten measured dispatches per process. Batch 2 changes median
16.968 to 16.572 ms (−2.33%); batch 4 changes 16.971 to 16.377 ms (−3.50%).
Both win all five pairs with exact outputs; these exploratory runs have no
independent reference. Batch 2 is not selected for the larger shape campaign
because batch 4 gives the larger observed reduction. Neither is promoted.

Dequantization-unroll tuning is not implemented or measured: byte-identical
core programs are preferred for this narrower DMA hypothesis. Unbounded DMA
submission is excluded by the documented descriptor limit, not presented as
a new experiment. Context sharing, DeltaNet and host scheduling changes are
outside this draft and are not bundled as workarounds.
