# Causal candidate DeltaNet / FFN boundaries

Continue Phase 20 after the audited `refined-runtime/native-v1` experiment.
The fixed targets are L50/P205 (first failing depth), L54/P125 (earliest
failing position) and L60/P220 (worst residual and a regression). Do not
promote the refined ax candidate or weaken any numerical threshold.

## Native replay and exact controls

```bash
source utilities/env.sh
export OPENBLAS_NUM_THREADS=1
ironvenv/bin/python utilities/refined-sublayers/replay.py build \
  --out Models/qwen38-27b/refined-sublayers/harness-v1 \
  --xrt-include /home/prihlop/sources/xdna-driver/xrt/build/Release/opt/xilinx/xrt/include \
  --xrt-lib /home/prihlop/sources/xdna-driver/xrt/build/Release/opt/xilinx/xrt/lib
ironvenv/bin/python utilities/refined-sublayers/replay.py prepare \
  --out Models/qwen38-27b/refined-sublayers/native-v2 \
  --baseline Models/qwen38-27b/refined-runtime/native-v1 \
  --harness-build Models/qwen38-27b/refined-sublayers/harness-v1
ironvenv/bin/python utilities/refined-sublayers/replay.py run \
  --out Models/qwen38-27b/refined-sublayers/native-v2
ironvenv/bin/python utilities/refined-sublayers/replay.py audit \
  --out Models/qwen38-27b/refined-sublayers/native-v2
```

Use fresh versioned output directories. The local XRT development headers and
libraries are explicit build arguments; runtime discovery is via `env.sh`.
No service, installed model, registry, standard export or kernel source is
modified. The generic harness is rebuilt, not any xclbin. Compiler dependencies,
source hashes, executable and CMake cache are pinned in its build evidence.

Preparation verifies the full-depth report's bound audit, both private bundles,
CLI builds, all retained control/candidate capture hashes and the original
independent reference. Each target uses the unchanged candidate bundle's lx
binaries and one production-packed layer pool. Each fresh NPU process starts
with zero scratch/state and replays **every** incoming candidate residual from
position zero through its target. No captured state is injected. Outputs,
scratch and complete state are dumped at every position.

The exclusive device lock and device-user check apply. The shared four-layer
preflight reads available RAM (only its 64-layer mode enforces 35 GiB); this
experiment uses one layer, not a full resident model. Logical NPU
buffers total 257,622,016 bytes (about 246 MiB), independent of full-model
residency. All 206 + 126 + 221 outputs must match their corresponding retained
native residuals **bytewise**, including signed zero. A separate audit binds
all 1659 captures. Boundary interpretation is forbidden on any mismatch.

## Independent and conditioned analysis

```bash
OPENBLAS_NUM_THREADS=1 ironvenv/bin/python utilities/refined-sublayers/analyze.py run \
  --capture Models/qwen38-27b/refined-sublayers/native-v2 \
  --out Models/qwen38-27b/refined-sublayers/analysis-v1
OPENBLAS_NUM_THREADS=1 ironvenv/bin/python utilities/refined-sublayers/analyze.py audit \
  --out Models/qwen38-27b/refined-sublayers/analysis-v1
```

The analysis streams one layer's effective weights at a time. It carries
separate histories from zero:

* the original mixed-precision oracle on the original reference inputs,
  required to reconstruct the retained target reference bytewise;
* that oracle on the captured candidate input prefix;
* the independent unrounded FP64 entry/conv/recurrence reference on the
  candidate input prefix;
* the strict recurrence reference conditioned on captured records.

The existing DeltaNet validators check every prefix position, including strict
per-head failures. Target checks retain FP32 Z/post-output/out/residual,
FFN norm/combined activation and both down partials. Raw up/gate projections
are not present in production scratch and are **not** claimed as captured.

Signed vectors telescope through the nonlinear FP64 FFN: original-reference
storage rounding, current input, input history, mixed-oracle/FP64 arithmetic,
recurrence, Z, post-output, output projection, residual addition, FFN norm,
FFN activation, down projection and final addition. All effects use one
stored-reference scale. Their maxima must not be added; effects can cancel,
and this replacement order is not a unique nonlinear attribution. Propagated
drift and the implementation gap are reported separately. The latter includes
accumulated local recurrent error, not just the final position's arithmetic.

Each target retains the node/effect arrays. The audit recomputes independent
histories and every array/metric, rather than trusting a JSON verdict. Exit 1
with `complete: true` means numerical FAIL, not incomplete execution. Keep
`model_acceptance: false` even if all selected local gates pass. This is neither
a full-model correction nor a prefill/performance/serving acceptance run.

## Native checkpoint — 2026-10-29

Baseline commit `dea2a4f`; Python 3.14.4, GCC 15.2.0, CMake 4.2.3, XRT 2.26.0,
amdxdna 2.26.0_20260817, firmware 1.1.2.64. Toolchain discovery selects the
Python 3.14 MLIR-AIE/Peano installation; no AIE compilation was needed.

Fresh harness build and all 553 NPU dispatches pass. Independent audit confirms
553/553 residual identities and 1659 captures. The retained `native-v1` attempt
stopped before dispatch because the shared memory checker rejected a one-layer
argument; corrected preparation and execution are `native-v2`. Never relabel
the failed attempt as a successful run.

Six replay tests were RED before implementation; nine replay/analysis tests
pass. Full host suite: **2073 passed, 47 skipped** (baseline 2064/47).
Retained plain/trace shared CLIs each pass CTest **4/4**. Capture timings are
diagnostic only; TTFT, prefill throughput and performance acceptance were not
measured. L1/shim geometry, kernel binaries and catalogue entries are unchanged.

## Audited numerical checkpoint — 2026-10-29

`analysis-v1` completes and its separate audit reproduces all node/effect arrays
bytewise and all DeltaNet/FFN measurements. Both commands deliberately exit 1:
`complete: true`, `local_passed: true`, `numerical_passed: false`. All three
original reference outputs reconstruct bytewise. Peak materialized effective
weight copies total 4.2814 GiB; no full-model residency is needed.

| Target | Incoming error | Total residual error | Propagated effect | Local implementation gap |
|---|---:|---:|---:|---:|
| L50/P205 | 0.0018135651 | 0.0117765736 | 0.0117388648 | 0.0000377089 |
| L54/P125 | 0.0021624459 | 0.0066132466 | 0.0066087116 | 0.0000045350 |
| L60/P220 | 0.0388441285 | 0.0620054922 | 0.0619942402 | 0.0000112520 |

All numbers are maxrel; incoming error uses the input-reference scale, while
signed output effects use the final stored-reference scale. Do not add maxima.
Current-input contributions are 0.0117169752 / 0.0066125725 / 0.0619976021;
history contributions are only 0.0000294601 / 0.0000044469 / 0.0000170087.
Local implementation gap includes causal local state error and mixed-reference
arithmetic differences; it is not solely one operation's rounding.

All **553/553** entry/conv/composed-state cases and **26,544/26,544** strict
recurrence head cases pass. Selected Z, conditioned post/output, residual add,
FFN front and both down-partial gates all pass. Worst selected front-activation
maxrel is 0.0000325225; worst selected down partial is 0.0000311694. Neither
loosening tolerances nor changing these target kernels is justified by this
evidence. Their local errors are small compared with propagated input drift.

Next trace earlier sources of current-input drift along the candidate causal
prefix, separating new local error from amplification at each step. The retained
P205 trajectory grows from 0.0004254134 at L48 to 0.0018135651 at L49 before
the L50 failure; L31 is another earlier growth point across the retained target
positions. These observations nominate probes, not a proven faulty operation.
Keep all three audited targets as downstream amplification/regression controls.
Full-model numerical acceptance, reset/isolation and performance remain open;
no production arithmetic change or candidate promotion was made.
