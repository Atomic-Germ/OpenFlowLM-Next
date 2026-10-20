# Private per-layer prefill residual capture

This diagnostic builds a separate paired-input native CLI. It records the actual
FP32 residual after each layer at every real prompt position and one NPU
continuation. It does not reconstruct residuals from activation scratch. The
production engine, XCLBINs, weight packing and independent CPU references are
unchanged. The 257-token full-model capture is complete, with **777/777** baseline logit
files byte-exact. Capture validity passes; numerical acceptance still fails.

## Capture contract

- Sequential NPU uses the existing serial dispatch body. Queue-ahead is disabled
  in the private copy so the next layer cannot overwrite global `xres` before
  `read_back`. The mapped buffer is read only, never dirtied by the capture.
- Hybrid layer-major records `N` rows after the layer's FFN; block-major records
  `t_real` rows at the absolute block start. Padding is never written.
- Each `ROUTE_resL.bin` is contiguous little-endian FP32 `[N+1,5120]`, including
  continuation at row `N`. Four-layer slices and all 64 layers are supported.
- The writer rejects non-finite values, gaps, duplicates, width changes,
  existing output files and incomplete layer/position coverage.
- The CLI requires one request, position zero, explicit NPU decode, all prompt
  logits and exactly two generated tokens (one continuation input). Repeated
  requests, host/auto decode, changing decode schedules and benchmarks are
  rejected. Stage-major remains unsupported by the paired-input experiment.

Synchronization and readback change scheduling. **No trace is interpretable
until every prompt/continuation logit and alias matches the retained paired
baseline bytewise.** Timing from this build is not a production benchmark.

## Build (CPU only)

From the repository root, with GCC 15.2.0, CMake and the XRT development SDK:

```bash
ironvenv/bin/python utilities/prefill-residual-trace/build.py \
  --out Models/qwen38-27b/prefill-residual-trace/instrumented \
  --xrt-include /home/prihlop/sources/xdna-driver/xrt/build/Release/opt/xilinx/xrt/include \
  --xrt-lib /home/prihlop/sources/xdna-driver/xrt/build/Release/opt/xilinx/xrt/lib \
  --jobs 4
```

The output directory must not exist. The builder copies the standalone engine,
reuses the paired-input patch unchanged, inserts three capture sites, builds all
native targets and runs CTest. `build-evidence.json` pins original/staged sources,
helper hashes, 585 actual compiler dependencies, commands, compiler, CMake cache
and CLI. A failed build retains `complete:false`. The measured CLI hash is
`b3f0bae0a8838b21dd8b8726c7424de5f6cb08826f252fe105fb7b24f62ac1b6`.
CTest is **4/4**. No NPU process or model residency was needed for preparation.

The unchanged paired baseline is
`Models/qwen38-27b/prefill-paired-input/paired/build/open_qwen36_cli`, SHA256
`f2b8882051782078bc3d5717ada5762c1ea220a79a9bb3b8a09aa2ff917be054`.
Keep its build evidence and the existing boundary captures intact.

## Resident run

Require at least **35 GiB available RAM**, run one resident process at a time,
and use a fresh output directory. The following command loads the model. It was
executed on 2026-10-20 with 76.87 GiB available at preflight; `run-257` now exists
and must be preserved. Choose a different output directory for a repeat:

```bash
LD_LIBRARY_PATH=/opt/xilinx/xrt/lib OPENBLAS_NUM_THREADS=1 \
ironvenv/bin/python utilities/prefill-residual-trace/run.py \
  --build Models/qwen38-27b/prefill-residual-trace/instrumented \
  --baseline-build Models/qwen38-27b/prefill-paired-input/paired \
  --baseline Models/qwen38-27b/prefill-full/boundaries/257 \
  --out Models/qwen38-27b/prefill-residual-trace/run-257
```

The runner verifies both builds and the baseline report, copies the retained
independent references, then delegates to the existing prefill checker. That
checker strips ambient `OFLM_OPEN_*` flags, uses OMP=2/PASSIVE, loads the unchanged
41-set production-consistent export and runs sequential, layer-major and
block-major routes serially. The new residual files add about **0.95 GiB** for
257 prompt tokens plus continuation across all three routes.

Numerical failure is expected from the retained long-prefix fixture. A completed
run with failed numerical gates still reaches the trace auditor; a process or
preflight error does not. Existing directories are refused rather than mixing
new and stale captures. Run lengths 255 and 256 separately when needed; the
256-token hybrid continuation has a known final residual failure.

## Audit without NPU

```bash
OPENBLAS_NUM_THREADS=1 ironvenv/bin/python utilities/prefill-residual-trace/audit.py \
  --build Models/qwen38-27b/prefill-residual-trace/instrumented \
  --baseline-build Models/qwen38-27b/prefill-paired-input/paired \
  --baseline Models/qwen38-27b/prefill-full/boundaries/257 \
  --run Models/qwen38-27b/prefill-residual-trace/run-257 \
  --out Models/qwen38-27b/prefill-residual-trace/run-257/trace-audit.json
```

The audit verifies CLI/source/reference/capture provenance and recomputes the
existing logit gates. It requires exact baseline logit identity before reading
intermediate residuals, then enforces complete trace coverage, shape and finite
values. Every layer/position has its own maxrel gate **<0.005**; zero-reference
rows require exact zero. It records all rows, hashes, hybrid schedule identity
and the first failing position/layer on each route. `localization` separates
chronological onset from shallowest failing depth, and reports counts, first
failures and worst errors for every layer and position. A threshold crossing is a
localization result, not proof of the responsible operation.

`capture_valid:true` means the trace is complete and controlled;
`numerical_passed:true` additionally requires every residual and existing native
acceptance gate. Numeric failure returns 1 with measurements retained. Exceptions
invalidate both flags, including an old success report. Never substitute capture
validity, matching IDs or final-layer accuracy for full numerical acceptance.

## Tests and status

```bash
OPENBLAS_NUM_THREADS=1 ironvenv/bin/python -m pytest -q \
  specs/open-engine/tests/test_prefill_residual_trace.py
```

Tests exercise C++ writer output, poisoned padding at 255/256/257 boundaries,
continuation coverage, invalid captures, per-position errors hidden by global
scaling, control identity, audit failure handling and memory preflight. The first
15 tests failed before implementation, then passed. The expanded suite has 29
cases, including five new RED-to-GREEN localization tests. Full host suite:
**1476 passed, 47 skipped**; CTest rerun **4/4**. No numerical correction or
model acceptance is claimed.


## Full-depth measurements — 2026-10-20

All three routes completed 257 prompt positions and one NPU continuation.
The trace has **192 files**, **49536** layer/position gates, and **46314 PASS**.
All **777/777** prior logit files (including aliases) are byte-exact. The two
hybrid schedules have **64/64 byte-exact residual files** including continuation.
The logit gates reproduce **558/774 PASS**. Exit status 1 denotes measured
numerical failure, not a failed native process. The updated audit recomputed
these results without rerunning the NPU.

All indices below are zero-based; each maxrel uses that position's reference
maximum. Incoming/outgoing errors can have different denominators.

| Route | Residual gates | First temporal failure | Incoming / outgoing maxrel | Worst maxrel |
|---|---:|---|---|---|
| Sequential | 15510/16512 | P38/L60 (linear) | 0.0040749071 / 0.0090391466 | 0.5389499720 at P220/L58 |
| Hybrid layer-major | 15402/16512 | P7/L63 (full attention) | 0.0007421414 / 0.0058400461 | 0.4325418278 at P205/L58 |
| Hybrid block-major | 15402/16512 | P7/L63 (full attention) | 0.0007421414 / 0.0058400461 | 0.4325418278 at P205/L58 |

The shallowest failing layers are different: sequential P241/L33 (0.0050229750),
hybrid P162/L32 (0.0054094424). At the first failing **logit** position,
sequential P80 first crosses the residual gate at L35 (0.0051683727), while
hybrid P82 first crosses at L58 (0.0053988184). Earlier residual failures were
invisible to final-logit/final-continuation-only checks. Both continuation rows
pass all layers in this length-257 run; that does not cover the known hybrid
length-256 continuation failure.

The trace localizes accumulated error to layer outputs, not individual kernels
or arithmetic operations. Next capture sublayer boundaries at hybrid P7/L63,
sequential P38/L60 and sequential P80/L35 with causal history. Separate incoming
error from attention/DeltaNet, output projection, FFN and residual accumulation
before selecting an arithmetic change. No layer is yet proven faulty.

Evidence: `Models/qwen38-27b/prefill-residual-trace/run-257/{report,trace-audit}.json`,
the three route logs, trace files and root `localization-evidence.json`.
