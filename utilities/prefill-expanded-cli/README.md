# Expanded private prefill CLI validation

Build fresh `combined` control, `expanded` candidate and `expanded-repeat`
executables. The control retains the previous combined patch exactly. The
candidate additionally selects paired-weight full-attention Q/K/V/gate,
full-attention output and dense FFN down. Linear entry/output and FFN up/gate
retain their validated correction. The original single GEMM dispatch and
FP32 high/low summation helpers remain unchanged.

Use exact manifest role, kernel, argument and shape checks; reject split,
non-SiLU, sandwich, different epsilon, grouped norms or incompatible state/KV
layouts. Linear and full-attention outputs share geometry, but their arguments
are distinct. Both may use the same verified low-weight output kernel only
after role validation. Five low-weight contexts are loaded by the candidate.
This remains a private experiment; no production sources, manifests or bundles
are changed. No new XCLBIN is required.

```bash
xrt_sdk=/home/prihlop/sources/xdna-driver/xrt/build/Release/opt/xilinx/xrt
for variant in combined expanded expanded-repeat; do
  OPENBLAS_NUM_THREADS=1 ironvenv/bin/python utilities/prefill-expanded-cli/build.py \
    --variant "$variant" --out "Models/qwen38-27b/prefill-expanded-cli/v1/$variant" \
    --xrt-include "$xrt_sdk/include" --xrt-lib "$xrt_sdk/lib" || break
done
OPENBLAS_NUM_THREADS=1 ironvenv/bin/python utilities/prefill-expanded-cli/prepare.py \
  --out Models/qwen38-27b/prefill-expanded-cli/prepared-v1 \
  --builds Models/qwen38-27b/prefill-expanded-cli/v1 \
  --baseline Models/qwen38-27b/prefill-combined-cli/native-v1 \
  --replay Models/qwen38-27b/prefill-attention-replay/native-v2
```

Build records pin original/staged sources, full compiler dependencies for seven
CLI translation units, low-kernel build records and binaries, CLI/CMake cache
hashes, compiler version and CTest. Kernel reuse verifies generated sources,
compiler inputs and resource evidence as well as N/K/T/Q4_1/BREUSE. Existing
output directories are refused. Failed builds/preparation remain incomplete.

Preparation requires the accepted attention replay and identical CPU audit,
verifies all raw captures and preparation inputs, remeasures retained full-depth
controls/candidate, and checks the complete current model-derived manifest.
Copy 149 independent reference arrays plus IDs/metadata unchanged. Baseline
numerical failure remains recorded; build/preparation never makes it PASS.
The repeat executable uses the existing separate capture plumbing; engine
sources must be identical to the single-request candidate. Its request body,
including reset, remains unchanged. This is not checkpoint-correctness testing.

Preparation performs no engine residency or NPU dispatch. Its `native_validated`
and `model_acceptance` remain false. Resident validation requires at least 35 GiB available RAM and
exclusive device access, exact fresh control capture identity against retained
combined results, unchanged sequential results, fresh 64-layer 20-token plus
continuation residual/logit gates, hybrid schedule identity and two separate
resident requests after reset. Keep numerical acceptance distinct from capture
validity/repeatability. Full-depth long-prefix and production acceptance remain
open even if the short prompt passes.

The previous `prefill-combined-cli/run.py` expects three older variant names.
Use the expanded runner below, which binds `combined`, `expanded` and the
separate repeat executable to this preparation.

## Build checkpoint — 2026-10-23

All three fresh v1 executables pass CTest **4/4 each** with GCC 15.2.0.
Compiler dependency counts are **591 / 592 / 592** for combined, expanded and
expanded-repeat. The fresh combined binary is byte-identical to the retained
combined binary; fresh native capture identity still belongs to the resident
validation stage. Expanded and expanded-repeat bind the same five low kernels.

CLI SHA256:

- combined: `ce074d89130b3b0f02d919ee23d2119fbb3cc49b153812befbaf699ca7f1d466`
- expanded: `20abacdb11714d6c7ac6b634246eeb1672edb6fed4f78c79d1e2b173445fe903`
- expanded-repeat: `1fba18cc1ea8339bd01fc735120bcdee12297a1fab698872fbf36112d2e005b8`

Reused low kernels are the validated FFN up/gate, linear entry/output,
attention entry and FFN down builds. Their build metadata, sources, compiler
inputs, resource evidence and binary hashes are rechecked. Artifact root:
`Models/qwen38-27b/prefill-expanded-cli/`; build and preparation logs are retained.

`prepared-v1/` completes with **2006 input hashes** and **151 unchanged reference
files** (149 arrays, IDs and metadata). Current manifest and all retained
measurements revalidate; the baseline still records **4000/4032 residual FAIL**.
Repeat engine source inventories agree recursively, excluding only the CLI.
Seven RED-to-GREEN tests; full host suite **1730 passed / 47 skipped**.
No native model run or new numerical acceptance is claimed by this preparation.

## Full-depth resident validation

The expanded runner now consumes this preparation directly. It revalidates all
builds and historical/current evidence, requires the exact three-build inventory,
then runs the fresh combined control before expanded. All retained combined
captures must match bytewise before the candidate can start. Candidate
sequential captures must match the fresh control. Each native process checks
>=35 GiB available RAM and exclusive NPU access under the shared validation lock.

```bash
OPENBLAS_NUM_THREADS=1 ironvenv/bin/python utilities/prefill-expanded-cli/run.py preflight \
  --prepared Models/qwen38-27b/prefill-expanded-cli/prepared-v1 \
  --out Models/qwen38-27b/prefill-expanded-cli/preflight-v1
LD_LIBRARY_PATH=/opt/xilinx/xrt/lib OPENBLAS_NUM_THREADS=1 ironvenv/bin/python \
  utilities/prefill-expanded-cli/run.py run \
  --prepared Models/qwen38-27b/prefill-expanded-cli/prepared-v1 \
  --out Models/qwen38-27b/prefill-expanded-cli/native-v1
OPENBLAS_NUM_THREADS=1 ironvenv/bin/python utilities/prefill-expanded-cli/run.py audit \
  --out Models/qwen38-27b/prefill-expanded-cli/native-v1
```

Single-request run/audit return 1 for completed numerical failure. A passing
logit check cannot hide a failing intermediate residual. The independent audit
recomputes every route and control identity, binds exact commands, CLI hashes,
validator sources and preparation, and retains the original acceptance bounds.
Preflight completion only validates evidence: `memory_ready` is separate and
memory/device availability is checked again before every native process.

Repeat validation requires a complete independently audited single-request run,
including when its numerical result is FAIL. The repeat build comes from the
same immutable preparation; its engine source inventory must match the candidate.

```bash
LD_LIBRARY_PATH=/opt/xilinx/xrt/lib OPENBLAS_NUM_THREADS=1 ironvenv/bin/python \
  utilities/prefill-expanded-cli/repeat.py run \
  --baseline Models/qwen38-27b/prefill-expanded-cli/native-v1 \
  --out Models/qwen38-27b/prefill-expanded-cli/repeat-native-v1
OPENBLAS_NUM_THREADS=1 ironvenv/bin/python utilities/prefill-expanded-cli/repeat.py audit \
  --out Models/qwen38-27b/prefill-expanded-cli/repeat-native-v1
```

First-request captures must match the expanded single-request run; second-request
captures must match the first bytewise. Preserve raw marked logs, independently
score both requests and check all three routes. Repeat `passed` means identity;
`numerical_passed` requires both requests to pass numerical gates. Normal request
reset is exercised, not full-buffer clearing or independent checkpoint semantics.
All reports keep `model_acceptance=false`: short-prefix acceptance alone is
insufficient for long-prefix or production promotion. Reuse a fresh output path
if rerunning; failed experiments must remain recorded.

Seven additional RED-to-GREEN tests cover build inventory, failure preservation,
control-first ordering, memory refusal before dispatch, completed numerical
failure and independent repeat/accuracy aggregation. Host suite on 2026-10-24:
**1737 passed / 47 skipped**. `preflight-v1` validates immutable evidence but
records insufficient memory (24.3 GiB); this is not a native run result.

## Resident checkpoint — 2026-10-24

`native-v1` completes with expanded **4032/4032 residual gates and 63/63 logits
PASS** across sequential, hybrid layer-major and hybrid block-major on 64
layers, 20 prompt tokens and one continuation. Fresh combined **261/261 captures**
match retained controls, preserving their 4000/4032 numerical failure. Expanded
sequential **87/87 captures** remain exact; hybrid schedules agree in **86/86
logit/residual files**. Independent CPU audit confirms the measurements.

Expanded hybrid worst residual maxrel is **0.0012128028** (control 0.0121972127),
minimum logit correlation **0.9999979147**. Sequential worst residual remains
**0.0048820188**. Original <0.005 residual and >0.9999 correlation gates are
unchanged; all argmax checks and continuation IDs `[17,478]` agree.

`repeat-native-v1` completes two resident requests per route: **261/261 first
captures** match the single-request baseline, **261/261 second captures** match
the first after reset. Both requests independently pass 4032/4032 residual and
63/63 logit gates. The separate repeat CPU audit confirms both requests.
`passed=true`, `numerical_passed=true`; model_acceptance stays
false. Retain the initial insufficient-memory preflight; native runs started
only after about 61 GiB RAM became available. No new builds were needed.

Next validate 255/256/257-token full-depth propagation with the expanded policy,
retained independent references, fresh controls and causal prefix/schedule
identity. Production integration and serving remain open.

## Long-prefix validation

`boundaries.py` reuses the accepted short run and the retained independent
255/256/257-token references. It verifies the historical reference/source/build
chain without rewriting old hashes, checks current model/export geometry, and
validates every reference's size, finite values and CPU-selected continuation.
All 64 residual prefixes and every prompt logit must match across lengths.

The fresh paired-input control comes from the verified combined preparation.
Run all its lengths/routes before expanded, requiring exact historical logits
and final activation captures. Expanded sequential must then match that fresh
control, including every layer's residual trace. Both variants are independently
scored at all layers/positions; hybrid schedule identity and cross-length causal
logit/residual prefixes are mandatory. Continuations are scored independently,
not compared across prompt lengths.

```bash
OPENBLAS_NUM_THREADS=1 ironvenv/bin/python utilities/prefill-expanded-cli/boundaries.py prepare \
  --baseline Models/qwen38-27b/prefill-expanded-cli/native-v1 \
  --references Models/qwen38-27b/prefill-full/boundaries \
  --out Models/qwen38-27b/prefill-expanded-cli/boundaries-prepared-v1
LD_LIBRARY_PATH=/opt/xilinx/xrt/lib OPENBLAS_NUM_THREADS=1 ironvenv/bin/python \
  utilities/prefill-expanded-cli/boundaries.py run \
  --prepared Models/qwen38-27b/prefill-expanded-cli/boundaries-prepared-v1 \
  --out Models/qwen38-27b/prefill-expanded-cli/boundaries-native-v1
OPENBLAS_NUM_THREADS=1 ironvenv/bin/python utilities/prefill-expanded-cli/boundaries.py audit \
  --out Models/qwen38-27b/prefill-expanded-cli/boundaries-native-v1
```

This runs 18 native processes serially under the shared NPU lock, with >=35 GiB
available RAM and device ownership checked before each. Keep that headroom for
the entire run. A completed numerical failure returns 1 while preserving all
measurements; partial execution remains incomplete. CPU audit recomputes all
metrics, controls and prefix comparisons and binds commands, CLI/source hashes
and preparation. Overall model acceptance stays false even if this experiment
passes. Existing output directories are refused; use new paths for reruns.

### Boundary checkpoint — 2026-10-24

Preparation verifies 1155 reference arrays, 3504 inputs and 172 historical/current
entries. All 18 native processes complete; the independent CPU audit confirms
the measurements, identities, commands and provenance. Run and audit return 1
for the preserved numerical failure, with `complete=true`, `passed=false` and
`model_acceptance=false`.

Expanded hybrid routes pass **98,688/98,688 residuals and 1542/1542 logits**.
Worst hybrid maxrel is **0.0025616438** (control 0.4325418278), minimum correlation
**0.9999961799**, zero argmax mismatches. All 2331 historical control captures,
969 expanded sequential captures, 1917 cross-length prefix checks per variant
and 966 expanded hybrid schedule files are byte-exact.

Sequential remains unchanged: **46,357/49,344 residuals and 550/771 logits**
pass. First residual crossing is L60/P38; shallowest is L33/P241. Logits first
fail at P80, minimum correlation is 0.9351669042, and argmax still agrees. Worst
residual maxrel is 0.5389499720 at L58/P220. Preserve this overall FAIL; the
hybrid result does not establish sequential or production acceptance.

Seven RED-to-GREEN tests; full host suite **1744 passed / 47 skipped**. Artifacts
are `boundaries-prepared-v1/`, `boundaries-native-v1/` and `boundaries-*.log` under
the experiment root. Next diagnose sequential causal state/arithmetic from the
retained traces; long-prefix repeat/reset and production integration remain open.
