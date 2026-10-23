# Expanded private prefill CLI preparation

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

No engine residency or NPU dispatch in this stage. `native_validated` and
`model_acceptance` remain false. Next require at least 35 GiB available RAM and
exclusive device access, exact fresh control capture identity against retained
combined results, unchanged sequential results, fresh 64-layer 20-token plus
continuation residual/logit gates, hybrid schedule identity and two separate
resident requests after reset. Keep numerical acceptance distinct from capture
validity/repeatability. Full-depth long-prefix and production acceptance remain
open even if the short prompt passes.

The previous `prefill-combined-cli/run.py` expects three older variant names;
do not point it directly at this preparation. The next stage must adapt its
runner/audit and repeated-request checks to `combined` and `expanded`, binding
the repeat executable separately before resident execution.

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
