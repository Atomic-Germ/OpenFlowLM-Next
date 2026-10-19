# Paired BF16 input experiment for hybrid prefill

This private A/B experiment isolates loss of FP32 activations at the hybrid
GEMM input. It builds two standalone CLIs from copies of `src/open_qwen36`:

- `control` copies the production engine unchanged.
- `paired` calls the same GEMM twice: first with the original input (rounded by
  production `host::tile_x`), then with `x - bf16_rne(x)`. It sums the two
  FP32 outputs in host storage. Q4 weights and NPU kernels remain unchanged.

The original single-dispatch body, including Q8 split-weight folding, is
retained. Each output global has separate workspace. Copy the first output
before dispatching the remainder; never sum into mapped device output, because
`read_back` can flush a dirty host cache line over subsequent device results.
Finite inputs that overflow BF16 are rejected before dispatch.

This is diagnostic code, **not the default production prefill route**. Only
non-staged Qwen3.8 hybrid layer-major and block-major execution is validated.
Stage-major is explicitly rejected by the private candidate because its direct
GEMM path bypasses the experiment. Sequential NPU decode is unchanged. Two
GEMM dispatches per projection increase work; no speed claim follows from a
precision pass.

## Build

Use a fresh directory for each build; the builder refuses to overwrite evidence.
The examples use Linux GCC/OpenMP and the local XRT development headers. Run
from the repository root:

```bash
experiment=Models/qwen38-27b/prefill-paired-input
xrt_sdk=/home/prihlop/sources/xdna-driver/xrt/build/Release/opt/xilinx/xrt
for variant in control paired; do
  ironvenv/bin/python utilities/prefill-paired-input/build.py \
    --variant "$variant" --out "$experiment/$variant" \
    --xrt-include "$xrt_sdk/include" --xrt-lib "$xrt_sdk/lib" || break
done
```

Each build runs CTest and records source hashes, the staged source inventory,
compiler version, commands, CMake cache hash and CLI hash in
`build-evidence.json`. Production source anchors must match exactly; applying
the patch twice is refused. The shared CMake include hierarchy is preserved,
so the candidate's changed `Core` layout is compiled consistently in every TU.
No production source, model packing, manifest or XCLBIN is modified.

## Reuse the independent reference and run

The full-depth reference and previous native control live in
`Models/qwen38-27b/prefill-full/short`. Keep those captures intact. Copy only
reference inputs into each fresh experiment directory:

```bash
ironvenv/bin/python - <<'PY'
import json, shutil
from pathlib import Path
source = Path('Models/qwen38-27b/prefill-full/short')
meta = json.loads((source / 'reference.json').read_text())
for variant in ('control', 'paired'):
    out = Path('Models/qwen38-27b/prefill-paired-input') / variant / 'run'
    out.mkdir()
    for name in ('reference.json', 'ids.txt', *meta['references']):
        shutil.copy2(source / name, out / name)
PY

# Run serially; each native process holds all 64 layers and the LM head.
# At least 35 GiB available RAM is required before every process.
for variant in control paired; do
  OPENBLAS_NUM_THREADS=1 LD_LIBRARY_PATH=/opt/xilinx/xrt/lib \
    ironvenv/bin/python utilities/prefill-slice/check.py run \
    --out "Models/qwen38-27b/prefill-paired-input/$variant/run" \
    --cli "Models/qwen38-27b/prefill-paired-input/$variant/build/open_qwen36_cli"
done

OPENBLAS_NUM_THREADS=1 ironvenv/bin/python utilities/prefill-paired-input/audit.py \
  --out Models/qwen38-27b/prefill-paired-input \
  --baseline Models/qwen38-27b/prefill-full/short
```

The control intentionally returns status **1** on the retained failing fixture.
Inspect `report.json`: an execution/provenance failure is not the expected
numerical failure. Require all 66 control logit files to reproduce the historical
baseline exactly before interpreting the candidate. The auditor recalculates
scores from arrays, checks reference/build/log/output provenance, requires all
22 sequential files to stay exact, and checks both hybrid schedules for identity.
A failed or incomplete audit cannot retain a stale successful report.

Each route covers 20 prompt positions and one independent NPU continuation.
Thresholds remain correlation **>0.9999**, matching real-vocabulary argmax,
continuation residual maxrel **<0.005**, matching generated IDs and exact alias/
hybrid schedule files. Final residual is the sum of L63 residual and both down
partials. Full-depth block-boundary and all-layer prefill residual coverage
remain separate acceptance work.

Regression tests:

```bash
OPENBLAS_NUM_THREADS=1 ironvenv/bin/python -m pytest -q \
  specs/open-engine/tests/test_prefill_paired_input.py
```

## Measured result — 2026-10-19

Full 64-layer, 20-token closed-thinking chat prompt `What is 2 + 2?` plus one
NPU continuation, independent CPU references retained unchanged:

| Route | Control minimum correlation | Paired minimum correlation | Control / paired L63 continuation maxrel |
|---|---:|---:|---:|
| Sequential NPU | 0.9999943778 | 0.9999943778 | 0.0003932455 / 0.0003932455 |
| Hybrid layer-major | 0.9998908204 (FAIL) | 0.9999488108 (PASS) | 0.0018923571 / 0.0011956505 |
| Hybrid block-major | 0.9998908204 (FAIL) | 0.9999488108 (PASS) | 0.0018923571 / 0.0011956505 |

The worst correlation remains prompt position 17. The candidate passes **63/63**
position comparisons with matching real-vocabulary argmax and generated IDs
`[17, 478]`. All **66** control logits reproduce the historical baseline;
all **22** sequential captures are unchanged by the candidate; its two hybrid
schedules agree bytewise in **22/22** files. The control CLI itself is byte-exact
with the baseline. The experiment demonstrates an input-rounding contribution,
not that all remaining GEMM error is explained or that all prompts pass.

GCC **15.2.0**, Release build, native CTest **4/4 per variant**. TDD:
**11 new tests** (six RED→GREEN and five audit regressions), full host suite
**1436 passed, 47 skipped**. CLI SHA256:

- Control: `94a8f27651622ada4ace68f706376b3b8087134a7d45ef06961f29c4f9ac55fc`
- Paired: `f2b8882051782078bc3d5717ada5762c1ea220a79a9bb3b8a09aa2ff917be054`

The same production export is reused: **41 sets / 82 binary hashes**, build key
`ac29fbcd21859648ff9d59f79e77f018b99daec0888f77a693e7ff18b2237032`.
Evidence is under `Models/qwen38-27b/prefill-paired-input/`, including both
build records, all six native runs, 149 shared reference files per variant,
reports, TDD failures and final suite log. `comparison.json` audits the
experiment separately from the intentionally failing control report.

Next validate full-depth 255/256/257-token boundaries and intermediate residuals
before selecting a production precision implementation. The doubled projection
cost and the unmodified staged path must be addressed before any default change.
