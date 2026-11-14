---
name: open-qwen38-qkv-precision
description: Build isolated Q4 QKV control and consistent-sum probes, capture activation tables, and validate causal record precision without full-model residency.
---

# Isolated QKV affine-sum precision

Use after the production factored recurrence passes strict head checks but
the full input gate fails at L34/T0 normalized K. The retained export is
`Models/qwen38-27b/production-factored-v2/kernels`, build key
`f5d922fd91d14b6cfa12f85d5c284a8f39efa56c305d2d572cd6a7009e4586b4`.
Its decode fixture supplies incoming residuals and captured FP32 entry norms.

## Arithmetic and scope

The production affine Q4 epilogue uses
`d * sum(code * x_quantized) + m * sum(x_original)`.
Its two terms therefore use different activations. Even a block with zero
effective weights (`d=0.25, code=8, m=-2`) can produce a nonzero result.
The private candidate computes the block sum from the same INT16 values and
power-of-two scale as the nibble product. It preserves the table size, INT16
values, scales, Q4 metadata, epilogue and weight packing.

This changes the FP32 direct table preparation only, in a generated private
header. The production header/recipe/export remain unchanged. BF16 preparation
is untouched. The probe runs the complete 10240×5120 QKV projection on one core,
with the production one-chunk calls and row-band law. It emits both QKV and the
complete 11520-byte table. Controls must reproduce production QKV **bytewise**.

The acceptance oracle is an independent FP64 projection from the incoming
residual and unrounded norm. Each projected variant has its own causal CPU
conv history from zero across all three positions. Primary Q/K/V/decay/beta
record bounds remain **0.005**. The control may pass its identity gate while
its numerical primary gate fails; both statuses are retained explicitly.
The candidate must pass the primary projection/record gates.

These are **isolated projections with CPU record preparation**, not new NPU
full-layer state/output or full-model propagation. Full-layer memory fit,
all affected projection routes, export integration and fresh full-depth tests
remain separate requirements. No throughput claim follows from this probe.

## Build and audit

Native Strix XDNA2; Python 3.14, mlir-aie 1.4.2, Peano
21.0.0.2026080301+c9c5ecb7, XRT `/opt/xilinx/xrt`.
Compiler subprocesses and hardware execution require tool escalation on this
host. Do not modify Python sources while a prepared replay or sweep is running:
their hashes are pinned and checked before scoring.

```bash
export PATH="$PWD/ironvenv/lib/python3.14/site-packages/llvm-aie/bin:/opt/xilinx/xrt/bin:$PATH"
export XILINX_XRT=/opt/xilinx/xrt
export OFLM_KEEP_FAILED=1
for mode in control consistent; do
  OPENBLAS_NUM_THREADS=1 ironvenv/bin/python utilities/qkv-precision/qkv_build.py \
    --mode "$mode" --out "Models/qwen38-27b/qkv-precision/$mode-build"
  ironvenv/bin/python utilities/attention-entry-replay/resources.py \
    --build "Models/qwen38-27b/qkv-precision/$mode-build" \
    --out "Models/qwen38-27b/qkv-precision/$mode-resources.json"
done
```

The builder generates complete private sources, refuses changed source anchors,
checks the actual compiler dependency files and records binary/source hashes.
Keep generated sources and `final.prj`; ignore the compiler's `__pycache__`
directory when checking source coverage. Stacks remain 6144 bytes. Measured
control/candidate code sizes are **3248/3344 bytes**, maximum L1 buffer end
**32512** for both. This does not establish fit in a production lx core.

## Reproduce the causal sweep

```bash
OPENBLAS_NUM_THREADS=1 LD_LIBRARY_PATH=/opt/xilinx/xrt/lib \
  ironvenv/bin/python -u utilities/qkv-precision/sweep.py \
  --model-dir Models/qwen38-27b/Qwen3.8-27B-NPU2 \
  --kernel-dir Models/qwen38-27b/production-factored-v2/kernels \
  --fixture Models/qwen38-27b/production-factored-v2/decode \
  --pools Models/qwen38-27b/production-full/pools \
  --build-root Models/qwen38-27b/qkv-precision \
  --out Models/qwen38-27b/qkv-precision-sweep \
  --harness /tmp/qwen38-fp32-harness/run_kernel
```

This loads one layer's weights at a time. It verifies tensor/pool byte identity,
pins every input, poisons output buffers, clears stale outputs/scores, runs
serially and checks config/build/source/capture provenance. It requires equal
control/candidate inputs and bit-exact INT16/scale table fields. `progress.json`
is progress only; only a complete `results.json` is the final sweep result.

For a single layer, run `qkv_replay.py` with the same model/kernel/fixture/pools,
`--build .../consistent-build --out SEPARATE_DIRECTORY --layer 34`; dispatch
its `run.cfg` with the harness, then repeat preparation arguments plus `--score`.
The control uses `--build .../control-build`. Numerical failures write the
complete JSON and return exit 1. Identity/provenance errors abort the run.

## Diagnostic model caveat

`diagnose.py` accepts model/kernel/fixture paths, `--layer 34` and `--out FILE`.
It compares original and consistent table arithmetic against captured QKV and
unrounded primary records. This CPU arithmetic model is diagnostic only:
AIE FP32 block reductions differ slightly from NumPy FP32 additions. The
original table's INT16/scales are exact, but its sum fields are checked against
FP64 with an FP32 reduction plus BF16-pair rounding bound. Candidate sums come
from quantized integers and must match the CPU table bytewise. The independent
primary record threshold is never changed to accommodate this distinction.

At L34/T0 the CPU model predicts K error 0.00583195 → 0.000387164.
The actual isolated NPU candidate gives **0.0003864200**, versus the exact
production projection fed through the same CPU conv oracle at **0.0058319225**.
The original NPU conv/record capture measures **0.0058359393**. Keep these
three comparisons distinct; the candidate has not yet run NPU conv/recurrence.
L34/T1 and T2 candidate K errors are **0.0001036760 / 0.0000464706**.

TDD covers zero-weight cancellation, packing lane order, integer ties and
saturation, malformed inputs, source drift, control identity versus numerical
acceptance, and table validation. Host suite: **1317 passed, 47 skipped**.

## Completed sweep — 2026-10-17

`Models/qwen38-27b/qkv-precision-sweep/results.json` is authoritative for the
48-layer × 3-position experiment (288 NPU dispatches across both variants).
All **144 control QKV outputs are byte-exact** to the production fixture.
Control primary acceptance remains **143/144**, failing only L34/T0 K.
Candidate primary projection/records pass **144/144**, at unchanged bounds.
All 144 candidate tables match the CPU table bytewise; control INT16/scales
are exact with bounded FP64 sum errors. All 144 control/candidate INT16/scale
table prefixes are byte-identical.

| Candidate metric | Worst normalized error | Layer / position |
|---|---:|---|
| QKV projection | 0.00004684383 | 60 / 2 |
| normalized K | 0.00044778651 | 42 / 0 |
| normalized Q | 0.00031085916 | 41 / 0 |
| V | 0.00025657110 | 34 / 0 |
| decay | 0.00001356882 | 41 / 0 |
| beta | 0.00001348578 | 20 / 1 |

Keep `qkv-precision/` build/resource/diagnostic evidence and the complete sweep
tree together. The earlier single-L34 runs in `qkv-precision/control` and
`consistent` predate the corrected CPU table-validation contract; use the
fresh sweep captures and reports for final acceptance. Independent oracle
modules are pinned in `oracle-provenance.json` and match base commit `5cb550d`.

Next: a private production-layout lx build with this arithmetic, dependency and
resource audits, causal full-layer replay and all affected FFN/projection gates.
Only after that should the validated recipe select it and regenerate full-depth
evidence. Resident lifecycle, prefill, serving and final acceptance remain open.
