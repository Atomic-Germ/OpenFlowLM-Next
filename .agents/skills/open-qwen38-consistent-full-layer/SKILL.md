---
name: open-qwen38-consistent-full-layer
description: Build private full-lx consistent Q4 activation sums with production factored recurrence, then validate causal DeltaNet, Z/out projections, FFN and residuals with streamed weights.
---

# Consistent Q4 sums in the complete linear layer

Use after [isolated QKV validation](../open-qwen38-qkv-precision/SKILL.md).
The experiment changes the direct FP32 activation-table sum from original FP32
values to reconstructed INT16 values, so both Q4 affine terms use the same
activation. It preserves table ABI, quantized values, scales, GEMV epilogue,
weight packing, FIFOs, scheduling and production compensated recurrence.
BF16 preparation is unchanged. Production recipe selection is unchanged.

Unlike the older factored-recurrence experiment, generate entry TUs directly
from `gen_kernels.files(recipe(spec))`: do NOT overlay the legacy `DNX` wrappers.
The current recipe must retain `dnx_factored.h` and `dnx_precise.h`.
IRON copies TUs into `final.prj`; the private generated directory must be FIRST
in the include path. Audit actual `.o.d` paths for the private GEMV/table and
unchanged production recurrence headers. `dense_down_acc` is not compiled by
this split-down recipe; do not require a nonexistent dependency file.

## Build and resources

Native Ryzen AI 9 365 / Strix XDNA2, Python 3.14, mlir-aie 1.4.2,
Peano 21.0.0.2026080301+c9c5ecb7, XRT `/opt/xilinx/xrt`.
Compiler and hardware commands require execution-tool escalation on this host.
Do not run source-generating tests during compilation. Run NPU jobs serially.

```bash
export PATH="$PWD/ironvenv/lib/python3.14/site-packages/llvm-aie/bin:/opt/xilinx/xrt/bin:$PATH"
export XILINX_XRT=/opt/xilinx/xrt
export OFLM_KEEP_FAILED=1
ironvenv/bin/python utilities/qkv-full-layer/consistent_lx.py \
  --kernel-dir Models/qwen38-27b/production-factored-v2/kernels \
  --mode consistent --out Models/qwen38-27b/qkv-full-layer/consistent-build-v2
ironvenv/bin/python utilities/attention-entry-replay/resources.py \
  --build Models/qwen38-27b/qkv-full-layer/consistent-build-v2 \
  --out Models/qwen38-27b/qkv-full-layer/resources.json
ironvenv/bin/python utilities/qkv-full-layer/consistent_lx.py \
  --kernel-dir Models/qwen38-27b/production-factored-v2/kernels \
  --mode control --out Models/qwen38-27b/qkv-full-layer/control-build
```

Measured candidate: **15936 / 16384 bytes** maximum text, **448 bytes** headroom,
**62336** maximum L1 buffer end, all **11 cores** fit with unchanged **6144-byte
stacks**. The earlier `consistent-build/` has no valid build record: compilation
succeeded, but its dependency audit incorrectly required `dense_down_acc.o.d`.
Use `consistent-build-v2/` and retain the rejected attempt/log for provenance.

## Causal replay and independent checks

```bash
set -e
for mode in control consistent; do
  build_dir=Models/qwen38-27b/qkv-full-layer/control-build
  if [ "$mode" = consistent ]; then
    build_dir=Models/qwen38-27b/qkv-full-layer/consistent-build-v2
  fi
  OPENBLAS_NUM_THREADS=1 ironvenv/bin/python -u utilities/qkv-full-layer/consistent_replay.py prepare \
    --model Models/qwen38-27b/Qwen3.8-27B-NPU2 \
    --kernels Models/qwen38-27b/production-factored-v2/kernels \
    --fixture Models/qwen38-27b/production-factored-v2/decode \
    --pools Models/qwen38-27b/production-full/pools --build "$build_dir" \
    --isolated Models/qwen38-27b/qkv-precision-sweep \
    --original Models/qwen38-27b/production-factored-v2/input-sha256.json \
    --out "Models/qwen38-27b/qkv-full-layer/run-$mode"
  HARNESS_TIMEOUT_MS=30000 LD_LIBRARY_PATH=/opt/xilinx/xrt/lib \
    /tmp/qwen38-fp32-harness/run_kernel \
    "Models/qwen38-27b/qkv-full-layer/run-$mode/run.cfg"
  OPENBLAS_NUM_THREADS=1 ironvenv/bin/python -u utilities/qkv-full-layer/consistent_replay.py score \
    --out "Models/qwen38-27b/qkv-full-layer/run-$mode"
done
OPENBLAS_NUM_THREADS=1 ironvenv/bin/python -m pytest -q specs/open-engine/tests
```

The input fixture covers 48 linear layers × three positions. Each layer starts
at zero; later positions carry only that variant's conv/recurrent state. Incoming
residuals are forced from the retained production run. Model and all 48 pool
hashes must match original production provenance. Sources, input buffers,
compiler dependencies, binaries, config and outputs are pinned. Do not edit
replay sources between preparation and scoring. Never overwrite or relabel
production fixtures. Preparation clears old reports and expected dumps.

Control acceptance requires byte-exact complete act/state/residual buffers.
Its numerical failures remain visible separately. Candidate acceptance requires
unchanged entry norm and QKV byte-exact to the validated isolated candidate,
independent FP64 record/conv/composed state and output checks, strict per-head
conditioned recurrence, Z/post/output projection checks, FFN front and both down
partials, plus an independent full-layer residual reference.

The primary record reference carries its own unrounded FP64 history/state from
zero. Z also uses the independent entry norm. Post-recurrence norm/gate and
output projection checks use captured inputs. FFN checks are conditioned on the
captured pre-FFN residual and activation. The whole-layer residual reference
uses the existing `replica_qwen35` mixed FP64/FP32 arithmetic, its own causal
state, and the bundle's effective weight formats. No NPU state seeds an oracle.

Bounds remain maxrel <0.005 for projections/residuals, and per-head recurrence
maxrel <1e-4, cosine >0.9999999. One layer's weights are materialized at a time;
no full-model residency is needed. `progress.json` is not a final result.
Only a complete `score.json` represents a finished run.

Passing isolated full-layer checks does not establish cross-layer propagation.
Production recipe integration, a fresh export/full-depth regression, resident
identity/reset, block-prefill, serving and performance acceptance remain distinct
requirements. Do not reuse old full-depth outputs for changed kernel binaries.

## Completed validation — 2026-10-17

Authoritative evidence: `Models/qwen38-27b/qkv-full-layer/summary.json`,
`resources.json`, and both `run-*/score.json` reports. Both scoring commands
completed successfully, including final input/source hash verification.

- **288 NPU calls**, 48 linear layers × three positions × two variants.
- **432/432 control act/state/residual buffers byte-exact** to production.
  Control numerical input acceptance remains **143/144**, failing L34/T0 K;
  identity success does not suppress that numerical failure.
- Candidate entry norms **144/144 byte-exact** to production and QKV
  **144/144 byte-exact** to the validated isolated consistent-sum probe.
- Candidate input/conv/composed state/output **144/144 PASS**. All **6912**
  strict head state/output cases pass; worst head errors
  **8.68502361e-7 / 2.94514686e-6**. History shifts are exact and padding zero.
- L34/T0 K: **0.0058359392854 → 0.0003889343454** in the complete NPU layer.
  Worst candidate K **0.0004466512456** (L42/T0), QKV **0.0000468438280**.
- Z/post/output, FFN front, both down partials and residual **144/144 PASS**.
  Worst FFN front **0.0002409316**, down partials **0.0000742412 / 0.0000805389**,
  residual **0.0006191985**. Post output remains BF16; its worst error
  **0.0036760336** passes. Not every individual metric improves.
- Model and all 48 pools match original provenance. **864 capture hashes**
  verified again after scoring. All **144 independent residual reference files
  are byte-exact** across control/candidate. CPU oracle sources are unchanged.
- TDD: **13 new tests**, host suite **1330 passed, 47 skipped**.

The production export is still keyed
`f5d922fd91d14b6cfa12f85d5c284a8f39efa56c305d2d572cd6a7009e4586b4`.
Next: select consistent sums only for the validated wide dense Q4 geometry,
build a fresh complete lx/ax export and rerun all full-depth gates. This private
result must not be presented as full-model propagation or resident acceptance.
