---
name: open-qwen35-fp32-down-replay
description: Reproduce the Qwen3.8 layer-60 FFN down-partial error with one resident layer and build/test direct FP32 activation-table preparation.
---

# Qwen3.5 dense down FP32 precision replay

The strict partial checker found layer 60 / position 2 / K[8192:17408] maxrel
0.0073258458 from captured FP32 `h`. The original kernel rounds `h` to BF16
before INT16 table preparation. Compare against FP32 input; BF16 diagnostics
must not turn a failed primary gate into PASS.

Keep the verified production export intact. Experiments below use separate
build directories under ignored `Models/qwen38-27b/ffn-replay/`. Read the
production-preflight skill for the model, export and capture provenance.
No new model packing or closed library is needed.

## Toolchain and generated files

On this host: Python 3.14.4, mlir-aie 1.4.2, Peano under
`ironvenv/lib/python3.14/site-packages/llvm-aie/bin`, XRT under `/opt/xilinx/xrt`.
Native builds and device calls require sandbox escalation. Run NPU jobs serially.

The standalone `build_design.py` does NOT regenerate layer kernel translation
units. Always run `gen_kernels.py` for the same model-derived spec first.
A stale `gemv_q4_gy.cc` with PER_CALL=2 instead of 1 produced non-finite values
before the FFN despite a successful build. Do not use such a build as evidence
for or against a numerical correction. Do not run source-generating tests or
another family export while a layer build is in progress.

```bash
OPENBLAS_NUM_THREADS=1 ironvenv/bin/python - <<'PY'
import sys
from pathlib import Path
sys.path.insert(0, 'open_kernels')
from recipes.load import spec_from_model_dir
out = Path('Models/qwen38-27b/ffn-replay')
out.mkdir(parents=True, exist_ok=True)
(out / 'spec.json').write_text(spec_from_model_dir(
    Path('Models/qwen38-27b/Qwen3.8-27B-NPU2')).to_json())
PY
export OPEN_KERNELS_SPEC="$PWD/Models/qwen38-27b/ffn-replay/spec.json"
export PATH="$PWD/ironvenv/lib/python3.14/site-packages/llvm-aie/bin:/opt/xilinx/xrt/bin:$PATH"
export XILINX_XRT=/opt/xilinx/xrt
export OFLM_KEEP_FAILED=1
ironvenv/bin/python open_kernels/designs/layer_x/gen_kernels.py
ironvenv/bin/python open_kernels/build_design.py open_kernels/designs/layer_x/lx.py \
  Models/qwen38-27b/ffn-replay/lx-direct
ironvenv/bin/python open_kernels/build_design.py open_kernels/designs/layer_x/ax.py \
  Models/qwen38-27b/ffn-replay/ax-direct
ironvenv/bin/python utilities/core_sizes.py Models/qwen38-27b/ffn-replay/lx-direct
ironvenv/bin/python utilities/core_sizes.py Models/qwen38-27b/ffn-replay/ax-direct
```

Check every core's .text < 16384. The direct path retains the 2.25*K table ABI
and existing buffer sizes. Only split dense FP32 preparation uses it; single-piece dense models,
BF16 inputs and other legacy FP32 preparation entries retain their routes.

## Isolated replay

```bash
OPENBLAS_NUM_THREADS=1 ironvenv/bin/python utilities/qwen35_layer_replay.py \
  --model-dir Models/qwen38-27b/Qwen3.8-27B-NPU2 \
  --kernel-dir Models/qwen38-27b/production-kernels \
  --fixture Models/qwen38-27b/production-multitoken \
  --pools Models/qwen38-27b/production-full/pools \
  --out Models/qwen38-27b/ffn-replay/direct \
  --candidate Models/qwen38-27b/ffn-replay/lx-direct
HARNESS_TIMEOUT_MS=30000 LD_LIBRARY_PATH=/opt/xilinx/xrt/lib \
  open_kernels/harness/out/run_kernel Models/qwen38-27b/ffn-replay/direct/run.cfg
```

Run the same preparation command with `--score` afterwards. For baseline,
omit `--candidate` and use a separate `--out .../baseline` directory. Baseline
scoring must return nonzero for the reproduced precision failure. Candidate
scoring additionally requires unchanged captured `h` and pre-FFN residual.

The default linear replay uses one layer's pool (~233 MB), constants (~21 MB),
recurrent state (~3.3 MB) and the previous layer's captured residual at each
position. For full-attention layers, select `--layer` and `--candidate .../ax-direct`:
the replay allocates KV state, loads the position table and patches `attnpos`.
State starts at zero and persists through all three tokens. No full-model
residency is required. It does not prove full-model/state/head accuracy.

## Table boundary probe

```bash
ironvenv/bin/python open_kernels/build_design.py utilities/gemv-fp32-prep/probe.py \
  Models/qwen38-27b/ffn-replay/prep-probe
ironvenv/bin/python utilities/gemv-fp32-prep/check.py \
  --out Models/qwen38-27b/ffn-replay/prep-fixture \
  --build Models/qwen38-27b/ffn-replay/prep-probe
HARNESS_TIMEOUT_MS=30000 LD_LIBRARY_PATH=/opt/xilinx/xrt/lib \
  open_kernels/harness/out/run_kernel Models/qwen38-27b/ffn-replay/prep-fixture/run.cfg
ironvenv/bin/python utilities/gemv-fp32-prep/check.py \
  --out Models/qwen38-27b/ffn-replay/prep-fixture \
  --build Models/qwen38-27b/ffn-replay/prep-probe --score
```

The 32 blocks test zero, both signs, power-of-two endpoints, cancellation and
scales from 2^-100 to 2^100. Shifts and INT16 rounding/saturation must be exact.
Block sums use the existing BF16 hi/lo format; AIE conversion flushes subnormal
low parts, so the sum bound includes the minimum normal FP32/BF16 value as an
absolute floor. This representation check is separate from the unchanged
0.005 FFN acceptance gate.

## All-layer isolated sweep

After both candidate builds complete, run all 64 layers serially:

```bash
LD_LIBRARY_PATH=/opt/xilinx/xrt/lib OPENBLAS_NUM_THREADS=1 \
  ironvenv/bin/python utilities/gemv-fp32-prep/sweep.py \
  --model-dir Models/qwen38-27b/Qwen3.8-27B-NPU2 \
  --kernel-dir Models/qwen38-27b/production-kernels \
  --fixture Models/qwen38-27b/production-multitoken \
  --pools Models/qwen38-27b/production-full/pools \
  --candidate-root Models/qwen38-27b/ffn-replay \
  --out Models/qwen38-27b/ffn-replay/sweep
```

Each layer gets separate preparation, hardware and score logs. `replay.json`
pins the kernel, pool, constants, state/position-table and input hashes; scoring
refuses changed inputs/configuration. `score.json` records output hashes and
requires exact FP32 `h` and pre-FFN residual identity, including signed zeros.
The sweep summary is invalidated on restart and only published after all layers.
This does not promote the candidates to `production-kernels/` or establish
full corrected-model logits, state/head correctness, or prefill/serving parity.

## Verified results — 2026-10-11

On Ryzen AI 9 365 / Strix XDNA2, all 64 isolated layers × 3 positions passed
(192 dispatches); every FFN input/pre-FFN residual matched the historical
capture bytewise. Worst maxrel: partial0 0.0003117401 (L54/T1), partial1
0.0001995445 (L48/T1), sum 0.0003320237 (L54/T1), closing residual
0.0000744078 (L62/T1). The original failing L60/T2 partial changed from
0.0073258458 to 0.0000988872. Bounds and FP32 reference are unchanged.

Accepted core .text maxima: lx 15584 B, ax 14400 B (<16384). The lx main
GEMV cores use 15072 B; a helper determines the lx maximum. The 32-block
hardware table probe passes. Open-engine tests: 914 passed, 47 skipped.
Artifacts: `ffn-replay/{lx-direct,ax-direct,prep-probe}`, build logs,
`core-sizes-{lx,ax}.txt`, and `sweep/results.json` with per-layer capture audits.
Only source comments and generator selection for unsplit models were edited
after the accepted builds; the generated 27B compute code is unchanged.

The isolated correction is complete. Keep the new kernels experimental until
a verified complete export and full-model propagation regression pass. Roughly
35–40 GiB available RAM is needed for that resident run; this sweep instead
ran with about 8–13 GiB available. No controlled timing/throughput claim is made.
