---
name: open-norm-scale-carry
description: Build compensated wide RMSNorm scaling, reproduce the real layer5 BF16 boundary, and validate unchanged full-model references and prior norm behavior.
---

# Compensated RMSNorm scaling

Read `specs/open-engine/plans/qwen38-norm-scale-precision.md` for evidence,
resource measurements and remaining failures. This is an opt-in precision
experiment. Keep model fixtures, weights, seeds and acceptance bounds unchanged.

The RED frame is `build_activation_boundary/carry_add/layer5_norm`: one local
BF16 mismatch at3295, zero propagated differences. Source for a fresh replay
is `build_activation_boundary/carry_add/full`, tag `cold-0-layer5`, field `xn`.
The input/weight product loses a residual before multiplication by inverse root.

`LN_SCALE_CARRY=1` requires `LN_NORM_RNE=1`, compensated streamed LN and exact
residual addition. The generic compensation algorithm uses twelve-bit splits
with exact FP32 operations; it is intended for finite normal-range products.
Mean/statistics/inverse root are unchanged. It does not promise globally exact
BF16 norms or overflow/underflow-safe triple products. Keep arithmetic helpers
shared noinline COMDAT functions to avoid duplicated code across workers.

```bash
base=open_kernels/designs/wide_deltanet/build_norm_finish
LN_N=5120 LN_STREAM_COMPENSATED=1 LN_RESIDUAL_RNE=1 LN_NORM_RNE=1 \
  LN_SCALE_CARRY=1 PATH=/opt/xilinx/xrt/bin:$PATH \
  ironvenv/bin/python open_kernels/build_design.py \
  open_kernels/designs/ln/ln.py "$base/carry"
ironvenv/bin/python utilities/test-wide-ln.py prepare --build-dir "$base/carry" --exact-residual
HARNESS_TIMEOUT_MS=30000 LD_LIBRARY_PATH=/opt/xilinx/xrt/lib \
  open_kernels/harness/out/run_kernel "$base/carry/ln.cfg"
ironvenv/bin/python utilities/test-wide-ln.py compare --build-dir "$base/carry"
```

Prepare strict guarded real-frame replays with `utilities/diagnose-wide-ln.py
prepare --source <source full> --kernel <validated LN> --out <fresh>
--tag cold-0-layer5 --field xn --index 3295`. Run `replay.cfg` with the harness,
then `compare --out <fresh>`. All5120 BF16 values, repeat and residual must
match. Also replay `build_attention_boundary/carry/full` layer4/index786.
Rebuild without `LN_SCALE_CARRY` and compare all36 captures for18 primitive
inputs against `build_norm_boundary/rne`; they must be byte-identical.

For statistics add `LN_TRACE_STATS=1 LN_TRACE_INDEX=3295` at build and
`--trace --index 3295` at prepare. Trace replaces xn with FP32 diagnostics;
its compare deliberately exits1 and cannot establish acceptance. Corrected
trace output is0.02532958984375, while sum, mean, inverse root and rounded
weighted input match the baseline trace. Normal/trace text is13056/13392 B,
data plus stack57600 B; DMA instructions are unchanged. Measured dispatch
is slower (about2 ms on this short sample), not an application benchmark.

Run `ironvenv/bin/python -m pytest specs/open-engine/tests -q`.
`test_norm_scale_carry.py` compiles the production compensation template with
scalar IEEE operations and compares200000 bounded random triples and both
signs of the real boundary against long-double products. Separate integer
add/mul tests validate the operations used by the NPU backend.

Full replay uses `utilities/replay-wide-model.py` with:

```bash
wide=open_kernels/designs/wide_deltanet
ironvenv/bin/python utilities/replay-wide-model.py \
  --source "$wide/build_full_model" --out "$wide/build_norm_finish/full" \
  --output-projection "$wide/build_ffn_boundary/carry/projection_k6144" \
  --ffn "$wide/build_activation_boundary/carry_add/ffn" \
  --ln "$wide/build_norm_finish/carry" \
  --attention "$wide/build_model_precision/attention" \
  --attention-projection "$wide/build_attention_boundary/carry/projection_k5120"
```

Use fresh output directories, run `decode.cfg`, then
`test-wide-full-model.py compare --out <full>` and
`diagnose-wide-model-rounding.py --out <full>`, with `OPENBLAS_NUM_THREADS=1`.
Recorded result:3564 calls/tokens/reset pass;18895/18896 slice and4121/4122
decode checks pass. Final residual/norm still fail, though their maxrel falls.
PR4 remains incomplete. The first eighteen norms are exact; layer9 xn has14
propagated differences. Inspect layer8 FFN with the validated
`build_activation_boundary/carry_add/ffn_trace` before changing another norm.
The retained `build_norm_finish/layer8_ffn` trace has one local activation
BF16 error at749, exact up/gate at that channel and no projection-induced BF16
errors. Repeat and recorded h/fo equality pass; `--require-exact-h-bf16`
correctly fails. Reproduce with `diagnose-wide-ffn.py --out <full>
--tag cold-0-layer8 --trace-build <validated ffn_trace> --trace-out <fresh>`,
run `trace.cfg`, then `--out <trace directory> --compare-trace
--require-exact-h-bf16`. Keep this failing fixture for the next stage.
