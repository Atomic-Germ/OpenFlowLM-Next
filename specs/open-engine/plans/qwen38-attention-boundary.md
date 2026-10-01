# Real attention projection and gate rounding

2026-10-01. Follow-up to [exact residual addition](qwen38-residual-precision.md).
The first attention frame, cold0/layer3, now has exact new K/V and all6144 gated
attention output values against the independent model reference. This stage
reuses existing opt-in arithmetic modes on a newly built Q/K/V/gate projection;
it does not change production defaults or the model oracle.

## Isolation and TDD

`utilities/diagnose-wide-attention.py` replays a guarded captured frame twice.
It verifies primitive acceptance, artifact hashes, immutable model metadata,
references and input guards. The conditional FP64 attention oracle consumes
the captured projections and cache; differences from this oracle and from the
independent model trajectory are reported separately.

The optional projection path requires an exact model xn boundary. It copies
the original contiguous Q/K/V/gate weight span without repacking, executes the
projection and uses the production DMA slices for attention input. Outputs are
guarded and repeat checked. The optional strict gate requires exact BF16 og
against the conditional oracle; model differences remain separately visible.

Tests first failed for the absent utility and full-replay replacement argument.
They cover conditional versus propagated error, malformed/nonfinite captures,
fixture hashes, repeat corruption and projection geometry. A round-trip test
also exposed NumPy's opaque V2 serialization of custom BF16: reference archives
now store exact FP32 expansions. CPU regression: **828 passed,47 skipped**.

## Real-frame evidence

All variants use the same captured layer3 inputs from
`build_residual_precision/full`, with exact BF16 xn. Counts below are BF16
values; conditional and propagated counts need not add on arbitrary frames.

| Variant | Local og differences | Propagated og differences | og versus model | New K/V versus model |
|---|---:|---:|---:|---:|
|Original projection and attention|21|6|27|2|
|Original projection, precise attention|0|6|6|1|
|Block-carry projection, precise attention|0|0|0|0|

The baseline replay exactly reproduces the source captures. All repeated
captures are bit-exact. The original qg and kvn have8132 and1774 differing FP32
values, with maximum absolute errors9.5367431641e-7 and1.0132789616e-6.
Block carry leaves only two qg differences (maximum1.4901161194e-8), while kvn
is exact. Those two differences do not change BF16 output on this frame.

## Build and primitive acceptance

Build instructions and full replay arguments are in
[open-attention-boundary](../../../.opencode/skill/open-attention-boundary/SKILL.md).
Use the existing ironvenv mlir-aie/Peano toolchain and open XRT harness. The
new K5120/N14336 layer_x projection enables `--projection-correction`,
`--product-correction` and `--block-carry`; no C++ arithmetic was changed.
Build layer_x variants sequentially because their generated headers are shared.

The new projection passes **14/14** primitive inputs, including five exact
cancellation cases. Worst normalized max error is1.0152264681e-7. Actual
per-core text is6288 B; allocated data plus reserved stack is63552 B.
Precise attention is the previously validated `ATTN_PROBE_PRECISE=1` artifact
from `build_model_precision/attention`, reused without recompilation.

| Artifact | SHA256 |
|---|---|
|Q/K/V/gate xclbin|`e7434816c0f5acbf0d514b6ff5113fd737080141f407054f77da885f45c40f3a`|
|Q/K/V/gate instructions|`1f2c808c29f714e0b2c497848dc543289b8f1f9fe0ba6bea1c512cc62a6d6029`|
|Reused precise attention xclbin|`c6241216940ed662add9f23485439a02c7822111e71a7489f4030946554fd8e9`|
|Reused precise attention instructions|`9902ffdb0158c29d7ad37af5af3a11c1c147947e6e90d48866545d354863dc7c`|

Artifacts are under `open_kernels/designs/wide_deltanet/build_attention_boundary`:
`baseline_v2`, `precise_v2`, `carry/projection_k5120`, `carry/layer3` and
`carry/full`. The original `baseline` and `precise` directories contain the
superseded V2 reference serialization and must not be used for comparisons.

## Full-model replay and next boundary

The full replay changes Q/K/V/gate projection and attention relative to
`build_residual_precision/full`. It retains block-carry output projection,
segment-carry FFN and exact residual LN. All3564 NPU calls complete against
the unchanged original full-model fixtures, seeds and thresholds.

The first eight norm boundaries, through layer3 post norm, are now exact.
The next difference is cold0/layer4 input norm at channel786: device
0.059814453125 versus conditional and model reference0.0595703125, for input
0.06631811708211899 and weight1.0703125. It is one local norm difference with
no propagated differences at this boundary. The next step is to isolate
RMSNorm statistics/scaling at this frame; this observation alone does not
identify the faulty arithmetic operation.

All384 residual additions remain exact (0/1966080 differing values). Across
384 norm boundaries,827992 values differ from the model, versus831413 before;
26 are local norm differences. These diagnostic counts overlap with propagated
errors and must not be added or substituted for numerical acceptance.

Full numerical acceptance is **FAIL**:18891/18896 slice checks and4121/4122
decode checks pass. Six checks fail versus11 in the preceding variant, but
the final residual and normalized-vector errors increase. This is not an
accepted default. All tokens still match248045 ->8678 ->198 ->2; reset replay,
state isolation, cache preservation and canaries pass. Minimum full-logit
correlation is0.999986361351, above the inherited0.9999 bound.

| Frame | Field | maxrel | Required maxrel | Cosine |
|---|---|---:|---:|---:|
|cold-0-layer63|head0 og|0.022857142857|<0.02|0.999951749568|
|cold-1-layer47|head1 og|0.021276595745|<0.02|0.999805431920|
|cold-1-layer47|head10 og|0.013636363636|<0.02|0.999866171764|
|cold-1-layer47|head11 og|0.029850746269|<0.02|0.999893937483|
|cold-1-layer63|y|0.028256613553|<0.005|0.999987338656|
|cold-1|final norm|0.017094017094|<0.008|0.999986887322|

Attention-head checks also require cosine>0.9999: head10 fails this condition
despite passing maxrel. Both result files retain `passed: false`; the full
compare exits1. PR4 still needs numerical closure, runtime integration and
benchmarking. Matching these three tokens is not a prompt-quality evaluation.

Build/validation logs: `/tmp/attention-qkvg-{build,prepare,compare}.log`,
`/tmp/attention-unit-final.log` and `/tmp/attention-full-{prepare,compare,rounding}.log`.
The hardware log and machine-readable checks are retained in `carry/full`.
