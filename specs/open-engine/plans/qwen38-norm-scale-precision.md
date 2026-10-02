# Real RMSNorm scale compensation

Follow-up to [FFN activation rounding](qwen38-activation-precision.md).
The cold0/layer5/channel3295 norm error is closed. The full-model numerical
gate still fails two checks; PR4 remains incomplete.

## Evidence and implementation

The previous guarded layer5 replay has one local BF16 error, zero propagated
norm errors and exact residual/repeated outputs. Its input is
0.028717929497361183, weight1.078125. A trace at index3295 isolates scaling:

| Quantity | Previous device | Independent FP64 |
|---|---:|---:|
|Sum of squares|7649.93701171875|7649.93689067496|
|Mean plus epsilon|1.4941293001174927|1.494129298959953|
|Inverse root|0.8180990815162659|0.8180990888296058|
|Input times weight|0.030961517244577408|0.030961517739342526|
|Pre-BF16 output|0.02532958798110485|0.025329589451337797|

The two individually rounded FP32 products lose the weight-product residual.
Keeping that residual with the unchanged device inverse root yields FP32
0.02532958984375, exactly the BF16 midpoint. Ties-to-even then selects
0.025390625, as does the independent FP64 reference after its required FP32
cast. The previous result selected0.0252685546875.

`LN_SCALE_CARRY=1` requires `LN_NORM_RNE=1` and its existing prerequisites.
The shared `fp32_scale_carry.h` algorithm splits operands into twelve-bit
significand components, recovers multiplication residuals and carries them
through the second multiplication before rounding the compensated result.
NPU operations reuse exact integer-lane FP32 add/sub/mul. This is finite,
normal-range compensated arithmetic, not an overflow/underflow-safe general
triple-product implementation or a guarantee of globally exact BF16 norms.
Statistics, scalar mean, inverse root, buffers and DMA remain unchanged.

The host test first failed for the absent header; recipe cache tests also
failed before their dependencies were updated. The production compensation
algorithm then passed200000 bounded random triples with BF16 weights and
both signs of the real boundary, against a long-double oracle: exact BF16
and at most one FP32 ulp. The strict real-frame RED gate rejects the previous
kernel's single differing BF16 value and passes the new kernel.

## Validation and resources

| Check | Result |
|---|---|
|Strict residual RMSNorm primitive|18/18 pass|
|Real layer5 norm|5120/5120 BF16 exact, repeat/residual exact|
|Earlier layer4/channel786 regression|5120/5120 BF16 exact, repeat/residual exact|
|Rebuilt previous mode|18/18 pass, all36 y/xn captures byte-identical|
|CPU suite|836 passed,47 skipped|
|Normal / trace program text|13056 /13392 B|
|Allocated data plus reserved stack|57600 B|
|DMA instructions|Byte-identical to previous norm mode|

The primitive's inherited tolerances pass despite two BF16 differences
(`random1`, `exponent_gaps0`), versus one for the previous mode. These are
reported separately from strict real-frame acceptance. Median dispatch over
18 calls is2.0605 ms versus0.5935 ms in this short comparison; it is not an
application benchmark. The mode remains opt-in.

| Artifact | SHA256 |
|---|---|
|Normal xclbin|`bfd11fcdb65673f6192254a8bc0d94ed8beae2df831907b4ea3ed92d4a4c00e5`|
|Instructions|`29fe75a6fbfc46032ec6adfef5cbbdd97d4cdd7ae42a3c0c16de019211eef05e`|

## Full-model replay

All3564 NPU calls complete. Three-token feedback, reset, canaries, state/cache
preservation and isolation pass. Both acceptance files retain `passed: false`:
18895/18896 slice and4121/4122 decode checks pass, with the same two failures.

| Failing field | Previous maxrel | New maxrel | Bound |
|---|---:|---:|---:|
|cold-1-layer63 residual y|0.03829945878403673|0.022905298362964267|<0.005|
|cold-1 final_norm|0.021367521367521368|0.01282051282051282|<0.008|

Logit correlations are0.9999978226354423,0.9999864910870133 and
0.9999963803744126; greedy tokens remain248045 ->8678 ->198 ->2.
All384 residual additions remain bit-exact (1966080 values). Local norm
differences fall16 ->11, total device/model norm differences826145 ->810497,
and propagated-input differences826144 ->810495. Local and propagated counts
overlap; changes across different device inputs are not a standalone accuracy
proof.

The first eighteen norm boundaries (layers0 through8, both xn and xm) are
BF16-exact. The first changed boundary is now layer9 xn:14 differences, all
propagated from input. Layer9 xm has46 model differences and2 local ones.
The next investigation starts at layer8 FFN, whose input xm is exact.
Its separate guarded trace reproduces one BF16 activation error at749,
zero projection-induced BF16 errors, exact repeat and bitwise agreement with
the full run's h/fo captures. Up0.18203915655612946 and
gate-0.11944595724344254 equal their independent FP32 references. Device h
is-0.010223388671875 versus reference-0.010223387740552425. The strict
`--require-exact-h-bf16` comparison correctly exits1. This supplies the next
stage's RED case without attributing the propagated norm errors to RMSNorm.
Matching greedy tokens does not close the numerical gate or establish runtime,
prefill, long-context, benchmark or catalogue readiness.

## Reproduction

Use [open-norm-scale-carry](../../../.opencode/skill/open-norm-scale-carry/SKILL.md).
Artifacts are under `open_kernels/designs/wide_deltanet/build_norm_finish`:
`trace`, `layer5_trace`, `carry`, `layer5`, `layer4`, `regression`, `carry_trace`,
`layer5_carry_trace`, `full` and `layer8_ffn`. Trace comparisons intentionally
exit1 with `passed: false`; they are diagnostics, not acceptance kernels.
The corrected trace changes only output at3295 to0.02532958984375; all
statistics and the rounded weighted intermediate match the earlier trace.

The full replay uses unchanged `build_full_model` fixtures, corrected
projection/attention bindings and `build_activation_boundary/carry_add/ffn`.
No model weights, reference tensors, seeds or tolerances were changed.
Logs are `/tmp/norm-finish-*.log`; full hardware output is `full/hardware.log`.
Hardware: NPU Strix. Toolchain: Python3.14.4, NumPy2.5.3, mlir-aie1.4.2,
llvm-aie21; Peano is under `ironvenv/lib/python3.14/site-packages/llvm-aie/bin`.
