# Real DeltaNet projection boundary

2026-10-03. Continue PR4 precision diagnosis after the sigmoid experiment and
main integration. Baseline `build_main_oct3/full` matches all8190 captures of
`build_activation749/full`; six numerical failures remain.

## Conditional diagnosis at cold-0-layer10

The independent input norm xn is exact. A new read-only diagnostic separates
errors introduced at each operator from errors already present in its inputs.
It uses the existing FP64 reference with the same FP32/BF16 boundaries, validates
fixture/kernel hashes and capture guards, and decodes production padded state.
It always reports diagnostic_only=true and passed=false, even for exact data.

| Boundary | Device/reference differences | Local differences | Propagated differences |
|---|---:|---:|---:|
|xn BF16|0|0|0|
|QKV FP32|8027|8027|0|
|Z FP32|4785|4785|0|
|AB FP32|177|177|0|
|conv-state BF16|2|0|2|
|records FP32|15812|12090|14486|
|recurrent output FP32|5751|4357|5743|
|gated output BF16|6|0|6|
|output projection FP32|4947|1395|4948|
|residual FP32|4733|0|4733|
|xm BF16|7|0|7|

Local and propagated sets overlap. Conv-state is shifted raw QKV history, not
the convolution activation. Two mismatched stored inputs arise from projection
rounding. QKV/Z maximum absolute errors are both9.5367431640625e-7. Gated output
matches the conditional post reference exactly: substituting only device Z
into ideal O produces four BF16 differences; only device O produces two.
Using conditional recurrence from device records retains all six. Thus direct
post arithmetic is not the current source of those six rounding differences;
QKV/Z accuracy is the next testable intervention. This does not claim AB,
record construction or recurrence are mathematically exact.

## TDD and candidate

Tests first fail for the missing diagnostic, missing logical-padding argument,
and missing DeltaNet replay option. The implementation then passes tests for
local/propagated overlap, per-head post normalization, corrupt/nonfinite captures,
poison padding, a complete synthetic fixture, and immutable-fixture rejection.
Replay tests cover QKV/Z binding, exact geometry, failed primitive acceptance,
and changed kernel hashes. CPU suite:860 passed,47 skipped.

The candidate uses the existing product-corrected, block-carry Q4 projection
at K5120/N16384 (QKV10240 then Z6144). The full-model harness previously used
activation correction alone for this role. No packing, reference or default
runtime mode changes. All14 primitive inputs pass, including five cancellation
cases. Main-core text is6288 B.

Artifacts: `open_kernels/designs/wide_deltanet/build_deltanet_boundary/carry/projection_k5120`.
- xclbin: `68b8b49c1d66751fb9842724ea99b6a30d7d45b9073850c937ad9e4b32150c01`
- instructions: `3e1f5403b12c50e9c51102458e5ddac4e59203152f2ccdcc0be2e37132c9f886`

Build/replay workflow:
[open-deltanet-projection-carry](../../../.opencode/skill/open-deltanet-projection-carry/SKILL.md).

## Candidate layer10 result and next isolated target

On the candidate's exact layer10 xn, QKV differs from the independent FP32
reference at only index1961 (maxabs1.862645149230957e-9), versus8027 values before.
Z is now completely exact. The two BF16 conv-state differences close. Gated
output differs only at index604 (head4), versus six indices before; xm has
three propagated differences at2315,2508,2814, with zero local norm error.

The post reference on captured O/Z reproduces that single gated-output error.
Replaying conditional recurrence and then conditional glue also retains it.
With reference records and device state/Z it disappears. Substituting device
QKV alone into reference AB likewise removes it; substituting device AB alone
into reference QKV reproduces it. These controlled CPU diagnostics identify
AB as sufficient to reproduce this remaining BF16 boundary on this frame;
they do not establish that all other primitive errors are absent.

For head4, captured versus independent AB values are:

| Field | Device | Reference FP32 |
|---|---:|---:|
|alpha|1.1147561073303223|1.1147568225860596|
|beta logit|1.9046249389648438|1.9046193361282349|
|decay|0.8549836874008179|0.854983389377594|
|beta|0.8704140782356262|0.8704134821891785|

The next bounded step is separating banked AB dot-product rounding from its
nonlinearities, starting with layer10/head4. Diagnostic evidence is retained
in `full/cold-0-layer10-deltanet-boundary.json`; no model references are replaced.

## Full-model result: four numerical failures, no promotion

All3564 dispatches complete. Tokens remain248045 ->8678 ->198 ->2; reset,
state isolation, cache preservation and canaries pass. All384 residual additions
remain exact. Full-logit correlations are0.9999977734076714,
0.999986508594468 and0.9999950526238309.

**18893/18896 slice and4121/4122 decode checks pass**, four numerical failures
versus six for the preceding stage. Three layer47 head failures disappear,
but a new layer61 residual failure appears. This is an opt-in experiment,
not production acceptance; defaults and catalogue remain unchanged.

| Failing capture / field | Normalized max error | Bound |
|---|---:|---:|
|cold-1-layer47/head11_og|0.024875621890547265|<0.02|
|cold-1-layer61/y|0.005366896977317769|<0.005|
|cold-1-layer63/y|0.020431852984951898|<0.005|
|cold-1/final_norm|0.01282051282051282|<0.008|

Final residual maxrel decreases from0.026122822719335435; final norm maxrel
is unchanged. The first21 norm boundaries remain exact. Norm diagnostics count
804385 model differences (previously811003), 10 local (previously9), and804384
propagated (previously811000); these sets overlap. The first local norm error
is now layer19/xn, while the first model divergence remains the three propagated
layer10/xm values described above. Reducing one local error does not ensure
monotonic improvement at every later boundary.

Full results are retained in `build_deltanet_boundary/full`, including
slice-results.json, decode-results.json, rounding-diagnosis.json and the
layer10 diagnostic. Comparator exits1 as required; PR4 remains incomplete.
Logs: `/tmp/dn-boundary-{full-hardware,full-compare,rounding,layer10-candidate}.log`.
