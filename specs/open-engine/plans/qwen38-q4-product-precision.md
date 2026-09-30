# Qwen38 Q4 block product precision

2026-09-28–30. Follow-up to [full-depth precision experiments](qwen38-model-precision.md).
This stage isolates a reproducible Q4 cancellation error. Runtime/catalogue
promotion and performance tuning remain outside this probe.
**Full-model acceptance remains FAIL; PR4 is not complete.**

## Reproduction and change

Before editing the arithmetic, an NPU regression reproduced a zero result
where the packed weights imply a nonzero exact answer. In every K32 block,
set d=1, m=-1 and the first two nibbles to1, giving W=[0,0,-1,...]. Set
x=[1,2^-24,2^-24,0,...]. Each block contributes exactly -2^-24, so K6144
must return -192*2^-24. The previous corrected kernel failed all four nonzero
cases (positive, negative, doubled and repeat); the zero case passed.

The opt-in `--product-correction` mode:

- Multiplies coarse and residual integer dots separately, avoiding their
  premature FP32 sum.
- Uses a compensated reduction for the activation block sum, retaining its
  low component when splitting into three BF16 factors.
- Compensates the nine BF16 products within each block before merging that
  block into the existing persistent Kahan sum.

The first experiment carried an unnormalized two-component sum across all K
blocks. Ordinary random inputs passed, but the repeated cancellation result
retained only one block's contribution. That variant was rejected before a
full-model execution; artifacts remain in `build_product_precision` for diagnosis.
The accepted local-block variant is in `build_product_precision_v2`.

No weight format, model pool, BF16 boundary, independent reference, seed token
or acceptance threshold changes. Default arithmetic is unchanged. The new
flag requires corrected projection or FFN tables and rejects unrelated scopes.

## Primitive checks and resources

Tests first demonstrated the hardware failure and the missing automatic
cancellation-fixture behavior. Product-enabled projection preparation now
adds the five cancellation inputs to the original nine and hashes their
separate weight pool. A passing ordinary fixture therefore covers the
cancellation regression before full-model replay.

- Projection:14/14 pass; all five cancellation outputs are bit-exact.
  Worst ordinary maxrel1.7348139161e-7.
- Full FFN:26/26 checks on the unchanged13 inputs pass. Worst maxrel
  7.1539954653e-6 against1e-4. This is higher than the previous corrected
  FFN's2.8614466311e-6; the new mode is not a universal accuracy improvement.
- Rebuilt old corrected projection with product correction disabled:9/9
  pass and all nine complete captures are byte-identical to the previous build.
- CPU suite:810 passed,47 skipped.

| Probe | Table | Data + reserved stack/core | Text/core |
|---|---:|---:|---:|
| K6144/N5120 projection |27008 B|62976 B|5568 B|
| H5120/FF17408 FFN |22592 B|63552 B|15616 B|

These are actual linked core placements. Scratch sizes, stack reservation and
the FFN4096+4096+4096+4096+1024 segmentation are unchanged.

## Reproduction

Use [.opencode/skill/open-q4-product-precision/SKILL.md](../../../.opencode/skill/open-q4-product-precision/SKILL.md)
for sequential builds and primitive commands. Use the existing immutable
`build_full_model` fixture and `utilities/replay-wide-model.py` for full runs.
The output projection and FFN use the following artifact hashes:

| Artifact | SHA256 |
|---|---|
| projection xclbin | `e7612165e2c16e1d40098fbcb2a1ca165c4dbb9c86e7eb174a524b3425d4bde3` |
| projection instructions | `76a1fa8f739def18653c8a8bd3d6fdfc2286908b7cd2c37d030fa8b583ccf2d9` |
| FFN xclbin | `8e1bd13db6233fa210198bca5a7d86e167ca05f3baaceacaa7caef4b236991c7` |
| FFN instructions | `e8cc08c29bbca5afbdfe59493baf6d4655e6853f0a4c369b25847e91337ce708` |

## Full-model results

The unchanged three-token64-layer replay includes3564 NPU calls and an exact
reset replay in each variant. All three completed product variants still fail
numerical acceptance, while token sequence248045 ->8678 ->198 ->2 matches and
all logits correlations exceed0.9999.

| Replacement | Failed slice/decode checks | Token1 layer63 y maxrel | Token1 final norm maxrel |
|---|---:|---:|---:|
| Product-corrected output only |2 /1|0.03704817062|0.02136752137|
| Product-corrected output + FFN |4 /1|0.02151062805|0.01282051282|
| Product-corrected output + FFN + compensated LN |2 /1|0.02205037713|0.01282051282|

Output-only also fails layer47 head1_og at0.02127659574 (limit0.02).
Output+FFN fails residual y at layers55,59,61,63. The residual limit is0.005;
the final norm limit is0.008. These remain failures, regardless of matching
argmax or primitive passes.
The compensated-LN variant also fails layer47 head1_og: its maxrel is within
0.02, but cosine0.99988917395 is below the required0.9999. The other two
failures are layer63 y and the final norm. All reset captures, state/cache
copies, guards and token feedback checks pass in all three runs.

Against the prior output+FFN variant, the first cold layer's maximum absolute
output error drops from3.814697265625e-6 to1.1920928955078125e-7 (32x).
Its output-projection error drops from1.9073486328125e-6 to5.960464477539063e-8.
This local improvement does not imply improved accuracy at every later layer.

The output+FFN rounding diagnosis counts834828 differing BF16 values over384
norm boundaries, with27 local norm differences; propagated-input counts overlap
and are not additive. At the very first changed boundary (cold0/layer1/xn),
there is one local norm difference and zero propagated-input differences.
This supplies a specific reason to test the already validated compensated LN
with the new product kernels before modifying more Q4 arithmetic.

That third replay removes the local channel1914 error at layer1
(-0.2080078125 becomes the expected -0.208984375). Both entry and post norm
outputs of layers0 and1 are now bit-exact. The first changed boundary is
layer2 entry norm, with19 differences entirely explained by its input;
layer2 post norm has435 propagated differences and no local error.
Across the full run there are845600 changed norm values and30 local errors.

Next investigate the layer1 residual/FFN output feeding layer2, using the
saved `full_ffn_ln` captures and unchanged references. Do not enable all
precision switches by default: the previous output-only variant still has
the smallest measured number of full-model failures (two).
