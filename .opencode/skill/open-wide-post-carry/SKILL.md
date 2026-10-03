---
name: open-wide-post-carry
description: Trace DeltaNet post RMSNorm/gate rounding, build compensated five-factor products for 32/48 heads, and replay immutable full-model fixtures.
---

# DeltaNet post product carry

Read [the measured report](../../../specs/open-engine/plans/qwen38-post-precision.md).
Baseline is `wide_deltanet/build_ab_precision/full`; the isolated target is
cold-0-layer10/channel604. Earlier trajectories have propagated AB errors at
the same channel, so use the compensated AB/QKV kernels when reproducing it.

Use ironvenv (Python3.14, mlir-aie1.4.2, llvm-aie21) and XRT. This design does
not rewrite layer_x generated sources. Keep accepted binaries immutable and
prepare fresh fixture directories.

```bash
wide=open_kernels/designs/wide_deltanet
base=$wide/build_post_precision
for heads in 48 32; do
  PATH=/opt/xilinx/xrt/bin:$PATH POST_CARRY=1 POST_TRACE=0 DN_POST_HEADS=$heads \
    ironvenv/bin/python open_kernels/build_design.py \
    open_kernels/designs/dn_post/post.py "$base/carry$heads"
  ironvenv/bin/python utilities/test-wide-post.py prepare \
    --heads "$heads" --kernel "$base/carry$heads" --out "$base/carry${heads}_acceptance"
  HARNESS_TIMEOUT_MS=30000 LD_LIBRARY_PATH=/opt/xilinx/xrt/lib \
    open_kernels/harness/out/run_kernel "$base/carry${heads}_acceptance/post.cfg"
  ironvenv/bin/python utilities/test-wide-post.py compare \
    --out "$base/carry${heads}_acceptance" --require-exact
done
```

For real replay add `--source "$wide/build_ab_precision/full"` to prepare;
use a separate output directory. For stage traces build with POST_TRACE=1
and pass `--trace` to prepare. Trace adds a fifth BO: six FP32 planes per
1024-channel group (sum_sq, inv, normalized, weighted, silu, result).
With carry enabled the middle three are rounded diagnostic previews; final
result uses the compensated five-factor product, not those previews.
The comparator checks trace-to-BF16 reconstruction, repeated outputs,
canaries and fixture hashes. Real replay is always diagnostic_only.

POST_CARRY defaults off and keeps the four-BO normal ABI. It retains product
residuals across O * inv * weight * Z * sigmoid(Z), with exact integer-lane
FP32 operations and the existing small-argument sigmoid series. Sum-of-squares
and inverse root are unchanged. This is finite normal-range compensation,
not a general overflow/underflow-safe product or a proof of exact nonlinear math.

Check default rebuild against `build_layer/post`, normal versus traced output,
and the 32-head prefix of the 48-head fixtures. All matched bytewise in this
stage. Synthetic inputs share a fixed 48-head generator before slicing.
Standalone acceptance has three cases: random, zero O and alternating Z=+/-20.

Full replay uses the original references and all preceding precision overrides:

```bash
ironvenv/bin/python utilities/replay-wide-model.py \
  --source "$wide/build_full_model" --out "$base/full" \
  --output-projection "$wide/build_ffn_boundary/carry/projection_k6144" \
  --ffn "$wide/build_main_oct3/ffn" --ln "$wide/build_norm_finish/carry" \
  --attention "$wide/build_model_precision/attention" \
  --attention-projection "$wide/build_attention_boundary/carry/projection_k5120" \
  --deltanet-projection "$wide/build_deltanet_boundary/carry/projection_k5120" \
  --deltanet-ab "$wide/build_ab_precision/carry5120" \
  --deltanet-post "$base/carry48_acceptance"
HARNESS_TIMEOUT_MS=30000 LD_LIBRARY_PATH=/opt/xilinx/xrt/lib \
  open_kernels/harness/out/run_kernel "$base/full/decode.cfg"
OPENBLAS_NUM_THREADS=1 ironvenv/bin/python utilities/test-wide-full-model.py compare --out "$base/full"
OPENBLAS_NUM_THREADS=1 ironvenv/bin/python utilities/diagnose-wide-model-rounding.py --out "$base/full"
OPENBLAS_NUM_THREADS=1 ironvenv/bin/python utilities/diagnose-wide-deltanet-boundary.py --out "$base/full"
```

Replay accepts only a passed, non-traced, non-conditional 48-head primitive
fixture, with hashes checked. Token agreement and the isolated channel fix do
not replace full numerical acceptance. Do not promote this experimental switch
to the runtime/catalogue based on conditional results. Record remaining gates
and the next boundary in the report. Logs are `/tmp/post-precision-*.log`.

Recorded full replay: layer10 gated output and xm are exact; first23 norm
boundaries match. Next is layer11/xm[4872], a propagated residual difference
with exact conditional norm and gated attention. All4122 decode checks pass,
including final_norm, but slice18892/18896 fails four checks. The combined
result stays failed: layer47/head1, layer55/y, layer63/y at cold-1, plus
cold-2/layer39/head4. Investigate earlier residual versus output-projection
counterfactuals before attributing layer11 to local RMSNorm. See the report
for bounds, changed failure set, resource sizes and measured precision cost.
