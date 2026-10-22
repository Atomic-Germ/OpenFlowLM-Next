# Private combined prefill CLI

Build three fresh native executables from the current source:

- `paired`: paired-BF16 inputs and residual tracing, the original control.
- `weighted`: the exact validated weighted-FFN wrapper plus residual tracing.
- `combined`: weighted FFN plus paired-weight **linear** QKV/Z and output.

The full-attention output shares N5120/K6144 with linear output. Select the
correction by manifest role and arguments, not by dimensions alone. Full
attention projections and FFN down retain paired-input arithmetic. Reject
stage-major, split projections and unsupported geometry. The native policy
test loads a current 27B manifest and exercises accepted and rejected roles.

The original `gemm_run_once` is preserved verbatim. Reuse the two paired-input
workspaces and the weighted wrapper's host output copy; sum weight halves in
FP32 host memory. Each low-weight kernel has its own XRT context. Production
sources, manifests and kernels are not edited. Existing low kernels must pass
source, generated-source, compiler dependency, resource and binary hash checks,
with exact N/K/T, Q4_1 format and BREUSE=0 before compilation.

```bash
xrt_sdk=/home/prihlop/sources/xdna-driver/xrt/build/Release/opt/xilinx/xrt
for variant in paired weighted combined; do
  OPENBLAS_NUM_THREADS=1 ironvenv/bin/python utilities/prefill-combined-cli/build.py \
    --variant "$variant" \
    --out "Models/qwen38-27b/prefill-combined-cli/v1/$variant" \
    --xrt-include "$xrt_sdk/include" --xrt-lib "$xrt_sdk/lib" || break
done
```

Only standalone native executables are built; there is no model residency or
NPU dispatch. Each fresh output retains commands, source and staged hashes,
all seven translation units' compiler dependencies, CMake cache, compiler
version, CLI identity and CTest result. Existing directories are refused and
failed builds retain incomplete evidence. `complete` means the build completed;
`native_validated` and `model_acceptance` remain false until separate NPU gates.

Prepare the retained independent references after all three builds finish:

```bash
OPENBLAS_NUM_THREADS=1 ironvenv/bin/python utilities/prefill-combined-cli/prepare.py \
  --out Models/qwen38-27b/prefill-combined-cli/prepared-v1 \
  --builds Models/qwen38-27b/prefill-combined-cli/v1 \
  --paired-baseline Models/qwen38-27b/prefill-paired-input/paired/run \
  --weighted-baseline Models/qwen38-27b/prefill-weighted-cli/v2/candidate/run
```

This rechecks native build dependencies, retained kernel geometry and binary
identity, historical control builds/captures and all reference files. It hashes
the model in chunks without loading weights into an engine. Source hashes that
changed after the fixture revision are checked against historical Git blobs and
current HEAD separately. Build artifacts and model weights have no historical
exception. References and their original metadata are copied unchanged;
`preparation.json` retains the separate historical/current evidence. The complete
retained manifest must equal the current recipe derived from the actual model,
including the tokenizer's real vocabulary size. The preparation does not run
the model or claim fresh control identity.

For the later resident step require at least 35 GiB available RAM and exclusive
NPU access. First require fresh paired control versus the retained paired-input
baseline, then fresh weighted control versus the retained weighted-FFN baseline.
Compare sequential logits, scratch and all residuals bytewise. Only then run
the combined candidate with independent 64-layer, 20-token references and one
continuation. Require complete logits and all residual gates, hybrid schedule
identity, repeated requests and reset. The current tracer only accepts one
request: repeated-request/reset instrumentation remains a separate prerequisite.
Keep maxrel <0.005 and original logit gates. Passing builds or isolated replays
do not establish cross-layer or long-prefix model acceptance.

## Preparation result (2026-10-23)

All three v1 builds complete, CTest **4/4 each**, with **589/590/591** compiler
dependencies respectively. CLI SHA256:

- paired: `13883df54f0a22a7b9b6314cc1a9eb3b75443d1ea9c46024fed08e535cc90e56`
- weighted: `fd8d3607d464ec16c8cd479697a1efc4ee55d2d6a80af422565411035c84844f`
- combined: `ce074d89130b3b0f02d919ee23d2119fbb3cc49b153812befbaf699ca7f1d466`

`prepared-v1/preparation.json` verifies 652 input hashes, 192 retained weighted
residual files and the complete current model-derived manifest. It retains 151
byte-identical files (149 reference arrays, IDs and original metadata). The
history graphs preserve old/current identities separately. Six new TDD tests;
full host suite **1672 passed / 47 skipped**. No NPU run or model residency.
