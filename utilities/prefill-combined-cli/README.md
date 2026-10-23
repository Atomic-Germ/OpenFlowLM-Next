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
identity, repeated requests and reset. The v1 tracer only accepts one request;
use the separate repeat build below to capture two requests without overwrites.
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

## Full-depth native validation

The runner verifies the immutable preparation, historical/current identities,
current native dependencies and exported kernels. It serializes device use,
checks free memory before every process, and runs two controls before the
candidate. Paired control must reproduce retained logits/scratch; weighted
control must also reproduce every retained residual. Sequential captures must
match across fresh variants, and hybrid schedules must agree. Independent
all-layer gates remain mandatory even when logits pass. Reports distinguish
completed numerical failure from execution/preflight failure.

```bash
LD_LIBRARY_PATH=/opt/xilinx/xrt/lib OPENBLAS_NUM_THREADS=1 ironvenv/bin/python \
  utilities/prefill-combined-cli/run.py run \
  --prepared Models/qwen38-27b/prefill-combined-cli/prepared-v1 \
  --out Models/qwen38-27b/prefill-combined-cli/native-v1
OPENBLAS_NUM_THREADS=1 ironvenv/bin/python utilities/prefill-combined-cli/run.py audit \
  --out Models/qwen38-27b/prefill-combined-cli/native-v1
```

Both commands return 1 for a completed numerical failure. Inspect `complete`,
`passed` and per-route gates; never treat failed accuracy as failed capture.
No historical reference metadata is rewritten to bypass source checks.

## Two requests on one resident engine

`repeat.py build` copies the verified combined stage and changes only CLI capture
plumbing. It requires `--twice`, separates request prefixes/writers and flushes
stdout so two marked request logs can be extracted from the preserved raw log.
The existing `request()` body, including `Core::reset()`, stays unchanged.
The snapshot/restore calls before request 2 are retained, but the reset that
follows means this is not an independent checkpoint-correctness test.

```bash
xrt_sdk=/home/prihlop/sources/xdna-driver/xrt/build/Release/opt/xilinx/xrt
OPENBLAS_NUM_THREADS=1 ironvenv/bin/python utilities/prefill-combined-cli/repeat.py build \
  --parent Models/qwen38-27b/prefill-combined-cli/v1/combined \
  --out Models/qwen38-27b/prefill-combined-cli/repeat-build-v1 \
  --xrt-include "$xrt_sdk/include" --xrt-lib "$xrt_sdk/lib"
LD_LIBRARY_PATH=/opt/xilinx/xrt/lib OPENBLAS_NUM_THREADS=1 ironvenv/bin/python \
  utilities/prefill-combined-cli/repeat.py run \
  --build Models/qwen38-27b/prefill-combined-cli/repeat-build-v1 \
  --baseline Models/qwen38-27b/prefill-combined-cli/native-v1 \
  --out Models/qwen38-27b/prefill-combined-cli/repeat-native-v1
OPENBLAS_NUM_THREADS=1 ironvenv/bin/python utilities/prefill-combined-cli/repeat.py audit \
  --out Models/qwen38-27b/prefill-combined-cli/repeat-native-v1
```

The first request must reproduce all single-request captures. The second must
match all first-request logits, residuals and scratch bytewise. Independently
recompute both requests' numerical gates and verify the two extracted logs
against the raw log. Repeat `passed` denotes exact repeatability;
`numerical_passed` separately reports accuracy. Neither implies long-prefix
acceptance. This exercises normal request reset, not zeroing every scratch or
KV buffer. No production kernel or model artifact is modified.

## Native results (2026-10-23)

`native-v1/report.json` and its independent `audit.json` record nine completed
resident runs (three variants, three routes, 64 layers, 20 prompt tokens plus
one continuation). Fresh paired and weighted controls reproduce **69/69** and
**261/261** retained captures respectively. Sequential captures are byte-exact
across variants; hybrid schedules agree in all logits and residuals.

The combined candidate passes **63/63 logit gates** and **4000/4032 residual
gates**. Each hybrid route passes **1328/1344**, versus **1315/1344** for weighted
FFN; sequential passes **1344/1344**. Worst hybrid residual error is
**0.0121972127**, above the unchanged **0.005** bound. Sixteen formerly failing
rows improve, but three new failures appear at L53/P18, L56/P17 and L63/P1.
L63/P1 grows from **0.0049026924 to 0.0096485275**. Both the run and independent
audit correctly record numerical FAIL; this is not full-model acceptance.

`repeat-native-v1/` completes three two-request resident processes. All
**261/261** first-request captures match the single-request candidate and all
**261/261** second-request captures match after normal request reset. The
independent repeat audit passes, with `numerical_passed=false` and
`model_acceptance=false`. The repeat CLI SHA256 is
`f2f9ca67238e44e0f73c7bcb6030087a71785a87fb3a17228fbf78002490186e`;
its build pins **591** compiler dependencies and passes **CTest 4/4**.
Eight new RED-to-GREEN tests cover capture rejection, residual acceptance,
control identity, preflight evidence and separate complete repeated requests.
Full host suite: **1680 passed / 47 skipped**.
