# Standard paired-prefill export and smoke test

For the standard FP32 DeltaNet post-output export, use the fresh-control
workflow below. The earlier smoke/all-layer commands in this document bind
historical BF16 output hashes and require their recorded sources and export.
Do not relabel those captures for a new arithmetic contract.

For resident reset on the FP32 export, first complete and separately audit the
fresh-control workflow (`post_validate.py`). Build another fresh capture CLI
with `build.py --repeat`, keeping the shared request/reset implementation. Then:

```bash
OPENBLAS_NUM_THREADS=1 ironvenv/bin/python utilities/prefill-production/post_reset.py run \
  --build Models/qwen38-27b/post-prefill-production/repeat-v1 \
  --baseline Models/qwen38-27b/post-prefill-production/short-v1 \
  --out Models/qwen38-27b/post-prefill-production/reset-v1
OPENBLAS_NUM_THREADS=1 ironvenv/bin/python utilities/prefill-production/post_reset.py audit \
  --out Models/qwen38-27b/post-prefill-production/reset-v1
```

The runner binds both baseline report and audit, rechecks current builds and
export inputs, and recomputes the baseline's independent numerical gates.
Each route runs two identical short requests in one resident process. Separate
captures and ordered raw-log markers identify each request. The first must
match all baseline traces; the second must match the first. Both independently
pass every residual and logit gate. A separate audit recomputes all results;
failed audits invalidate previous success. The shared NPU lock, exclusive
device check and 35 GiB memory preflight apply to every route. This validates
ordinary request reset, not different-prompt isolation or full-buffer zeroing.
Block boundaries, long prefixes, serving and performance remain separate gates.

## FP32 block-boundary validation

Reuse the verified plain/trace builds from the passing FP32 short run. Prepare
the retained independent 255/256/257-token references without loading the model:

```bash
OPENBLAS_NUM_THREADS=1 ironvenv/bin/python utilities/prefill-production/post_boundaries.py prepare \
  --baseline Models/qwen38-27b/post-prefill-production/short-v1 \
  --references Models/qwen38-27b/prefill-full/boundaries \
  --out Models/qwen38-27b/post-prefill-production/boundaries-prepared-v1
OPENBLAS_NUM_THREADS=1 ironvenv/bin/python utilities/prefill-production/post_boundaries.py run \
  --prepared Models/qwen38-27b/post-prefill-production/boundaries-prepared-v1 \
  --out Models/qwen38-27b/post-prefill-production/boundaries-native-v1
OPENBLAS_NUM_THREADS=1 ironvenv/bin/python utilities/prefill-production/post_boundaries.py audit \
  --out Models/qwen38-27b/post-prefill-production/boundaries-native-v1
```

Preparation checks all 1155 reference files, CPU-selected continuation tokens,
their metadata binding to the passing reference audit, unchanged model/oracle
hashes, shared prompt prefixes and the current FP32 export/build identities.
Reference metadata remains immutable; runtime commands select the current
bundle. Successful preparation is not numerical acceptance of native execution.

The native gate executes 18 resident processes: three prompt lengths, three
routes, and plain/traced controls. Each requires at least 35 GiB available RAM
and exclusive NPU access under the shared validation lock. Both plain and traced
outputs are independently scored; every trace must reproduce its fresh plain
logits and final scratch. All-layer residual/logit bounds stay unchanged.
Cross-length identity covers prompt logits and every prompt residual row,
excluding continuations selected from different final prompt tokens. Both
hybrid schedules must agree. A completed numerical failure is recorded with
`complete: true, passed: false` and exit status 1; its separate audit must retain
that failure. Incomplete or changed evidence raises an error and invalidates
prior audit success. This does not validate long-prefix reset or performance.

Measured on 2026-10-27 in `boundaries-native-v1`: all 18 processes complete,
and a separate audit confirms the overall numerical FAIL. Both hybrid routes
pass 98,688/98,688 residual rows and 1542/1542 logits in total. Sequential passes
49,058/49,344 residuals and 759/771 logits; worst residual maxrel is 0.0992221863
against 0.005, and minimum logit correlation is 0.9963255072 against 0.9999.
All 2331 ordinary/trace files, 966 hybrid schedule identities and 1917 causal
prefix identities match. Argmax and generated tokens match; every continuation
residual row passes. Do not interpret these identities as numerical acceptance.

The first temporal residual crossing is L54/P125, the shallowest is L41/P239,
and the worst is L58/P205. Use the current captures for causal conditioned replay
before attributing drift to a kernel. The full report and a compact summary are
retained alongside the captures. No native binary or kernel changed in this run.

The ordinary qwen35 recipe exports manifest v4 and 46 kernel sets for the
validated dense Q4 H5120/FF17408 geometry. Five GEMM companions compute the
BF16 residual of dequantized weights; both components read the same packed
Q4NX file. Other sizes and MoE recipes retain their existing exports.

Build with the normal exporter, using a fresh output directory:

```bash
source utilities/env.sh
export OFLM_KEEP_FAILED=1
ironvenv/bin/python open_kernels/export_qwen36_kernels.py \
  --model-dir Models/qwen38-27b/Qwen3.8-27B-NPU2 \
  --build-root Models/qwen38-27b/prefill-production/build-v3 \
  --out Models/qwen38-27b/prefill-production/kernels-model-v1 -j2
```

Use the actual model to derive its spec. The checked-in geometry fixture has
`real_vocab=248320`; the local converted model has `real_vocab=248077` with the
same padded vocabulary and kernels. Do not hand-edit a manifest to bridge that
difference. `--build-root` also isolates source staging and lock files from
Docker-owned shared build directories. Do not run source-generating tests
while an export is compiling.

The unpatched shared CLI must include manifest-v4 runtime support. Run the
bounded integration check against retained audited data:

```bash
OPENBLAS_NUM_THREADS=1 ironvenv/bin/python utilities/prefill-production/smoke.py \
  --cli Models/qwen38-27b/prefill-runtime/cli-v2/open_qwen36_cli \
  --kernels Models/qwen38-27b/prefill-production/kernels-model-v1 \
  --baseline Models/qwen38-27b/prefill-expanded-cli/native-v1 \
  --out Models/qwen38-27b/prefill-production/smoke-v2
```

The runner takes the shared NPU validation lock, checks device availability and
requires 35 GiB available RAM before each resident run. It verifies model,
bundle and reference hashes, then runs a 20-token prompt plus continuation
through 64 layers with layer-major and block-major hybrid scheduling. Each
route must reproduce all 22 retained logit files byte for byte and pass 21
independent CPU logit comparisons (correlation >0.9999, matching argmax).
Missing/extra files, wrong dimensions, NaNs, changed reference metadata and
changed golden hashes fail. Each output directory is new; failures are kept.

This checks integration and short-prompt logits. It does not capture every
intermediate layer, establish long-prefix/reset acceptance, exercise serving,
or demonstrate that hybrid host stages run on the NPU. The full application
uses the same Core, but needs the new kernel bundle to enable paired prefill;
an executable rebuild alone does not upgrade an installed older bundle.

## All-layer and resident reset gate

Build a capture-only copy of the shared runtime with `build.py`, supplying
`--out`, `--xrt-include` and `--xrt-lib`. Build another fresh directory with
`--repeat`. These builds add the existing synchronized residual writer and
separate per-request output names. They keep the current shared projection
implementation and the body of `request()` unchanged. No kernels are rebuilt.
The verifier reconstructs the permitted changes and checks source/compiler,
CMake and executable hashes.

With `trace-v1` and `repeat-v1` built under the production evidence directory:

```bash
OPENBLAS_NUM_THREADS=1 ironvenv/bin/python utilities/prefill-production/validate.py run \
  --build Models/qwen38-27b/prefill-production/trace-v1 \
  --smoke Models/qwen38-27b/prefill-production/smoke-v2 \
  --baseline Models/qwen38-27b/prefill-expanded-cli/native-v1 \
  --kernels Models/qwen38-27b/prefill-production/kernels-model-v1 \
  --out Models/qwen38-27b/prefill-production/all-layer-v1
OPENBLAS_NUM_THREADS=1 ironvenv/bin/python utilities/prefill-production/validate.py audit \
  --out Models/qwen38-27b/prefill-production/all-layer-v1

OPENBLAS_NUM_THREADS=1 ironvenv/bin/python utilities/prefill-production/validate.py run \
  --build Models/qwen38-27b/prefill-production/repeat-v1 \
  --smoke Models/qwen38-27b/prefill-production/smoke-v2 \
  --baseline Models/qwen38-27b/prefill-expanded-cli/native-v1 \
  --kernels Models/qwen38-27b/prefill-production/kernels-model-v1 \
  --repeat-baseline Models/qwen38-27b/prefill-production/all-layer-v1 \
  --out Models/qwen38-27b/prefill-production/reset-v1
OPENBLAS_NUM_THREADS=1 ironvenv/bin/python utilities/prefill-production/validate.py audit \
  --out Models/qwen38-27b/prefill-production/reset-v1
```

Each route (sequential and both hybrid schedules) captures 64 residual arrays,
22 logit files and the final working-buffer slice. The single request must
reproduce all historical captures and satisfy independent residual/logit gates.
Only an independently audited single run can serve as the repeat baseline.
For reset, each process keeps one model resident for two requests; every first
capture must reproduce the single run, and every second capture must reproduce
the first. Ordered raw-log markers bind each request to its separate files.
Auditing recomputes numerical measurements and identities from the files.
A missing route or failed numerical gate cannot be hidden by identical repeats.

This validates ordinary request reset with one short prompt. It does not test
long-prefix reset, full-buffer zeroing, independent checkpoint/restore semantics,
serving request isolation or full model acceptance. Hybrid host stages remain
explicit; the known sequential long-prefix accuracy failure remains open.

## Fresh controls after FP32 post-output integration

`build.py --plain` stages the shared CLI without changing core or CLI source.
The default build adds only synchronized residual tracing. Build both in fresh
directories, then use `post_validate.py run` with `--plain-build`, `--trace-build`,
`--baseline`, `--kernels` and `--out`; follow with `post_validate.py audit --out`.
See the [build/run skill](../../.agents/skills/open-qwen38-post-prefill-validation/SKILL.md)
for full commands and the retained reference location.

This runner checks the current standard export, model and unchanged independent
oracle against pinned historical CPU references. It runs a 20-token prompt plus
continuation through all 64 layers on sequential NPU and both hybrid schedules.
All trace logits and the final scratch dump must match a fresh unpatched CLI
run of the same export byte-for-byte. Every residual row and logit must also
pass the existing independent numerical gates. Both hybrid schedules must match
on all residual/logit files. Missing or extra captures, changed inputs, failed
numerical gates and mismatched tracing controls invalidate the result.

The separate audit recomputes measurements and checks build/source identities.
It does not grant reset, block-boundary, long-prefix, serving or performance
acceptance. No production runtime or NPU kernel arithmetic is modified here.
