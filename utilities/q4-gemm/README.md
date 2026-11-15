# Q4 prefill GEMM DMA batching

This experiment changes only the device instruction sequence: a bounded group
queues two or four consecutive output visits before reclaiming its DMA tasks.
The default is one visit. Host dispatch order, physical block size, FIFO depths,
weight layout, dequantization, FP32 accumulation and paired-product reduction
order do not change. Both high and low weight companions use the same sequence.

`dma_schedule.py` preserves row-group-major / token-tile-minor traversal and
never pads a partial group. All drains remain awaited before descriptors are
freed. The compiler's actual shim descriptor allocation, not the planner's
batch bound alone, decides whether a shape builds. Never collect all visits in
one group: each shim tile has only 16 descriptors.

The build-time `GQP_DMA_BATCH` option accepts 1, 2 and 4. It does not add a runtime
setting or change the kernel ABI. Ordinary recipe exports clear ambient values;
only an explicit recipe selection can promote a new default. A private build
does not install itself or establish model acceptance.

## Isolated build and comparison

Use a fresh output directory for every invocation. The builder snapshots the
design, records compiler dependencies and hashes, and checks all 32 compute
cores against program/L1 budgets. Source headers are unchanged between arms.

```bash
source utilities/env.sh
ROOT=Models/qwen38-27b/pr3-gemm
for BATCH in 1 4; do
  ironvenv/bin/python utilities/q4-gemm/build.py \
    --n 14336 --k 5120 --batch "$BATCH" --out "$ROOT/example-$BATCH"
done
# Repeat with --low; never omit the low-weight acceptance arm.

OPENBLAS_NUM_THREADS=1 ironvenv/bin/python utilities/q4-gemm/fixtures.py \
  --n 14336 --k 5120 --out "$ROOT/example-fixture"
OPENBLAS_NUM_THREADS=1 ironvenv/bin/python utilities/q4-gemm/compare.py \
  --before "$ROOT/example-1" --after "$ROOT/example-4" \
  --weights "$ROOT/example-fixture/weights.bin" \
  --activations "$ROOT/example-fixture/activations.bin" \
  --reference "$ROOT/example-fixture/reference-high.bin" \
  --n 14336 --k 5120 --tokens 256 \
  --harness "$ROOT/harness/run_kernel" --out "$ROOT/example-comparison"
```

Build the harness with its existing CMake workflow. Comparison serializes on
the same NPU lock as the model oracles. It requires at least five alternating
pairs; each process excludes one warm-up dispatch and measures ten dispatches
by default. It checks every reported completion and the final output of every
process. Outputs must be finite, complete and byte-identical across all arms
and repetitions. An optional reference uses the existing `rel_fro <= 5e-3`
kernel gate. Omitting that reference establishes equivalence only, not an
independent numerical pass. Timings include host submit-to-completion, not BO
synchronization, context creation or model loading. They are not device-only
execution times or end-to-end throughput.

## Full-model comparison

Use private full kernel directories that differ only in the ten GEMM streams.
Retain the base manifest byte-for-byte, including its DeltaNet head mapping;
do not regenerate unrelated layer kernels for a GEMM-only experiment.

The existing benchmark/state driver accepts `--before-kernels` and
`--after-kernels`. Supplying both holds the OpenMP startup policy enabled on
both arms; its original policy comparison remains unchanged without them.

After building all five projection shapes and both weight components into
`$ROOT/sets/batch1` and `$ROOT/sets/batch4`, stage each complete private arm:

```bash
for BATCH in 1 4; do
  ironvenv/bin/python utilities/q4-gemm/stage.py \
    --base Models/qwen38-27b/production-v2/kernels-model \
    --variants "$ROOT/sets/batch$BATCH" --out "$ROOT/model-batch$BATCH"
done
```

```bash
utilities/prefill-block/build-cli.sh "$ROOT/cli"
CLI="$ROOT/cli/open_qwen36_cli"
OPENBLAS_NUM_THREADS=1 ironvenv/bin/python utilities/prefill-block/omp_policy.py \
  --before "$CLI" --after "$CLI" \
  --before-commit 836072db --after-commit 836072db \
  --before-kernels "$ROOT/model-batch1" --after-kernels "$ROOT/model-batch4" \
  --model oflm-data/models/Qwen3.8-27B-NPU2 --tokens 256 \
  --out "$ROOT/model-256"
# Repeat at 512 and 1024 tokens. build.json records the measured source snapshot,
# not a fabricated measurement commit SHA.

OPENBLAS_NUM_THREADS=1 ironvenv/bin/python utilities/prefill-block/omp_policy.py \
  --check boundaries --before "$CLI" --after "$CLI" \
  --before-kernels "$ROOT/model-batch1" --after-kernels "$ROOT/model-batch4" \
  --model oflm-data/models/Qwen3.8-27B-NPU2 --layers 64 --generate 16 \
  --lengths 256 --out "$ROOT/full-state"
```

The full-model driver retains the existing logit/state comparison thresholds and
requires exact logit and state bytes. Four-layer boundary sweeps do not replace
the 64-layer gate. A numerical or isolated-kernel pass alone must not promote
the recipe default; end-to-end performance and integration gates also apply.

For a context-creation failure, isolate the unchanged export before changing
the model, manifest or driver:

```bash
ironvenv/bin/python utilities/q4-gemm/context_probe.py \
  --kernels Models/qwen38-27b/production-v2/kernels-model \
  --harness "$ROOT/harness/run_kernel" --out "$ROOT/context-probe"
```

This probes each context alone and all of them together, with no inference.
Its result is diagnostic evidence, not a numerical pass. Do not bypass paired
precision or silently coalesce contexts to make a performance gate run.

## Evidence and review checks

```bash
# Portable checks of the committed summary: paired coverage, numerical gates,
# variant/resource limits, timing ranges, percentages and effective throughput.
ironvenv/bin/python utilities/q4-gemm/verify.py

# Also require the original local files: hashes, raw logs, outputs, compiler
# inputs and identical compute-core images. Missing evidence is an error here.
OPENBLAS_NUM_THREADS=1 ironvenv/bin/python utilities/q4-gemm/verify.py --local-evidence

OPENBLAS_NUM_THREADS=1 ironvenv/bin/python -m pytest specs/ -q
# Run on the clean committed branch; this also checks the merge with its target.
utilities/check-settings-ci.sh
```

The summary-only check reports local evidence as `NOT CHECKED`, not a pass.
The source hashes describe the measured build snapshot; the review commit also
contains validation tools and documentation. Updating review notes does not
retag a historical measurement as a new benchmark. The
[measured report](../../specs/open-engine/reports/qwen38-q4-gemm-dma.md)
lists the full-model acceptance gates that remain open before promotion.

When `origin` is a fork, the helper's origin-first fetch can otherwise check an
older fork main instead of the real PR target. With an `upstream` remote, use
a temporary per-command URL override (no repository configuration change):

```bash
env GIT_CONFIG_COUNT=1 GIT_CONFIG_KEY_0=remote.origin.url \
  GIT_CONFIG_VALUE_0="$(git remote get-url upstream)" \
  utilities/check-settings-ci.sh upstream/main
git log -1 --oneline refs/remotes/ci-check/main
```
