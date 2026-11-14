# Private FP32 DeltaNet post-output

The standard wide dense recipe now selects the shared FP32 path. For a fresh
production export and full-depth checks, use the
[integration skill](../../.agents/skills/open-qwen38-post-output-integration/SKILL.md).
The private builder still supports the original BF16 control and FP32 candidate:
it explicitly restores the control's scratch layout and does not append the
FP32 region twice. New private builds require a compatible current export.
The commands and measurements below describe the retained historical experiment;
reproduce them from their recorded checkout, without rewriting old evidence.

This experiment retains FP32 post-recurrence norm/gate output through direct
FP32 preparation of the Q4 output projection. It follows the signed L0 rounding
attribution. At that checkpoint production recipes, exports, weights and
acceptance bounds were unchanged. Both variants compile a complete private lx from generated
entry sources, retaining factored recurrence and consistent Q4 activation sums.

The control retains BF16 output. The candidate appends 24,576 bytes to activation
scratch (393,216 -> 417,792 bytes), widens only the post output stream to 4 KiB,
and consumes six FP32 activation chunks through the existing streaming preparer.
Original scratch offsets and state layout remain unchanged. Stack/program/L1
limits are not increased.

## Build

See the [build skill](../../.agents/skills/open-qwen38-fp32-post-output/SKILL.md).
Run host tests before compiling; do not run source-generating tests concurrently.
Use fresh directories and preserve failed attempts. `control-v1` compiled but
its dependency audit incorrectly expected `post.o.d` instead of IRON's renamed
`post_fn.o.d` / `post_fn.cc`. The corrected auditor compares that copied source
against `generated/post.cc`; `control-v2` is the fresh retry.

```bash
source utilities/env.sh
export OFLM_KEEP_FAILED=1
ironvenv/bin/python utilities/deltanet-post-fp32/build.py \
  --kernels Models/qwen38-27b/production-consistent/kernels --mode control \
  --out Models/qwen38-27b/deltanet-post-fp32/control-v2
ironvenv/bin/python utilities/deltanet-post-fp32/build.py \
  --kernels Models/qwen38-27b/production-consistent/kernels --mode fp32 \
  --out Models/qwen38-27b/deltanet-post-fp32/fp32-v1
```

`build.json` binds generated sources, copied translation units, production
recurrence/Q4 headers, compiler dependencies and binaries. `resources.json`
checks every compute core's linked program and assigned L1 buffers. Both are
required before replay.

## Causal control and candidate

The replay consumes the exact audited original embeddings and one L0 weight
pool. State is initialized once and carried through all 243 positions.
No full-model residency is required. Use installed XRT runtime libraries and
an idle NPU; the runner holds the common validation lock.

```bash
export XILINX_XRT=/opt/xilinx/xrt
export LD_LIBRARY_PATH=/opt/xilinx/xrt/lib
export OPENBLAS_NUM_THREADS=1
ironvenv/bin/python utilities/deltanet-post-fp32/replay.py run \
  --capture Models/qwen38-27b/prefill-sequential-drift/native-l0-v2 \
  --build Models/qwen38-27b/deltanet-post-fp32/control-v2 \
  --harness Models/qwen38-27b/prefill-sequential-drift/harness-l0-v1/run_kernel \
  --out Models/qwen38-27b/deltanet-post-fp32/control-run-v1
ironvenv/bin/python utilities/deltanet-post-fp32/replay.py audit \
  --out Models/qwen38-27b/deltanet-post-fp32/control-run-v1
ironvenv/bin/python utilities/deltanet-post-fp32/replay.py run \
  --capture Models/qwen38-27b/prefill-sequential-drift/native-l0-v2 \
  --build Models/qwen38-27b/deltanet-post-fp32/fp32-v1 \
  --control Models/qwen38-27b/deltanet-post-fp32/control-run-v1 \
  --harness Models/qwen38-27b/prefill-sequential-drift/harness-l0-v1/run_kernel \
  --out Models/qwen38-27b/deltanet-post-fp32/fp32-run-v1
ironvenv/bin/python utilities/deltanet-post-fp32/replay.py audit \
  --out Models/qwen38-27b/deltanet-post-fp32/fp32-run-v1
```

Control requires byte-exact complete act/state/residual buffers. Candidate
requires byte-exact entry norm, QKV/Z, records, recurrence output and complete
state against production at every position. Compare every residual against the
retained independent reference. Selected P36–38 and P240–242 also require
post/output projection, FFN front and both down-partial gates. These selected
checks are conditioned on captured inputs; full-layer residual references remain
independent. Preserve P242 cancellation regressions and all failure reports.
`model_acceptance` is always false: L0 is not full-depth propagation.

## Resource results

Both accepted builds contain 11 compute cores and pass the original resource
limits. Control/candidate maximum linked text is 15,936 / 16,000 bytes; the
maximum assigned L1 buffer end is 62,336 bytes for both. Each build records
849 compiler dependency entries. Scratch grows by 24 KiB; state remains
3,317,760 bytes. These checks establish fit, not a throughput measurement.

## L0 results

Control-run-v1 matches all 729 retained buffers exactly. FP32-run-v1 preserves
all 243 upstream/state cases bytewise, passes all 243 independent residuals
and all six selected post/output/FFN cases. Worst residual maxrel falls from
0.0015055132 to 0.0000357078; median from 0.0003848893 to 0.0000083408.
P37/P241 improve about 45x/338x, and P242 improves 1.65x. Of 243 positions,
242 improve; P202 rises slightly from 0.0000257763 to 0.0000260274, within
the unchanged bound. Separate re-audits pass for both runs, reproducing
all metrics and capture hashes. Verify each audit.json before reuse.

The two accepted runs total 486 dispatches. Thirteen new RED-to-GREEN tests
and the complete host suite pass (1814 passed, 47 skipped). These results
justify broader private layer coverage; they do not close full-model sequential
long-prefix acceptance or promote the experiment to production.


## Shared implementation checkpoint — 2026-10-26

The private builder now selects `Linear.POST_F32` in the shared `lx` and
`xlayer` code. This one selection controls `DN_POST_FP32`, the post kernel
argument, output FIFO, dense post DMA and direct FP32 output preparation.
Both modes compile the same `dn_post/post.cc`; the builder no longer rewrites
its arithmetic or the stream/preparation code. Scratch placement remains
private pending recipe/manifest promotion. Published recipes retain BF16 post
output, including all five Qwen3.5 sizes and Qwen3.6 MoE.

Build fresh `shared-control-v1` / `shared-fp32-v1` directories using the commands
in the [shared post skill](../../.agents/skills/open-qwen38-shared-post-output/SKILL.md).
Use `prefill-production/kernels-model-v1` as `--kernels`, because the current
recipe includes paired prefill. The earlier commands/results above describe
the historical private implementation and must not be used to overwrite it.

```bash
ironvenv/bin/python utilities/deltanet-post-fp32/identity.py \
  --build Models/qwen38-27b/deltanet-post-fp32/shared-control-v1 \
  --reference Models/qwen38-27b/deltanet-post-fp32/control-v2 \
  --reference-run Models/qwen38-27b/deltanet-post-fp32/control-run-v1 \
  --out Models/qwen38-27b/deltanet-post-fp32/shared-control-identity-v1.json
ironvenv/bin/python utilities/deltanet-post-fp32/identity.py \
  --build Models/qwen38-27b/deltanet-post-fp32/shared-fp32-v1 \
  --reference Models/qwen38-27b/deltanet-post-fp32/fp32-v1 \
  --reference-run Models/qwen38-27b/deltanet-post-fp32/fp32-run-v1 \
  --out Models/qwen38-27b/deltanet-post-fp32/shared-fp32-identity-v1.json
```

Both comparisons pass: instruction bytes are exact, and complete XCLBINs
match except for documented build stamps (79 / 77 bytes). Each reference is
bound to its previously passing causal audit, build record, two binary hashes
and all 729 captured buffers. Current builders and compiler dependencies are
verified separately; historical sources are not reinterpreted as current ones.
The comparison refuses overwritten reports and changed or incomplete archives.

Both new full-lx builds pass all 11 core resource checks with 849 pinned
compiler dependency entries. Maximum text is 15936 / 16000 bytes and maximum
assigned L1 buffer end is 62336, with unchanged 6144-byte stacks. Toolchain:
mlir-aie 1.4.2, llvm-aie 21.0.0.2026080301+c9c5ecb7.

Eight stream/store tests failed before the change and pass after it; thirteen
additional cases cover archive integrity and published-recipe ABI preservation.
Full host suite: **1878 passed, 47 skipped**. No new resident model run or
performance measurement was performed in this checkpoint. Recipe-owned FP32
scratch/manifest export and full-depth propagation remain the next gates.
