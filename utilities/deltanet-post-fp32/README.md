# Private FP32 DeltaNet post-output

This experiment retains FP32 post-recurrence norm/gate output through direct
FP32 preparation of the Q4 output projection. It follows the signed L0 rounding
attribution. Production recipes, exports, weights and acceptance bounds remain
unchanged. Both variants compile a complete private lx from current generated
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
