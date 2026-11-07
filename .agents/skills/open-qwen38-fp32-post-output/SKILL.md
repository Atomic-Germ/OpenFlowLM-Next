---
name: open-qwen38-fp32-post-output
description: Build private full-lx FP32 DeltaNet post-output and direct FP32 output preparation with exact production controls and causal L0 validation.
---

# Private FP32 post-output

Use after signed L0 attribution in the sequential-L0 replay skill. Preserve
production sources, binaries, packing and model metadata. Generate current
recipe entry TUs with factored recurrence and consistent direct-FP32 Q4 sums.
Never overlay legacy recurrence wrappers.

Control preserves BF16 post output/preparation. Candidate changes post storage
to FP32, the output FIFO/kernel argument to 4096 bytes, six post drains to FP32,
and output-projection preparation to the existing direct-FP32 routine. Append
24576 bytes to scratch (393216 -> 417792) for FP32 A_OG; retain all original
offsets and state ABI. Six 4-KiB activation elements stream one at a time through
the depth-two FIFO. Preserve 6144-byte stacks and program/L1 limits.

Run tests before compilation; do not run source-generating tests during builds.
The builder requires fresh output, checks source anchors and compiler dependency
resolution, retains logs, and runs the ELF/L1 resource auditor.

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

Compiler and device access may require escalation. Use Peano from ironvenv via
PEANO_INSTALL_DIR and installed /opt/xilinx/xrt runtime, not the source-tree
runtime without the XDNA shim. Run NPU jobs serially with the shared lock.

Before interpreting a candidate, require all 243 control act/state/residual
captures byte-exact to prefill-sequential-drift/native-l0-v2. Candidate entry,
QKV/Z, records, recurrence output and complete state must remain exact. Compare
all residuals against independent references and selected post/output/FFN gates
at P36–38 and P240–242. Retain failures and P242 cancellation regressions.
One L0 pool is sufficient; this is not full-model or production acceptance.


## Replay and audit

Follow `utilities/deltanet-post-fp32/README.md` for exact control/candidate
commands. `replay.py run` uses an existing built standard harness and original
L0 inputs, holds the shared device lock, and rejects overwritten runs. Control
must be separately audited before candidate dispatch. `replay.py audit` rereads
all captures and repeats numerical measurements with pinned inputs and builds.

The initial control-v1 compilation succeeded but was rejected by a dependency
checker expecting post.o.d. IRON renames the TU to post_fn.cc/post_fn.o.d.
Keep the failed record; control-v2 uses a corrected, regression-tested mapping
and compares the compiled TU bytes against generated/post.cc.

## Measured checkpoint

Control-v2 and fp32-v1 pass all 11 core resource checks: maximum text
15936 / 16000 bytes, maximum assigned L1 buffer end 62336 bytes, 849 pinned
compiler dependency entries each. Control-run-v1 reproduces all 729 retained
buffers; fp32-run-v1 preserves all 243 upstream/state cases exactly and passes
243 independent residual gates plus six selected post/output/FFN cases.
Worst residual maxrel drops 0.0015055132 -> 0.0000357078; P37/P241 improve
45x/338x. P242 improves too, while P202 rises slightly within bounds.
Separate numerical re-audits pass for both runs, reproducing all metrics and
capture hashes. Verify each audit.json before reusing the captures. Host suite:
1814 passed, 47 skipped (13 new RED-to-GREEN tests). This is one-layer evidence;
production promotion and full-depth sequential precision remain open.
