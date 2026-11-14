---
name: open-qwen38-factored-full-layer
description: Build private full-lx factored DeltaNet with shared arithmetic helpers, audit compiler dependencies and core budgets, then validate all linear layers with causal replay.
---

# Factored DeltaNet in a full linear layer

Use after [isolated factored readout](../open-qwen38-factored-readout/SKILL.md).
This stage keeps production binaries, recipes, manifests, state ABI and scheduling
unchanged. The private full-lx design changes only recurrence headers and helper
linkage. It reuses all generated entry TUs from the production recipe.

## Compiler and resource findings

IRON copies C++ TUs into `final.prj`. A quoted include therefore cannot rely on
the original private source directory. Put the private generated directory FIRST
in the include search path. `build_full_layer.py` verifies all five `dnx_*.o.d`
files contain both private `dnx.h` and `precise.h`. A build using the production
header is rejected, even if its resource sizes look plausible.

The direct factored full layer overflows program memory: **18592 / 16384 bytes**
on each main core. It contains three copies of `compensated_mac` (1136 bytes
each), plus four local `mac3` copies. The `--shared` build changes only
`static __attribute__((noinline))` to `inline __attribute__((noinline))` in the
two private headers. Linker deduplication then retains one copy of each helper.
No arithmetic expression, loop, FIFO, stack size or geometry is changed.

Shared full-lx: **15840 bytes** maximum text, **544 bytes** program headroom,
**62336** maximum L1 buffer end; all **11 cores** fit with **6144-byte stacks**.
Both the direct failure and shared success must be retained. A successful
standalone build cannot predict this full-layer budget.

## Build

Native Ryzen AI 9 365 / Strix XDNA2; Python 3.14, mlir-aie 1.4.2,
Peano 21.0.0.2026080301+c9c5ecb7; XRT `/opt/xilinx/xrt`. Compiler and hardware
commands require execution-tool escalation on this host. Do not run source-
generating tests during compilation. Run hardware programs serially.

```bash
export PATH="$PWD/ironvenv/lib/python3.14/site-packages/llvm-aie/bin:/opt/xilinx/xrt/bin:$PATH"
export XILINX_XRT=/opt/xilinx/xrt
export OFLM_KEEP_FAILED=1
ironvenv/bin/python utilities/deltanet-replay/build_full_layer.py \
  --kernel-dir Models/qwen38-27b/production-fp32-linear/kernels \
  --out Models/qwen38-27b/deltanet-full-layer/shared-v2 --mode factored --shared
ironvenv/bin/python utilities/attention-entry-replay/resources.py \
  --build Models/qwen38-27b/deltanet-full-layer/shared-v2 \
  --out Models/qwen38-27b/deltanet-full-layer/shared-resources.json
ironvenv/bin/python utilities/deltanet-replay/build_full_layer.py \
  --kernel-dir Models/qwen38-27b/production-fp32-linear/kernels \
  --out Models/qwen38-27b/deltanet-full-layer/control --mode control
```

To reproduce the program-memory failure, omit `--shared` and use a separate
output directory. Preserve `attempt.json`, compiler log and final core ELFs.
There must be no valid `build.json` after a failed build. `utilities/core_sizes.py`
reads the retained linked ELFs even if xclbin generation failed.

## Controlled replay

The fixture contains 48 linear layers × 3 positions. Each layer gets the
retained incoming production residuals and starts from zero; later positions
carry only that variant's own conv/recurrent state. Pool weights are loaded one
layer at a time. Validate model/pool hashes against the retained production
`input-sha256.json` before accepting the run.

```bash
for mode in control factored; do
  build_dir=Models/qwen38-27b/deltanet-full-layer/control
  if [ "$mode" = factored ]; then
    build_dir=Models/qwen38-27b/deltanet-full-layer/shared-v2
  fi
  OPENBLAS_NUM_THREADS=1 ironvenv/bin/python utilities/deltanet-replay/full_layer_replay.py prepare \
    --model Models/qwen38-27b/Qwen3.8-27B-NPU2 \
    --kernels Models/qwen38-27b/production-fp32-linear/kernels \
    --fixture Models/qwen38-27b/production-fp32-linear/decode \
    --pools Models/qwen38-27b/production-full/pools --build "$build_dir" \
    --out "Models/qwen38-27b/deltanet-full-layer/run-$mode" || break
  HARNESS_TIMEOUT_MS=30000 LD_LIBRARY_PATH=/opt/xilinx/xrt/lib \
    /tmp/qwen38-fp32-harness/run_kernel \
    "Models/qwen38-27b/deltanet-full-layer/run-$mode/run.cfg" || break
done
OPENBLAS_NUM_THREADS=1 ironvenv/bin/python -u utilities/deltanet-replay/full_layer_replay.py reference \
  --out Models/qwen38-27b/deltanet-full-layer/run-control \
  --refs Models/qwen38-27b/deltanet-full-layer/reference
for mode in control factored; do
  OPENBLAS_NUM_THREADS=1 ironvenv/bin/python utilities/deltanet-replay/full_layer_replay.py score \
    --out "Models/qwen38-27b/deltanet-full-layer/run-$mode" \
    --refs Models/qwen38-27b/deltanet-full-layer/reference \
    --single-result Models/qwen38-27b/deltanet-factored/factored-results.json \
    --single-run Models/qwen38-27b/deltanet-factored/run-factored || break
done
OPENBLAS_NUM_THREADS=1 ironvenv/bin/python -m pytest -q specs/open-engine/tests
```

Prepare invalidates prior audit/score and expected dumps. It refuses output
inside the production fixture, model, pools, build or export. Inputs, binaries,
compiler dependencies, generated sources, run config and outputs are pinned.
Keep control and candidate artifacts separate.

Residual reference uses the existing `replica_qwen35` (mixed FP64/FP32) with
independent conv/recurrent carry and the manifest's effective quantization.
Weights are cached for one layer across its three positions, then discarded.
The acceptance bound stays maxrel <0.005. The stricter recurrence oracle carries
independent FP64 state from zero at maxrel <1e-4 and cosine >0.9999999 per head.
Neither reference is seeded from captured state.

The control must reproduce the complete act/state/residual buffers bytewise.
For the candidate, records/QKV/conv history must remain byte-identical to
production, and every head's state/output must also match the validated isolated
factored kernel bytewise. This catches source-selection or linkage changes
that a whole-tensor numerical score could hide.

## Acceptance scope

These are isolated full-layer replays with forced incoming residuals. They do
not propagate candidate outputs across the 64-layer model. Production export
integration, fresh full-depth regression and resident runtime identity/reset
remain separate gates. Full resident reset requires 35–40 GiB available RAM;
this experiment uses one layer's weights at a time.

Evidence lives in `Models/qwen38-27b/deltanet-full-layer/`. Initial `direct/` and
`shared/` builds are explicitly rejected header-selection attempts; use only
`direct-v2/` for overflow evidence and `shared-v2/` for the measured candidate.

## Measured validation — 2026-10-16

- Direct build: program-memory overflow at **18592 B**; no valid build record.
  Shared build: **125.1 s**, all 11 cores pass the budgets above.
- Control: **144/144 full-layer cases**, all **432 act/state/residual captures
  byte-identical** to production. Its **92/6912** head-output numerical failures
  remain visible separately from control identity acceptance.
- Candidate: **144/144 residual and recurrence cases PASS**; **6912/6912** head
  state/output checks pass. Worst head state/output maxrel
  **8.12033576e-7 / 2.69657816e-6**. Conv history shifts are bit-exact; padding zero.
- All **13824** candidate head-state/output captures match the isolated factored
  run bytewise. Records/QKV/conv history match production for all 144 cases.
- Independent full-layer residual worst maxrel: control **0.0012884741**,
  candidate **0.0013026962**, both below **0.005**. The recurrence correction
  does not improve every residual score; both remain within the original gate.
- All **49 model/used-pool hashes** match retained production provenance.
  The same shared input hashes and 144 independent residual references serve
  both variants. Both scoring commands exit 0 with complete JSON.
- TDD: **9 new cases**, full host suite **1279 passed, 47 skipped**. Regressions
  cover source injection, shared linkage, header-search order, dependency-file
  verification, causal configuration, capture length/bit identity and output
  isolation. Production key remains
  `dc6c7d83b46ba6f25fe2cd0a641527737ada1f1de6422bbe156c0d51617b096f`.

The next stage is production integration with a fresh export and full-depth
regression. Do not reuse old full-depth/resident evidence for the new binaries.
