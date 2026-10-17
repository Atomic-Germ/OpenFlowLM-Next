---
name: open-qwen38-factored-readout
description: Build and validate isolated compensated factored DeltaNet readout with causal state, byte-exact production control and unchanged head bounds.
---

# Compensated factored DeltaNet readout

Use after the triple-BF16 recurrence experiment to avoid readout of a rounded
updated state. This validates an isolated kernel conditioned on captured input
records. It does not establish full-layer fit, propagation or model acceptance.

## Arithmetic and layout

`utilities/deltanet-replay/factored.h` includes the pinned `precise.h` and reuses
its pass 1, delta and state-update arithmetic. Instead of projecting the stored
updated state, output is computed as:

```text
delta = beta * (v - decay * S_old^T * k)
output = (decay * S_old^T * q + (k^T * q) * delta) / sqrt(128)
```

Three BF16 components and all nine products remain in use. TwoSum accumulates
products in high/low FP32 vectors for both dot products and their final
combination. The stored FP32 state is unchanged from the triple candidate.
This distinction matters: exact readout of its rounded state still failed for
one head in the earlier experiment. A more accurate readout of that state alone
could not close the primary gate.

The production-layout single-head design is unchanged: 10 rows/slice, 13 slices,
130 padded rows, separate input/output state BOs, 5120-byte input elements and
1280-float scratch. Additional scratch uses floats 768..895 (low readout) and
896..897 (dot high/low), below delta at 1088..1215. Stack stays 6144 bytes.
Each head starts from zero and carries its own state for all three positions.
The primary oracle remains `deltanet_state.step_reference` with independent
FP64 carry. A separate FP64 algebra check validates factorization; it never
replaces that oracle or relaxes maxrel <1e-4, cosine >0.9999999 or zero padding.

## Reproduce

Native Ryzen AI 9 365 / Strix XDNA2; Python 3.14, mlir-aie 1.4.2,
Peano 21.0.0.2026080301+c9c5ecb7, XRT `/opt/xilinx/xrt`. Native compiler/NPU
commands require execution-tool escalation on this host. Run hardware serially.
No model weights are loaded.

Prepare the complete production-record fixture and control binary with the
[recurrence replay skill](../open-qwen38-recurrence-replay/SKILL.md).
An existing control can be reused only if configure verifies every source and
binary hash against the current production header/design and fixture build key.

```bash
export PATH="$PWD/ironvenv/lib/python3.14/site-packages/llvm-aie/bin:/opt/xilinx/xrt/bin:$PATH"
export XILINX_XRT=/opt/xilinx/xrt
export OFLM_KEEP_FAILED=1
ironvenv/bin/python utilities/deltanet-replay/build_recurrence.py \
  --kernel-dir Models/qwen38-27b/production-fp32-linear/kernels \
  --out Models/qwen38-27b/deltanet-factored/factored --mode factored
ironvenv/bin/python utilities/attention-entry-replay/resources.py \
  --build Models/qwen38-27b/deltanet-factored/factored \
  --out Models/qwen38-27b/deltanet-factored/resources.json
```

Run a fresh control, then the candidate. Preserve JSON and exit statuses.
Configure deletes expected stale outputs; never reconfigure evidence to keep.

```bash
for mode in control factored; do
  build_dir="Models/qwen38-27b/deltanet-factored/$mode"
  if [ "$mode" = control ]; then
    build_dir=Models/qwen38-27b/deltanet-replay/control
  fi
  run_dir="Models/qwen38-27b/deltanet-factored/run-$mode"
  ironvenv/bin/python utilities/deltanet-replay/recurrence_replay.py configure \
    --inputs Models/qwen38-27b/deltanet-replay/inputs --build "$build_dir" --out "$run_dir" || break
  HARNESS_TIMEOUT_MS=30000 LD_LIBRARY_PATH=/opt/xilinx/xrt/lib \
    /tmp/qwen38-fp32-harness/run_kernel "$run_dir/run.cfg" \
    > "Models/qwen38-27b/deltanet-factored/$mode-hardware.log" 2>&1 || break
  OPENBLAS_NUM_THREADS=1 ironvenv/bin/python utilities/deltanet-replay/recurrence_replay.py compare \
    --inputs Models/qwen38-27b/deltanet-replay/inputs --build "$build_dir" --out "$run_dir" \
    > "Models/qwen38-27b/deltanet-factored/$mode-results.json" || break
done
OPENBLAS_NUM_THREADS=1 ironvenv/bin/python -m pytest -q specs/open-engine/tests
```

Source pins include both candidate and shared triple header; control sources
and production export are untouched. Comparison validates complete input/output
coverage, finite values, original captures, build identity and run configuration.
Its control exit status measures byte identity, not the control's numeric gate.

## Measured results — 2026-10-16

- Fresh control: 6912/6912 steps byte-identical for state and output; same 92
  numeric head-output failures, no state or padding failures.
- Factored candidate: **6912/6912 state/output/padding PASS** at unchanged bounds.
  Worst state maxrel **8.12033576e-7**, output **2.69657816e-6**.
  Minimum state/output cosine **0.9999999999999113 / 0.9999999999994235**.
- All 6912 state capture hashes exactly match the triple candidate. Its four
  failed outputs now pass: L25/T0/H10, L41/T0/H39,H40,H41, errors respectively
  **7.6518e-8 / 7.7571e-8 / 8.4828e-8 / 1.4046e-8**.
- FP64 factored-vs-primary output worst maxrel **9.2059e-13**.
- Standalone .text **8720 bytes**, L1 buffer end **22016**, stack **6144**;
  compilation **22.3 s**. The control has .text 5376; triple candidate 4768.
  Standalone fit is not full-lx fit. No production promotion in this experiment.
- TDD: three failing tests first, then all 24 replay tests pass; full host suite
  **1270 passed, 47 skipped**. Five cases added, including header/binary tampering.

Evidence: `Models/qwen38-27b/deltanet-factored/`, with complete comparison JSON,
resource audit, source/binary/input/output pins, hardware logs and evidence.json.
Production key stays `dc6c7d83b46ba6f25fe2cd0a641527737ada1f1de6422bbe156c0d51617b096f`.

Next: measure full-lx code/data fit and isolated full-layer causal propagation
before changing the production builder. Do not infer throughput from these
single-head validation dispatches. Full-depth export regression and resident
identity/reset remain required; the latter needs 35–40 GiB available RAM.
