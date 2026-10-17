---
name: open-qwen38-fp32-linear-integration
description: Build the integrated FP32 DeltaNet entry/history export, validate the state ABI and host prefill carry, then run full-depth decode and independent boundary gates.
---

# Production FP32 DeltaNet entry and history

Use after the isolated linear-entry and conv-history experiments. The validated
wide dense geometry (H5120, 16 key/48 value heads, dim 128, four conv taps,
split norm/down) selects both corrections. Other geometries retain BF16.
The historical private experiments require checkout `46c845a`; their builders
refuse the now-integrated recipe rather than reinterpret its buffers.

## ABI and implementation

The manifest declares `linear_entry_dtype: float32` and `linear_conv_dtype:
float32`. Entry `xn` occupies five 4096-byte elements; the following activation
regions move by 8192 bytes. Linear act is 393216 bytes, state is 3317760 bytes,
and recurrent-state offset is 122880. Conv history uses three rows of FP32,
with 512-channel halves over the unchanged 2-KiB FIFOs. AB carries five FP32
blocks with both high/low parts in the BF16 MAC; its 17 side fills use
`Pipeline(3)` and rolled core loops. Do not raise the global shim budget.
Stack stays 6144 bytes. Weights and all pool/constant packing are unchanged;
`q4nx-build` consumes the new recipe without manual model conversion.

Host block prefill reads the same history dtype and preserves FP32 QKV history
across short blocks. The manifest parser rejects mismatched block dtype,
offsets and state sizes. Snapshots/reset copy the complete declared buffer.
This storage compatibility does not establish block-prefill numerical parity.
Shared DeltaNet checkers read dtype from the validated manifest and keep the
unrounded FP64 primary reference and previous numerical bounds.

For GCC/Clang, host DeltaNet and its scalar oracle compile with
`-ffp-contract=off` in both standalone and main CMake builds. This preserves
explicit multiply-then-add behavior; implicit FMA contraction otherwise makes
two legacy vector/tail bit-equality tests fail even on the baseline source.
Explicit FMA intrinsics elsewhere remain available.

## Build and CPU checks

Use native Python 3.14, mlir-aie 1.4.2, Peano
21.0.0.2026080301+c9c5ecb7 and XRT from `/opt/xilinx/xrt`.
Do not run source-generating tests concurrently with export. AIE forkserver
compilation and NPU dispatch require execution-tool escalation on this host.

```bash
OPENBLAS_NUM_THREADS=1 ironvenv/bin/python -m pytest specs/open-engine/tests -q
cmake --build /tmp/qwen38-paired-runtime -j 2
ctest --test-dir /tmp/qwen38-paired-runtime --output-on-failure
OPENBLAS_NUM_THREADS=1 ironvenv/bin/python open_kernels/model/replica_block.py \
  --fixture /tmp/qwen38-linear-block-fixture
OMP_NUM_THREADS=2 /tmp/qwen38-paired-runtime/block_host_test /tmp/qwen38-linear-block-fixture
export PATH="$PWD/ironvenv/lib/python3.14/site-packages/llvm-aie/bin:/opt/xilinx/xrt/bin:$PATH"
export XILINX_XRT=/opt/xilinx/xrt
export OFLM_KEEP_FAILED=1
ironvenv/bin/python -u open_kernels/export_qwen36_kernels.py \
  --model-dir Models/qwen38-27b/Qwen3.8-27B-NPU2 \
  --out Models/qwen38-27b/production-fp32-linear/kernels \
  --build-root Models/qwen38-27b/production-fp32-linear/build --max-ctx 2048 -j 2
ironvenv/bin/python utilities/audit-kernel-export.py \
  --export Models/qwen38-27b/production-fp32-linear/kernels \
  --build-root Models/qwen38-27b/production-fp32-linear/build \
  --out Models/qwen38-27b/production-fp32-linear/build-audit.json
```

Record all 41 sets/82 binary hashes. Audit lx and ax core .text/L1 with
`utilities/attention-entry-replay/resources.py`; use per-build `build.json`
with the two artifact hashes, as the previous integration did. Stack must not
be reduced to make a failed build fit.

## Full-depth replay

Verify model and 65 reused pool hashes against retained provenance before
`--reuse-pools`, which only checks sizes. Regenerate independent CPU references;
all 195 should match the retained reference because its arithmetic is unchanged.
A new export requires a new fixture and metadata, not relabeling old captures.

```bash
OPENBLAS_NUM_THREADS=1 ironvenv/bin/python -u open_kernels/model/make_decode.py \
  --model-dir Models/qwen38-27b/Qwen3.8-27B-NPU2 \
  --kernel-dir Models/qwen38-27b/production-fp32-linear/kernels \
  --layers 64 --tokens 3 --max-ctx 2048 --capture-boundaries \
  --out Models/qwen38-27b/production-fp32-linear/decode \
  --pool-dir Models/qwen38-27b/production-full/pools --reuse-pools
ironvenv/bin/python utilities/stream-decode-weights.py \
  --cfg Models/qwen38-27b/production-fp32-linear/decode/run_decode.cfg \
  --out Models/qwen38-27b/production-fp32-linear/decode/run_streamed.cfg
HARNESS_TIMEOUT_MS=30000 LD_LIBRARY_PATH=/opt/xilinx/xrt/lib \
  /tmp/qwen38-fp32-harness/run_kernel \
  Models/qwen38-27b/production-fp32-linear/decode/run_streamed.cfg
OPENBLAS_NUM_THREADS=1 ironvenv/bin/python open_kernels/model/compare_decode.py \
  --out Models/qwen38-27b/production-fp32-linear/decode --tokens 3
```

CPU preparation may run without `--kernel-dir` during compilation; then bind
with `--kernel-dir --cfg-only` after the complete export passes audit. Run
`deltanet_inputs.py`, `deltanet_state.py`, `lm_head.py`, `attention_heads.py`,
`attention_projections.py`, `ffn_front.py`, and `ffn_partials.py` against the
same model/export/decode directories. Both FFN checkers require all 64 layer
indices explicitly. Preserve JSON, input/weight/capture hashes and exit status.
Report primary and head-local diagnostics separately. A local or residual PASS
does not imply every head, prefill or serving passes.

Resident runtime identity/reset needs approximately 35–40 GiB available RAM;
streamed harness full-depth validation fits much less. Rebuild the native CLI,
run the resident cfg, compare native logits bytewise for three positions, and
run `--det-step 3 --det-full` with IDs 248045,8678,198 when memory allows.


## Verified integration — 2026-10-14

Build key `dc6c7d83b46ba6f25fe2cd0a641527737ada1f1de6422bbe156c0d51617b096f`.
41 sets / 82 binary hashes verified. Max .text lx/ax 15248/13760 bytes;
L1 buffer ends 62336/59920; stack 6144. All 66 model/pool hashes and 195
regenerated independent CPU reference hashes match retained provenance.
Host suite: 1231 passed, 47 skipped; native CTest 4/4; host prefill tests PASS.

Streamed full-depth run uses 2.963 GiB logical buffers: 198 dispatches,
192/192 residuals PASS, worst 0.0032961313 at L58/T0. Logit correlations
0.9999989778/0.9999981051/0.9999984696 and argmax 8678/198/2 match CPU.
Composed DeltaNet 144/144 PASS, worst state/output 0.0011601912/0.0018687877.
Conditioned state/output worst 1.47720485e-5/2.53313001e-5, all history shifts
bit-exact and all padding zero. All 6912 head-state diagnostics pass, but
92 head-output diagnostics fail (worst failed maxabs 1.51510633e-7). These
remain separate unresolved diagnostics; no threshold was relaxed.

Attention heads 1152/1152 and composed branches 48/48 pass (worst branch
0.0012155947). Final norm/LM-head 3/3 pass, worst composed logit error
0.0013846288. FFN front and split-down each pass 192/192 cases, worst front
0.0009043463 and partial 0.0003953126. All seven checker processes exit 0;
JSON checkers pin capture and effective-weight hashes.

Only about 16 GiB RAM was available. Native four-layer full-buffer reset
passed with 0/3 differences; full-depth resident identity/reset is pending.
The limited lifecycle regression used the rebuilt CLI:

```bash
LD_LIBRARY_PATH=/opt/xilinx/xrt/lib OMP_NUM_THREADS=2 \
  /tmp/qwen38-paired-runtime/open_qwen36_cli \
  --model Models/qwen38-27b/Qwen3.8-27B-NPU2 \
  --kernels Models/qwen38-27b/production-fp32-linear/kernels \
  --layers 4 --max-ctx 2048 --ids 248045,8678,198 \
  --det-step 3 --det-full --decode-route npu --pmode none
```

Artifacts live in `Models/qwen38-27b/production-fp32-linear/`. Do not promote
this result to full model acceptance: head-output diagnostics, full-depth
resident lifecycle, block-prefill parity, serving and performance remain open.


## Follow-up: localize head-output errors without model residency

Use `utilities/deltanet_readout.py` with `--model-dir`, `--kernel-dir` and `--out`
pointing to the same production model, export and completed decode fixture as
above. It loads no model weights and runs no NPU kernels. Preserve stdout JSON
and the exit status: unresolved primary head diagnostics deliberately return 1.
The unrounded recurrence remains independent of captured state; only a separate
readout diagnostic uses the updated state snapshot.

On 2026-10-16, all 6912 primary head metrics and 289 capture hashes reproduce
the retained production recurrence report. Primary whole-tensor checks still
pass and 92 head-output diagnostics still fail. Exact captured-state readout
versus observed output has 69 failures; exact captured-state readout versus
independent primary output has 47. These sets overlap: do not add their counts.
Even ideal readout alone therefore does not close the primary head gate.

A CPU three-term BF16-pair representation estimate matches observed output
within the existing head bounds in 6909/6912 cases. Adding low×low leaves 47
representation-to-exact-readout failures, versus 67 for three terms. The estimates
use FP64 accumulation, not bit-exact hardware emulation; the fourth term alone
is not a validated fix. Failed heads have median cancellation ratio 79.08,
versus 2.60 for passed heads. Next, isolate higher-precision state update and
readout with a byte-exact hardware control and causal state before promotion.
The production kernels and build key remain unchanged.

Artifacts: `Models/qwen38-27b/deltanet-readout/`. Full-depth resident/reset still
requires additional available RAM. The diagnostic does not close that gate.

Diagnostic follow-up tests: 15 new cases; full host suite 1246 passed, 47 skipped.
