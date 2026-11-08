---
name: open-qwen38-consistent-integration
description: Build production consistent Q4 activation sums for full Qwen3.8 lx/ax and validate fresh full-depth decode, independent boundaries and resident identity/reset.
---

# Production consistent Q4 activation sums

Use after the [private full-layer replay](../open-qwen38-consistent-full-layer/SKILL.md).
The production recipe selects `GEMV_Q4_CONSISTENT_SUM=1` only in the generated
`dense_prep_f32.cc` TU for the validated wide dense Q4_1 geometry. Both affine
terms now use the same INT16-reconstructed activation. Original FP32 sums can
leave a nonzero result even for a constant zero-effective-weight block.

Selection requires both `factored_recurrence(spec)` and
`attention_entry_fp32(spec)`: H5120, FF17408, 16/48 linear heads, 128 linear head
dim, four-tap conv, 24/4 attention heads, 256 attention head dim, rotation64,
Q4_1 attention/linear/linear-output/FFN, split norm/down and paired attention.
Other geometries, attention-only, 9B, MoE and mixed/Q8 roles keep original sums.
The correction applies to direct FP32 prep in both lx and ax, including FFN.
Compensated recurrence, weight packing, buffer sizes, streams and stacks stay
unchanged. No new conversion or manual `q4nx-build` packing step is needed.

The manifest declares `q4_activation_sum: quantized_fp32` beside
`linear_recurrence: factored_compensated`. Python bundle validation rejects an
older arithmetic contract even if buffers fit. Create a fresh fixture with
the matching export; never relabel historical control data. Old private
experiments need their recorded checkout, binaries and source identity.

## Build and audit

Native Ryzen AI 9 365 / Strix XDNA2, Python 3.14, mlir-aie 1.4.2,
Peano 21.0.0.2026080301+c9c5ecb7, XRT at `/opt/xilinx/xrt`.
Compiler subprocesses and NPU dispatch need execution-tool escalation on this
host. Do not run source-generating tests while compilation is active.

```bash
OPENBLAS_NUM_THREADS=1 ironvenv/bin/python -m pytest -q specs/open-engine/tests
export PATH="$PWD/ironvenv/lib/python3.14/site-packages/llvm-aie/bin:/opt/xilinx/xrt/bin:$PATH"
export XILINX_XRT=/opt/xilinx/xrt
export OFLM_KEEP_FAILED=1
ironvenv/bin/python -u open_kernels/export_qwen36_kernels.py \
  --model-dir Models/qwen38-27b/Qwen3.8-27B-NPU2 \
  --out Models/qwen38-27b/production-consistent/kernels \
  --build-root Models/qwen38-27b/production-consistent/build --max-ctx 2048 -j 2
ironvenv/bin/python utilities/audit-kernel-export.py \
  --export Models/qwen38-27b/production-consistent/kernels \
  --build-root Models/qwen38-27b/production-consistent/build \
  --out Models/qwen38-27b/production-consistent/build-audit.json
ironvenv/bin/python utilities/audit-consistent-sum.py \
  --export Models/qwen38-27b/production-consistent/kernels \
  --build-root Models/qwen38-27b/production-consistent/build \
  --out Models/qwen38-27b/production-consistent/arithmetic-audit.json
```

Audit all 41 sets/82 hashes, actual copied `final.prj/dense_prep_f32.cc`, production
GEMV headers in lx/ax `.o.d`, and both factored/precise headers for all five DNX
objects. A recipe flag alone is insufficient. Use
`utilities/attention-entry-replay/resources.py` with per-build `build.json`
recording hashes for `final.xclbin` and `insts.bin`; inspect stack sizes in
`final.prj/aie.mlir`. Do not alter the 6144-byte stack or resource limits.

## Fresh CPU reference and streamed NPU run

Verify model and all 65 pool hashes before `--reuse-pools` (which checks sizes,
not contents). Regenerate all 195 independent references and compare their hashes
to retained CPU provenance: the oracle arithmetic is unchanged. Preparation
can omit `--kernel-dir` during compilation; bind the completed audited export
with the same arguments plus `--cfg-only` afterward.

```bash
OPENBLAS_NUM_THREADS=1 ironvenv/bin/python -u open_kernels/model/make_decode.py \
  --model-dir Models/qwen38-27b/Qwen3.8-27B-NPU2 \
  --kernel-dir Models/qwen38-27b/production-consistent/kernels \
  --layers 64 --tokens 3 --max-ctx 2048 --capture-boundaries \
  --out Models/qwen38-27b/production-consistent/decode \
  --pool-dir Models/qwen38-27b/production-full/pools --reuse-pools
ironvenv/bin/python utilities/stream-decode-weights.py \
  --cfg Models/qwen38-27b/production-consistent/decode/run_decode.cfg \
  --out Models/qwen38-27b/production-consistent/decode/run_streamed.cfg
```

Build `open_kernels/harness` with CMake Release. On this host XRT headers are
under `/home/prihlop/sources/xdna-driver/xrt/build/Release/opt/xilinx/xrt/include`
and link libraries in its sibling `lib`; pass `XRT_INCLUDE_DIR`/`XRT_LIB_DIR`.
Use current sources, not an unverified stale harness binary.

```bash
HARNESS_TIMEOUT_MS=30000 LD_LIBRARY_PATH=/opt/xilinx/xrt/lib \
  /tmp/qwen38-consistent-harness/run_kernel \
  Models/qwen38-27b/production-consistent/decode/run_streamed.cfg
OPENBLAS_NUM_THREADS=1 ironvenv/bin/python open_kernels/model/compare_decode.py \
  --out Models/qwen38-27b/production-consistent/decode --tokens 3
```

Run all seven boundary checkers in `open_kernels/model/`: `deltanet_inputs.py`,
`deltanet_state.py`, `attention_heads.py`, `attention_projections.py`,
`ffn_front.py`, `ffn_partials.py`, `lm_head.py`, each with the same `--model-dir`,
`--kernel-dir` and `--out`. FFN checkers require `--layers` listing 0 through 63.
Retain JSON, stderr and exit codes. Also require
`deltanet_state.json["head_diagnostics_passed"]`, which is stricter than its
process exit code. Do not change thresholds or substitute conditioned checks
for composed causal state/output and independent input gates.

## Resident runtime identity and lifecycle

Run NPU processes serially. Approximately 35–40 GiB of available RAM is needed;
do not overlap resident runs with weight-loading CPU checkers. Build the native
CLI from current sources (`src/open_qwen36`). Its manifest test must accept the
actual new export, and native CTest must pass. Commands below use the native
build retained in `/tmp/qwen38-main-merge-next/build`.

```bash
LD_LIBRARY_PATH=/opt/xilinx/xrt/lib OMP_NUM_THREADS=2 OMP_WAIT_POLICY=PASSIVE \
  /tmp/qwen38-main-merge-next/build/open_qwen36_cli \
  --model Models/qwen38-27b/Qwen3.8-27B-NPU2 \
  --kernels Models/qwen38-27b/production-consistent/kernels \
  --ids 248045 --max-tokens 3 --layers 64 --max-ctx 2048 \
  --dump-logits Models/qwen38-27b/production-consistent/engine \
  --twice --decode-route npu --pmode none
OPENBLAS_NUM_THREADS=1 ironvenv/bin/python open_kernels/model/compare_decode.py \
  --out Models/qwen38-27b/production-consistent/decode --tokens 3 \
  --runtime-prefix Models/qwen38-27b/production-consistent/engine
LD_LIBRARY_PATH=/opt/xilinx/xrt/lib OMP_NUM_THREADS=2 OMP_WAIT_POLICY=PASSIVE \
  /tmp/qwen38-main-merge-next/build/open_qwen36_cli \
  --model Models/qwen38-27b/Qwen3.8-27B-NPU2 \
  --kernels Models/qwen38-27b/production-consistent/kernels \
  --ids 248045,8678,198 --layers 64 --max-ctx 2048 \
  --det-step 3 --det-full --decode-route npu --pmode none
```

`--twice` checks repeated request IDs and exercises checkpoint/restore/reset;
its logits files contain the second request. Require all three logits byte-exact
to the harness. `--det-full` checks every written buffer over forced-token
replays; this is deterministic lifecycle validation, not an independent CPU
oracle. Keep it alongside the numerical checks above. Full-depth three-position
decode does not establish block-prefill parity, long-context accuracy, serving
isolation or throughput acceptance. Paired-KV block attention still uses host
attention; no fully NPU prefill claim is implied.

## Measured production export — 2026-10-17

Build key `ac29fbcd21859648ff9d59f79e77f018b99daec0888f77a693e7ff18b2237032`.
All 41 sets / 82 binary hashes pass, with actual compiled FP32 prep and production
GEMV/factored header dependencies verified. lx/ax have 11/15 compute cores;
maximum text 15936/13760 bytes, L1 buffer end 62336/59920, stack 6144. lx has
448 bytes of program headroom. Host suite 1350 passed/47 skipped; native CTest 4/4
and actual 27B manifest test pass. No model repacking was needed.

All 66 model/pool hashes and 195 freshly regenerated CPU reference files match
provenance. Streamed validation uses 2.963 GiB logical buffers and 198 dispatches.
All 192 residuals pass, worst 0.0030399006483 at L58/T0. Logit correlations are
0.9999989597 / 0.9999982780 / 0.9999986060; CPU/NPU argmax 8678 / 198 / 2.

All seven boundary checker processes pass; both DeltaNet head flags are true.
Input/composed state/output pass 144/144, with worst K 0.0009785606623 at L34/T0,
composed state/output 0.0002698188 / 0.0011590659. All 6912 strict head cases
pass, worst state/output 1.2624e-6 / 2.3722e-6 at unchanged maxrel<1e-4 and
cosine>0.9999999. History is byte-exact and padding zero. This new propagation
uses different incoming residuals than the private forced-input experiment;
do not compare its K value as though the inputs were identical.

Attention heads 1152/1152, projection/sublayer 48/48, LM-head 3/3 and FFN front/down
192/192 each pass. Worst FFN front 0.0002516594; down partials 0.0000958907 /
0.0000693562 and sum 0.0000777669. Full resident repeated request reproduces its
IDs; all three second-request logits are byte-exact to the harness. Full-buffer
reset reports 0/3 differing replays on all 64 layers. The native process loaded
64 weight pools plus LM-head successfully; it was not run concurrently with CPU
weight checkers. No throughput result is claimed from this short test.

Retained evidence is under `Models/qwen38-27b/production-consistent/`: seven
boundary JSON reports, `checker-status.json`, `decode-summary.json`,
`runtime-identity.log`, `reset-full.log`, resource/build/arithmetic audits and
`stage-evidence.json`. This directory and model weights stay outside Git.
Separate all-layer FFN up/gate captures, remaining source/packing checks,
block-prefill parity, serving isolation and final performance acceptance remain
open. Composite FFN front/down gates do not replace individual projection gates.
