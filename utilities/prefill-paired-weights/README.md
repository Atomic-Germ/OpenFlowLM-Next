# Private paired-weight GEMM experiment

This diagnostic retains Q4NX and the production GEMM layout. A control build
uses unchanged copied design/dequant sources. The low-weight build changes
only the fused dequant conversion: BF16_RNE(W - BF16_RNE(W)), evaluated from
the FP32 Q4 dequant accumulator. It preserves stream geometry and L1 buffers.
Separate dispatches allow the existing BF16 matrix multiplication to consume
the two weight parts without widening its ABI. Production files are untouched.

## Build

Use fresh directories. The builder sanitizes GQP flags and selects N34816,
K5120, T256, Q4_1, no B reuse. It records source, generated-file, compiler-
dependency and binary hashes and checks 32 core code/data budgets. The private
header must appear in compiler dependency files. Failures retain build.json
with complete=false and the failed compiler project.

```bash
mkdir -p Models/qwen38-27b/prefill-paired-weights
OPENBLAS_NUM_THREADS=1 ironvenv/bin/python utilities/prefill-paired-weights/build.py \
  --mode control --out Models/qwen38-27b/prefill-paired-weights/v2/control
OPENBLAS_NUM_THREADS=1 ironvenv/bin/python utilities/prefill-paired-weights/build.py \
  --mode low --out Models/qwen38-27b/prefill-paired-weights/v2/low
```

Toolchain: local ironvenv (Python 3.14, mlir-aie), bundled llvm-aie/Peano;
XRT under /opt/xilinx/xrt. The builder sets PATH internally. Kernel compilation
and hardware dispatch require host access in the managed environment.

## Validation contract

Reuse the verified artifacts in Models/qwen38-27b/prefill-upgate-replay/run
and that stage's standard harness build. Require exact individual high/low
control outputs against the retained production dispatches, not only equality
of their sum. Preserve T256 slots 7 and 135, zero inactive columns, the same
111411200 weight bytes and the original FP32-input/exact-weight reference.

Candidate arithmetic is FP32((Wh*Xh + Wh*Xl) + (Wl*Xh + Wl*Xl)), with each
product supplied by a separate native dispatch. The control and low builds
use the same geometry and inputs. Keep diagnostics against paired-input or
rounded-weight references separate from primary accuracy. Never infer model
acceptance from a passing projection; up/down errors partly cancel upstream.

No full-model residency is needed. The per-process logical NPU weight/input/
output footprint is about 0.1395 GiB, excluding XRT overhead. Execute serially
with exclusive NPU access. No throughput claim from the extra dispatches.

Any failed build or control remains a failed experiment; do not replace the
reference or relax gates to proceed.

## Native replay

```bash
LD_LIBRARY_PATH=/opt/xilinx/xrt/lib OPENBLAS_NUM_THREADS=1 \
ironvenv/bin/python utilities/prefill-paired-weights/run.py \
  --out Models/qwen38-27b/prefill-paired-weights/replay \
  --baseline Models/qwen38-27b/prefill-upgate-replay/run \
  --build-root Models/qwen38-27b/prefill-paired-weights/v2 \
  --harness-build Models/qwen38-27b/prefill-upgate-replay/harness
```

The runner checks the retained control's preparation/output hashes, both new
kernel builds and compiler dependencies, and harness provenance. It dispatches
control and low-weight kernels with high/low inputs in both slots: eight runs.
All candidate slots must match bytewise. Separate up/gate primary gates must
pass, with strictly lower maxrel and rel_fro than control. Equality is not an
improvement. The paired-input reference is diagnostic only. Output JSON always
sets model_acceptance=false, including a successful isolated candidate.

The first low build failed because AIE API to_float does not accept a BF16
vector. The corrected private header widens through an accfloat accumulator,
then subtracts FP32 values before rounding the residual to BF16. Both variants
are rebuilt under v2; original builds/logs are retained rather than rebinding
old provenance. The additional conversion creates no new persistent L1 buffer.

## Measured result — 2026-10-20

All **8/8 NPU dispatches** complete. Every fresh control high/low output matches
its retained production output bytewise, including all inactive columns. Both
candidate slot results are byte-identical. Primary gates pass with strictly
lower maxrel and rel_fro for each projection, against the original FP32-input /
exact-effective-weight reference.

| Projection | Control maxrel | Candidate maxrel | Candidate rel_fro |
|---|---:|---:|---:|
| Up | 0.0019482680 | 0.0000015091 | 0.0000010450 |
| Gate | 0.0002586737 | 0.0000013845 | 0.0000012487 |

Against the paired-input diagnostic reference, candidate up/gate maxrel is
0.0000011376 / 0.0000016002. The diagnostic reference does not replace the
primary gate. No full FFN, layer residual, causal state or logits were rerun
with the candidate. The earlier full-model long-prefix accuracy failure is
still unresolved (558/774 retained logit gates). Existing weights need no
repacking, and production recipes/binaries are unchanged.

Both private builds pass all **32-core** resource checks. Maximum text size is
**4784 bytes** for control and **5568 bytes** for low; maximum buffer end is
**61448 bytes** for both, with 4096-byte configured stacks. Each build records
**801 compiler dependencies**. Toolchain: Python 3.14, mlir-aie 1.4.2,
llvm-aie 21.0.0.2026080301+c9c5ecb7. The source and generated hashes are checked
before hardware execution.

| Artifact | SHA256 |
|---|---|
| Control XCLBIN | c846df595d6ae46bd4dc41ee6ca00f0e5e4365c9458c737eac264a4d4481c74b |
| Low XCLBIN | 244bb9411526c110cc3e36852c8118d1209fd1f986315b9998a5269a475e1071 |
| Both insts.bin | 74ba0e0fe1114a9480f566bd6b88f76d38b39903b3b70a9f71dc134a393794ca |

Tests: **9 new RED-to-GREEN cases**, full host suite **1522 passed / 47 skipped**.
The initial compile failure and all original builds are retained. Evidence is
under Models/qwen38-27b/prefill-paired-weights/: v2/*/build.json, resource ELF/
MLIR and dependency files, replay/report.json, captures, logs and validation-
evidence.json. Never edit these sources while reproducing a pinned run.

Next replay the complete FFN with the improved up/gate and the original down
path on captured causal inputs, preserving an exact control. Then check layer
outputs and other selected positions before integration. Four GEMM products
replace two for this projection; no throughput claim is made.
