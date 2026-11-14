# Isolated causal up/gate replay

Replay the retained hybrid L63/P7 FFN projection without full-model residency.
The existing production `gemm_n34816_k5120` kernel receives the same up/gate
pool bytes and captured FP32 norm. Two BF16 dispatches reproduce the private
paired-input route; their outputs are added in FP32 on the host.
No production source, kernel or model weights are changed.

## Build and prepare

Use fresh output directories. The standard harness is built with CMake Release
and the local XRT SDK; evidence pins its source, compiler dependencies, cache,
compiler version and executable. It needs no new XCLBIN.

```bash
mkdir -p Models/qwen38-27b/prefill-upgate-replay
OPENBLAS_NUM_THREADS=1 ironvenv/bin/python utilities/prefill-upgate-replay/check.py build \
  --out Models/qwen38-27b/prefill-upgate-replay/harness \
  --xrt-include /home/prihlop/sources/xdna-driver/xrt/build/Release/opt/xilinx/xrt/include \
  --xrt-lib /home/prihlop/sources/xdna-driver/xrt/build/Release/opt/xilinx/xrt/lib

OPENBLAS_NUM_THREADS=1 ironvenv/bin/python utilities/prefill-upgate-replay/check.py prepare \
  --out Models/qwen38-27b/prefill-upgate-replay/run \
  --capture Models/qwen38-27b/prefill-sublayer-trace/run-257 \
  --capture-build Models/qwen38-27b/prefill-sublayer-trace/verified \
  --baseline Models/qwen38-27b/prefill-residual-trace/run-257 \
  --kernels Models/qwen38-27b/production-consistent/kernels \
  --pools Models/qwen38-27b/production-full/pools \
  --original Models/qwen38-27b/production-consistent/input-sha256.json
```

Preparation revalidates the sublayer provenance chain, full baseline logit/
residual identity and the complete export's binary hashes. It independently
repacks the manifest's two FFN operations and requires byte equality with the
previously hashed L63 pool. Dequantized packed weights must equal the source
container's effective matrices elementwise. This checks effective local weights,
not original upstream model conversion.

The captured norm occupies slot 7 in a T256 block; all other rows are zero.
The second layout moves the same input to slot 135, across the token-tile/core
boundary. Both outputs must reproduce the original native capture bytewise
before any diagnostic is accepted. That proves the zero-row substitution for
this selected result; it is not general dense-prompt coverage.

## Run

Run serially with exclusive NPU access:

```bash
LD_LIBRARY_PATH=/opt/xilinx/xrt/lib OPENBLAS_NUM_THREADS=1 \
ironvenv/bin/python utilities/prefill-upgate-replay/check.py run \
  --out Models/qwen38-27b/prefill-upgate-replay/run \
  --build Models/qwen38-27b/prefill-upgate-replay/harness
```

Four dispatches run in separate processes: high/low input at each slot.
Each process holds approximately **0.1395 GiB** of logical weight/input/output
buffers, excluding XRT overhead. CPU preparation holds one projection's unpacked
weights at a time. The model is memory mapped, not loaded resident on the NPU.

The high dispatch receives `BF16_RNE(x)`. Low starts with the FP32 subtraction
`x - BF16_RNE(x)` and uses the same BF16 tiler. Add the two output arrays in
FP32; FP64 addition would not reproduce native arithmetic. Require complete,
finite output arrays and zero inactive columns in each individual dispatch,
even when their errors could cancel in the sum.

## Interpretation

Report separate up and gate errors against:

1. Exact effective Q4 weights and the original FP32 norm (primary reference).
2. The same weights and the sum of the two BF16 input parts.
3. BF16-rounded weights and the paired input (rounding diagnostic).

Also compare each high/low dispatch to BF16-rounded weights independently.
`gemm_q4_dequant.h` emits BF16 dequantized weights; the third reference tests
the effect of this boundary. It must never replace the primary reference.
Conditioned projection gates stay `maxrel <0.005` and `rel_fro <=0.005`.
`control_valid` means identity and provenance passed, not full-model acceptance.

An exact sparse control and small residual against BF16-rounded weights can
localize a rounding contribution without proving an isolated correction will
improve propagated model outputs. Earlier FFN attribution showed that up and
down errors partly cancel. Preserve the independent references and repeat
full causal gates after any eventual production change.

Old report success is invalidated on any exception. Expected dumps are removed
before dispatch, and reports pin logs, output hashes, preparation and build
evidence. Artifacts remain in the ignored model directory.

```bash
OPENBLAS_NUM_THREADS=1 ironvenv/bin/python -m pytest -q \
  specs/open-engine/tests/test_prefill_upgate_replay.py
```

## Measured result — 2026-10-20

**4/4 NPU dispatches complete**. Both slot layouts reproduce all 34816 captured
up/gate values bytewise; inactive columns are exactly zero in each high/low
output. Packed pool identity and effective source-weight identity both pass.

| Reference / diagnostic | Up maxrel | Gate maxrel |
|---|---:|---:|
| Exact weights, FP32 input (primary) | 0.0019482680 | 0.0002586737 |
| Exact weights, paired BF16 input | 0.0019495638 | 0.0002587289 |
| BF16-rounded weights, paired input | 0.0000011273 | 0.0000015830 |
| CPU weight-rounding effect relative to exact weights, paired input | 0.0019497818 | 0.0002588257 |

Results are identical across slots. Primary up/gate rel_fro is
**0.0004853703 / 0.0006707061**. The primary gates already pass; the prior vector
attribution showed how this local error contributes to a failing layer output.
BF16 weight rounding accounts for most of the measured projection discrepancy
on this row. The remaining difference against the rounded-weight CPU product
is much smaller. This distinguishes a weight-rounding contribution from
incorrect effective packing for the tested projection; it is not a general
proof about every GEMM or every input.

Next build an isolated higher-precision weight-dequantization/GEMM candidate,
retaining this exact control and original-input/exact-weight reference. Test
the candidate before integrating it or drawing a propagated-accuracy conclusion.
Full-model long-prefix accuracy remains **FAIL (558/774 retained logit gates)**;
no production correction was made by this stage.

Harness: GCC **15.2.0**, CMake Release, **326 compiler dependencies**,
SHA256 `acd59b215f0276e80c667e8e3cc9c67bd24cabdbde38342073ee82c7887ca74d`.
Existing 41-set export build key:
`ac29fbcd21859648ff9d59f79e77f018b99daec0888f77a693e7ff18b2237032`.
No new kernel set, ABI or packing format. The harness has no CTest target;
its verification here is the four completed native dispatches and exact
capture identity. Host tests: **1513 passed, 47 skipped**, including **13 new
RED-to-GREEN cases**. Evidence is in `run/audit.json`, `run/report.json`,
`harness/build-evidence.json`, test/build logs and `validation-evidence.json`
under `Models/qwen38-27b/prefill-upgate-replay/`.
