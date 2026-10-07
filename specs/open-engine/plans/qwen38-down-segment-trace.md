# Real down-segment trace

2026-10-04. Follow-up to [the residual boundary](qwen38-residual-boundary.md).
This stage isolates the remaining layer0 FFN down error; it does not change
production arithmetic or claim PR4 acceptance.

## Implementation and TDD

Added a down-only trace using the existing segmented worker, packed weights,
scratch, five K4096 segments (last K1024), block carry and final RNE. Each band
records segment high/low and accumulated high/low through a separate output
FIFO. The low Q4 plane is restored to logical row order before capture. SiLU
is not linked, leaving instruction space for the next isolated experiment.

The first two tests failed before the decoder/analysis existed, then passed.
Additional tests reject malformed traces, nonfinite data, wrong CLI scope,
tampered references, damaged guards and non-deterministic replay. A synthetic
cancellation case distinguishes losing a segment's low component from losing
it during reduction. Default segment references still use K8192; corrected
down probes derive their partition from recorded K4096 metadata.

## Hardware evidence

The synthetic trace covers 11 inputs, including zero, ones, sparse boundary
inputs and a repeated first input. All 66 checks pass: 55 cumulative outputs
and 11 trace/output reconstructions. Guard and repeat checks also pass.

The real replay takes the exact captured h from
`build_residual_precision/final_rne_layer0`. Both repeated outputs and traces
are byte-identical. h is unchanged, reconstructed trace equals output, and the
final 5120-float vector is byte-identical to the full FFN. The same three FP32
differences remain. This establishes that isolation preserves the error.

The default uncorrected K8192 down probe was also rebuilt and passes all 33
checks on its original 11 inputs. The retained trace source is `down_trace.cpp`:
unlike generated `layer_x/*.cc`, it is tracked, and its content enters the probe
source hash. The retained build/replay reproduces the first diagnostic result.

| Artifact | SHA256 xclbin | SHA256 instructions |
|---|---|---|
| `retained/down` | `bfb4d424a3fccd6f134b4fcbed6be9d0a6853352d2f36b733e2ade4df799cfa7` | `df238508cb3e1e8efd8b3838967e73cd3ab92e2f230ee1c6753b3a74f027588a` |
| `default/down` | `78cfcb22c71c79303e2d234afbc0fed40c154336dd51c0eb3f8ab41e1fef8076` | `c4bd8183b72ef9c16596a6d9e9fca4e794619c11872b50feb3bfd6331629aea6` |

| Channel | Segment (zero-based) | K range | Segment high+low error | Cross-segment reduction error |
|---|---:|---|---:|---:|
| 4872 | 3 | 12288..16383 | +2.3283064365386963e-10 (2^-32) | 0 at all five stages |
| 392 | 1 | 4096..8191 | -1.1641532182693481e-10 (-2^-33) | 0 at all five stages |
| 1769 | 3 | 12288..16383 | +1.1641532182693481e-10 (2^-33) | 0 at all five stages |

The other four segments of each selected channel agree with the independent
FP64 partials. Even FP64 summation of the captured segment high/low values
rounds to the same wrong FP32 output. In particular, channel4872 remains
0.010909566655755043 instead of 0.010909565724432468. Replacing the segment
reducer alone therefore cannot close this boundary.

## Scope and next step

Trace text is 10016 B per core, below 16384 B. The preceding full FFN trace
was 15552 B; fitting this diagnostic does not establish room for a production
correction. Full-model fixtures, tolerances and binaries are unchanged. No new
full-model run is claimed: the last result remains 18892/18896 slice checks and
4122/4122 decode checks, with combined acceptance failing.

Next trace Q4 blocks inside K12288..16383 for channel4872, including block
product error and high/low renormalization. Test a correction there before
spending the full FFN's instruction budget. Do not pursue a larger segment
reducer based on the now-disproved explanation for these three channels.

Reproduction: [open-down-segment-trace](../../../.opencode/skill/open-down-segment-trace/SKILL.md).
Artifacts are under `open_kernels/designs/wide_deltanet/build_down_trace`;
logs are `/tmp/down-trace-*.log`. CPU suite: 877 passed, 47 skipped.

Follow-up: [Q4 block traces](qwen38-down-block-trace.md) locate the channel4872
error at block53/K13984..14015. The captured activation sum and persistent
block reduction are exact; investigate local weighted-block evaluation next.
