# Q4 block localization inside the real down segment

2026-10-04. Continue from [segment high/low traces](qwen38-down-segment-trace.md).
No production arithmetic or acceptance tolerance changes in this stage.

## Implementation

A standalone K4096/32-row probe calls the production `gemv_q4_tile` with the
same correction/product/block/segment carry flags. An opt-in, default-disabled
`GEMV_Q4_BLOCK_TRACE` hook records each 32-element block's local high/low result
and the subsequent persistent high/low accumulator. The output also includes
the tile-interface high/low values and the activation table's three BF16 sum
components. The read-only trace never feeds data back into arithmetic.

The probe copies the selected 32-row chunks bytewise from the original pool;
there is no weight requantization. It uses the exact BF16-rounded activation
from the preceding real FFN capture. One core and reduced diagnostic scratch
replace the full FFN composition; hardware equality of both final components
with the preceding full-width segment trace is therefore a mandatory check.

The retained format has 16 records of 1120 floats: 8 blocks x 4 planes x 32
lanes, 32 high values, 32 low values, then 8 high/8 low/8 tail activation sums
and 8 zeros. Block lanes and low interface lanes are even/odd-permuted; high
interface lanes become logical only on the final tile. The decoder tests pin
both orders. The earlier 1088-float build is superseded by `build_down_blocks/sums`.

## Measured result

Each selected 32-row segment ran real, zero, real. Every replay passes exact
repeat, zero-state reset, final segment-high equality and segment-low equality.

| Channel | Segment | Local block | Global K range | Local block error |
|---|---:|---:|---|---:|
| 4872 | 3 | 53 | 13984..14015 | +2^-32 |
| 392 | 1 | 107 | 7520..7551 | -2^-33 |
| 1769 | 3 | 53 | 13984..14015 | +2^-33 |

Each 128-block x 32-row capture has exactly one local block error. In all three
captures, accumulation of the captured block components has zero error against
their FP64 cumulative sum. The selected channel's final interface normalization
also has zero error. These blocks reproduce the complete error found in the
preceding segment trace.

Channels4872 and1769 share an activation block and have Q4 minimum factors
-0.03125/-0.015625. This suggested a possible activation-sum error. Capturing
the three sum components disproved it: **all 128 activation sums are exact in
each of the three cases**. Their minimum-times-sum error is zero. Do not infer
an activation-table sum defect from the proportional output errors.

The fault is now bounded to local weighted-block evaluation before persistent
block accumulation. The nine BF16 product components, integer dot/scaling and
local compensation still need individual measurement; this trace does not yet
identify the faulty instruction or justify changing their math.

## Tests, artifacts and limits

TDD: decoder/reference tests failed before implementation; the activation-sum
layout test also failed before extending the record. Four new tests cover lane
order, sum-component order, independent packed-weight FP64 block references,
hash/guard rejection and zero-state failure. Full suite: **881 passed,47 skipped**.

Rebuilt the original corrected down trace with block tracing disabled:
**66/66 synthetic checks pass**. All11 output arenas, all11 traces and insts.bin
are byte-identical to `build_down_trace/retained/down`. No default math changes.

| Artifact | Text | SHA256 xclbin | SHA256 instructions |
|---|---:|---|---|
| `build_down_blocks/sums` | 6288 B | `c9321325b8d1a9b62c430ae1b749114ad46ffa4dd27558681c5500270ffcc0d7` | `6ccd2f1cce0696688fdb86b60a4194a9caa6f56588a7d6fd4bed5c1d60b86ca0` |
| `build_down_blocks/regression/down` | 10016 B | `ed33fb5520a0407fa7df6c4d262e92247a40959acc19dc58862e126012bc2ce0` | `df238508cb3e1e8efd8b3838967e73cd3ab92e2f230ee1c6753b3a74f027588a` |

Toolchain: ironvenv Python3.14, mlir-aie1.4.2, llvm-aie21; XRT NPU harness.
Logs: `/tmp/down-block-*.log`. Real results:
`build_down_blocks/sums/channel{4872,392,1769}/blocks-results.json`.
The report's `passed` is diagnostic reproduction, not numerical acceptance.

Full-model references and binaries are unchanged; no new full-model run is
claimed. PR4 stays open with the preceding four slice failures. Next capture
the nine local products and compensation steps in block53/channel4872, and
verify the integer activation dot representation before selecting a correction.

Reproduction: [open-down-block-trace](../../../.opencode/skill/open-down-block-trace/SKILL.md).
