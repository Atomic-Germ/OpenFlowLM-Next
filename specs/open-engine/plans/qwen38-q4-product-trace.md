# Local Q4 product accumulation: measured cancellation error

2026-10-07. Continue from [block traces](qwen38-down-block-trace.md).
This stage identifies the operation responsible for the three layer0 down
differences. It adds instrumentation and a reproducible fixture, not a numerical
correction or model-support claim.

## Instrumentation

`build-products` builds a fresh standalone K4096 probe and records the selected
block plus kernel hashes in `product-trace.json`. The selected block records
all nine product terms: BF16 operand a, BF16 operand b, their FP32 product,
accumulator high and low after `q4_product_add`. The optional macro is disabled
by default and includes its block selection in the specialization hash.

The record expands from1120 to2560 floats per tile; output FIFO depth1 keeps
the diagnostic within L1. Unselected product records are explicitly zeroed.
The real/zero/real reset check ignores weight operands in the zero frame but
requires products, sums and the entire base trace to clear. The ninth product
trace's high/low must equal the earlier local-block trace in all32 lanes.

## Result on NPU Strix

For channels4872 and1769 (block53, K13984..14015), and channel392 (block107,
K7520..7551): all nine captured products equal independent FP64 a*b exactly;
their FP64 sum equals the independent packed-weight reference. No missing
operand tail or product error explains the discrepancy. The compensated sum
is exact through the first six terms. Error first appears on the seventh:
the `m * xsum_high` term. Its low compensation remains unchanged.

| Channel | High before term7 | Term7 | IEEE FP32 sum | Captured high | Error |
|---|---:|---:|---:|---:|---:|
| 4872 | 0.0038717917632311583 | -0.00390625 | -3.4458236768841743e-5 | -3.445800393819809e-5 | +2^-32 |
| 392 | -0.0017625255277380347 | 0.001953125 | 0.00019059947226196527 | 0.00019059935584664345 | -2^-33 |
| 1769 | 0.0010458918986842036 | -0.001953125 | -0.0009072331013157964 | -0.0009072329849004745 | +2^-33 |

Every expected high in this table is exactly representable in FP32, so ordinary
rounding of these two operands cannot explain the measured deviation. The
following two terms preserve the error. The existing native `aie::add/sub`
TwoSum sequence does not recover it. This identifies the failing numerical
operation; it does not establish which compiler lowering or hardware-internal
step causes it.

`tests/fixtures/q4_product_cancellation.json` retains the three actual operand,
high/low and expected-high cases. CPU checks pin exact representability, unchanged
low compensation and the measured error. The complete captured nine-term trace
is in each `blocks-results.json` under `build_down_products/block53` or `block107`.

## Validation and next implementation

TDD: product-layout and operand/compensation separation tests failed before the
new helpers, then passed. The third test uses the measured cancellation fixture.
The new trace preserves the preceding segment high/low bitwise, passes repeat
and zero reset, and matches the base block trace for all three real cases.
Full CPU suite: **884 passed,47 skipped**. Rebuilt corrected down with both
block/product hooks disabled: **66/66 synthetic checks pass**; all11 activation
arenas, all11 segment traces and insts.bin match `build_down_trace/retained/down`
bytewise. Regression xclbin SHA256:
`e2699e904222ba7306d2e34189411bb915d43e582067a11ea304c514fdfba719`.

Production references, weights and numerical gates remain untouched. No new
full-model run is claimed; the previous four slice failures remain open.

Next implement exact FP32 addition/error recovery for local Q4 product
accumulation behind an opt-in flag. Test all three captured cases, cancellation,
small tails and zero before an NPU replay. Native subtraction cannot be assumed
safe merely because its exact result is representable: these measurements
invalidate that shortcut for the observed add/sub path. Verify segment and full
FFN instruction size, default-mode regression and unchanged full-model acceptance.

| Probe | .text | SHA256 xclbin | SHA256 instructions |
|---|---:|---|---|
| block53 | 6896 B | `dd5c7aba3d69093bd7b1260488e7742425005dea5470257e89f832823e42c9d0` | `9b15d7a3e43d832a08c4ea718d000c0068c968e4470b231186625a4198913d07` |
| block107 | 6896 B | `401dbe7c8563f89a1497610f1536320499f7bc7806dee6601c60b017e86c7333` | `9b15d7a3e43d832a08c4ea718d000c0068c968e4470b231186625a4198913d07` |

Toolchain: ironvenv Python3.14, mlir-aie1.4.2, llvm-aie21, XRT NPU Strix.
Reproduction: [open-q4-product-trace](../../../.opencode/skill/open-q4-product-trace/SKILL.md).
Logs: `/tmp/down-product*.log`.
