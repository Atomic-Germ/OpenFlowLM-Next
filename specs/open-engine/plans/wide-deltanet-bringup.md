# Shared WideDeltaNet bring-up — 2026-09-24

Authoritative work order: user's `LLM_Coding_Agent_Plan.md`, Track A before B
(27B) before C (Flash-Next/qwen4exp). Branch: `implement-qwen38-27b-support`.
This report records the separate AB milestone. The subsequent
[A7 chain milestone](wide-deltanet-a7.md) now passes the inherited synthetic
acceptance gates, with the stricter head-local diagnostic limitation recorded
there. Neither model is supported yet.

## Actual compile/place findings

The phase-4 worker's declared 60032 bytes did not establish physical feasibility.
The phase-5 host schedule now issues bank-major weights separately from replayed
xn, with exact bounds/order tests at H=5120 and H=2560. Compiler probes show:

1. Whole `lx`, H=5120 and explicitly synthetic FFN=8192: `Number of elements
   to acquire 3 must be smaller than depth 2` in `xcommon.prep_bands`. This is
   the main GEMV xn FIFO, independent of glue and the real FFN=17408 L1 blocker.
2. Isolated actual glue worker: `tile (2, 3) requires 3 input/1 output DMA
   channels, but only 2 input/2 output available`. No fused xclbin was built.

Reproduce with `ironvenv/bin/python utilities/probe-qwen35-wide.py --scope layer`
or `--scope glue`. The utility writes spec, package versions and compiler logs
under `open_kernels/designs/layer_x/build_wide_probe/` (ignored). It extracts the
real worker, not a replacement numerical implementation. Synthetic FFN and
compile-only status are explicit in metadata. `OPEN_KERNELS_WIDE_GLUE_PROBE=1`
is confined to this diagnostic invocation; ordinary wide recipes report
`not implemented`, including with `OPEN_KERNELS_UNVALIDATED=1`.

## Implemented shared primitive

- `recipes/wide_deltanet.py`: explicit `WideDeltaNet`, bank width 32, derived
  bank count and active tails, xn chunks and value-to-key grouping. At both
  required widths: 16 key/48 value heads, dims128, banks `(0,32),(32,16)`.
- `designs/wide_deltanet/ab.py` and `ab_store.cc`: standalone two-input NPU
  dispatch, existing `glue_ab_e` and `glue_small_bank` arithmetic, private
  bf16 xn chunk of 2048 elements, reusable alpha/beta accumulators of 32 floats.
  Result contains four fp32 arrays of 48: alpha, beta logits, decay, beta.
  Both inputs come from DDR; results go to DDR. CPU prepares fixtures and
  computes the comparison oracle; there is no CPU computation fallback in
  the kernel path and no closed model kernel/library is used.
- Input stream: each bank's Wa, Wb, then a padded small-constants element.
  The weight stream uses one fill, xn uses four whole-vector replay fills,
  result one drain. `Pipeline(3)` throttles the fourth replay. There are two
  physical shim/core input channels, not one per logical fill.
- Existing `transpose_banked` is the fixed-32 packing primitive referred to
  as `transpose_banked32` in the new plan. Keeping its persisted opcode avoids
  unnecessary manifest churn. Python/C++ tests cover arbitrary hidden width,
  all active tails 1..32, H=2560/5120, exact bytes and destination canaries.

Upstream checked: [Qwen3.5 implementation](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_5/modeling_qwen3_5.py)
and [Qwen4Exp implementation](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen4_exp/modeling_qwen4_exp.py).
Both repeat query/key by the derived value/key head ratio. The open container
stores `A=-exp(A_log)`; the reference computes `exp(A*softplus(alpha+dt_bias))`
and `sigmoid(beta_logits)`. The downloaded 27B fixture remains pinned in the
[earlier report](qwen35-27b-bringup.md). No Flash-specific parser or QSA/PLE work
is included in this milestone.

## Hardware evidence

Strix aie2p 6x8, PCI0000:67:00.1, firmware1.1.2.64, XRT2.26.0
(`e9db9ab15f10173f8d2fc93ff92ab4c7eb09d2e6`), amdxdna2.26.0_20260817.
Python3.14.4; mlir-aie1.4.2; Peano21.0.0.2026080301+c9c5ecb7; NumPy2.5.3.
Installed from `ironvenv-requirements.txt`. Exact procedure is in the
[build skill](../../../.opencode/skill/open-wide-deltanet-ab/SKILL.md).

`utilities/test-wide-deltanet-ab.py` creates deterministic weights/inputs,
uses the production packer, and compares every output head against float64
math over identical bf16 inputs and f32 constants. Each width runs three
random dense vectors and impulses at xn offsets 0,2047,2048,H-1. Every result
buffer is poisoned with NaNs before dispatch. Inactive weight lanes are zero.
This checks chunk boundaries, both banks and all tails with fresh outputs;
the seven dispatches are **not** a recurrent-state test.

| Measurement | H=5120 | H=2560 |
|---|---:|---:|
| Successful NPU dispatches | 7/7 | 7/7 |
| Field comparisons | 28/28 | 28/28 |
| Worst max-absolute error / max-absolute reference | 1.684e-6 | 7.805e-7 |
| Smallest cosine | >0.999999999999 | >0.999999999999 |
| First dispatch, wall time | 0.979 ms | 0.795 ms |
| Subsequent dispatches, wall time | 0.480–0.530 ms | 0.407–0.446 ms |
| Core ELF text | 14368 B | 10960 B |
| Placed buffers + reserved stack | 21504 B | 21504 B |
| Core input/output DMA channels | 2 / 1 | 2 / 1 |
| Instruction stream | 1000 B | 1000 B |

Acceptance keeps dn_glue's existing thresholds: finite, max-relative <1e-4,
cosine >0.99999, per field. Times are standalone dispatch wall times, not model
latency or throughput claims. `input_with_addresses.mlir` places stack4096,
xn FIFO4096, weight FIFO8192, xn scratch4096, result768 and accumulators256
within offsets0..21504. No inferred 64-lane accumulator or hidden full-vector
scratch. Compile/place and real execution both succeeded.

Artifacts stay ignored in `open_kernels/designs/wide_deltanet/build_ab_h{5120,2560}`:
`final.xclbin`, `insts.bin`, `insts.elf`, `final.prj`, `build.log`, `run.log`,
`ab-fixture.json` (versions/geometry/hashes), `ab-results.json` (all metrics),
binary fixtures/results and `ab.cfg`.

Xclbin SHA256 for this run:

- H5120: `e51d92f7037ce73fe97420639de91866e9365b6298a9ff0c67e5212f851c6ec4`
- H2560: `384d3a4de93effddd6dd96dcddf9c06cb217e09ddddfaf9cfbc52c796ba57a46`

## TDD and regression

New geometry tests first failed on the absent capability; actual-worker tests
then failed on the absent AB design. Both pass after implementation. A failing
recipe test also preceded the explicit guard for the physically impossible
fused design. Existing legacy topology/packing tests remain in place.

Full `ironvenv/bin/python -m pytest specs/open-engine/tests -q`:
**668 passed, 47 skipped**. SciPy is present in ironvenv, so the earlier two
environmental vision failures also pass. Python tests execute the actual AB
worker with checked FIFO substitutes and exact integer arithmetic, and check
derived mapping for all 48 heads. They do not simulate AIE math or placement.
The expanded C++ `pools_test` also **PASS (0 failures)**, including both widths,
all 32 tail sizes and refusal before writing on malformed geometry. Reproduce
with the C++ build command in the earlier 27B report.

## Next stage after AB (completed in the A7 report)

1. Compose the separate AB outputs with a two-input conv/record dispatch;
   compare all 48 records including value-to-key mapping against reference.
2. Chain the existing open DeltaNet recurrence and validate persistent state
   over multiple tokens at H5120 and H2560. Only then close A7.
3. Resume Track B: resolve main-core xn buffering, segmented FFN
   8192/8192/1024, required primitive points, real-layer and 64-layer comparisons,
   converter/model packaging and LLM suite. Track C follows the stable 27B path.

No catalogue entries, q4nx builder registration, manifests or shipped model
assets were promoted. Follow the [A7 report](wide-deltanet-a7.md) for the current
record/state acceptance evidence and Track B next steps.
