# Segmented dense FFN — 2026-09-24

Historical stage results below. The subsequent
[precision stage](dense-ffn-precision.md) resolves the reported full-FFN failure
and passes13 synthetic inputs with and without up/gate tracing.

Based on `51ba7f6`, Track B B2–B4 implementation and down-projection validation.
**Segmented down passes hardware comparison. Full FFN end-to-end acceptance
remains open:** one synthetic output has normalized error1.123084e-4 against
an unchanged1e-4 limit. This is not whole-layer or model support.

## Implementation

The dense recipe selects segmentation only when the full activation table
cannot fit the existing61440-byte L1 budget even at one Q4 chunk per weight
FIFO element. Fitting legacy geometries, including K12288, retain their full
table, weight-element size and worker path. The MoE path is unchanged.

For H5120/FF17408 the segments are `(0,8192)`, `(8192,8192)`, `(16384,1024)`.
`recipes/segmented_dense.py` computes generic segments and contiguous byte
slices inside each64-row band of the existing standard Q4 matrix. No pack
format, Q4 decoder, model-name branch or K17408 catalogue point was added.
The q4nx packer's existing up/gate/down operations remain sufficient; this
stage does not export a model or require a new manual conversion step.

`layer_x/xcommon.py` streams each h segment once and prepares it with the
existing f32-to-bf16 table kernel. It traverses all ten bands per core before
moving to the next segment. Existing `gemv_q4_gms` computes each segment band
into ms. Two small generated vector kernels add/copy its64 results into ds
and emit completed bands. All640 per-core partial sums remain in ds, whose
DeltaNet use has ended before FFN. The first segment overwrites every slot
on every token, avoiding stale/NaN accumulation. Attention-only recipes reserve
sufficient ds storage even though they have no DeltaNet work.

`ffn_body` and `ffn_sequence` integrate this path after the existing
up/gate/SiLU stages. DMA drains precede weight fills, weight transfers use
Pipeline(3), and each segment's input transfers complete before the next.
No CPU neural computation occurs in the NPU path; host math is comparison only.
Mixed Q8 is explicitly not implemented for segmentation. The recipe now has
an explicit unvalidated guard for segmented FFN, independent of whether each
individual GEMV width already exists in the catalogue.

## TDD and regression

Baseline affected tests:96 passed. Added geometry/packing tests first:
15 failures, then15 passes. Added actual worker/schedule tests:2 failures,
then17 passes. Additional tests cover invalid ranges, attention-only scratch,
recipe segment checks, and an acceptance guard (observed red before its fix).

Tests execute the actual worker functions using bounded FIFO models. They
check the segment-major call order, all bands, reset across two tokens, one h
read per segment, drains-before-fills, and exact one-time coverage of every
original pool chunk. Existing dense activation tests now supply the new empty
segment field for their legacy FFN fixture.

```text
ironvenv/bin/python -m pytest specs/open-engine/tests -q
718 passed, 47 skipped

test_segmented_dense.py: 24 passed
```

## Hardware validation

Three isolated programs use actual layer_x buffers, kernels and schedules on
all eight cores. Geometry is the real H5120/FF17408, weights are synthetic Q4.
Same Strix/XRT/Peano environment as [dense-wide-input.md](dense-wide-input.md):
Python3.14.4, mlir-aie1.4.2, llvm-aie21.0.0.2026080301+c9c5ecb7,
NumPy2.5.3, XRT2.26.0, amdxdna2.26.0_20260817, firmware1.1.2.64.
Runs used the open harness directly on the host NPU. Three xclbins were built,
no libraries or model packages. No catalogue promotions.

The down tests use seed38417: two random f32 inputs, ones, zero, impulses at
0/8191/8192/16383/16384/17407, and a repeat of the first random input. CPU
reference uses float64 math over the exact pool bytes and bf16-rounded input,
matching the existing f32 table-prep contract. It indexes original matrix
coordinates independently of the segment DMA helper. Every output starts
poisoned with NaNs;64-byte trailing canaries and exact transfer lengths pass.

| Comparison | Checks | Maximum normalized error | Minimum cosine |
|---|---:|---:|---:|
| Partial after K8192 | 11/11 | 6.116402e-6 | 0.9999999999854 |
| Partial after K16384 | 11/11 | 4.818076e-6 | 0.9999999999895 |
| Partial after K17408 | 11/11 | 4.633203e-6 | 0.9999999999901 |
| Production down, final only | 11/11 | 4.633203e-6 | 0.9999999999901 |

Thresholds: max(abs(got-ref))/max(abs(ref)) <1e-4, cosine >0.9999999,
finite values; zero-reference output requires exact equality. No thresholds
were relaxed. Final-only outputs are bit-identical to the final diagnostic
snapshots for all11 inputs. Repeating the first input is bit-deterministic
in both programs. Total:22 successful down dispatches,44 numerical gates.

### Full FFN accuracy limitation

Two additional NPU dispatches run the actual full FFN, including assembling h
in DDR and reading it back through segmented preparation. Both dispatches
complete and both h comparisons pass. The complete output gate is **not passed**:

| Input | h max-relative | FFN output max-relative | FFN output gate |
|---|---:|---:|---|
| 0 | 4.997904e-6 | 4.096821e-5 | PASS |
| 1 | 1.092418e-5 | 1.123084e-4 | FAIL |

The first observed differing tensor is h (within its gate). After rounding to
bf16,149/130 values differ from the rounded independent h reference. Example
for input1, h[68]: NPU0.1674811095 vs reference0.1674791723, which round to
0.16796875 and0.1669921875. The down reference computed from **device-produced
h** passes at2.624785e-6 /9.984182e-6. This localizes the larger output difference
to upstream h differences crossing bf16 boundaries; it does not justify
substituting device h for the end-to-end reference.

The comparator keeps the failing independent output in `checks`, leaves
`passed: false`, exits nonzero, and reports the localized comparison separately
in `diagnostics`. No seed, weights, tolerance, or reference was changed to hide
the failure. The existing up/gate/SiLU kernels and their approximation math
were not modified. Full-block accuracy needs further work before acceptance.

## Resources and artifacts

Actual compiler placement includes18432 B table,512 B ms,5120 B ds,
two10240 B weight elements, two4096 B activation elements, two256 B outputs
and6144 B stack: **59392 B/core** for all three designs. No FIFO depth increase.

| Program | ELF text per core |
|---|---:|
| Diagnostic segmented down | 6976 B |
| Final-only segmented down | 6064 B |
| Full FFN | 11296 B |

These are isolated FFN programs; combined attention/DeltaNet layer text size
has not been established. Fused wide glue still has its known3-input/2-channel
DMA blocker and requires integration of the separate A7 chain.

Commands and constraints are in the new
[build skill](../../../.opencode/skill/open-segmented-dense/SKILL.md).
Ignored artifacts under `open_kernels/designs/layer_x/build_segmented/`:
`down/`, `down_final/`, `ffn/`, each retaining0/1-build.log, toolchain/spec
metadata, final.prj, xclbin, instructions, original packed pool, input/reference
fixtures, segmented.cfg, run.log, compare.log, segmented-fixture.json and
segmented-results.json. The comparison utility verifies fixture hashes.

Validated/run xclbin SHA256:

- down: `cc64de61d99c7a6d6b85c93768f55987961142deead90eca6c0f7b491d9880e0`
- down_final: `02f8f3489e88a8fe812bc8624ba2797dfd314866f7e161b0e4f7f1514754b0b1`
- ffn (failed strict output gate): `06c3460f4ce69ea6b4d4bdeedec97eeb61370a40fd1ad8cf2e7a9dc3ea8ab33d`

Next: resolve the full-FFN accuracy gate with first-diverging-tensor analysis,
then integrate the separate DeltaNet chain and finish the remaining wide
normalization/LM-head/attention gates before real layers and model validation.
