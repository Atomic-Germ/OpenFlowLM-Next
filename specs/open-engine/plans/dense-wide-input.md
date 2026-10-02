# Dense wide activation streaming — 2026-09-24

Track B prerequisite and partial B5 primitive validation, based on `3d26ab5`.
The main-core depth-two FIFO can now consume three or more activation chunks.
Q4 projections at K5120 and K6144 pass hardware comparison. Segmented FFN,
whole-layer integration and full-model support remain pending.

## Implementation and TDD

`layer_x/xcommon.py` prepares wide xn/xm one 4096-byte bf16 chunk at a time,
releasing each chunk immediately after preparing its table range. GEMV reads
the prepared table. Existing one/two-chunk paths keep their acquire/release
ordering. FIFO depth, generated C++ math and packed weight representation are
unchanged. The FFN's existing streamed f32 h input is also unchanged.

`test_dense_activation_stream.py` executes the actual Python worker functions
against a depth-two FIFO model. It checks all prepared blocks, input lifetime,
and the transition from xm through up/gate to h/down. Before implementation:
**4 failed, 6 passed** (three wide projection cases and H5120 xm). Afterwards:
**10 passed**. Full regression:

```text
ironvenv/bin/python -m pytest specs/open-engine/tests -q
694 passed, 47 skipped
```

The full synthetic FFN8192 layer probe passes the former `acquire(3)` failure
and reaches physical placement. It still fails on tile (2,3): three input DMA
channels requested, two available. The separate A7 chain still needs layer
integration. FFN xm is unit/IR validated here; its complete up/gate/down path
was not run on hardware in this stage.

## Hardware and reference

`layer_x/projection_probe.py` uses the actual `X.prep_bands`, Q4 GEMV functions,
and main-core scratch (`tab`, `ms`, `ds`) on eight cores. Two output bands per
core give N1024, exercising resets between bands. The synthetic FFN8192 keeps
the full 8192-wide table allocation. This is a diagnostic projection program,
not a model kernel. The specialization hash includes source and geometry.

`utilities/test-dense-projection.py` uses the existing Q4 packer and computes
the CPU reference from those exact packed bytes and bf16 inputs. Seed2961;
nine inputs per width: two random, ones, zero, and impulses at indices2047,
2048,4095,4096,K-1. K5120's unused final-chunk padding contains NaNs. Output
buffers are poisoned before every dispatch and followed by64-byte canaries.
All outputs are finite, all canaries intact, and all lengths match exactly.
CPU reference data is never loaded into the NPU output or computation.

| Check | K5120 | K6144 |
|---|---:|---:|
| Successful NPU dispatches/checks | 9/9 | 9/9 |
| Maximum normalized error | 5.884593e-6 | 4.782513e-6 |
| Minimum cosine | 0.9999999999856 | 0.9999999999807 |
| Zero and five impulses | exact | exact |
| Placed buffers + stack per core | 59392 B | 59392 B |
| ELF text per core | 4112 B | 3568 B |

Acceptance uses the existing Q4 thresholds: max(abs(got-ref))/max(abs(ref))
<1e-4 and cosine >0.9999999. Zero-reference output requires exact equality.
No numerical threshold was weakened.

Resources are from `final.prj/input_with_addresses.mlir` and `llvm-size`, not
just estimates. Each core has6144 B stack,18432 B table, two10240 B weight
elements,5120 B ds, two4096 B activation elements,512 B ms and two256 B output
elements. This fits the61440 B recipe budget with FIFO depth two.

Environment: Strix aie2p6x8, Ryzen AI9 365, PCI0000:67:00.1; XRT2.26.0
(`e9db9ab15f10173f8d2fc93ff92ab4c7eb09d2e6`), amdxdna2.26.0_20260817,
firmware1.1.2.64. Python3.14.4, mlir-aie1.4.2,
Peano21.0.0.2026080301+c9c5ecb7, NumPy2.5.3. Execution used the open XRT
harness directly on the host NPU. Two xclbins were built; no libraries or model
packages were built. No catalogue entries were promoted.

## Reproduction and artifacts

Commands and build constraints are in the new
[skill](../../../.opencode/skill/open-dense-activation-stream/SKILL.md).
Build directories are ignored:

```text
open_kernels/designs/layer_x/build_streamed_xn/
  0-build.log, 1-build.log                 # full-layer diagnostic failure
  projection_k5120/
  projection_k6144/
```

Each projection directory retains build logs, toolchain/spec metadata,
`projection.cfg`, binary fixtures, `run.log`, `compare.log`,
`projection-fixture.json` (kernel/instruction/weight/program hashes) and
`projection-results.json` (all numerical checks). Compile-only metadata keeps
`hardware_validated: false`; the separate results file records the actual run.

Validated xclbin SHA256:

- K5120: `99ef54ddf2621bd984bfde8a6c36282c1a0eb205a05505aa9294ea4c98b401a4`
- K6144: `e4c93d4518900b9124544f7b6a434ecaf35e748b6d7eada7601b170125e2fb8e`

Subsequent stage: [segmented dense FFN](segmented-dense-ffn.md) implements and
validates the down projection; its full-FFN accuracy gate remains open.

Next at the time of this stage: segmented down GEMV8192+8192+1024, with segment-major scheduling across
output bands, local partial accumulation and unchanged packed matrix layout;
integrate the separate AB/conv/recurrence chain. LN5120, LM head5120, attention
24/4/256/64, real layers and model validation remain outstanding. These tests
do not establish mixed-Q8 support or change the A7 head-local precision caveat.
