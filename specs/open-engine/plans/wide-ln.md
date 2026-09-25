# Standalone wide RMSNorm — 2026-09-25

Track B5's **LN width=5120** primitive now passes hardware comparison at
epsilon 1e-6. Legacy widths 2048 and 4096 also pass. This is a prerequisite
for full-layer integration; B5 as a whole and model support remain pending.

## Blocker and implementation

Following `d045953`, an actual build of the previous `ln.py` at N5120 failed
basic-sequential buffer allocation. Its five-input acquire lowers to six
consumer buffers: 6144-byte stack + 10240-byte output + 6*10240-byte inputs
= **77824 bytes**, beyond physical tile memory. Evidence remains locally in
`/tmp/ln5120-baseline.log` and `ln/build_wide_baseline`.

The new path for N>4096 uses `ln_stream.py` for both the worker and matching
host schedule. Each fp32 half occupies one N*2-byte element. Input order is
x0, add0, x1, add1, w. One acquire at a time and depth-two input buffering
allow a private fp32 residual vector to replace retained input buffers.
Half0 resets the 32 statistics lanes, half1 continues accumulation, and the
last kernel normalizes the entire residual using their common inverse RMS.
Output order remains y0, y1, xn with a depth-one FIFO. The five-BO ABI is
unchanged. Three separate C++ translation units implement copy, add/squares,
and normalization, using the existing LN/vecmath arithmetic.

Widths <=2048 retain the fused kernel; 2048<N<=4096 retain split outputs.
Widths must be divisible by 64; the streamed helper rejects scratch above
60 KiB explicitly, including N6144. This is not a claim of hardware support
for every intermediate width. Recipe cache dependencies now include the new
helper for qwen35, dense and qwen36moe, and the standalone specialization
also covers `ln.h` and `ln_stream.py`.

No catalogue entry, model format, q4nx builder, or runtime manifest changed.
The existing recipe already supplies LN_N/LN_EPS to this design. No manual
model packing step, closed binary, or CPU neural fallback was introduced.

## Tests and reference comparison

TDD: three worker/schedule/budget tests failed before implementation, then
passed. Three cache-coverage cases then failed before dependency updates.
The acceptance utility's zero/nonfinite/outlier/mismatch test also failed
before implementation. All **7 new tests pass**. The initial focused
regression was 82 passed; final open-engine suite: **730 passed, 47 skipped**.

`utilities/test-wide-ln.py` prepares 12 inputs per width: three random pairs,
entry norm (add=0), exact cancellation, unequal half scales, ones, tiny
epsilon-sensitive values, impulses at half-1/half/last, and the first random
pair repeated after the others. Each call poisons outputs with NaNs and
checks trailing canaries. All fixtures, references, xclbin, instructions and
cfg are hashed; stale output files are cleared on preparation. FP64 reference
math follows the existing LN fixture, with fp32 residual and bf16 norm output.

The inherited `ln/compare.py` gates are unchanged: residual normalized max
error <1e-6; xn normalized max error <8e-3, cosine >0.999999, and bf16 bit
mismatches <N//20. Explicit zero/finite handling avoids an undefined cosine
for cancellation. The fp32 FFN's 1e-4 gate is not the bf16 LN gate.

| N | Max residual error | Max xn error | Min xn cosine | Max bf16 mismatches |
|---|---:|---:|---:|---:|
| 5120 | 8.917515e-8 | 4.048583e-3 | 0.9999999700774986 | 5/5120 |
| 4096 | 9.413480e-8 | 1.937984e-3 | 0.9999999887500519 | 4/4096 |
| 2048 | 8.764947e-8 | 2.732240e-3 | 0.9999999644514225 | 5/2048 |

All **36 dispatches / 72 tensor comparisons** pass. Guards remain intact;
repeating the first input gives identical output bytes at all three widths.
Existing vecmath addition is approximate: random residuals are within the
gate but are not bit-identical to correctly rounded IEEE addition. Entry,
cancellation, unequal halves, ones, tiny and impulse residuals are exact.

## Actual resource placement

N5120 `final.prj/input_with_addresses.mlir` on tile(0,2):

| Allocation | Start | Bytes |
|---|---:|---:|
| Stack | 0 | 6144 |
| Saved residual | 6144 | 20480 |
| Output element | 26624 | 10240 |
| Input element 0 | 36864 | 10240 |
| Input element 1 | 47104 | 10240 |
| Statistics | 57344 | 128 |
| **End** | **57472** | |

This leaves 3968 bytes within the conservative 61440-byte data budget.
ELF `.text` is 3200 bytes, with no `.data` section. Legacy rebuilt text sizes
are 3424 bytes at4096 and 3344 at2048. These measurements are for the
standalone primitive, not whole-layer placement. One input and one output
shim DMA channel are used. Each invocation transfers 51200 input bytes and
30720 output bytes at N5120 (five fills, two host drains).

## Reproduce

Exact workflow: [.opencode/skill/open-wide-ln/SKILL.md](../../../.opencode/skill/open-wide-ln/SKILL.md).
Build dirs under `open_kernels/designs/ln/`: `build_wide_stream`,
`build_regression_4096`, `build_regression_2048`. Each contains `ln.cfg`,
`ln-fixture.json` and `ln-results.json`. Build logs are `/tmp/ln5120-stream.log`,
`/tmp/ln4096-build.log`, `/tmp/ln2048-build.log`; hardware logs use the same
width with `-hardware.log`. Hardware execution was sequential on the host.

Toolchain: Python3.14.4, mlir-aie1.4.2,
llvm-aie21.0.0.2026080301+c9c5ecb7, XRT2.26.0
(`e9db9ab15f10173f8d2fc93ff92ab4c7eb09d2e6`). Device: Ryzen AI9 365,
Strix aie2p6x8, PCI0000:67:00.1, amdxdna2.26.0_20260817,
firmware1.1.2.64. No Docker or proprietary kernel library was needed.

SHA256 for the N5120 artifacts (generated outputs remain ignored):

```text
final.xclbin   5aa7d3e3b0773511353c93674d0147563b856eb45bc5ebe72165e1ce88102edd
insts.bin      29fe75a6fbfc46032ec6adfef5cbbdd97d4cdd7ae42a3c0c16de019211eef05e
ln-results.json 36d5dcaed5697de4bc17d22a7f158127cc9c6c9487f0e82d26e8fc57fefe9c14
```

## Remaining gates

Next B5 primitives: LM head K5120 and attention Q24/KV4/head_dim256/rotary64.
Then integrate separate AB/conv/recurrence with projections, post norm and
segmented FFN, measuring full-layer data and program memory. The existing
fused `lx` path still exceeds physical shim input DMA resources; its embedded
norm is not replaced by this standalone implementation. Recurrent state
layout also needs reconciliation: standalone recurrence stores128 rows per
head, whereas the layer uses padded140 rows. Do not silently reinterpret it.

No full layer, 8-layer slice, 64-layer run, model packing, chat test, or
benchmark was performed in this phase. The A7 head-local precision caveat
remains documented in [wide-deltanet-a7.md](wide-deltanet-a7.md).
