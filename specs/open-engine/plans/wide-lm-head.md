# Wide LM-head hardware gate — 2026-09-25

Following [wide-ln.md](wide-ln.md), Track B5's **LM head K5120** now passes
synthetic hardware comparison at the full **248320-token vocabulary**, on
8 cores. The existing Q8 kernel and host schedule required no change.
Catalogue promotion and whole-model support remain pending other gates.

## Changes and TDD

The old `lm_head_q8/make_test.py` reference hardcoded32 chunks per128-row
band (K2048); using it for wider inputs addressed wrong rows. The new
`references` derives the count as4*(K/256), validates the full byte length,
and decodes a bounded batch for several input vectors at once. The original
single-input `reference` API remains available. The captured-fixture CLI
now takes its K from `LMHEAD_K`.

Four initial test failures covered K2560/4096/5120 and missing batch support;
K2048 already passed. Tests encode signed Q8 codes and bf16 scales in the
documented file raster, call the real production `pack.apply_op(lmhead_q8)`,
and compare the oracle to an independently formed dense FP64 matrix product.
They exercise columns0/255/256/2047/K-1, multiple row quarters and output
bands, as well as random batched inputs and malformed lengths. A subsequent
fixture/canary test failed before adding the hardware utility. All **6 new
tests pass**; the full open-engine suite is **736 passed, 47 skipped**.

`utilities/test-wide-lm-head.py` generates complete128-row bands through the
same production pack operation, writes the synthetic pool incrementally and
computes FP64 references from a memmap. There is no full dequantized matrix
allocation. Fixture geometry, seed2963, artifact/weight/input/reference/cfg
hashes, NaN output poison and trailing canaries are retained in the build
directory. Re-preparation removes stale output dumps.

The production qwen35 recipe already builds this design with hidden/vocab
and packs its head with the matching width. No model/container format,
q4nx builder, kernel arithmetic, C++ library, catalogue entry, or runtime
manifest changed. This is its established Q8 **head** format; it does not
validate mixed-Q8 attention, DeltaNet or FFN projections.

## Hardware evidence

K5120 uses248320 output rows (1940 bands,243/242 per core) and a
1350860800-byte synthetic pool. Thirteen calls cover two random bf16 inputs,
ones, zero, impulses0/255/256/2047/2048/4095/4096/5119, then repeat-first.
All output rows are compared; no sampling or truncation is used.

Regression K4096 uses1152 rows (9 bands across8 cores), a5013504-byte pool,
and11 inputs (the same applicable boundaries). This exercises one two-band
worker and seven single-band workers; it is not a full-vocabulary regression.

| Geometry | Checks | Max normalized error | Min cosine |
|---|---|---:|---:|
| K5120/N248320 | Whole output | 4.372786e-6 | 0.9999999999910298 |
| K5120/N248320 | Per-core output | 5.142826e-6 | 0.9999999999909388 |
| K4096/N1152 | Whole output | 4.581833e-6 | 0.9999999999865823 |
| K4096/N1152 | Per-core output | 5.707915e-6 | 0.9999999999832018 |

All **24 dispatches / 216 comparisons** pass the inherited LM-head gate:
normalized max error <1e-4 and cosine >0.9999999. All impulse and zero
outputs are exact; guards remain intact. Repeat-first outputs are identical
bytes in both builds, verifying reset across unrelated invocations.

Harness run times for the full synthetic head were34.012–40.016ms. These
are individual kernel measurements, not a full-model prefill/decode
benchmark, and exclude fixture preparation and initial host weight upload.

## Actual resources

`build_wide_5120/final.prj/input_with_addresses.mlir`, per compute core:

| Allocation | Start | Bytes |
|---|---:|---:|
| Stack | 0 | 4096 |
| Weights0 | 4096 | 17408 |
| Weights1 | 21504 | 17408 |
| Activation table | 38912 | 11520 |
| Broadcast x | 50432 | 10240 |
| Result0 | 60672 | 512 |
| Result1 | 61184 | 512 |
| **End** | **61696** | |

Eight compute tiles occupy columns0–1, rows2–5. The8 weight streams and
broadcast activation use5 shim columns0–4; drains are split per worker.
Each core's ELF `.text` is2432 bytes; no `.data` section. The existing
standalone design fits physical64 KiB with3840 bytes free, but exceeds
the whole-layer recipe's conservative61440-byte budget by256 bytes.
Do not treat it as available placement for added layer scratch.

Per invocation:1350860800 weight bytes +10240 activation bytes from the host,
and993280 result bytes; nine fills and eight drains. Weights remain in one
host-visible BO between calls, not in AIE-local memory.

## Reproduce and provenance

Commands: [open-wide-lm-head skill](../../../.opencode/skill/open-wide-lm-head/SKILL.md).
Build directories under `open_kernels/designs/lm_head_q8/`:
`build_wide_5120` and `build_regression_4096`. Fixtures/results are
`lm-fixture.json` and `lm-results.json`, with `lm.cfg` and all output dumps.
Logs: `/tmp/lm5120-{build,prepare,hardware,compare}.log`,
`/tmp/lm4096-{build,prepare,hardware}.log`, `/tmp/wide-lm-suite.log`.

Same host/toolchain as [wide-ln.md](wide-ln.md): Strix aie2p6x8, Ryzen AI9
365, XRT2.26.0, amdxdna2.26.0_20260817, firmware1.1.2.64,
Python3.14.4, mlir-aie1.4.2 and llvm-aie21.0.0.2026080301+c9c5ecb7.
NPU executions were sequential; no closed binary or CPU runtime fallback.

SHA256, full K5120:

```text
final.xclbin 7a3b6bf6c93331e7691e8ef24b68a7923f7b70e5f1b6b041bac091881f5626f2
insts.bin 203044bbf77c58356c2ddacc513cb54b5cc55defe209072c2a3f999667cf2899
weights.bin 604da835dbedce3c06cce7642ae55a4b37e4731698abb5c14d96775b03914d0e
lm-results.json 0a9be7359325502d843b8c40cd9f379f419301d8b98731c73065ee548417fe0b
```

## Next gate

Attention Q24/KV4/head_dim256/rotary64 remains to be validated at the exact
tuple. Full-layer integration still requires replacing the fused wide glue
DMA schedule, using a fitting norm schedule inside or outside the layer,
and reconciling128-row standalone versus140-row padded recurrent state.
No 8-layer/64-layer model, converter packaging, chat test or full-model
benchmark is claimed. The A7 head-local precision caveat remains open.
