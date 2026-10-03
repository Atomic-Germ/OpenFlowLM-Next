# Wide attention primitive — 2026-09-25

Track B5 attention **Q24/KV4/head_dim256/rotary64** now passes synthetic
hardware comparison through position2048, including all query heads and
eight sequential tokens. The existing production arithmetic and worker body
needed no change. This follows [wide-lm-head.md](wide-lm-head.md).

## Implementation and TDD

The old standalone `attn.py` fixes Q16/KV2 and a separated K/V cache. It
cannot test this tuple by changing an environment variable. Instead,
`wide_probe.py` takes geometry from the real `qwen36moe.attn(ModelSpec)`
used by qwen35, and `probe_support.py` extracts `ax._attn` through AST.
The exact worker consumes the same broadcast stream and calls the original
`attn_meta/q/k/v/init/step/step_new/fin` kernels. No attention math or Q4
decoder was duplicated. Six workers own four heads each; GQA groups contain
six query heads, so several KV groups cross worker boundaries.

The diagnostic uses six BOs and row-interleaved bf16 KV storage, matching
the production worker's stream. It places workers at(c,2); the full layer's
(2+c,3) placement and surrounding workers are not built here. A fixed `nf`
controls the input transfer length; `pos` is supplied per invocation. Thus
this gate does not test the runtime's driver-patched DMA lengths or offsets.

New tests first failed for missing worker extraction/schedule (3), missing
FP64 reference (2), and missing device-cache fixture (1). A failure-path
serialization test then exposed a NumPy boolean in the metric result, fixed
by converting its relative error to a Python float. All **7 new tests pass**.
The complete open-engine suite: **743 passed, 47 skipped**.

Unit tests execute the actual worker with controlled FIFO endpoints for
Q24/KV4 and legacy Q16/KV4. They check stream depletion, cache-row ownership,
global head offsets, and the correct gate elements for every worker. The
schedule test covers all output head ranges and the interleaved window.
Reference tests cover grouped values, masked future NaNs, sigmoid gating,
half-split RoPE over only64 of256 dimensions, and zero/nonfinite metrics.

## Independent reference and fixture

`utilities/wide_attention_reference.py` implements FP64 head RMSNorm,
partial RoPE using the supplied fp32 cos/sin values, GQA, stable softmax,
and sigmoid output gating. Projected q/g/k/v inputs are fp32 and effective
norm weights are bf16. New K/V are rounded to bf16 before joining the cache,
as the device does. The unrounded FP64 attention output is compared as an
fp32 reference against the device's bf16 result. This is test tooling only.

`utilities/test-wide-attention.py` hashes artifacts, cfg, inputs and
references, clears stale outputs, poisons outputs and guards their tails.
Cold cases use random cached K/V, with **all future rows NaN**. Additional
cases distinguish all four KV groups, saturate alternating head gates at
±20, and use zero values. This catches masking/grouping errors that a single
whole-tensor random comparison could miss.

Each build also runs eight warm tokens. Only byte copies of device-produced
K/V update the device cache. The CPU independently evolves its ideal bf16
reference cache, never feeding reference state to the device. Final cache
bytes must equal the concatenated device row dumps plus untouched future
rows/guard. Repeating the first input after the warm chain must reproduce
the original output bytes.

## Hardware evidence

| Build | Geometry | Streamed rows | Calls |
|---|---|---:|---:|
| build_wide_257 | Q24/KV4, HD256, ROT64 | 257 | 23 |
| build_wide_2048 | Q24/KV4, HD256, ROT64 | 2048 | 26 |
| build_regression_16_4 | Q16/KV4, HD256, ROT64 | 257 | 23 |

Cold positions include0/1/2/15/16/17/63/64/255/256 and the last streamed
row count. The longer build additionally tests1023/1024/2047. Every build
includes8 warm tokens and the three semantic cases plus repeat-first.

The inherited `attn_chain/compare_attn.py` thresholds are unchanged:
normalized max error<1e-2 for K/V and<2e-2 for gated output, cosine>0.9999.
This probe additionally applies the output gate to each query head, requires
bit-exact fp32→bf16 V conversion, and explicitly handles zero vectors.
The bf16 output gate differs from the fp32 GEMV/FFN gate.

| Build | Max K error | Max output error | Max head-local error | Min head-local cosine |
|---|---:|---:|---:|---:|
| Q24/nf257 | 2.463055e-3 | 3.521631e-3 | 3.982140e-3 | 0.9999980383659497 |
| Q24/nf2048 | 3.875969e-3 | 3.424391e-3 | 3.648784e-3 | 0.9999981342108490 |
| Q16/nf257 | 4.784689e-3 | 3.575209e-3 | 3.705125e-3 | 0.9999978182984419 |

All **72 dispatches,216 whole-tensor checks and1544 head-local checks pass**.
V is bit-exact in every call. Output guards, final device-cache byte checks
and repeat-first comparisons pass for all three builds. No tolerances were
relaxed to accept these results. The independent DeltaNet A7 head-local
precision caveat remains separate and unresolved.

## Actual resources

Q24 `final.prj/input_with_addresses.mlir`, each compute tile:

| Allocation | Start | Bytes |
|---|---:|---:|
| Stack | 0 | 6144 |
| Split query hi/lo | 6144 | 24576 |
| Local output accumulator | 30720 | 4096 |
| Two output elements | 34816 | 4096 |
| Three lowered input buffers | 38912 | 6144 |
| New K, V | 45056 | 4096 |
| Temporary norm head | 49152 | 1024 |
| qn, kn | 50176 | 1024 |
| cos/sin | 51200 | 256 |
| Softmax max/sum | 51456 | 256 |
| Position block | 51712 | 16 |
| **End** | **51728** | |

The source input FIFO depth4 lowers to three consumer buffers here. The
actual end leaves9712 bytes against the conservative60 KiB data budget.
Each Q24 core's `.text` is14400 bytes, with no `.data`; margin against the
historical16 KiB program budget is1984 bytes. Q16 regression ends at43536
data bytes/core with13872 text bytes. These are isolated-worker results,
not full-layer placement evidence.

The host makes6 fills and7 drains at Q24. Input bytes per call are
4096 meta +49152 q/gate +8192 k/v +4096*nf cache:1114112 bytes at nf257,
8450048 bytes at nf2048. Output is4096 new-cache bytes +12288 gated output
bytes. The warm harness additionally copies4096 device bytes into the cache
between calls. No CPU attention computation runs inside the dispatch loop.

## Reproduce

Use [open-wide-attention skill](../../../.opencode/skill/open-wide-attention/SKILL.md).
Generated builds/fixtures/results remain in ignored directories under
`open_kernels/designs/attn/`. Each contains `attention.cfg`,
`attention-fixture.json`, `attention-results.json`, output row dumps and
`warm-cache.bin`. Logs are `/tmp/attn-{wide,long,regression}-{build,hardware,compare}.log`
and `/tmp/wide-attention-suite.log`.

Same host/toolchain as [wide-ln.md](wide-ln.md): Ryzen AI9 365, Strix aie2p,
PCI0000:67:00.1, XRT2.26.0, amdxdna2.26.0_20260817, firmware1.1.2.64,
Python3.14.4, mlir-aie1.4.2 and llvm-aie21.0.0.2026080301+c9c5ecb7.
Hardware runs were sequential with the open XRT harness. No closed library
or converted model package was used.

SHA256 for Q24/nf2048:

```text
final.xclbin 0a54f31747899abffc35b9838cb19e712828b668c715bcc04172a709e836b182
insts.bin 2b91d61036ab1697e274508c99f92f5730cdd875fc2f16655e211c8e7074b883
attention-results.json 190941358e92a5bdd41165b49ccd4ede99509168295d45abc49abdb6990ced45
```

Q24/nf257 xclbin:
`deedb0dff5d91402b8ec0c45e02bc4e98472a4ef42c9605a8a865c70d64b2041`.
Q16/nf257 xclbin:
`8aee215f7ac8c509c79ee674a442d2f61cf2ddbf5c7a381ade65f4c9af3a7def`.

## Next stage and limitations

The remaining listed B5 attention primitive now has standalone synthetic
evidence. Next, follow B7's order: compose and compare one DeltaNet layer,
then one attention layer, before an8-layer slice. The wide fused glue still
exceeds physical input DMA resources, the embedded layer norm still needs
a fitting schedule, and standalone128-row recurrent state must be reconciled
with the layer's padded140-row representation. Full-layer program/data
placement must be measured with all workers present.

No catalogue entry or runtime manifest was promoted. q4nx-build's model
format and packing are unchanged. No full layer, projection-to-logits model,
8-layer/64-layer run, model packaging, chat check or full-model benchmark
is claimed by this primitive validation.
