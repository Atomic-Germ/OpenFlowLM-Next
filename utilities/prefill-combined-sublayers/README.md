# Combined prefill sublayer diagnosis

Copy the verified combined CLI and add read-only captures at L63/P1,
L42/P17 and L52/P17. Preserve its projection policy, four-product arithmetic,
causal host operations and residual writer. Do not modify production sources,
prior builders, references or kernels. Pin parent build, sources, staged files,
all compiler dependencies, CMake cache and executable. Existing outputs fail.

L63 is full attention: capture all 20 real rows of norm/Q/K/V/gate/og and
paired-BF16 KV after the block. L42/L52 are linear: capture norm/QKV/Z/og
prefixes and raw state before/after the block. All targets retain selected
input, pre-FFN residual, FFN norm/up/gate/activation/down/output projection.
Sequential runs retain selected activation scratch. Refuse a partial first
block or nonzero block start; state-after refers to P19, not selected P1/P17.

```sh
xrt_sdk=/home/prihlop/sources/xdna-driver/xrt/build/Release/opt/xilinx/xrt
OPENBLAS_NUM_THREADS=1 ironvenv/bin/python utilities/prefill-combined-sublayers/build.py \
  --parent Models/qwen38-27b/prefill-combined-cli/v1/combined \
  --out Models/qwen38-27b/prefill-combined-sublayers/build-v1 \
  --xrt-include "$xrt_sdk/include" --xrt-lib "$xrt_sdk/lib"
LD_LIBRARY_PATH=/opt/xilinx/xrt/lib OPENBLAS_NUM_THREADS=1 \
 ironvenv/bin/python utilities/prefill-combined-sublayers/run.py \
  --build Models/qwen38-27b/prefill-combined-sublayers/build-v1 \
  --baseline Models/qwen38-27b/prefill-combined-cli/native-v1 \
  --out Models/qwen38-27b/prefill-combined-sublayers/native-v1
OPENBLAS_NUM_THREADS=1 ironvenv/bin/python utilities/prefill-combined-sublayers/check.py \
  --run Models/qwen38-27b/prefill-combined-sublayers/native-v1 \
  --out Models/qwen38-27b/prefill-combined-sublayers/native-v1/sublayer-audit.json
```

Require >=35 GiB available RAM, exclusive NPU access and the shared resident
lock. Run sequential, hybrid and block-major serially. All **261** baseline
logits/residual/scratch files must match the previously independently audited
combined candidate. Require **83** selected captures and **40** hybrid pairs.
The runner's `passed` means instrumentation identity, not numerical acceptance.

Re-audit all full-depth residual/logit gates from captures. Reconstruct three
selected independent CPU outputs bytewise before interpreting conditioned
metrics. Check actual FFN input, output/residual reconstruction, FP32 conv tail,
state padding and paired KV extent. Compare projection predictions on actual
input and captured norm; report per-head gates so large heads cannot hide small
head errors. A FP64 causal reference driven by captured projections diagnoses
host arithmetic only; it never replaces independent model references.

Preserve failed audits; fix CPU checks without repeating NPU captures. Report
FFN, local attention/projection and accumulated input drift separately. All
limits remain unchanged. Passing instrumentation or conditioned checks does
not close full-depth/long-prefix numerical failures or permit production promotion.

## Validated checkpoint (2026-10-23)

GCC 15.2; CTest **4/4**; **592 compiler dependencies**. CLI SHA256:
`e6e53ce672ff447655cb0345736a2b6e10a9dcbdcdb980c9e77b47a4b57aaa20`.
Three resident routes complete: **261/261 baseline identities**, **83 captures**,
**40/40 hybrid pairs**. Three selected CPU outputs reconstruct primary references
bytewise. Independent audit: **54/54 FFN** and **30654/30654 projection/host
gates** pass, including **29760 per-head gates**, with unchanged limits.

The completed audit is `native-v1/sublayer-audit-v2.json`. The first audit failed
on the CPU adapter passing ModelSpec's frequency list into a NumPy oracle;
position-zero list multiplication produces an empty list. Convert frequencies
to an FP64 array. A synthetic regression checks this and causal attention.
The failed JSON/log are retained; the corrected audit reuses the same captures.

| Hybrid target | Input maxrel | Pre-FFN maxrel | Output maxrel | Ideal FFN on actual residual |
|---|---:|---:|---:|---:|
| L63/P1 | 0.0017499814 | 0.0026927327 | 0.0096485275 | 0.0070258073 |
| L42/P17 | 0.0029677911 | 0.0047458357 | 0.0057900192 | 0.0052107169 |
| L52/P17 | 0.0119563497 | 0.0106894266 | 0.0121972127 | 0.0123487787 |

All three ideal-FFN diagnostics still fail 0.005, so a local FFN correction
alone cannot close these captured errors. L63 additionally has down-from-H
error **0.0039805268**, versus up/gate **1.2174e-6 / 2.1872e-6**. Linear
pre-FFN residuals match ideal causal state driven by native inputs within
**1.5940e-6 / 7.6249e-7** at L42/L52. Full-attention conditioned host output
has worst per-head error **3.8945e-6**; selected out projection error is
**0.0006512213**. These are local diagnostics, not replacement model references.

Full-depth residual result remains **4000/4032 FAIL**, `model_acceptance=false`.
Next attribute effective/BF16 weight rounding in FFN down and full-attention
entry/output from these captures before selecting a new NPU precision replay.
Ten new tests; host suite **1690 passed / 47 skipped**. No production edits or
new kernels, and no timing/throughput claim.
