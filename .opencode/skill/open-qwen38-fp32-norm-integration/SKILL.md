---
name: open-qwen38-fp32-norm-integration
description: Build and validate complete Qwen3.8 lx/ax kernels with FP32 post-attention norm streaming, the widened scratch ABI and full-depth decode regression.
---

# Integrated FP32 FFN norm input

Use after the isolated front experiment. Split dense FFN with split norm now
emits two FP32 norm halves through `ln_nr32` and prepares up/gate tables directly
from FP32. Layer-entry attention norm and unsplit dense recipes retain BF16.
The existing FP32 down preparation is reused; weight packing is unchanged.

## ABI boundary

`Ffn.XM_F32` selects five 4-KiB elements for Qwen3.8's 5120-value norm input,
versus three for BF16. Each act buffer grows by 8192 bytes and H/down offsets
move. `manifest.layout.ffn_norm_dtype = "float32"` records this explicitly.
The C++ engine treats dense act buffers as opaque, sized from the manifest.
The front comparator uses this field to read captures; absent means bfloat16.

Old exports remain intact but are deliberately rejected by current recipe
compatibility checks. Do not edit an old export/fixture's manifest or build key,
or interpret its act captures with the new offsets. Regenerate fixtures from a
new complete export. Existing model and weight pools can be reused only after
confirming their hashes and unchanged packing. Historical BF16-only diagnostic
commands need the source revision matching their ABI (through `1031a45`).

## Build and reference

Use the production-preflight skill for native harness/CLI build commands and
model provenance. Actual Peano is under Python 3.14 on this host. Native build
forkserver and NPU calls require sandbox escalation. Do not run source-generating
tests or competing exports while this exporter is generating/compiling kernels.

```bash
export PATH="$PWD/ironvenv/lib/python3.14/site-packages/llvm-aie/bin:/opt/xilinx/xrt/bin:$PATH"
export XILINX_XRT=/opt/xilinx/xrt
export OFLM_KEEP_FAILED=1
ironvenv/bin/python -u open_kernels/export_qwen36_kernels.py \
  --model-dir Models/qwen38-27b/Qwen3.8-27B-NPU2 \
  --out Models/qwen38-27b/production-fp32-front-kernels -j 1
OPENBLAS_NUM_THREADS=1 ironvenv/bin/python -u open_kernels/model/make_decode.py \
  --model-dir Models/qwen38-27b/Qwen3.8-27B-NPU2 \
  --kernel-dir Models/qwen38-27b/production-fp32-front-kernels \
  --layers 64 --tokens 3 --out Models/qwen38-27b/production-fp32-front-multitoken \
  --pool-dir Models/qwen38-27b/production-full/pools --reuse-pools
```

Require 41 sets, 82 verified hashes and every core .text <16384. The source
build key and per-core ELF sizes belong with the artifact audit. A successful
build alone does not establish numerical acceptance or block-prefill accuracy.

For this build, CPU preparation ran without `--kernel-dir` while the exporter
was compiling. After both completed, the same `make_decode.py` command above
was rerun with `--kernel-dir` and `--cfg-only`, verifying and binding the export
before any hardware run. All 195 regenerated CPU reference hashes matched the
previous reference. Do not change reference metadata by hand.

## Hardware and local gates

Finish CPU preparation first; check at least roughly 35–40 GiB available before
loading the resident 64-layer NPU model. Run NPU jobs serially.

```bash
HARNESS_TIMEOUT_MS=30000 LD_LIBRARY_PATH=/opt/xilinx/xrt/lib \
  /tmp/qwen38-fp32-harness/run_kernel \
  Models/qwen38-27b/production-fp32-front-multitoken/run_decode.cfg
LD_LIBRARY_PATH=/opt/xilinx/xrt/lib OMP_NUM_THREADS=4 OMP_WAIT_POLICY=PASSIVE \
  /tmp/qwen38-main-status-runtime/open_qwen36_cli \
  --model Models/qwen38-27b/Qwen3.8-27B-NPU2 \
  --kernels Models/qwen38-27b/production-fp32-front-kernels \
  --ids 248045 --max-tokens 3 --layers 64 --twice \
  --dump-logits Models/qwen38-27b/production-fp32-front-multitoken/engine
OPENBLAS_NUM_THREADS=1 ironvenv/bin/python open_kernels/model/compare_decode.py \
  --out Models/qwen38-27b/production-fp32-front-multitoken --tokens 3 \
  --runtime-prefix Models/qwen38-27b/production-fp32-front-multitoken/engine
```

Run CLI `--det-step 3 --det-full` with the seed and first two generated IDs for
reset repeatability; this is an NPU reference, not an independent CPU state gate.
Score all captured front/down boundaries with the same model/kernel/out arguments:

```bash
for checker in ffn_front ffn_partials; do
  OPENBLAS_NUM_THREADS=1 ironvenv/bin/python "open_kernels/model/$checker.py" \
    --model-dir Models/qwen38-27b/Qwen3.8-27B-NPU2 \
    --kernel-dir Models/qwen38-27b/production-fp32-front-kernels \
    --out Models/qwen38-27b/production-fp32-front-multitoken \
    --layers "$(seq -s, 0 63)" \
    > "Models/qwen38-27b/production-fp32-front-multitoken/$checker.json"
done
```

Keep the original bounds: logit correlation >0.9999 with equal argmax; all
residual/front/down maxrel <0.005. Front correctness is conditioned on captured
pre-FFN residual, down correctness on captured h. State/head and full-chain
accuracy remain separate. The full layers retain combined h, not individual
up/gate projections; separate projection evidence comes from the isolated probe.

## Recorded full-depth result — 2026-10-11

Export build key:
`sha256:b707550f6c1d82a71a12998672e41e248ae70d4171757bfb5d1714432be3b3ae`.
The exporter records base source commit `1031a45`; the integration changes used
for this build are retained as `integration-source.patch` with its SHA256 in
`build-audit.json`. All 41 sets / 82 hashes match the retained build binaries.
Every core fits program memory; maxima lx 15584, ax 14400 bytes.

CPU references were regenerated: all 195 hashes unchanged. Model SHA256 is
`5f5c282ae81b63df34e5f9ce42c616d73b454691c4b8d253027b2e08a1fe3d84`;
all 65 reused pool hashes were verified. The manifest/config C++ check passes.

Full-chain **PASS** for 64 layers × 3 positions (198 dispatches): all 192
residuals below 0.005, worst **0.0044951978** at L61/T1. Logit correlations
are **0.9999986446 / 0.9999957289 / 0.9999973938**, argmax **8678 / 198 / 2**.
All three runtime dumps match harness bytes, the repeated request reproduces
IDs, and full-buffer reset replay reports **0/3 differing runs**. No threshold
was relaxed. This is three-position sequential decode evidence, not a claim
of independently validated recurrent state or block prefill.

Evidence is in `production-fp32-front-multitoken/`: `build-audit.json`,
`input-sha256.json`, `manifest-changes.json`, `results.json`, reference/capture
files and `logs/`. Tests: **969 passed, 47 skipped**; CTest **3/3**.
Do not report these timings as a controlled performance benchmark: power mode
and system workload were not controlled.

All 192 propagated down checks pass: worst partial0/partial1/sum/closure
maxrel **0.0003103112 / 0.0003736917 / 0.0003362342 / 0.0000686012**.
The independent FP64 reference uses captured h. Results and capture hashes
are in `ffn_partials.json`.

All 192 front checks pass as well: worst norm **0.0000172679**, primary
combined h **0.0009100331**, captured-norm activation diagnostic
**0.0009182833**. L11/T0 primary h is **0.0001456326** on the newly propagated
input. `ffn_front.json` retains every metric and capture hash. Separate full-layer
up/gate outputs are still not captured; the isolated probe supplies the limited
individual-projection evidence.

## Additional attention acceptance check

Run `open_kernels/model/attention_heads.py` with the same `--model-dir`,
`--kernel-dir` and `--out` to score every retained full-attention head. This
requires no new NPU run. The bound is **absolute** error <0.02 plus cosine
>0.9999; normalized error and BF16-rounded references are diagnostic only.

The recorded fixture passes 1109/1152 heads; 43 fail absolute error in layers
51/55/59/63, worst 0.2226126094. All cosine gates pass. Ideal BF16 output
rounding alone exceeds the bound in 24 cases. See `attention-heads.json` and
`attention-heads-summary.json`; the export has a scoped `VALIDATION_FAILED.txt`
marker. Earlier full-chain/FFN PASS results remain valid but do not close this
local gate. Next isolate attention precision; do not loosen the threshold or
replace its unrounded FP64 reference. Recurrent state still needs captures.
