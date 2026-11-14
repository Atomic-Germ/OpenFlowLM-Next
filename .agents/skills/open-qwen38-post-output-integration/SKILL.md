---
name: open-qwen38-post-output-integration
description: Build the standard Qwen3.8 FP32 DeltaNet post-output export and validate scratch compatibility, compiler resources and full-depth decode with independent references.
---

# Production FP32 DeltaNet post-output

Use after open-qwen38-shared-post-output. The standard dense Q4 H5120/FF17408
recipe selects POST_F32 together with its validated consistent-sum geometry.
Other geometries and mixed Q8 roles retain their existing storage. The new
24576-byte output region starts at 393216; act is 417792 bytes. All other
scratch offsets and the 3317760-byte recurrent state are unchanged. Packing
does not change, and no model conversion or manual packing step is required.

Manifest v4 declares per-type post_output dtype/offset/bytes. The native reader
rejects unsupported dtype, alignment, state kind and out-of-bounds regions.
Python bundle validation requires the current recipe. Do not edit an old
manifest to make historical kernels appear compatible.

Run host tests and native CTest first. Do not run source-generating tests while
export compilation is active. Use a fresh export root and retain failed builds.
On this host compiler subprocesses and XRT dispatch require execution-tool
escalation. Source utilities/env.sh for toolchain discovery; Peano stays out of
PATH. Retain toolchain.json with the export.

```bash
source utilities/env.sh
export OFLM_KEEP_FAILED=1
ironvenv/bin/python -u open_kernels/export_qwen36_kernels.py \
  --model-dir Models/qwen38-27b/Qwen3.8-27B-NPU2 \
  --build-root Models/qwen38-27b/production-post-fp32/build-v1 \
  --out Models/qwen38-27b/production-post-fp32/kernels-v1 --max-ctx 2048 -j 2
ironvenv/bin/python utilities/audit-kernel-export.py \
  --export Models/qwen38-27b/production-post-fp32/kernels-v1 \
  --build-root Models/qwen38-27b/production-post-fp32/build-v1 \
  --out Models/qwen38-27b/production-post-fp32/build-audit.json
```

Require all 46 sets, current source identity, compiled FP32 post/preparation
and consistent/factored headers. Inspect the copied compiler dependencies,
including isolated staging paths; compare staged header contents to production
sources. Audit lx/ax text below 16384 bytes, assigned L1 end at most 65536 and
the existing 6144-byte stack. A recipe flag alone is not binary evidence.

Follow open-qwen38-consistent-integration for a fresh 64-layer, three-token CPU
reference, streamed harness, all seven independent boundary checks and serial
resident repeat/reset validation. Use production-post-fp32 for new artifacts.
Verify retained model/pool hashes before reusing pools. The independent CPU
oracle must remain unchanged; compare its 195 reference files to the archived
production-consistent fixture. Use current sources for the harness and CLI.
Take the shared NPU validation lock. Resident runs require at least 35 GiB
available RAM and must not overlap CPU model-weight checkers.

The previous BF16 native output hashes are historical controls, not expected
outputs for this arithmetic upgrade. Do not modify their reports or relax
numerical thresholds. Three-token propagation does not establish long-prefix,
block-prefill, serving or performance acceptance; keep those gates open.

## Measured integration — 2026-10-26

The command above builds 46 sets / 92 verified binary hashes with mlir-aie
1.4.2 and llvm-aie 21.0.0.2026080301+c9c5ecb7. Build key:
67f3c4395334a0bc3aa0dcb2bcff398cfa01147451cddd3b022afca4215c7a13.
Standard lx has exact instructions and whole-XCLBIN equivalence (80 stamp bytes)
to the audited private fp32-v1 / fp32-run-v1 pair. The archived comparison
verifies all 729 capture hashes and binds its original passing report/audit.
Use identity.py's archived_build and exporter's xclbin_equivalent for this
comparison; do not run the historical private source verifier after promotion.

The build/arithmetic audits pass. The arithmetic auditor now derives the
exporter's isolated staging root and checks staged headers against canonical
production bytes. Resource records live in build-v1/{lx,ax}/build.json, with
sha256 for final.xclbin and insts.bin. The existing resources.py auditor reads
these records. lx/ax: 11/15 cores, text maxima 16000/13760, L1 ends 62336/59920;
all 26 stacks remain 6144. The copied post_fn.cc matches shared post.cc.

All 66 model/pool hashes match production-consistent/input-sha256.json, itself
bound to its stage-evidence.json. All 195 freshly generated CPU references match
production-consistent/decode. New streamed decode completes 198 dispatches with
2.964 GiB logical buffers. All 192 residuals pass, worst 0.0001072489472 at L63/T1
(previous standard maximum 0.0030399006483). Logit correlations are
0.9999993162 / 0.9999987908 / 0.9999988379; CPU/NPU argmax 8678 / 198 / 2.
All seven boundary checkers pass, including both DeltaNet head flags. Counts:
144 input/state, 6912 strict recurrence heads, 1152 attention heads, 48 attention
projections, 192 FFN front/down each, 3 LM-head cases.

Fresh native builds are /tmp/qwen38-post-production-{native,harness}. They use
XRT_INCLUDE_DIR=/home/prihlop/sources/xdna-driver/xrt/build/Release/opt/xilinx/xrt/include
and its sibling lib as XRT_LIB_DIR. The CLI's --twice output reproduces the
first request; all three dumped logits match the streamed harness. --det-step 3
--det-full over IDs 248045,8678,198 reports 0 of 3 differing replays. Both commands
use --decode-route npu --pmode none, max-ctx 2048 and all 64 layers. Hold the
shared resident-validation lock and use /opt/xilinx/xrt/lib at run time.

Host suite: 1886 passed, 47 skipped; native CTest 4/4 and actual export manifest
load pass. Production-post-fp32 retains command logs, numerical JSON, source,
reference and artifact hashes. These are short decode/lifecycle checks. Fresh
short prefill, block-boundary and long-prefix acceptance remain open, as do
serving and performance. Installed bundles and running services were untouched.
