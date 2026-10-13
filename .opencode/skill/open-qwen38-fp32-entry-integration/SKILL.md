---
name: open-qwen38-fp32-entry-integration
description: Build and validate production FP32 attention-entry norm, its scratch/stream ABI and full-depth Qwen3.8 decode with independent projection gates.
---

# Production FP32 attention entry

Use after the controlled entry replay. `attention_entry_fp32` selects the validated
paired Qwen attention geometry with split norm/down. Only full-attention entry
changes: linear entry remains BF16, post-attention FFN norm remains FP32.

## Contract

- The manifest declares `attn_entry_dtype: float32`. Full attention's `xn` uses
  five 4096-byte elements rather than three, moving every later scratch region
  by 8192 bytes. Full-attention act size is 274432 bytes at H5120.
- The norm helper emits two FP32 halves through `ln_nr32` and direct FP32
  preparation feeds Q/gate/K/V. Norm weights remain BF16; model packing is
  unchanged. Paired KV, gated output and output-projection preparation remain
  as before. Keep the existing FIFO topology and 6144-byte stack.
- `attention_projections.py` reads entry dtype from the validated manifest;
  incompatible old layouts are refused. Regenerate fixtures for a fresh complete
  export. Do not edit old manifests or reference hashes to relabel an old run.
- The historical private replay requires checkout `dfefe4d`; it now refuses
  already-integrated recipes. Its successful forced-input experiment does not
  establish propagation, runtime/reset or prefill acceptance for a new export.

## Build and prepare

Use the native toolchain and XRT SDK from
`../open-qwen38-production-preflight/SKILL.md`. AIE forkserver compilation and
NPU dispatch need sandbox escalation on this host. Do not run source-generating
tests concurrently with the exporter.

```bash
export PATH="$PWD/ironvenv/lib/python3.14/site-packages/llvm-aie/bin:/opt/xilinx/xrt/bin:$PATH"
export XILINX_XRT=/opt/xilinx/xrt
export OFLM_KEEP_FAILED=1
ironvenv/bin/python -u open_kernels/export_qwen36_kernels.py \
  --model-dir Models/qwen38-27b/Qwen3.8-27B-NPU2 \
  --out Models/qwen38-27b/production-fp32-entry/kernels \
  --build-root Models/qwen38-27b/production-fp32-entry/build \
  --max-ctx 2048 -j 2
ironvenv/bin/python utilities/audit-kernel-export.py \
  --export Models/qwen38-27b/production-fp32-entry/kernels \
  --build-root Models/qwen38-27b/production-fp32-entry/build \
  --out Models/qwen38-27b/production-fp32-entry/build-audit.json
OPENBLAS_NUM_THREADS=1 ironvenv/bin/python -u open_kernels/model/make_decode.py \
  --model-dir Models/qwen38-27b/Qwen3.8-27B-NPU2 \
  --kernel-dir Models/qwen38-27b/production-fp32-entry/kernels \
  --layers 64 --tokens 3 --max-ctx 2048 \
  --out Models/qwen38-27b/production-fp32-entry/decode \
  --pool-dir Models/qwen38-27b/production-full/pools --reuse-pools
```

Verify model and all reused pool hashes against retained provenance first;
`--reuse-pools` checks size only. CPU preparation can proceed without
`--kernel-dir` during compilation. Then repeat with `--kernel-dir --cfg-only`
to validate the existing reference contract and select exported binaries.
Compare all 195 regenerated reference hashes with the retained independent
fixture; no CPU reference arithmetic changes are needed for this integration.

## Hardware and numerical gates

When memory is insufficient for resident loading, preserve full-depth state
and reload one weight BO using the streamed harness:

```bash
ironvenv/bin/python utilities/stream-decode-weights.py \
  --cfg Models/qwen38-27b/production-fp32-entry/decode/run_decode.cfg \
  --out Models/qwen38-27b/production-fp32-entry/decode/run_streamed.cfg
HARNESS_TIMEOUT_MS=30000 LD_LIBRARY_PATH=/opt/xilinx/xrt/lib \
  /tmp/qwen38-fp32-harness/run_kernel \
  Models/qwen38-27b/production-fp32-entry/decode/run_streamed.cfg
OPENBLAS_NUM_THREADS=1 ironvenv/bin/python open_kernels/model/compare_decode.py \
  --out Models/qwen38-27b/production-fp32-entry/decode --tokens 3
```

Run `attention_projections.py`, `attention_heads.py`, `ffn_front.py` and
`ffn_partials.py` against the same model/kernel/decode directories. Supply
`--layers 0,1,...,63` to both FFN checkers; the attention checkers default to
all full-attention layers. Record exit status as well as JSON metrics. Do not
weaken bounds: normalized maximum residual/projection error <0.005; per-head
absolute error <0.02 and cosine >0.9999. Logits must pass correlation and argmax.

Rebuild the native CLI and run CTest before resident execution. With roughly
35–40 GiB available, use the original resident cfg and CLI `--twice`, compare
runtime logits bytewise, then force IDs 248045,8678,198 with `--det-step 3
--det-full`. Streamed hardware results do not replace resident identity/reset.
Independent state/LM-head, block-prefill, serving and final resource/performance
gates remain separate even when the full-depth residual/logit checks pass.

## Verified result — 2026-10-13

Build key `63d7cd0879422f0f31f0b9d1e018799b8db868a3f110e3ee5f908a3faa5fc8da`.
41 sets / 82 hashes pass. Max .text lx 15584, ax 13760 bytes; max L1 end lx
64384, ax 59920 with stack 6144. `memory-audit.json` records both builds using
`utilities/attention-entry-replay/resources.py`'s generic `audit` function and
per-build `build.json` binary hashes (mode `production-entry-integration`).

All 195 regenerated CPU references, model and 65 weight-pool hashes match.
Streamed hardware: 198 dispatches, residual 192/192 PASS, worst 0.0040853661
(L63/T1). Logit correlations 0.9999988740/0.9999971796/0.9999982962, argmax
8678/198/2. Attention heads 1152 PASS, maxabs 0.0004209284, minimum cosine
0.999999998443142; composed branch 48 PASS, worst 0.0005515331. Front/down
192 each PASS; worst front 0.0013213623, down partial 0.0003836580.

Host suite 1097 passed, 47 skipped; native CTest 4/4. Resident identity/reset
was not run (about 30 GiB available). State/LM-head, prefill, serving and final
resource/performance acceptance remain open. `decode-results.json` and the
four boundary JSON files pin the numerical evidence; this scoped PASS must
not be promoted to complete model support.
