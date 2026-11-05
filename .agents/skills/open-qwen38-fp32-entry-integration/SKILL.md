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

## Follow-up: tail boundaries and resident validation

Use `make_decode.py --capture-boundaries` when preparing a new fixture, or with
`--cfg-only` after copying compatible reference/input files to a separate
fixture directory. This adds one BF16 `y_norm` and one full `y_stateL` per
linear layer at each position. `stream-decode-weights.py` retains these dumps.
Run `open_kernels/model/lm_head.py --model-dir ... --kernel-dir ... --out ...`
to validate the unrounded FP64 norm/head chain and captured-norm projection.
It streams native Q8 weight slabs and records exact coverage/input/weight hashes.
Do not treat the BF16-conditioned or rounded diagnostic as the composed gate.

The follow-up fixture is `Models/qwen38-27b/head-state-validation/decode`.
All 387 original activation/residual/logit captures reproduce the integrated
export byte for byte; 144 state snapshots (468910080 bytes total) and three norm
captures are retained. LM-head passes all three positions: worst norm maxrel
0.0021158452, composed logits 0.0008821369, conditioned projection 0.0000110584;
minimum composed correlation 0.9999987783, matching argmax 8678/198/2. State
snapshots still need an independent numerical oracle.

When available RAM rose to about 60 GiB, the same export was also run with all
weights resident, in `head-state-validation/resident`. Its 387 captures match
the streamed run byte for byte. Native CLI `--ids 248045 --max-tokens 3
--layers 64 --max-ctx 2048 --twice --dump-logits .../resident/engine` reproduces
the token sequence; `compare_decode.py --runtime-prefix .../resident/engine`
checks all three logit dumps bytewise and all independent full-depth gates.
Use the existing CLI `/tmp/qwen38-paired-runtime/open_qwen36_cli` only after
verifying its hash against the integrated export's `integration-evidence.json`.
No new kernel or native binary build was required for this follow-up.

The subsequent reset run uses `--ids 248045,8678,198 --layers 64 --max-ctx 2048
--det-step 3 --det-full`: 0 of 3 replays differ. This closes resident repeatability
for the integrated export, superseding the earlier memory-deferred status;
independent state accuracy remains open. The power-mode request failed, so do
not present these validation timings as a performance benchmark. Host suite:
1130 passed, 47 skipped.
