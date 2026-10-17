---
name: open-qwen38-paired-attention-integration
description: Build and validate full Qwen3.8 layers with paired-BF16 KV, FP32 gated attention output and direct FP32 output-projection preparation.
---

# Integrated paired attention

Use after the paired-KV replay. The qwen35 H5120/NH24/KVH4/HD256/ROT64
geometry selects `Attn.KV_PAIR`, with VEXP=1, RB=1 and four heads per core.
Other geometries retain the existing attention ABI. Model tensors and weight
packing do not change.

## Boundaries

- The manifest declares `attn_kv_dtype: bfloat16_pair` and
  `attn_output_dtype: float32`. KV rows are `[K_hi,K_lo,V_hi,V_lo]`, 8192 bytes.
  The position table and attention input/output FIFO elements remain 2048 bytes.
- Each core emits two FP32 heads per output element, consuming one gate element
  for each. Core 0 emits all four cache planes before its two output elements
  on the same FIFO. No extra DMA channel or reduced stack is needed.
- `AA_OG` grows from 12288 to 24576 bytes, shifting later full-layer scratch
  offsets. `o_proj` receives six 4-KiB FP32 elements through `dense_prep_f32`.
  BF16 conversion must not be reintroduced before table preparation.
- Old exports remain usable with their matching source/ABI. Current recipe
  checks deliberately reject old capture layouts. Rebuild a complete export
  and regenerate the fixture; never relabel old manifests or reference hashes.
- Host block attention reads/writes both KV planes. The BF16-only NPU block
  attention path is not implemented for paired cache, so this geometry uses
  host attention between NPU GEMMs. Independent block-prefill accuracy remains
  a separate gate. Public BF16 K/V inspection returns the high plane with the
  original logical head count; snapshots/reset retain the full cache bytes.

## Build

Use the native production-preflight toolchain: mlir-aie 1.4.2, Peano under
Python 3.14, runtime `/opt/xilinx/xrt/lib`. XRT development headers and linker
symlinks are in the driver SDK, not the runtime-only `/opt` installation.
Native compiler forkserver and NPU access require sandbox escalation here.

`--build-root` keeps generated compiler projects in a separate writable tree.
It avoids container-owned `final.prj` directories without changing ownership,
deleting previous artifacts or changing the recipe's build identity.

```bash
export PATH="$PWD/ironvenv/lib/python3.14/site-packages/llvm-aie/bin:/opt/xilinx/xrt/bin:$PATH"
export XILINX_XRT=/opt/xilinx/xrt
export OFLM_KEEP_FAILED=1
ironvenv/bin/python -u open_kernels/export_qwen36_kernels.py \
  --model-dir Models/qwen38-27b/Qwen3.8-27B-NPU2 \
  --out Models/qwen38-27b/production-paired-attention/kernels \
  --build-root Models/qwen38-27b/production-paired-attention/build \
  --max-ctx 2048 -j 2
ironvenv/bin/python utilities/audit-kernel-export.py \
  --export Models/qwen38-27b/production-paired-attention/kernels \
  --build-root Models/qwen38-27b/production-paired-attention/build \
  --out Models/qwen38-27b/production-paired-attention/build-audit.json
```

The audit verifies manifest compatibility, current source identity, exported
and build-directory binary hashes, and every retained core ELF's program size.
Do not run source-generating tests during the exporter. A `--only ax` export
is useful for fit checks but is incomplete for full-model execution.

Build the native CLI with the SDK paths documented in
`../open-qwen38-production-preflight/SKILL.md`, using a separate build directory
such as `/tmp/qwen38-paired-runtime`, and run CTest. The parser tests also reject
inconsistent cache/output dtype pairs and wrong global/per-layer row sizes.

## Full-depth fixture

Verify the unchanged model and reused pool hashes. Prepare independent CPU
references and bind the complete export before dispatch:

```bash
OPENBLAS_NUM_THREADS=1 ironvenv/bin/python -u open_kernels/model/make_decode.py \
  --model-dir Models/qwen38-27b/Qwen3.8-27B-NPU2 \
  --kernel-dir Models/qwen38-27b/production-paired-attention/kernels \
  --layers 64 --tokens 3 --max-ctx 2048 \
  --out Models/qwen38-27b/production-paired-attention/decode \
  --pool-dir Models/qwen38-27b/production-full/pools --reuse-pools
```

CPU preparation can run without `--kernel-dir` while compilation proceeds;
afterward repeat the command with `--kernel-dir` and `--cfg-only`. This checks
existing reference hashes and selects the export's binary paths, without
rewriting references. Full NPU execution needs roughly 35–40 GiB available;
run NPU jobs serially.

Use the native harness on `decode/run_decode.cfg`, then the native CLI with
`--ids 248045 --max-tokens 3 --layers 64 --twice --dump-logits .../decode/engine`.
Run `compare_decode.py --out .../decode --tokens 3 --runtime-prefix .../decode/engine`
and all three local checkers `attention_heads.py`, `ffn_front.py`,
`ffn_partials.py` with the matching model/kernel/out arguments. Front/down need
`--layers 0,1,...,63`; attention defaults to every full-attention layer.

Keep original residual/FFN bounds and unrounded attention reference, including
absolute per-head max error <0.02 and cosine >0.9999. Conditional diagnostics,
short generated answers and reset repeatability do not replace independent
state/head, block-prefill or serving acceptance.

## Limited-memory harness

If resident execution lacks memory, transform the generated cfg into a separate
streamed-weight cfg with `utilities/stream-decode-weights.py --cfg .../run_decode.cfg
--out .../run_streamed.cfg`. This retains every layer state and all capture/
dispatch ordering, but reloads one shared layer-weight BO before each layer run.
For 64 layers at context capacity 2048, logical BO sizes fall from 16.618 to
2.960 GiB. Use the same native harness and local numerical checkers. The tool
records source/streamed cfg hashes and never rewrites the original cfg.

This is full-depth sequential hardware validation with streamed weights. It
must not be reported as resident-runtime identity, repeated-request acceptance,
reset acceptance or a performance result. Those require the original cfg and
native CLI once enough memory is available. Never stop unrelated user workloads
to make room for them.

## Recorded result — 2026-10-12

Build key `sha256:fa34220099d565bfeb514765e3c72665f768a42ee17e9ba37dd7fa3402788405`;
41 sets, 82 hashes, max program lx 15584 / ax 13760 bytes. Source base is
`2dad494`; `integration-source.patch` SHA256 is
`cc3b5bc73878b9614ea3e774134946cac1c82bb542a4e007c25525bc6fda92bf`.
`integration-evidence.json` pins the patch, audits and native binaries.
Host tests: 1049 passed, 47 skipped. Native CTest: 4/4. All 195 regenerated CPU
reference hashes and model/65 pool hashes agree with the retained inputs.

The streamed 64×3 run completes 198 dispatches. Attention: 1152/1152 PASS,
maxabs 0.0003238841, min cosine 0.9999999988341115. Front/down: 192 each PASS;
worst norm/front 0.0000164919/0.0005658096; down partial0/partial1/sum/closure
0.0002353918/0.0001898039/0.0001780994/0.0000365645.
Logits corr 0.9999986691/0.9999964904/0.9999982652, argmax 8678/198/2.

**Full-chain FAIL:** only 189/192 residual checks pass. Position 1 layers
59/61/63 exceed 0.005, worst 0.0058082229. Preserve the export's
`VALIDATION_FAILED.txt`; do not promote local attention success to model support.
Resident runtime/reset was not run because available memory was about 28 GiB.
Do not reuse older export/runtime PASS as acceptance for this build.
