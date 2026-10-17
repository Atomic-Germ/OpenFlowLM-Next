---
name: open-qwen38-fp32-production
description: Build the complete Qwen3.8-27B kernel export with direct FP32 split-down preparation and validate full-depth three-token propagation, runtime identity and FFN partials.
---

# Corrected Qwen3.8 production decode

Use this after the isolated direct-FP32 split-down replay passes. It checks
propagation through corrected layers, beyond replay from historical inputs.
The model remains the shared `qwen35` recipe; packing and weights are unchanged.
Independent up/gate/norm, state/head, prefill and serving gates remain separate.

## Build and provenance

On this Linux host use Python 3.14 / mlir-aie 1.4.2 and Peano
21.0.0.2026080301+c9c5ecb7. Model assets are under ignored
`Models/qwen38-27b/Qwen3.8-27B-NPU2`. Preserve earlier verified exports.
Read `../open-qwen38-production-preflight/SKILL.md` for the model's provenance
and native harness/CLI build commands. `/tmp` host builds may need recreating. For this run the harness was built in
`/tmp/qwen38-fp32-harness` with the same XRT include/lib settings; the runtime
was rebuilt in `/tmp/qwen38-main-status-runtime`.

```bash
export PATH="$PWD/ironvenv/lib/python3.14/site-packages/llvm-aie/bin:/opt/xilinx/xrt/bin:$PATH"
export XILINX_XRT=/opt/xilinx/xrt
export OFLM_KEEP_FAILED=1
ironvenv/bin/python -u open_kernels/export_qwen36_kernels.py \
  --model-dir Models/qwen38-27b/Qwen3.8-27B-NPU2 \
  --out Models/qwen38-27b/production-fp32-kernels -j 1
```

The exporter regenerates kernel translation units before compiling. Do not run
source-generating tests or another family export concurrently. Build all 41
sets; `--no-build` or a partial export does not establish current-source provenance.
The forkserver needs a local socket outside this machine's sandbox.
If an old build fails opening `final.prj/.lock`, inspect ownership: previous
container builds can leave files owned by `nobody`. Preserve those build
folders by renaming them before a clean build; keep verified exports intact.
This run saved old folders as `<build_dir>.before-fp32`.

Verify every exported manifest file against the 82 hashes in `toolchain.json`
and its source commit/build key. `decode_bundle.load_bundle` performs the
compatibility and binary checks when `make_decode.py --kernel-dir` loads it.
Inspect every retained core ELF with `utilities/core_sizes.py`: .text <16384.
Do not infer numerical or prefill acceptance from successful compilation.

## Full-depth fixture and hardware

Allow about 35–40 GiB available RAM for the resident model. Finish CPU reference
preparation before starting NPU runs. Use a new fixture bound to the new export;
never edit an old reference's build key to relabel it.

```bash
OPENBLAS_NUM_THREADS=1 ironvenv/bin/python -u open_kernels/model/make_decode.py \
  --model-dir Models/qwen38-27b/Qwen3.8-27B-NPU2 \
  --kernel-dir Models/qwen38-27b/production-fp32-kernels \
  --layers 64 --tokens 3 --out Models/qwen38-27b/production-fp32-multitoken \
  --pool-dir Models/qwen38-27b/production-full/pools --reuse-pools
HARNESS_TIMEOUT_MS=30000 LD_LIBRARY_PATH=/opt/xilinx/xrt/lib \
  /tmp/qwen38-fp32-harness/run_kernel \
  Models/qwen38-27b/production-fp32-multitoken/run_decode.cfg
```

Pool reuse is valid here because container bytes and packing are unchanged;
`--reuse-pools` itself checks only sizes. Record pool/weight hashes. The CPU
reference must finish with 195 pinned captures (64 residuals plus logits at
three positions), and the harness executes 198 dispatches. The model file is a
hard link: do not reconvert or overwrite it in place.

Run the rebuilt CLI against exactly the same export, serially with the harness:

```bash
LD_LIBRARY_PATH=/opt/xilinx/xrt/lib OMP_NUM_THREADS=4 OMP_WAIT_POLICY=PASSIVE \
  /tmp/qwen38-main-status-runtime/open_qwen36_cli \
  --model Models/qwen38-27b/Qwen3.8-27B-NPU2 \
  --kernels Models/qwen38-27b/production-fp32-kernels \
  --ids 248045 --max-tokens 3 --layers 64 \
  --dump-logits Models/qwen38-27b/production-fp32-multitoken/engine --twice
OPENBLAS_NUM_THREADS=1 ironvenv/bin/python open_kernels/model/compare_decode.py \
  --out Models/qwen38-27b/production-fp32-multitoken --tokens 3 \
  --runtime-prefix Models/qwen38-27b/production-fp32-multitoken/engine
```

Keep correlation >0.9999, equal argmax and all residual maxrel <0.005.
`--runtime-prefix` requires byte identity at every step. `--twice` repeats IDs
and overwrites the dumps on the second request. Also use `--det-step 3
--det-full` with `--ids` equal to the seed and first two reference-generated
IDs. This checks reset repeatability against NPU, not independent CPU state.

Score down partials on the NEW propagated captures, rather than isolated inputs:

```bash
OPENBLAS_NUM_THREADS=1 ironvenv/bin/python open_kernels/model/ffn_partials.py \
  --model-dir Models/qwen38-27b/Qwen3.8-27B-NPU2 \
  --kernel-dir Models/qwen38-27b/production-fp32-kernels \
  --out Models/qwen38-27b/production-fp32-multitoken \
  --layers "$(seq -s, 0 63)" \
  > Models/qwen38-27b/production-fp32-multitoken/ffn-partials.json
```

All four boundaries must pass independently at all 192 layer/position pairs.
This FP64 reference is conditioned on captured FP32 `h`; it does not validate
up/gate/norm. Keep build/reference/hardware/runtime/comparison logs with the
export hashes and result metrics under the ignored fixture directory.

## Recorded result — 2026-10-11

Build source `c718cbe`; build key
`sha256:e9cf679f2870b23c689ab09ab3820713a6345d6dfe62defadabaf63a5d069e39`.
All 41 sets / 82 hashes verified; core maxima lx 15584, ax 14400, ln 3424,
LM head 2432, attention products 1984, GEMM 4784 bytes. CPU references were
regenerated, all 195 hashes unchanged. Model SHA256 remains
`5f5c282ae81b63df34e5f9ce42c616d73b454691c4b8d253027b2e08a1fe3d84`.

**Full-chain acceptance FAIL.** Positions 0 and 2 pass. At position 1,
layers 55/56/57/58/59/61/63 fail the 0.005 residual bound; worst 0.0083300389
at L63. All three logits pass (minimum correlation 0.9999932894, IDs
8678/198/2). Runtime logits match harness bytes; repeated request and 3 reset
replays pass. Do not promote this export or loosen the residual bound.
It carries `VALIDATION_FAILED.txt` despite its valid build and binary hashes.

All 192 propagated FFN down checks pass. Worst partial0/partial1/sum/closure:
0.0002570118 / 0.0002904925 / 0.0002560792 / 0.0001103267. This excludes a
failing local down boundary in these captures, not accumulated upstream error.
The cause requires independent up/gate/norm and attention/state checks.

Evidence: `production-fp32-multitoken/{results,build-audit,input-sha256,ffn-partials}.json`
and `logs/`. Open-engine suite: 914 passed / 47 skipped; CTest 3/3.
No performance claim: power-mode request failed and CPU partial scoring ran
concurrently with runtime validation. Prefill and serving remain unvalidated.

## Localizing the FFN front from retained captures

No new NPU run or full residency is required for this checker:

```bash
OPENBLAS_NUM_THREADS=1 ironvenv/bin/python open_kernels/model/ffn_front.py \
  --model-dir Models/qwen38-27b/Qwen3.8-27B-NPU2 \
  --kernel-dir Models/qwen38-27b/production-fp32-kernels \
  --out Models/qwen38-27b/production-fp32-multitoken \
  --layers "$(seq -s, 0 63)" \
  > Models/qwen38-27b/production-fp32-multitoken/ffn-front.json
```

The current run intentionally exits 1: layer 11 / position 0 combined activation
maxrel 0.0071636206 exceeds 0.005. Norm passes all 192 cases (worst 0.0036529193).
Combined activation from captured BF16 norm also passes all 192 (worst
0.0016461914), but cannot override the primary unrounded-input gate.

With `--layers 11`, position 0's BF16-rounded-norm diagnostic falls to
0.0003203473; ideal rounding alone predicts 0.0074496288 activation error.
This identifies local norm-input rounding loss, not the root cause of all seven
full-chain residual failures. Separate up/gate outputs are not captured, so
combined-activation PASS is not independent proof of each GEMV. Next inspect
FP32 norm-to-up/gate preparation with isolated replay and unchanged acceptance
bounds. Captured `A_XM` / `AA_XM` is BF16; `RES` and `H` are FP32. Do not read
all three as FP32. The checker retains scope and hashes in its JSON output.
