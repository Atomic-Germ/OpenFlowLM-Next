---
name: open-dflash2-drafter
description: Build, gate and time the DFlash 2 block drafter (Ternary Bonsai 2's t2 drafter) on the open NPU path. Use when converting a drafter GGUF with q4nx-build --dflash2-drafter, rebuilding the dflash2 kernel family (gemv_t2n), running the drafter-vs-reference gate, or swapping in newer fine-tuned drafter weights.
---

# DFlash 2 drafter on the NPU (OPEN-DRAFT-DFLASH2)

The drafter is a 1.9B-parameter block drafter: five Qwen3 layers read the target's residual at five tap
layers and propose a 7-token path through a top-16 selector. Ours is **ternary**: every matrix is t2 in the
1024-block Hadamard basis (trained by `utilities/dflash2_train.py`), so it runs on the same 2-bit chunks
(OPEN-QUANT-T2) as the target. Spec: `specs/open-engine/spec.md` (OPEN-DRAFT-DFLASH2, -WALK, OPEN-GEMV-T2N,
OPEN-CONVERT-DFLASH2). Kernels are built, not checked in.

## Why a new kernel and not the prefill GEMM

`gemm_q4_prefill` is built at T = 256 and refetches the activation per row-block group: at 8 rows it streams
~4x the weight bytes and computes 256 rows. `designs/gemv_t2n` is `lm_head_t2`'s dataflow (8 cores, one shim
stream each) on gemv_t2.h's chunks with the mmul's four rows as a token QUAD (two quads = 8 tokens); the host
builds the int16 table for one K slice (<= 8 k-tiles, so it fits L1) and adds the slices' partial y. One
xclbin serves every shape (the core program reads its counts from the table's header); the shape is the
instruction stream. Core program: 4.2 KB of 16 KB.

## Steps

1. Convert (6 min, 1.4 GB):
   `python -m q4nx.cli -i <drafter.gguf> --dflash2-drafter --target-gguf <Ternary-Bonsai-2-27B-PQ2_0.gguf>
   --drafter-codes <drafter_t2_best.pt> -o <dir>` (from `utilities/q4nx-build`). Without `--drafter-codes`
   the codes are recovered from the GGUF's bf16 weights (refused if they are not ternary in the basis).
   New fine-tuned weights are the same format: rerun this, nothing else changes.
2. Kernels (WSL `ironenv142`, ~1 min, 8 shapes): `python open_kernels/export_qwen36_kernels.py --model-dir <dir>
   --out src/xclbins/<name>/open_kernels` with `PATH=~/xrt-tools/bin`, `LD_LIBRARY_PATH=~/xrt-tools/lib`.
3. Kernel test: `designs/gemv_t2n/make_test.py --n N --k K`, `run_kernel.exe test/run.cfg`, `--compare`.
4. Taps: d01's `tf3.exe` over PrismML llama.cpp (HIP) with `DACC_DUMP`, 3-8 min a prompt.
5. Reference: `utilities/dflash2_ref.py drafts` (batched, CPU, ~1 min a prompt); NPU:
   `open_qwen36_cli --layers 1 ... --draft <dir> --draft-kernels <set> --draft-run <run> --draft-pids ...`;
   `utilities/dflash2_ref.py compare`.

## Using it (OPEN-DECODE-SPECULATIVE method `dflash2`)

- Install: `oflm-add <Bonsai container dir> --tag <tag> --speculative dflash2 --draft-model <drafter container dir>`
  copies the drafter beside the model, links its kernel set (found by spec_hash: build it with step 2 first) and
  writes `"speculative": {"method": "dflash2", "draft": "<drafter dir name>"}`. Refused, before the registry is
  written, without a verify program in the target's kernel set (`open_kernels_w4h`: verify width 4), for a
  drafter trained for another target, or with no kernel set installed for it.
- Serve: `oflm serve <tag> --spec` (or `OFLM_OPEN_SPEC=1`; `OFLM_OPEN_SPEC_WIDTH=2..4`). Greedy requests only.
  `OFLM_OPEN_DRAFT_KERNELS_DIR` points at a drafter kernel set that is not linked.
- CLI: `open_qwen36_cli --model <Bonsai container> --kernels <its set> --ids-file p.ids --max-tokens 256
  --gemm-block --spec --spec-method dflash2 --draft <drafter container> --draft-kernels <dflash2 set> [--spec-width 3]`.
- Time it only on an idle box, spec off / lookup / dflash2 interleaved; width 4 is the default and the best.

## Traps

- `cmd.exe /c script.cmd` from Git bash needs `MSYS_NO_PATHCONV=1` and the script's full Windows path, or it
  opens an interactive shell and the "build" never runs.
- The llama.cpp GGUF lists tap layers as the INPUT of 6, 20, ... = the OUTPUT of 5, 19, ... (config.json's
  `target_layer_ids`, which is what the engine reads).
- Block rows are non-causal inside the block, so a draft of k tokens is the walk's first k of an 8-row block,
  not a (k+1)-row block: agreement with llama.cpp's k = 3 drafts is lower than with the 8-row reference.
- `gemv_t2n.cc` at 8 float accumulators live spills ~2x; keep only the four mmul accumulators live and
  accumulate y in memory (`t2n_half`). Eight mmul accumulators (split octets) spill worse (681 vs 424 bundles).
- Timings on this box are meaningless while other sessions run CPU jobs (the NPU clock follows the package
  power budget): check CPU load before reading any number.
