---
name: open-qwen3-8b-kernels
description: Build, verify and serve the open XDNA2 kernel set for Qwen3-8B-NPU2 (the dense recipe at hidden 4096), including the 8-bit dit_gemm prefill route (`fast`) and the q4_1 route kept as `lean`. Use when re-exporting the set, after touching recipes/dense.py, designs/dit_gemm or the bfp16_dit pack op, or when checking prefill speed, accuracy or memory against the previous route.
---

# Qwen3-8B-NPU2 on the dense recipe

Spec: `specs/open-engine/spec.md` OPEN-FAMILY-QWEN3 (the set) and OPEN-PREFILL-GEMM8 (the 8-bit
prefill route). Spec file: `open_kernels/recipes/specs/qwen3-8b.json`. Archived plan with the
kernel measurements: `specs/open-engine/plans/archive/prefill-gemm8.md`.

## Export (WSL, serial)

```bash
source ~/ironenv142/bin/activate
export PATH=~/xrt-tools/bin:$PATH LD_LIBRARY_PATH=~/xrt-tools/lib     # both, or xclbinutil dies late
cd /mnt/c/code/openflowlm-next
python open_kernels/export_qwen36_kernels.py --spec open_kernels/recipes/specs/qwen3-8b.json
```

The set lands in `src/xclbins/Qwen3-8B-NPU2/open_kernels` with `manifest.json` and
`toolchain.json`. It carries the decode kernels, the q4_1 block route (`gemm_n*_k*`, the `lean`
variant) and the 8-bit one (`dit_n*_k*`, all on one `dit` context). The build cache skips
what the `build_key` already covers; `--force` rebuilds. `dit_gemm` sources are part of the key,
so editing `designs/dit_gemm/` rebuilds the dense sets.

`--check DIR` against a previous export: `insts.bin` byte-identical, `final.xclbin` equal apart
from the ~80 stamp bytes. Use it when the change should not move a compiled byte.

## Verify the manifest

- `manifest_version` is 4; `layer_types.dense.gemm_block` is `gemm: "dit"` at `t` 1024 and
  `gemm_block_variants.lean` is the q4_1 route at `t` 256.
- `python -m pytest specs/open-engine/tests/test_prefill_gemm8.py specs/open-engine/tests/test_build_cache.py`
- `manifest_test` and `pools_test` (`src/open_qwen36/build.cmd`) for the parser and the
  `bfp16_dit` pack op.

## Serve and test

```powershell
$env:OFLM_OPEN_KERNELS_DIR = "C:\code\openflowlm-next\src\xclbins\Qwen3-8B-NPU2\open_kernels"
$env:OFLM_SERVE_PORT = "52626"          # 52625 is often taken
oflm serve qwen3:8b                      # --prefill-mode lean for the q4_1 route
oflm-test --llm
```

The route line at load reads `T = 1024 (fast)` or `T = 256 (lean)`. Kill `oflm.exe` afterwards
or the next Ninja build fails in its copy step.

## fast against lean

`fast` (the default) makes a 9-bit-a-weight copy of the five projections at load (~7.8 GB;
load ~6 s longer) and swaps it for the q4 route's per-layer weight buffers, so the peak is
13.3 GiB against 9.8 GiB for `lean`. On a box that cannot spare that, run `--prefill-mode lean`.
Measured 981-token prefill: 4.7 s (`fast`) against 11.1 s (`lean`).

## Accuracy gate (manual, when the 8-bit route changes)

Teacher-forced NLL over ~2048 tokens of natural text with `--prefill-logits`, `fast` within
0.5 % relative of `lean`; report KL, top-1 agreement and the worst sampled logits correlation
but do not gate on correlation (the 8-bit route moves single distributions more than the bf16
one). Also check byte-identical logits at 700 and 2000 tokens between scores-then-values and one
sub-block at a time (OPEN-PREFILL-ATTN).

## Traps

- The 8-bit route refuses `OFLM_OPEN_ATTN_BLOCK=0`, `OFLM_OPEN_HOST_ATTN` and
  `OFLM_OPEN_HOST_ATTN_DECODE` at load; the message names `--prefill-mode lean`.
- Time prefill under the shared timing lock on a quiet box; another session's build shifts a
  run by 5 % or more.
- A shape that does not fit (`K % 512`, `N % 1024`; Qwen3-4B's o_proj is 2560 wide) or a family
  with no `attn_block.prep` gets no `dit` route and stays below manifest version 4.
