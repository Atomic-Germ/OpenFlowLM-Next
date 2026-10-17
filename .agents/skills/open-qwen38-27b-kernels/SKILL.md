---
name: open-qwen38-27b-kernels
description: Build, verify and ship the open XDNA2 kernel sets (lx ax ln lm_head_q8 + the block prefill route) that run Qwen3.8-27B -- the Qwen3.5 dense recipe at hidden 5120, FFN 17408, 48 DeltaNet value heads. Use when rebuilding them, adding another Qwen3.5/3.8 size wider than the 9B, converting a Qwen3.5-family GGUF whose value heads outnumber its key heads 3:1 or more, or when a Qwen3.5 model's kernel compares pass but its text is garbage.
---

# Qwen3.8-27B on the Qwen3.5 dense recipe

The 27B is the fifth size of `open_kernels/recipes/qwen35.py`. Spec: OPEN-FAMILY-QWEN35
(its 2026-10-01 result) and OPEN-CONVERT-QWEN35-VHEADS in `specs/open-engine/spec.md`;
engine notes: `src/open_qwen36/README.md` ("The 27B"); plan:
`specs/open-engine/plans/archive/qwen35-27b.md`.

## Read this first: check the CONTAINER, not only the kernels

The kernels and `model/replica_qwen35.py` read the same `model.q4nx` bytes. A converter bug
therefore passes the slice, the engine's bit-identity and a 48-layer prompt compare -- and
the model still answers in fragments. That is exactly what happened with the first published
`Atomic-Germ/Qwen3.8-27B-NPU2`. Before any chat test on a new container:

```
# the HF shard holding layers 0 (linear) and 3 (full): see the repo's model.safetensors.index.json
python open_kernels/model/container_vs_hf.py --model-dir <container> --hf-shard <shard.safetensors>
```

`ALL MATCH` (quantized projections >= 0.99 correlation) is the bar. A value-head scramble shows
as every head-indexed DeltaNet tensor near 0 and everything else ~0.997.

**The bug that was:** llama.cpp stores Qwen3.5's value heads tiled (GGUF head `r*num_k + kh`
= HF head `kh*grp + r`). `q4nx-build`'s GGUF path untiled with `grp = 2` and took qkv's value
rows as "the second half" -- right for 32 heads over 16, wrong for 48 over 16. Fixed in
`q4nx/models/qwen35.py` (`v_untile`, `untile_qkv`, head counts from the GGUF's `qwen35.ssm.*`
keys); `tests/test_qwen35_vheads.py` pins it. The HF-safetensors path never reorders (HF is
already grouped).

## Convert

```
# unsloth/Qwen3.8-27B-GGUF, Qwen3.8-27B-Q8_0.gguf (27 GB)
cd utilities/q4nx-build
python convert.py -i <Q8_0.gguf> -o %USERPROFILE%\.flm\models\Qwen3.8-27B-NPU2 -s Atomic-Germ/Qwen3.8-27B-NPU2 -t language
```

~50 min of mostly one core, ~20 GB peak RAM (it holds every tensor until export). The 5120
width matches no `QWEN35_VARIANT_DIMS` entry and routes to the 9B's name map -- correct, the
names are the same. Text-only (the skeleton has no `vision_weight.q4nx`).

## Build (native Windows)

```
cd C:\dev\mlir-aie; . .\iron_env.ps1
cd <repo>
python open_kernels\export_qwen36_kernels.py --model-dir %USERPROFILE%\.flm\models\Qwen3.8-27B-NPU2
```

41 sets. The four the decode step runs (`lx` ~3 min, `ax` ~7, `ln`, `lm_head_q8`) plus the
block prefill route: 32 `ag_s*` / `ag_pv*` at AG_M 1536 and five GEMMs (`gemm_n5120_k6144`,
`gemm_n5120_k17408`, `gemm_n14336_k5120`, `gemm_n16384_k5120`, `gemm_n34816_k5120`). The route is
~25 min of CPU at BelowNormal (the attention products 30-90 s each); the engine refuses a
manifest whose kernels are missing, so either build it all or test on a copy of the kernel dir
whose `manifest.json` has the `gemm_*` / `ag_*` kernels, contexts and globals and every
`gemm_block` removed (sequential prefill). A killed export reports the set it was on as
`build FAILED (1)` with no aiecc output -- rerun it before suspecting the design.

**Never run two exports in one checkout at once**: each regenerates `designs/layer_x`'s kernel
TUs for its own spec. Use a `git worktree` for a parallel baseline.

Program memory is close: `lx`'s glue core 15 584 B and main cores 14 624 B, `ax` 14 400 B of
16 384. `OFLM_KEEP_FAILED=1` makes `build_design.py` keep `final.prj` when aiecc fails, so
`llvm-size -A final.prj/elfs_*/*.elf` can say which core overflowed.

## What is different at this width (all gated; other sizes compile byte for byte)

| wall | fix | where |
|---|---|---|
| FFN 17408's table (39 KB) overflows a main core | down GEMV as K 8192 + 9216, strided pool taps, `out2` + `out2b` | `qwen36moe.down_split`, `xcommon.ffn_body` / `ffn_sequence` / `piece_tap` |
| norm helper: 6 x 10 KB elements + stack > 64 KB | streamed residual (`ln_add2`), norm read back via `ln_nr`; close with `ln_add3` | `norm_split`, `xcommon.ln_split_*`, `designs/ln/ln.py` SPLIT |
| 48 value heads vs a 32-lane accumulator | 64-lane tile, 32 rows per 4 KB element | `designs/dn_glue/glue_ab_w.cc`, `Linear.AB_LANES/AB_ROWS` |
| 3 xn halves = 14 side fills > 13 | half-outer walk, 11 fills | `glue_fills`, `Linear.GLUE_HALF_OUTER`, lx.py |
| 3-element xn / xm / og in a 2-deep x fifo | prepare + release one element at a time | `xcommon.prep_stream` |

Attention (24 heads over 4 kv, HD 256, gated) needed no change: six cores x 4 heads, RB 1.
q8 `ssm_out_proj` narrows to q4_1 (5120 is not in `MIXED_CORE_FITS`).

## Verify

1. `OPEN_KERNELS_UNVALIDATED` is no longer needed (the points are in the catalogue).
2. Slice: `python open_kernels/model/make_decode.py --model-dir <m> --layers 8 --tokens 3 --out open_kernels/model/out_q27`,
   `open_kernels\harness\out\run_kernel.exe open_kernels/model/out_q27/run_decode.cfg`,
   `python open_kernels/model/compare_decode.py --tokens 3 --out open_kernels/model/out_q27`.
   2026-10-01: 0.999998 / 0.999997 / 0.999998, residual 0.999999.
3. Engine: `src\open_qwen36\out\open_qwen36_cli.exe --model <m> --kernels <k> --ids 248045 --max-tokens 3 --layers 8 --dump-logits <p> --twice`
   then compare `<p>_t*.bin` with the harness's `y_logits*.bin`: 0.000e+00.
4. Whole-prompt depth check (any depth N): `model/replica_prompt.py --layers N` against the CLI's
   `--layers N --dump-logits` on the same ids. ~30 s per layer of fp64 on the CPU.
5. `container_vs_hf.py` (above), then `chat.py` with `PYTHONIOENCODING=utf-8` (a redirected
   stdout is cp1252 and the tokenizer emits UTF-8).

6. The block route (OPEN-PREFILL-BATCH step 7): each GEMM shape through
   `designs/gemm_q4_prefill/make_test.py --shape nN_kK --tokens 256`, `run_kernel.exe`,
   `compare.py` (rel_fro 1.64-1.77e-3); then the CLI with and without `--gemm-block` on a
   prompt of MORE than 256 ids (`--ids-file`, comma-separated), `--layers 4 --prefill-logits`,
   plus `--gemm-block --block-major` (byte-identical to layer-major); then all layers,
   `--max-tokens 8`. `chat.py`'s short prompts never take the route -- the engine needs >= 64
   prompt tokens -- so a chat compare says nothing about it.

Speed, a quiet box, a 300-token prompt: prefill 50 ms/token on the route (386 one token at a
time, 7.8x), decode 419-450 ms/token. Check `Get-Counter '\Processor(_Total)\% Processor Time'`
first: under another CPU-bound job the same decode read 594-815 ms/token.
