---
name: open-granite-kernels
description: Build, verify and ship the open XDNA2 kernel sets (dx ln lm_head_q4, the prefill GEMMs and the attn_fa prefill attention) that run IBM Granite 4.2 3B on the dense recipe. Use when rebuilding those xclbins, adding another Granite size, debugging "no open kernels found" for a Granite tag installed with oflm-add, debugging or speeding up attn_fa (the NPU prefill attention), or when a Granite container's attention_multiplier is refused at load.
---

# Granite 4.2 3B on the dense recipe

Granite is the **fifth family** on `open_kernels/recipes/dense.py`, and the
first at **head_dim 64** — the point every shipped OpenFlowLM design refused,
because head_dim is intrinsic to RoPE and cannot be padded. Nothing in the
design changed for it: `ATTN_HD` / `ATTN_NH` are compile-time macros and
`designs/attn/attn.h` already carried HD 64's `kScale = 0.125f`.

Kernels are **built, not checked in** (`.gitignore`: `src/xclbins/BERT-h*/` and
the model's `open_kernels/`). Reference docs: `specs/open-engine/spec.md`
(OPEN-FAMILY-GRANITE), `src/open_qwen36/README.md`, `open_kernels/PROVENANCE.md`.

## The fold — read this before anything else

Granite is Llama plus four scalars: `attention_multiplier` (replacing the
implicit `hd**-0.5`), `embedding_multiplier`, `residual_multiplier`,
`logits_scaling`. `ModelSpec` expresses none of them and `attn.h` hard-codes
`1/sqrt(HD)`.

They do not need expressing, because **q4nx-build folds all four into the
weights** at conversion: `q_proj *= attention_multiplier * sqrt(hd)`,
`o_proj`/`down_proj *= residual_multiplier`, `embed_tokens *=
embedding_multiplier`, `lm_head /= logits_scaling`. For 4.2-3B the only
non-unit factor is `attention_multiplier = 0.015625` at hd 64, so the fold is
`q_proj *= 0.125` and the folded `config.json` reads `attention_multiplier =
0.125 = 64**-0.5` exactly. A power of two, so it is exact in bf16. The
container keeps the originals under `q4nx_folded_multipliers`.

**An unfolded container is refused twice**, by name, and both refusals matter
because an unfolded model does not crash — it returns plausible garbage:

- `recipes/spec.py::_granite_scale_check`, at generation, from HF `config.json`
  and GGUF metadata alike.
- `hf_config_check` carries `attention_multiplier`, so the **engine** refuses at
  load. Swapping a different `model.q4nx` under a built kernel set is otherwise
  silent.

## Build — on Windows, not WSL

This is the opposite of the Qwen3.6 skill's advice, and deliberately so.
`aiecc` emits `insts.elf` through `aiebu-asm`, and `build_design.py` asks for it
unconditionally. **Windows has `aiebu-asm.exe` and `xclbinutil.exe` from
`C:\Xilinx\XRT`**, both on PATH after `iron_env.ps1`. WSL has neither: Ubuntu
24.04's own XRT is `202210.2.13.466` (2022, pre-NPU) and building Xilinx/aiebu
needs Boost + liblzma + liblz4, i.e. root. Nothing in the build path is
OS-specific — an xclbin is a device artifact with no host code in it.

```powershell
cd C:\dev\mlir-aie; . .\iron_env.ps1          # MUST be dot-sourced
cd <repo>
python open_kernels\export_qwen36_kernels.py --model-dir C:\Users\<you>\.oflm\models\Granite-4.2-3B-NPU2
```

Three sets, from build dirs `dense/build_granite_h2560`, `ln/build_2560_1e-05`,
`lm_head_q4/build_100352`, into
`src/xclbins/Granite-4.2-3B-NPU2/open_kernels/{dx,ln,lm_head_q4}` plus
`manifest.json`, `spec.json`, `toolchain.json`.

`OPEN_KERNELS_UNVALIDATED=1` is **no longer needed** — head_dim 64, num_heads 40
and gemv_q4 K 8192 entered `catalogue.py` with the 2026-09-06 run below. It is
needed again only for a *new* unvalidated point.

## The standalone test binaries

Neither ships built. Both need the same three variables:

```powershell
$env:XRT_INCLUDE_DIR = "C:\Xilinx\XRT\include"
$env:XRT_LIB_DIR     = "C:\Xilinx\XRT\lib"
$env:VCVARS64        = "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat"
cmd /c open_kernels\harness\build.cmd      # -> open_kernels\harness\out\run_kernel.exe
cmd /c src\open_qwen36\build.cmd           # -> src\open_qwen36\out\open_qwen36_cli.exe
```

`VCVARS64` must be set: the scripts default to VS 2022 **BuildTools**, and this
box has Community **18**.

## Verify — the ladder, in order

The three commands in step 1 are **one chain**, not three choices:
`make_decode.py` writes the fixture and the fp64 reference, `run_kernel.exe`
produces `y_*`, `compare_decode.py` scores one against the other.

```powershell
python open_kernels\model\make_decode.py --model-dir <model dir> --layers 4 --tokens 2 --out open_kernels\model\out_gr
open_kernels\harness\out\run_kernel.exe open_kernels\model\out_gr\run_decode.cfg
python open_kernels\model\compare_decode.py --tokens 2 --out open_kernels\model\out_gr

src\open_qwen36\out\open_qwen36_cli.exe --model <model dir> --kernels src\xclbins\Granite-4.2-3B-NPU2\open_kernels `
    --ids 100264 --max-tokens 3 --layers 4 --dump-logits out_gr\y
python src\open_qwen36\chat.py "Explain what an NPU is in two sentences." --model <model dir> --kernels src\xclbins\Granite-4.2-3B-NPU2\open_kernels
```

**Prompt id 100264 = `<|start_of_role|>`.** *Not* `config.json`'s
`bos_token_id` 100283, which decodes to `</documents>` and disagrees with
`tokenizer_config.json`'s own bos (`<|end_of_text|>`). The container carries
both; only one is how a Granite prompt starts. `make_decode.py`'s per-family
default is 100264 for this reason.

Granite also closes **both** roles with `<|end_of_text|>`, so the turn end and
the eos are the same token — the other three families have a separate pair, and
`chat.py`'s Granite branch says so explicitly.

## The combined kernel set (Uno decode + bf16 and bfp16 prefill)

One `export_qwen36_kernels.py` run builds everything Granite ships, no manual step: `dx`, `dx_attn`, `dx_f32`, `ln`,
`lm_head_q4`, `lm_head_q4_f32`, the four bf16 `gemm_n*_k*`, the four `gemm_bfp16_n*_k*`, `dxl`, `dxl_lora` and `lmhl`, with one
`manifest.json` (manifest_version 4). `dxl`'s weight fifo depth (`DXL_WDEPTH`) defaults to 4 where L1 holds it, so no env is needed.

```powershell
cd C:\dev\mlir-aie; . .\iron_env.ps1
cd <repo>      # (Get-Process -Id $PID).PriorityClass = 'BelowNormal' first; one build at a time
python open_kernels\export_qwen36_kernels.py --model-dir C:\models\Granite-4.2-3B-NPU2 --out <repo>\src\xclbins\Granite-4.2-3B-NPU2\open_kernels
```

About 2 h at BelowNormal on this box, the `dxl` / `dxl_lora` builds being the long ones. Then `open_qwen36_cli --kernels <set>`; pass it
explicitly, `C:\models\Granite-4.2-3B-NPU2\open_kernels` is a symlink to an old tree. `--prefill-mode bfp16` picks the bfp16 route.
`--uno N --gemm-block` prefills the prompt as blocks before the Uno cycle, so a 1024 / 2048-token context costs seconds, not a
token-at-a-time prefill. The plain-decode reference inside `--uno` prefills all but the last token the same way and steps the last
one, so both sides start from one cache (block-prefilling the seed's row instead differs in bf16 and flips near-tie argmaxes: Uno
then reads DIFFERS at token 27).

Measuring: `utilities/granite-combined-bench/matrix.py` runs prefill / decode / Uno-at-depth / the six-prompt sweep as small units, each
under `dibs run --measure`, rerunning a NOISY unit. A stray CPU-bound process (an orphaned `find`, a browser tab) makes every unit
NOISY; read which process `dibs` names before retrying.

## Result — 2026-09-06, mlir-aie 1.4.2.dev16+g7e00b57 / Peano 21.0.0.2026080301

Correctness, 4-layer slice: logits corr **0.999998 / 0.999990**, same argmax
(38457) and an **identical top-5** at both positions, residual corr
0.999990–0.999999 in every layer. A coherent two-sentence answer through
`chat.py`, ending on `<|end_of_text|>` at token 52.

**Performance: at zero context `dx` wins; past that, context dominates and it
is not Granite's fault.** `part0` 60.0 ms over 40 layers at position 0 is
**1500 µs/layer**, against the hand-written four-dispatch Granite kernels in
`vegah/OpenFlowLM@feat/kernels` at **1744.7 µs/layer** — 1.16×, the direction
one dispatch per layer was expected to give.

Everything above that is a term linear in position, and it is the **dense
recipe's**, not this family's. Measured the same evening with
`open_kernels/model/sweep_positions.ps1`:

| | at position 0 | per layer per position |
|---|---:|---:|
| Qwen3-4B (36 layers, 32 heads @ hd 128) | 1.58 ms/layer | 41.6 µs |
| Granite-3B (40 layers, 40 heads @ hd 64) | 1.50 ms/layer | 49.6 µs |

At position 2048 one token costs 3–4 seconds on either. See
**OPEN-ATTN-CONTEXT** in `specs/open-engine/spec.md` for the full table and for
what the cost is *not*: Granite reads half the KV bytes per position per layer
and does fewer MACs, yet its slope is 19% steeper, so neither bandwidth nor
arithmetic explains it. Head count (40/32 = 1.25 against a measured 1.19) is
what tracks.

**So do not quote a Granite tok/s number without the position.** 5.92 tok/s
over 63 tokens and 1500 µs/layer at position 0 are both true and are not the
same measurement.

## Decode cost versus context depth (2026-10-14)

Plain decode fell from 20.7 tok/s at context 0 to 10.7 at 2048 while Vulkan stayed at ~23. Nothing on the host grows with position
(`OFLM_OPEN_STEP_TRACE=1 open_qwen36_cli --ids 100264 --at-position P --bench-decode 5 --bench-step 10`): embed 0.03 ms, patch 0.00,
route 0.00, `ln` 0.2, `lm` 3.6, all flat; the 40 `dx` dispatches carry the whole growth, 0.54 us per layer per position. Attention is
inside `dx` (5 cores x 8 heads, RB 4, the whole KV window broadcast on `ain` after the q/k/v GEMVs), so the cost is that kernel's.

Where one layer's 1.12 ms of growth at 2048 sits (base kernel, `dx` min per dispatch, all CLEAN):

| piece | ms/layer at 2048 | how it was priced |
|---|---:|---|
| window stream alone (`ATTN_NULL=1`) | 0.28 | the cores take the rows and compute nothing |
| score phase | 0.41 | `ATTN_ABL=1` minus the stream |
| softmax | 0.19 | `ATTN_ABL=2` minus `ATTN_ABL=1` |
| value accumulation | 0.24 | the rest |
| pure compute, all three | 0.83 | `ATTN_REPEAT=3` minus 1, divided by 2 |

The stream and the compute **add** (0.28 + 0.83 = 1.11 against 1.12 measured). A deeper `ain` ring changes nothing, and that is why: the objectfifo lowers to
max acquire + 1 buffers whatever depth the design declares (declared 10 and 24 both gave 9 `ain_0_cons_buff_N` in
`input_with_addresses.mlir`; `lni` 5 -> 6, `x` 2 -> 3), so the DMA has one spare 1 KB buffer while the core holds eight. Giving it room did not help
either: copying each block into eight local 1 KB buffers (`acquire(8)`, a vector copy kernel, `release(8)`, then `attn_stepb` on the
copies) was bit-identical and slower, 2.080 against 1.986 ms a dispatch at 2048 (both with the rolled tree: beside the copy the
unrolled one overflows program memory, and it reads 1.929 alone). The copy cost what it should (~0.1 ms) and bought nothing, so the 0.28 ms is not DMA wait; suspect the per-block acquire / release and the
9-way pointer `switch` for each of the 8 elements (unmeasured). RB 2
is worse than RB 4 (+0.2 ms/layer), so the per-block cost is real but is not the bulk either.

The function's flat bundle count (419 for `attn_rowb_impl`) is a third of what a block costs: counting the per-head score and value loops
at their 8 trips gives ~2,770 cycles, which is the 1.6 us a block that `ATTN_REPEAT` measured if the core runs near 1.7 GHz (inferred,
not read from a counter). And **the bundle count of a function does not
include loop trip counts** (`utilities/dxl-asm/attn_bundles.py` prints bundles per basic block). The score loop also called
`__muldi3` (64-bit software multiply) once per head per block for `h / 5`, Granite's 5 q heads per kv head.

Per-change attribution, one `dx` dispatch at 2048 (min of 21, CLEAN; each row adds to the one above): base 2.212 ms; `ATTN_KVH_MUL` 2.092;
+ the rolled tree 2.057; + rows unrolled by 4 with `ATTN_EXP32` 1.996; + the conditional rescale exponential 1.929. The ablation rows
above come from builds that also change the DMA schedule, so the split of the 1.12 ms is a pricing, not a trace.

What shipped, all bit-identical to the old kernel (the logit files byte-identical over 70 + 20 and over 2100 + 20 positions; 10 prompts x 96 greedy tokens
identical; `--rows-check 17` ROWS PASS), all under `recipes/attnknobs.py: tree8()` for Granite only:
- `ATTN_KVH_MUL`: `h / 5` as a 32-bit multiply-shift (`attn_kvh`).
- `ATTN_TREE8`: the 8-heads-a-core score tree (`attn_score2g`, one tree across the block's heads instead of 32 `reduce_add`s), with
  `ATTN_T8_UNROLL=4` rows unrolled on `dx`, 1 on `dxl` (whose cores carry the skip / pack kernels: 4 overflows program memory, 2 stops
  the backend with "immediate operand out of range"); a core's 8 heads span two or three kv heads (5 q heads each), so the two heads of
  a pair can need different K rows.
- `ATTN_EXP32`: the 8-lane exponential runs on the 32-lane routine (`vexpN` is lane-wise), which frees ~1.5 KB of program memory; the
  unrolled tree needs it (15,472 of 16,384 bytes with it; rows unrolled by four overflowed without it, and the rolled tree is 13,072).
- Under `ATTN_TREE8` the rescale factor `a = exp(m - mn)` is computed only when some lane's max rose (it is exactly 1 otherwise): the
  second 32-lane exponential, ~257 bundles a block, ran every block before. Profile at 2048: step 82.0 -> 78.7 ms.
- Stack 0x3000 for the attention workers. **0x1800 builds and runs and reads back NaN** from the first blocked position (pos >= 4):
  the unrolled tree spills past it. A NaN `xres after layers` at `--ids` of 5+ tokens is this.

Probe variables for the next attention experiment (`PROBE_VARS`, in the build key): `ATTN_NULL`, `ATTN_ABL` (1 score, 2 + softmax,
now honoured by the block kernel), `ATTN_REPEAT=n` (the block computed n times), `ATTN_TREE8=0|1`. A `dx`-only build takes ~75 s:
`export_qwen36_kernels.py --model-dir <model> --only dx --out <scratch>`, then copy `dx/final.xclbin` and `dx/insts.bin` over a copy of
the union set (the manifest needs no edit unless `rb` changes). Check a probe took effect in
`dense/build_granite_h2560/final.prj`: `aie.mlir` (the objectfifo depth), `input_with_addresses.mlir` (`ain_0_cons_buff_N`: a depth-10
ring lowers to 9 buffers) and the `.text` size (`llvm-size -A elfs_main_core_2_3/*.elf`). A fifo-depth change moves only ~85 bytes
of the xclbin (the ring's wrap constants), so a near-identical xclbin proves nothing either way.

Not done, and where the rest is: the stream (0.28 ms) is 2 KB a position broadcast to every core, of which a core needs a quarter; a
stream per core on the five free shim MM2S channels would cut it, at the cost of a 3-D window BD the position patch
(`stream_patch.hpp attn_table`) does not know. Eight attention cores (one per kv head) would cut the compute by 3/8 but need a memtile
join for the og drains (the shim S2MM channels are all taken).

## Prefill attention on the NPU: attn_fa (2026-10-15, OPEN-PREFILL-ATTN-FA)

`open_kernels/designs/attn_fa/` is Granite's block attention as ONE dispatch a 256-token block: scores, causal mask, online
softmax and both products on the array. It is the default on both prefill routes; `OFLM_OPEN_ATTN_FA=0` is the host
attention for A/B (and skips loading the `af` context). The load log says `gemm-block attention: NPU (attn_fa ...)`.

Why not `attn_block` (two GEMMs a kv head, host softmax) at head dim 64: the fp32 score matrix is L / 64 times bigger
than Q or K, so writing it from the array, softmaxing it on the host and writing P back (twice for the Q hi / lo split)
costs more than the host attention it replaces. Not built.

Build: the exporter builds `af256` .. `af4096` (one xclbin; the core program does not depend on L, ~15 s a stream). For a
quick set, `AF_LMAX=2048` in the environment narrows the recipe to 8 streams; longer prompts then take the host attention.

```powershell
cd C:\dev\mlir-aie; . .\iron_env.ps1
cd <repo>                       # copy the union set to <set> first; --only builds just these into it
python open_kernels\export_qwen36_kernels.py --model-dir C:\models\Granite-4.2-3B-NPU2 --out <set> --only af256,af512,...,af4096
```

One shape on the array: `python open_kernels\designs\attn_fa\make_test.py --build <build dir> --out <t> --L 512`,
`run_kernel.exe <t>\run.cfg`, `python open_kernels\designs\attn_fa\compare.py <t>` (PASS at rel_fro ~1.1e-3 vs fp64).
The model: `utilities\dense-decode-probe\prefill_mode_check.py --arms bfp16:host,bfp16:npu` (or `fast:host,fast:npu`).
Timing with the stage split: `utilities\granite-combined-bench\attn_bench.py --lengths 512,2048 --dispatch-log`.

Traps hit building it (each cost a hardware round):
- **`broadcast_elem` on bf16 is `vextbcst.32`**: it broadcasts a 32-bit unit, i.e. a PAIR of bf16. Double the values first
  (`interleave_zip(v, v, 1)`) and broadcast element i of the doubled half. Without it every product reads the wrong value
  (rel_fro 1.25, no hang, no NaN).
- **The emulated bf16 `mmul<4,8,8>` spills**: with hi / lo operands its expanded A vectors (8 a mmul) go to the stack
  every iteration. The kernel uses element-wise MACs with the query rows in the 32 lanes instead (S^T layout), which also
  makes the softmax lane-wise.
- **Peano's 8-lane float `reduce_add` kills the backend** ("Unsupported register size" in RegBankSelect); a 4x8 block
  times ones on the MAC array is the row sum.
- **Stack**: four interleaved exponentials in `af_sm` take a 5 KB frame; at a 4 KB stack the array hung (state 8) or,
  slightly over, returned NaN in ~5 % of rows. `STACK = 0x2000`. Read the frame from `llvm-objdump -d` (`paddxm [sp], #N`).
- `AF_INFLIGHT=4` (four passes' BDs queued) hangs; 2 works.
- The harness's first runs are warm-up: take the min of 8 (make_test writes 8 runs).

Where the time goes (2026-10-15, CLEAN): a dispatch is ~0.1 ms + ~0.69 ms per 256 rows of window (af512 1.47 ms, af2048
5.6 ms; ~17 us a 64-key chunk a pass, about 1.7x the MAC-bound estimate). The first dispatch of a layer also pays the
switch into `af` (~1.7 ms) and the out projection the switch back (~1.2 ms). Ablation at L 512 before the softmax rewrite:
DMA and sync floor ~0.4 ms, softmax 0.62 ms, P.V 0.34, Q.K^T 0.17 (the softmax is what four interleaved chains fixed).

## Rules

- Never commit the xclbins; the distributed package ships them pre-built.
- A catalogue point enters `catalogue.py` **after** its hardware run, never
  before. `OPEN_KERNELS_UNVALIDATED=1` is how you get the run, not how you ship.
- Do not assert a validated set's membership in a test. Three tests had to be
  edited when Granite's points landed, all because they spelled the set out —
  which fails on the one event that is never a regression.
- Granite has **no closed engine** to fall back on (`llama_npu` whitelists
  `hidden_size` to {2048, 3072, 4096} and refuses 2560). A missing kernel set is
  a named refusal naming the build command, not a quiet substitution.
- Prefill is not decode-as-prefill and not `dxB`: the engine takes the block route layer-major (OPEN-PREFILL-HOST-ATTN,
  60 -> 220 tok/s at pp512), with the attention on the NPU where the set carries `attn_fa` (OPEN-PREFILL-ATTN-FA). Measure
  it with `open_qwen36_cli --gemm-block --ids-file`; a kernel dir carrying the Uno `dxl` / `lmhl` entries needs the Uno
  engine build to parse it.
