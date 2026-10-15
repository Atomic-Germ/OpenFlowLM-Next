---
name: open-granite-kernels
description: Build, verify and ship the open XDNA2 kernel sets (dx ln lm_head_q4) that run IBM Granite 4.2 3B on the dense recipe. Use when rebuilding those xclbins, adding another Granite size, debugging "no open kernels found" for a Granite tag installed with oflm-add, or when a Granite container's attention_multiplier is refused at load.
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

One `export_qwen36_kernels.py` run builds everything Granite ships, no manual step: `dx`, `dxh` (the head, a second stream
over `dx`'s image), `dx_attn`, `dx_f32`, `ln`, `lm_head_q4`, `lm_head_q4_f32`, the four bf16 `gemm_n*_k*`, the four bfp16
streams (`gemm_bfp16_n4096_k2560`, `_n3072_k2560`, `_n16384_k2560_glu`, `_n3072_k8192_h`: OPEN-PREFILL-BFP16-FFN's fused
gate|up and down-from-H), `dxl`, `dxl_lora` and `lmhl`, with one `manifest.json` (manifest_version 5). `dxl`'s weight fifo
depth (`DXL_WDEPTH`) defaults to 4 where L1 holds it, so no env is needed. The q4nx tail is `dxh`; `ln` and
`lm_head_q4_f32` stay for the GGUF-direct section's tail. To refresh only the bfp16 streams, copy a full set and run the
exporter with `--out <copy> --only <the four names>` (~2 min); an engine that reads manifest_version 4 refuses the result
by name.

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
then reads DIFFERS at token 27). `--spec N` does the same for the per-cycle choice of prompt lookup, Uno and a plain step
(OPEN-SPEC-LOOKUP; how to run and measure it: the open-dxl-rows skill).

Measuring: `utilities/granite-combined-bench/matrix.py` runs prefill / decode / Uno-at-depth / the six-prompt sweep as small units, each
under `dibs run --measure`, rerunning a NOISY unit. A stray CPU-bound process (an orphaned `find`, a browser tab) makes every unit
NOISY; read which process `dibs` names before retrying.

bfp16 prefill with the glu FFN (2026-10-15, CLEAN, min / median tok/s): pp512 760 / 768, pp1024 487 / 492, pp2048 308 / 329,
against the five-step route's 520 / 531, 381 / 388, 269 / 274. Host attention is two thirds of pp2048. The skill for the route
itself is `open-gemm-bfp16`; the numbers and the accuracy gates are OPEN-PREFILL-BFP16-FFN.

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

## Decode at context 0: the weight stream and one context (2026-10-15)

A ctx-0 token is ~2.13 GB of q4_1 weights: 40 x 49.2 MB of `dx` and a 160.6 MB head. What a step cost on the base set,
CLEAN (`--bench-decode 5 --bench-step 12`, `OFLM_OPEN_STEP_TRACE=1`):

| piece | ms | how it was read |
|---|--:|---|
| real step, submit-ahead (min / median) | 46.3-46.6 / 46.4-46.9 | `--bench-step` |
| `dx` x 40 in the pipelined step | ~1.04 each | 49.2 MB at 46.9 GB/s |
| the first `dx` of a token, after the `lm` context | +1.1 | the trace's `mean(sw)` against `mean(=)` |
| `ln` | 0.21-0.26 | 0.087 alone (`--bench-kernel ln:40`): the rest is its context change |
| `lm` | 3.63-3.83 | 3.107 alone, 51.7 GB/s on 8 streams: ~0.6 is its context change |
| host (embed, logits read, gaps) | ~0.1 | the trace's gap / submit columns |

**The timeline method.** `--bench-kernel NAME:40` is a kernel's same-context minimum (one warm call first);
`OFLM_OPEN_STEP_TRACE=2 --bench-decode 5 --bench-step 12` prints the serial per-kernel table, the real step under both
schedules and the last step dispatch by dispatch (gap, submit, start-to-done, `<- context change`). A queued dispatch's
start-to-done includes its wait behind the one in front, so a layer's cost in the step is (step - tail - first-layer
penalty) / 40, not that column. `xrt-smi examine -r all` shows Windows' own `WorkloadsSessionHost.exe` contexts on the
same partition; `-r platform` reports the power mode and no power reading ("Estimated Power: N/A").

**Power mode and a trap.** `--pmode turbo` against `performance` on this step, 3 rounds interleaved, CLEAN: 41.1 / 41.2
against 41.1 / 41.3 ms min / median, no effect. A first probe read turbo 2x slower, but a performance run in the same
unit already showed serial dispatches finishing on a ~16 ms grid (`dx` min 1.06, mean 15.6 ms) before turbo was set: a
transient host-side state, not the mode. When a serial table's mean is many times its min, rerun before reading it.

**The channel map.** `dx`'s weights are 8 streams, one per shim column (`w0..w7` on `Tile(c, 0)`); the second MM2S of
columns 0, 1, 2 carries `lni`, `x`, `ain`, so 11 of 16 MM2S are used (columns 3..7 have one free each). S2MM: `y0..y7`,
`lno`, `aout`, `og0..og4` = 15 of 16. The head streams on the same 8 and reaches 51.7-52.5 GB/s, within ~3% of the
15-stream array ceiling (53): more streams are worth ~1 ms a token at most, `dx` was the one below the line.

**What fixed `dx` (OPEN-DECODE-STREAM, bit-identical).** A main core's weight fifo is 2 x 10 KB of L1, so DDR stopped
20 KB into every wait on an activation (entry norm, attention, mid norm, h gather), and each phase's first fill was
issued only after that wait. Each column's weights now go shim -> a 16-deep ring in its memtile -> core, and the o, first
up | gate band and down fills queue ahead of the wait in front of their phase. Step 46.4 -> 44.4 ms median, `dx`
~1.04 -> ~0.97 ms a layer (50.9 GB/s).
- Pin the ring's memtile channels to 0 (in) and 1 (out): aie2p memtiles split their 48 BDs into even / odd pools of 24
  and a ring takes one per buffer on each side, so both on channel 0 overflow past depth 12 (`emit_mlir.py --lower`
  prints each memtile's BD use).
- Issue order is the trick, and a wrong one deadlocks: `Pipeline(3)` awaits a channel's oldest fill before queueing a
  fourth, so a fill may go early only where that await needs no transfer issued after it (o after the xn fill;
  up_0 / gate_0 after the og fill, never up_1, which awaits o; down after the last up | gate band).
- Did not help (CLEAN, same sitting, with one context): a 24-deep ring (41.0 / 41.6 against 41.2 / 41.4 ms min /
  median), dropping the layer's unused last xn and its drain (41.1 / 41.4; the probe was removed again).

**One context (OPEN-DECODE-ONE-CONTEXT-DENSE).** The final norm and the head are `dxh`, a second instruction stream over
`dx`'s image (the two xclbins differ only in build stamps), named in context `dx`; the engine queues it behind the last
layer like a layer. A token changes context zero times (was three, ~1.9 ms), and the head runs at 52.5 GB/s there (3.06
ms against 3.11 alone in its own context and ~3.7 in the step). Step 44.4 -> 41.4 ms median (24.2 tok/s).
- The head is the layer's own code: the ln core runs the entry norm with w = normw (`ln_nr` is `ln_xn`'s arithmetic at
  add = 0, so hn is the old tail's bits) and the main cores the q|k|v GEMV with 196 bands instead of 7 (the head is
  K = 2560 like q, the band split `lm_head_q4`'s), then skip the rest. Logits byte-identical.
- **Trap: an RTP word read right after the barrier is the previous dispatch's.** The first build read the mode there:
  layer -> layer and head -> head ran, layer -> head hung (ERT state 8, serial and queued alike). The cores read it only
  after the dispatch's first data has landed (x on the main cores, the entry norm's inputs on the ln core).
- **Overlay `dx` and `dxh` together.** `dxh`'s stream only runs on the `dx` image it was built with (a new `dx` still
  runs under an old manifest's `ln` + `lm` tail). A dx-only experiment is `--only dx,dxh` plus that export's
  `manifest.json`.

tg128 after a 0 / 512 / 1024 / 2048-token prompt, CLEAN medians: 20.61 / 17.42 / 15.04 / 12.19 -> 22.96 / 19.08 / 16.22 /
13.05 tok/s, ~5 ms a token at every depth. tg128 "at 0" is 2.2 ms a token slower than the position-0 step on either set
because it averages positions 1-128. **Uno did not move** (`dxl` / `lmhl` have neither change): plain decode now beats
it on the chat and code prompts (sweep 1.07x -> 0.95x geomean, all identical); the same ring and a `lmhl`-as-second-
stream on `dxl`'s image are the obvious next port.

**The host gap between tokens.** Timestamps around the step boundary (a scratch build of the CLI, 128 tokens, CLEAN):
from `dxh`'s wait returning to layer 0's start() returning took 0.82-0.89 ms a token, and 0.69-0.74 of it was the
CLI's `argmax` over 100352 logits -- `v[i] > v[best]` reloads v[best] through the index it just picked, a dependent
load chain, run at the clock a host core has after a 40 ms block. The rest: read back 0.01, memcpy 0.02, the per-step
stderr line 0.04, embed + patch 0.01, layer 0's submit 0.04-0.05 (stderr to a file instead of a pipe: no change). The
CLI's argmax is now AVX2 with the scalar loop's exact answer (200k edge-case vectors: NaN, +-0, +-inf, subnormals,
ties), and one-dispatch layers restart a kept `xrt::run` instead of building one per dispatch (`OFLM_OPEN_RUN_REUSE=0`
builds anew; ~0.1-0.2 ms a token). Still scalar 100k passes a token: `Engine::logits_view` (the app's path, not
measured) and the CLI's `argmax_bf16`, which the uno sweep's plain-decode column uses.

**Re-check with a control (end of the night).** dibs counts every track of one session as one holder, so a sibling's
`dibs run npu` can land inside a unit that still prints CLEAN (the night's `--rows-check` gate printed 126.5 ms a pass;
the same check alone reads 63.9). Every NPU run, timed or not, goes through `dibs run --measure`, and every unit carries
a control: the committed CLI's tg128 at ctx 0 on the final set, known 22.96 tok/s, the unit void if it is off by more
than 5% (`utilities/decode-gates/tg_control.py`: control, configs interleaved per depth, control). Re-checked CLEAN
medians, every control within 5%:

| tg128 after a prompt of | 0 | 512 | 1024 | 2048 |
|---|--:|--:|--:|--:|
| base set, committed CLI | 20.65 | 17.51 | 15.34 | 12.02 |
| final set, committed CLI | 23.03 | 19.33 | 16.69 | 13.08 |
| final set, this CLI (argmax, run reuse) | 23.40 | 19.62 | 16.93 | 13.19 |

Position-0 step min / median: base 46.0 / 46.2, final 41.1 / 41.2 ms (41.1 / 41.4 on the committed CLI). Uno sweep:
base 1.06x, final 0.95x geomean, all identical.

**Bytes.** The weight stream is now within ~1% of the 8-stream rate, so the next ctx-0 lever is fewer bytes. A
Q4_0-native chunk (`GEMV_Q4_0_NATIVE`, `LMHEAD_Q4_0=1`: 4608 B, no m plane, m = -8 d in-register) streams the head in
2.739 ms against 3.046 for the same blocks in 5120 B chunks, 52.8 against 52.7 GB/s, bit-identical
(`utilities/decode-gates/q4_0_head_probe.py`); re-checked with a control, 2.828 / 2.865 against 3.078 / 3.202 ms min /
median (8-10% for 10% fewer bytes). A model-wide Q4_0 layout would take ~8-10% off every layer's stream (estimate:
41.4 -> ~37.5-38 ms at position 0). Quality on the repo-docs corpus (llama.cpp Vulkan, KLD against bf16 over
24 x 512 tokens): Q4_0 0.155, Q4_1 0.158, Q4_K_M 0.124. Not a default: a proposal.

**Against llama.cpp Vulkan (b10944, the 890M), one sitting, NPU controls around each model.** Q4_1 tg128 23.55 +- 0.15,
at d2048 22.21 +- 0.17; Q4_0 23.14 / 21.91; this path 23.2-23.4 at ctx 0 in the same unit and 13.19 at 2048. Earlier
units without a control read Vulkan at 17-18 tok/s on the same binary and file, most likely a sibling's NPU run inside
the unit: compare to Vulkan in one sitting with a control, or not at all.

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
- Prefill is not decode-as-prefill and not `dxB`: the engine takes the block route with host attention, layer-major
  (OPEN-PREFILL-HOST-ATTN, 60 -> 220 tok/s at pp512). Measure it with `open_qwen36_cli --gemm-block --ids-file`; a
  kernel dir carrying the Uno `dxl` / `lmhl` entries needs the Uno engine build to parse it.
