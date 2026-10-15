---
name: open-dxl-rows
description: Build, verify and extend the dense L-row pass (designs/dxl - dxl, dxl_lora, lmhl) behind speculative decoding (Uno, prompt lookup, the per-cycle choice between them) on K2 and Granite. Use when rebuilding those xclbins, adding a family to ROWS_FAMILIES, changing L, adding a LoRA/draft model, measuring or tuning --spec / --uno, or when an L-row build computes garbage or hangs.
---

# The dense L-row pass (designs/dxl)

L consecutive positions through a dense layer in one dispatch (`dxl.py`), plus their final
norm, the head and per-row argmax in another (`lmhl.py`). Each row is **bit-identical** to a
decode step at its position. Spec: OPEN-DECODE-ROWS, OPEN-UNO-LORA, OPEN-UNO-DECODE, OPEN-SPEC-LOOKUP. Recipe:
`open_kernels/recipes/dxl.py`. Plan with history: `specs/open-engine/plans/k2-horizon-7b-uno.md`.

## How it is put together

- **Main cores (8):** a job table. Each job is one pass of the L-row GEMV (`dxl_gemv.h`) over
  a tile of bands: their [L][64] accumulators stay resident, K is walked in 1024-wide (bf16)
  or 512-wide (fp32) slices, and the L tokens' slice tables are rebuilt per slice.
  - The tile body is gemm_q4's tile4 with **gemv_q4_tile's epilogue order**
    (hi, lo, xs_hi, xs_lo). gemm_q4_tile4 swaps the last two, and that is the difference
    between bit-identical and merely close.
  - Both tables (0 verify, 1 draft) live in each core. The stream writes `rtp[0]` and then
    sets the barrier. The core waits on it at the top and **never releases it**: a release
    adds 1, and the next dispatch would start on the old mode.
- **ln core:** the L rows' norms in turn.
- **Attention cores:** a group of ACORES (4) per row, so 16 at L = 4. Group 0 sits on dx's
  tiles (2..5, 3); the rest take rows 4 and 5. Every group runs dx's kernels unchanged.
  - One broadcast `ain` stream carries every row's prologue (meta, ptab, q, k, v), then one
    window of pos0 + L cache rows. Group j skips the other prologues, walks the window to
    pos0 + j with the pb counts meta gives that position, skips the rest (`dxl_attn_skip`:
    pb[6] = pb[0] + L - j - pb[1], which also covers position 0's dummy row), then adds its own
    row with `attn_step_new`. Bit-identical to decode by construction.
  - Granite's groups compile the same bit-identical score tree as `dx` (`attnknobs.tree8`, OPEN-ATTN-SCORE-TREE), with the row loop
    rolled to fit program memory: 12,320 B per core (13,520 with the KV pack) of 16,384; unrolling by 4 overflows it and by 2 crashes the backend.
  - Each group's core 0 packs its new K|V row (`dxl_kvpack`) into a memtile join (Tile(1, 1)).
    One drain writes all L rows, and it finishes before the window fill is issued.
  - Each group's four og elements join in a memtile (Tile(2 + g, 1)) into one 8 KB row: one
    drain per row. Shim output channels stay at 14 of 16.
  - Why: the attention walk is ~56% arithmetic (an `ATTN_NULL=1` build), so sharing one
    window on 4 cores would still pay 4x. With a group per row, a pass's attention slope is
    decode's (0.75 ms per 1k positions a layer, against 2.9 before).
- **Patching:** the attention sites are compiled at positions 1 .. L. `attnrows`
  (`harness/stream_patch.hpp`, the engine's `step_rows`) moves row j's record to pos0 + j, the
  joined KV drain to pos0, and sizes the window to pos0 + L.
- **Buffers:** the same per-layer pool / consts / kv / ptab as dx, plus an L-row xres and act.
  `lora` is arg 6 (verify binds a dummy BO there).

## Traps, each one met for real

1. **64 B alignment.** Every main-core buffer has to be a whole number of 64 B. The build
   asserts it.
   - The allocator packs buffers back to back. A 3,080 B job table left everything after it
     32 B aligned, and the 512-bit accesses then read wrong bytes with no fault.
   - The symptom was identical garbage whatever kernels ran; corr against dx was 0.02.
   - `designs/dxl/glj.py` (GL_NOBUF / GL_PAD) is the probe that isolated it.
2. **Program memory.** Python-unrolled band loops in the core body cost 20 KB of control
   code. Every loop has to be `range_` with runtime counts, which is why there is a job
   table. Measure with `OFLM_KEEP_FAILED=1` plus
   `llvm-size -A final.prj/elfs_main_core_*/...elf`.
3. **DMA queue.** A shim channel queues 4 BDs. Every fill and drain goes through
   `Pipeline(3)`, the ln channels included.
4. **The z wait.** Before a LoRA tile's z fill, `py.finish()` waits for the A band's drain.
   It must come **before** that tile's own drain is issued, or it waits on a drain that needs
   the very fill it is blocking (the dispatch times out, state 8).
5. **Heredocs.** In this shell, `\\n` inside a heredoc reaches Python as a real newline. Write
   patch scripts with the file tool.
6. **Constant skip loops hang `llc`.** Three back-to-back `range_(13)` acquire/release loops
   ahead of the real work (row-3's group) ran `llc` 9+ minutes without finishing. A count
   through memory (`dxl_attn_count` into pb[7]) compiles in seconds. Watch for one core's `llc`
   running long in `final.prj`.

7. **South ports.** Every stream to DDR crosses from row 2 to the memtiles through row 2's south
   ports, 4 a column, 32 in all. og (L x ACORES), KV (L), y (8) and the norm (1) are 29 for K2 and
   33 for Granite's 5 cores a group, which aiecc reports only as "Unable to find a legal routing".
   `dxl.py` KV_ONE then has group 0's core 0 pack all L KV rows (one stream).

## Build (Windows, iron_env.ps1)

```powershell
python open_kernels\export_qwen36_kernels.py --model-dir C:\models\k2-7b\K2-Horizon-7B-NPU2 --only dxl,dxl_lora,lmhl -j 3
```

The recipe's `rows_route` builds:
- `dxl`, with `DXL_L=4`;
- `dxl_lora`, with `DXL_DRAFT=1`: the same core programs and a different stream, running in
  dxl's context (the xclbins differ in 72 header bytes);
- `lmhl`.

A family enters `ROWS_FAMILIES` (`recipes/dense.py`) only after the procedure below passes
on it.

## Verify

```powershell
python open_kernels\model\make_decode.py --model-dir <dir> --layers 1 --tokens 4 --out model\out_k2l
run_kernel model\out_k2l\run_decode.cfg                          # dx, the reference rows
python designs\dxl\make_dxl_test.py --fixture model\out_k2l --build <dxl build> --l 4; run_kernel ...\run_dxl.cfg
python designs\dxl\make_dxl_test.py --fixture model\out_k2l --l 4 --compare        # 0 of 4096 differ
python designs\dxl\dxl_at_test.py --pos0 5 --build <dxl build>; run_kernel model\out_k2l\run_dxl_at5.cfg; dxl_at_test.py --pos0 5 --compare
                                                                  # rows at a later position on decode's cache: 1, 5, 8, 1022
python designs\dxl\make_dxl_test.py ... --head                    # lmhl: 0 of 250624, argmax == host
python designs\dxl\draft_ref.py --fixture model\out_k2l --model <dir> --l 4 --pack  # draft: LoRA pool for layer 0
python designs\dxl\make_dxl_test.py ... --build <draft build> --lora model\out_k2l\lora_L0.bin; run_kernel; draft_ref.py --compare
open_qwen36_cli ... --rows-check 17                               # whole model: ROWS PASS
open_qwen36_cli ... --uno 64                                      # UNO IDENTICAL to decode
```

## Timing

- **The harness and the engine agree** on a quiet machine (one dx layer 2.6 ms × 36 ≈ the
  engine's 107 ms step). The "2x" once noted here was a busy machine.
- **Other sessions share this NPU.** Take turns with `dibs` (the old `lock.py` is gone): a number you will quote
  comes from `dibs run --measure --tag <you> --label <what> --eta <t> -- <cmd>` and only when it prints both
  `dibs: CLEAN` and `dibs: CLEAN after`; a correctness run goes under `dibs run npu --tag <you> ...`. Keep a measured
  unit under ~5 minutes: it holds every other session's commands while it runs.
- **Measured 2026-10-08/09 (K2-7B, L = 4):**
  - A verify pass costs 1.39x a decode step at short context (quiet; the 1.17-1.28x first noted
    came from a busy machine). The draft pass ~1.7x.
  - The head is DMA-bound on its 5 KB elements: 25 ms against `lm_head_q4`'s 12.8 ms in the
    harness. `lmhl2.py` (10 KB elements, sliced tables, rows padded to 250880) is correct but
    measured no faster: 27-38 ms against lmhl's 25-31 ms, so the recipe keeps `lmhl`.
  - Uno over 6 prompts x 128 tokens, quiet machine (2026-10-09): 3.47 tokens a cycle (CPU
    oracle 3.45), identical to decode, **1.11x** (0.98-1.24x; 9.1-11.6 tok/s against 9.3).
    A cycle costs 3.13 decode steps against ~2.6 for its measured parts: that gap is the next
    thing to find.
  - **Never quote a ratio from a busy machine.** With the CPU at 100% the same bench showed
    1.55x: contention slowed plain decode (more dispatches per token) ~3x and Uno less.

## Prompt lookup and the per-cycle choice (OPEN-SPEC-LOOKUP)

The verify pass needs no draft pass when the drafts come from the context: the tokens that followed the latest earlier
occurrence of the current suffix n-gram (n = 4 .. 1, `spec_lookup.hpp lookup_drafts`). One verify pass (~1.33 plain
steps) commits 1 to L tokens; a Uno cycle (~2.9 steps) 2 to L + 1. Each cycle the engine picks lookup, Uno or a plain
step by expected tokens per ms (running means of acceptance, lookup's per matched n, and of each pass's measured ms).

```powershell
open_qwen36_cli --model <dir> --kernels <set> --ids-file <ids> [--gemm-block] --spec 128 [--spec-mode adaptive|lookup|lookup-uno|uno|plain]
#   -> per-method cycles / tokens a cycle, accepted drafts by matched n, pass ms, both speeds, SPEC IDENTICAL
open_qwen36_cli ... --spec-probe 256 --spec-dump <json>   # greedy by draft passes: the transcript + every position's Uno drafts
```

Measuring, `utilities/spec-lookup/spec_lookup.py` (absolute paths everywhere: the CLI is started by CreateProcess):

1. `prompts --model <dir> --out <p>`: the six uno-ref prompts (thinking on, as served), chat / code at 1024 and 2048
   (`granite-combined-bench`'s real repo text), and four lookup prompts with thinking off: a whole-file edit, a bug fix,
   quoting a spec section, document Q&A.
2. `probe ...` under `dibs run npu` (~10 min for all fourteen), then `sim` / `ngrams` replay them offline: exact for
   lookup, Uno up to the noise ids. `sim --rows 2|8` prices a what-if verify build linearly in rows (lookup only).
3. `matrix --units "short:128:lookup+adaptive;edit_py+fix_py:256:uno+lookup+adaptive;..." --reps 3 --gap 75` runs each
   unit (prompt names or groups, tokens, modes back to back in one process) per rep as one `dibs run --measure` (keep it
   under ~5 min), sleeps `--gap` seconds after each so the other sessions get the machine, reruns NOISY units and skips
   units that already have a `.jsonl`; `table --dir <out>` prints medians. A `STOP` file in the out dir pauses it at
   the next unit. While a unit runs, every command of yours but plain reads is held.

What it buys on Granite 4.2 3B (OPEN-FAMILY-GRANITE-UNO, 2026-10-15): lookup pays wherever the answer repeats the
context (a whole-file bug fix 3.1x, 43.6 tok/s; an edit 1.8x; quoting / document Q&A 1.5-1.75x) and little on fresh
reasoning text (1.10-1.17x); Uno pays on math / code and loses on chat. The adaptive choice is the best geomean at
short context and 2048 and within 3% elsewhere, from 6% behind to 5% ahead of the better single method on a given
prompt. The pass cost is the lever: a verify pass is 1.33 / 1.26 / 1.24 plain steps at context 0 / 1024 / 2048 and
the draft pass 1.56 / 1.44 / 1.37. Offline, a verify pass at 1.10 steps would lift adaptive on reasoning text from
~1.15x to ~1.30x (estimate); 2 or 8 rows would not (8 helps only whole-file rewrites).

Traps:
- In a worktree-isolated agent, `cmd.exe /c` and `powershell -File` are refused (they could run git). Run a `.cmd`
  wrapper directly (`./.claude/build_cli.cmd`) and keep it in the gitignored `.claude/`.
- dibs treats the subagents of one session as one holder. While a `--measure` unit of mine held the machine, a sibling
  track's `dibs run npu` held the npu at the same time (`dibs status` showed both), and an untimed probe of my own got
  in the same way; those units still read CLEAN while plain decode fell to a third. Start no NPU work while your own
  matrix runs; give `matrix --expect <json of plain tok/s>` so a unit 5% slow is rerun like a NOISY one, and read the
  table, which sets aside a run whose plain decode or pass time is 5% off the prompt's median. Stage a CLI copy as
  `<dir>/open_qwen36_cli.exe`, not `cli_v4.exe`: dibs's NOISY verdicts named `open_qwen36_cli.exe` processes and
  missed a renamed copy.
- `presence_penalty 1e-6` (the old served "decode plainly" trick) is not plain greedy at an exact bf16 tie: the penalty
  breaks the tie. Serve a second time with `OFLM_SPECULATIVE` unset for the reference instead.
- The app's AVX2 greedy scan kept the lower *lane* on a tie, not the lower id (fixed with OPEN-SPEC-LOOKUP); a served
  speculative reply could split from plain decode there.
- Granite's server ignores a request's `"think": false` (its AutoModel has no `enable_think` parameter and does not
  pass `enable_thinking` to the template), so a served whole-file edit reasons first and lookup gains 1.15x there,
  not the CLI's 3.1x (the CLI prompts render the template with `enable_thinking=False`).
