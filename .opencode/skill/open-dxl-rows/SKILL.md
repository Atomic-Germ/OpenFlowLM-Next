---
name: open-dxl-rows
description: Build, verify and extend the dense L-row pass (designs/dxl - dxl, dxl_lora, lmhl) behind speculative decoding and K2-Horizon-7B-Uno. Use when rebuilding those xclbins, adding a family to ROWS_FAMILIES, changing L, adding a LoRA/draft model, or when an L-row build computes garbage or hangs.
---

# The dense L-row pass (designs/dxl)

L consecutive positions through a dense layer in one dispatch (`dxl.py`), plus their final
norm, the head and per-row argmax in another (`lmhl.py`). Each row is **bit-identical** to a
decode step at its position. Spec: OPEN-DECODE-ROWS, OPEN-UNO-LORA, OPEN-UNO-DECODE. Recipe:
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
- **Attention cores:** dx's kernels, one query row at a time. Row j's KV drain finishes
  before row j + 1's window fill is issued.
- **Patching:** the attention sites are compiled at positions 1 .. L. `attnrows`
  (`harness/stream_patch.hpp`, the engine's `step_rows`) moves row j to pos0 + j.
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
python designs\dxl\make_dxl_test.py ... --head                    # lmhl: 0 of 250624, argmax == host
python designs\dxl\draft_ref.py --fixture model\out_k2l --model <dir> --l 4 --pack  # draft: LoRA pool for layer 0
python designs\dxl\make_dxl_test.py ... --build <draft build> --lora model\out_k2l\lora_L0.bin; run_kernel; draft_ref.py --compare
open_qwen36_cli ... --rows-check 17                               # whole model: ROWS PASS
open_qwen36_cli ... --uno 64                                      # UNO IDENTICAL to decode
```

## Timing

- **Harness numbers run about 2x the engine's.** Compare ratios there; take absolutes from
  `--rows-check` / `--uno`.
- **Other sessions share this NPU.** Run timed work under their lock:

  ```
  python C:/Users/josha/AppData/Local/Temp/claude/c--code-openflowlm-next/343fcc59-a3ad-4984-a34e-b3d7ef3e8d75/scratchpad/agents/lock.py timing --label <you> -- <cmd>
  ```

  Use `npu` instead of `timing` for correctness runs. `timing` waits until CPU is under 25% and
  no NPU/GPU rival is running (LM Studio counts).
- **Measured 2026-10-08 (K2-7B, L = 4):**
  - A pass costs 1.17-1.28x one decode step.
  - The draft layer is +14% over verify.
  - The head is DMA-bound on its 5 KB elements: 25 ms against `lm_head_q4`'s 12.8 ms in the
    harness. Bigger elements need sliced tables; that is the obvious next speedup.
