---
name: open-phi3-nanbeige-kernels
description: Build, verify and serve the open XDNA2 kernel sets for Phi-4-mini (the phi3 recipe: a 96-of-128 rotation and longrope) and Nanbeige4.1-3B (the llama3 recipe at 20 heads over 4). Use when re-exporting either, adding another Phi-3 or Nanbeige size, when a Phi export dies in Peano on a rename, or when `flm serve` segfaults on the first request for a model whose adapter still casts to its closed engine class.
---

# Phi-4-mini and Nanbeige4.1-3B on the dense recipe

Both are `open_kernels/recipes/dense.py` families, exported and verified exactly like
Granite (`open-granite-kernels`), with three things that are new. Spec:
`specs/open-engine/spec.md` OPEN-FAMILY-PHI3 and the Nanbeige paragraph of
OPEN-FAMILY-LLAMA3; hardware log `.claude/plans/b-hw-results.md` (gitignored).

## Export (WSL, serial)

```bash
source ~/ironenv142/bin/activate
export PATH=~/xrt-tools/bin:$PATH LD_LIBRARY_PATH=~/xrt-tools/lib     # both, or xclbinutil dies late
cd /mnt/c/code/openflowlm-next
python open_kernels/export_qwen36_kernels.py --model-dir ~/.flm/models/Phi4-mini-Instruct-NPU2
python open_kernels/export_qwen36_kernels.py --model-dir ~/.flm/models/Nanbeige4.1-3B-NPU2
```

Sets: `dense/build_phi3_h3072` + `ln/build_3072_1e-05` + `lm_head_q4/build_200064`, and
`dense/build_llama3_h2560` + `ln/build_2560_1e-05` + `lm_head_q4/build_166144`, into
`src/xclbins/<model>/open_kernels`. The catalogue points these need (attn tuples
`(128, 24, 8, 96, F, F, F)` and `(128, 20, 4, 128, F, F, F)`, `gemv_q4` K 10752) are in
`recipes/catalogue.py` since 2026-09-10, so no `OPEN_KERNELS_UNVALIDATED`.

**Peano can die on `unable to rename temporary ... final.prj/ln_y.o: No such file or
directory`.** That is the /mnt/c mount, not the code: the same `ln_y.cc` compiles for every
other family, and a retry passes. Loop the export (`.claude/plans/build_b4.sh` does) rather
than reading it as a kernel error.

## What is new in the kernels

- **A 16-lane RoPE tail** (`designs/attn/attn.h`, `#if (ATTN_ROT / 2) % 32`). Phi rotates
  96 of 128 dims = 48 pairs; the loop does 32 and the tail 16. The rule is now "a multiple
  of 32". Families whose rotation is a multiple of 64 compile the identical loop -- the 35B
  and Qwen3-4B `--force --check` gates were byte-identical after the edit.
- **longrope on the position table, switched per row.** HF picks its short or long factor
  list per forward call from the running sequence length. A resident table can't re-select
  per call, but this engine computes one row per token as the context grows, so both
  tables are baked (`inv_freq` + `long_inv_freq` on the ptab global) and row r reads
  `long_inv_freq` once r reaches `switch_row` (= `original_max_position_embeddings`,
  4096 on Phi-4-mini) -- `--max-ctx` no longer decides which table ships, only how many
  rows exist. `ModelSpec.rope_scale()` (1.190 on Phi-4-mini) rides on the same global as
  `scale`, unconditionally; `pools.cpp`, `pack.ptab` and `replica_dense.rope` all multiply
  cos and sin by it. No key for other families -- `RowGlobal::switch_row` defaults to
  `kSwitchNever`.
- **The load-time compatibility check names `rope_theta` and the longrope config**, not
  just the shape fields every other family checks -- two same-shaped containers can differ
  only in their factor lists (different longrope fine-tunes), and that has to be caught at
  load, not run silently. Only checked when the source `config.json` actually carried the
  key, so an omitted-but-defaulted field (`partial_rotary_factor`) doesn't turn into a
  refusal for containers that never had it.
- Nothing new for Nanbeige beyond the two points: it declares `model_type: llama` and is one.

## Verify

The OPEN-FAMILY-QWEN3 procedure; `.claude/plans/validate_b.ps1` runs it end to end:

```powershell
.\validate_b.ps1 -Name Phi4-mini-Instruct-NPU2 -Tag ph -Token 200021 -Registry phi4-mini-it:4b
.\validate_b.ps1 -Name Nanbeige4.1-3B-NPU2 -Tag nb -Token 166100 -Registry nanbeige4.1:3b
```

Phi's first token is `<|user|>` (no bos); Nanbeige's is `<|im_start|>`. `chat.py` has both
templates. Nanbeige is a reasoning model with no off switch: every answer opens `<think>`,
so `flm-test --llm` needs a real generation budget (`--gen-lim 4000`; 600 leaves the answer
column empty) and its greedy opening "Weimplify is asked:" is the quantized model, not the
kernels (24/24 positions of a decode chain match the fp64 replica).

## Traps

- **`flm serve` segfaults on the first request** if the adapter reaches its closed engine
  through `dynamic_cast<xxx_npu*>` (Nanbeige did, for checkpoint/restore). On the open
  engine that cast is null. Use the `causal_lm` virtuals (`this->lm_engine->checkpoint()`).
- The app finds a set at `<root>/xclbins/<model>/open_kernels`; for a one-off serve check
  `FLM_OPEN_KERNELS_DIR=<dir>` wins. `FLM_SERVE_PORT=52626` (52625 is often taken).
- Kill `flm.exe` after a serve check or the next Ninja build fails in its copy step.
