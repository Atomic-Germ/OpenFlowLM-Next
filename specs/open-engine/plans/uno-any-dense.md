# Plan: Uno for any dense family, not just K2

**Status:** in progress, 2026-10-09. Branch `feat/uno-any-dense`, stacked on PR #184 (`feat/k2-horizon-7b`) and merged into
it by PR.

#184 brought IFM's K2-Horizon-7B-Uno to the NPU. Several pieces of it assume K2 where nothing about Uno requires that.
This plan makes those pieces family-neutral, with K2's output and kernels unchanged. Bringing up a second adapter
(Granite 4.2 3B, `Cyronius/granite-4.2-3b-uno`) is a separate issue and PR on top of this one.

## Changes

| # | What | Requirement | Verification |
|---|---|---|---|
| G1 | **The LoRA's B carries the base converter's folds.** A family whose builder folds a multiplier into a projection's weights (Granite: `attention_multiplier·√head_dim` into q, `residual_multiplier` into o and down) must fold the same factor into that projection's s·B, or the draft's update is off by that factor. `uno.py` reads the pre-fold multipliers from `config.json`'s `q4nx_folded_multipliers`. With none recorded (K2), nothing changes. | OPEN-UNO-LORA (modified) | test: `test_uno.py` |
| G2 | **The draft's noise range is the adapter's.** `--uno-noise-high N` records the exclusive upper bound of the noise ids an adapter was trained with in `uno.q4nx`'s metadata. `Q4nxFile` keeps the metadata, and `uno_cycle` and the CLI draw from `[1, N)`. Absent, it is today's `[1, vocab - 1]`. | OPEN-UNO-LORA, OPEN-UNO-DECODE (modified) | test: `test_uno.py` (the metadata); manual: K2 `--uno 128` unchanged and identical |
| G3 | **One Uno generate for every family.** `uno_applies` / `generate_uno` move from `modeling_k2.cpp` to the AutoModel base, and a family opts in with one override. K2 behaves exactly as before. | OPEN-UNO-DECODE (modified) | manual: `oflm serve` K2 greedy, sampled and tool-call requests as in #184 |
| G4 | **The L-row pass at any hidden width that is a multiple of 512,** not only 1024: `recipes/dxl.py` slices, job tables past 4 bands a core, and `lmhl` covering every column. Today `lmhl` silently drops the columns past the last whole 1024; it must refuse what it can't cover. K2's tables stay byte-identical. | OPEN-DECODE-ROWS (modified) | test: `tests/test_dxl.py` (a 2560-wide config; K2 tables unchanged); manual: K2 `--rows-check 17` PASS |
| G5 | **A Uno bench.** `utilities/uno-bench` runs `open_qwen36_cli --uno N` over a prompt file under the NPU lock's timing gate and tabulates tokens a cycle, Uno vs decode tok/s and the speedup. | none (tooling) | — |

**Order:**
- G1 and G5 first; they are pure Python.
- Then G2 and G3, which are engine changes and need a build.
- G4 last. It touches the `dxl` and `lmhl` designs that #184's speed work (`k2-uno-speed.md` Phase 3) is changing, so it rebases onto whatever that phase lands.

## Not in this plan (the Granite issue and PR)

- `ROWS_FAMILIES += granite`, and Granite's AutoModel opting into Uno (G3).
- Converting the Granite adapter with `--uno-noise-high 100256`, the first special id; it was trained on `[1, 100256)`.
- Granite's `--rows-check 17` and `UNO IDENTICAL`, and the bench numbers next to K2's.
- OPEN-FAMILY-GRANITE-UNO.

## Spec impact

- **OPEN-UNO-LORA (modified):** no longer K2-only. s·B carries the projection's base fold, and the noise range is adapter metadata.
- **OPEN-UNO-DECODE (modified):** noise is drawn from the adapter's range. One generate serves every family that opts in.
- **OPEN-DECODE-ROWS (modified):** hidden widths that are multiples of 512; `lmhl` refuses a width it does not cover.
