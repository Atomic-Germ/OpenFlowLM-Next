# Plan: Uno for any dense family, not just K2

**Status:** done, 2026-10-10. Pushed onto PR #184's branch (`feat/k2-horizon-7b`), agreed with #184's owner. Results
are in the spec (OPEN-DECODE-ROWS and OPEN-UNO-DECODE, 2026-10-10).
- **Done:** G1, G2, G3, G5, G6 and G4.
  - K2-7B on the NPU: the CLI's six prompts identical to decode with the app's pick (G6).
  - K2-3.7B (hidden 2560) on the NPU: `--rows-check 17` passes from a 20- and a 158-token prompt (G4).
  - `dxl`, `lmhl` and `rows_route` are this branch's while #184's speed Phases 3-7 wait on review and a settings
    design (#188).
- **Left for the Granite PR (#189):** the 2560-wide draft pass on a real adapter. No K2-3.7B Uno adapter exists, so
  the half slices' LoRA jobs have run only in the unit tables.
- **Found on the way:** `pools_test` aborted on Windows since #121; main fixed it in #181, which #184 picks up when it merges main.

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
| G6 | **Uno's greedy picks what the app's greedy picks.** The app argmaxes logits rounded to bf16, lowest id on a tie (`Engine::forward` is `buffer<bf16>`); `lmhl`'s verify argmax compared f32, so served Uno and served decode split where two logits share a bf16 value (K2, the train prompt, token 128: 25.494 vs 25.439). `lmhl` rounds each logit to bf16 and keeps the lowest id on a tie, within a chunk and across chunks; so does the CPU oracle, and the CLI's check compares bf16 argmax. | OPEN-UNO-DECODE (the known gap closed) | manual: the K2 serve check's 160-token train reply identical; `--uno` UNO IDENTICAL on the six prompts |
| G5 | **A Uno bench.** `utilities/uno-bench` runs `open_qwen36_cli --uno N` over a prompt file under the NPU lock's timing gate and tabulates tokens a cycle, Uno vs decode tok/s and the speedup. | none (tooling) | — |

**Order:**
- G1 and G5 first; they are pure Python.
- Then G2 and G3, which are engine changes and need a build.
- G6, then G4; both are in `lmhl`.

## Not in this plan (the Granite issue #189 and its PR)

- `ROWS_FAMILIES += granite`, and Granite's AutoModel opting into Uno (G3).
- Converting the Granite adapter with `--uno-noise-high 100256`, the first special id; it was trained on `[1, 100256)`.
- Granite's `--rows-check 17` and `UNO IDENTICAL`, and the bench numbers next to K2's.
- OPEN-FAMILY-GRANITE-UNO.

## Spec impact

- **OPEN-UNO-LORA (modified):** no longer K2-only. s·B carries the projection's base fold, and the noise range is adapter metadata.
- **OPEN-UNO-DECODE (modified):** noise is drawn from the adapter's range. One generate serves every family that opts in. The verify argmax is the app's: bf16, lowest id on a tie.
- **OPEN-DECODE-ROWS (modified):** hidden widths that are multiples of 512; `lmhl` refuses a width it does not cover.
