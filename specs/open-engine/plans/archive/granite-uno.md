# Plan: Granite 4.2 3B's Uno drafter on the NPU (#189)

**Status:** done, 2026-10-10. Branch `feat/granite-uno`, stacked on PR #184 (`feat/k2-horizon-7b`). Results are in
the spec (OPEN-FAMILY-GRANITE-UNO, 2026-10-10).
- R1 and R3 done: `ROWS PASS`, all six prompts identical to decode, 3.04 tokens a cycle against the CPU reference's
  3.15, and 0.94x on a quiet machine.
- R2 decided against for now: at 0.94x, Uno would slow Granite's served greedy requests. The opt-in is K2's one line,
  for when a pass costs less or L = 8 runs.
- Found on the way, fixed here:
  - q4nx-build never folded a GGUF Granite's config.json (#21).
  - Granite's `dxl` overran row 2's 32 south ports (`KV_ONE`).
- Fixed on #184: `ag_*` builds that no family without a prep could use, and Granite's could not tile.

The adapter is `Cyronius/granite-4.2-3b-uno`: a PEFT LoRA, r 128, alpha 8192 (s = 64), on q, k, v, o, gate, up and
down of all 40 layers, trained on noise ids `[1, 100256)`. CPU/GPU reference over 60 prompts at L = 4: 1.586 tokens a
forward, 3.17 a cycle.

## Changes

| # | What | Requirement | Verification |
|---|---|---|---|
| R1 | **Granite ships the L-row pass.** `ROWS_FAMILIES = ("k2", "granite")`; Granite's kernel set gains `dxl`, `dxl_lora` and `lmhl`. | OPEN-DECODE-ROWS (modified: K2 and Granite), OPEN-FAMILY-GRANITE-UNO (new) | test: `tests/test_dxl.py` (Granite's route); manual: `--rows-check 17` prints `ROWS PASS` |
| R2 | **Granite's AutoModel opts into Uno** for greedy requests, as K2 does: `generate` and `generate_with_prompt` take `_shared_generate_uno` when `_uno_applies()`. Sampling defaults are unchanged, so Uno serves requests that ask for greedy (top_k 1, no penalty). | OPEN-FAMILY-GRANITE-UNO | manual: `oflm serve` greedy, cut, stream and sampled requests, as K2's |
| R3 | **Convert and measure.** `q4nx-build -i <granite-4.2-3b> -f granite --quant Q4_1 --uno-adapter <adapter> --uno-noise-high 100256`. `open_qwen36_cli --uno 128` over `utilities/uno-ref/prompts.jsonl` prints `UNO IDENTICAL` on every prompt, and tokens a cycle are within ±5% of the CPU reference (`k2-diffusion/scripts/eval_tpf.py --blocks 4 --quant q4_1` on the same prompts). `uno-bench` numbers sit next to K2's. | OPEN-FAMILY-GRANITE-UNO | manual |

**Decision (R2):** Granite's default sampler stays as it is. Making greedy the default would change every Granite reply
to buy speed, and that is the user's call, not a side effect of this PR.

**Not in this plan:**
- Hosting the converted model or `uno.q4nx`; the adapter repo is private.
- L = 8, where this adapter reaches 3.53 tokens a cycle. It stays blocked for every family.

## Spec impact

- **OPEN-FAMILY-GRANITE-UNO (new):** Granite 4.2 3B with `uno.q4nx` decodes greedy requests by the cycle, identical to
  greedy decode; the procedure and the measured tokens a cycle and speed.
- **OPEN-DECODE-ROWS (modified):** `ROWS_FAMILIES` is K2 and Granite.
