# Bonsai 2 DFlash 2 drafter: measurement history (2026-10-12)

The requirements are OPEN-DRAFT-DFLASH2 (+ -WALK, -SYNC), OPEN-GEMV-T2N, OPEN-DECODE-TAPS,
OPEN-DECODE-SPECULATIVE(-VERIFY) and OPEN-REGISTRY-SPECULATIVE in `specs/open-engine/spec.md`. This file keeps
the numbers: what the drafter costs alone, and what the whole thing does end to end.

## The drafter alone (fine-tuned t2 drafter `nakli-tft-2a`, D4)

Steps 0, 4, 8 .. of each prompt's 512-token greedy trajectory, llama.cpp taps, 2495 drafts: k = 3 2485 identical
in all 3 tokens (99.60%), the other 10 near-ties, so 100% excluding them; k = 7 2455 identical (98.40%), 34
near-ties and 6 others (rows 5-7, selector margins 0.19-2.96 on the reference side), so 99.76% excluding
near-ties; teacher-forced tokens per pass NPU 3.1651 against the reference's 3.1671 at k = 3 and 4.6709
against 4.6737 at k = 7 (-0.06% both). The shipped weights are the longer fine-tune (`FT-3`, 8558 steps): on
chat_1, code_1 and doc_1 (stride 4, 373 drafts) the NPU gives the same draft as the torch reference in all 7
tokens every time (373/373 at k = 3 and k = 7; tokens per pass equal).

A drafter pass (8 rows, 31 dispatches) on a quiet box (CPU 9-16%, no other NPU process): 42.1-42.6 ms:
matrices 30.3 ms (head 11.3, gate|up 8.5, down 4.7, conv_proj 2.2, q|k|v 2.1, o 1.7), host tables and sums
4.8, attention at 27 rows of context 2.1, selector 0.9, norms / convs / SiLU / embedding the remaining ~4.
Injecting 8 committed rows (fc + every layer's K|V) takes 5.9-6.4 ms. The cost model's 15.4 ms (t2 weights
9 ms + head 6.4) assumed the DMA floor; the big matrices run at 25-30 GB/s, not 52, because eight tokens take
two mmul quads per code block (twice the head's compute, `gemv_t2n.cc` is ~424 bundles per 128-K group of a
chunk), and the 20 small dispatches are bound by the ~0.2 ms dispatch cost.

## End to end (D5): Ternary Bonsai 2 27B, `open_kernels_w4h` (verify width 4), drafter FT-3

Everything below was run with the drafter and the target on one NPU, taps on, 256 generated tokens, greedy,
`--pmode performance`, the box idle (system CPU 3-8% before each run; the CLI's own threads put the run at
17-27%); every run checked for another NPU or GPU process first (none), and the configs of a class run back to
back with the order reversed every other round. Every run's tokens equal spec off's, token for token.
Nothing was discarded: the three rounds of a cell differ by 0-0.1 tok/s (decode only; the one-time work below
is in them).

CLI, decode tok/s, median of 3 rounds (`c1_*.ids` prompts; chat and reasoning prompts are 28 and 78 tokens, code
and doc 1765 and 1671):

| class | off | lookup w2 | lookup w3 | lookup w4 | dflash2 w2 | dflash2 w3 | dflash2 w4 | w4 vs off | w4 vs lookup w2 |
|---|---|---|---|---|---|---|---|---|---|
| chat | 6.06 | 6.27 | 6.33 | 6.20 | 7.21 | 8.63 | 9.43 | 1.56x | 1.50x |
| code | 5.59 | 9.00 | 11.24 | 12.84 | 6.43 | 8.32 | 9.76 | 1.75x | 1.08x |
| doc | 5.61 | 6.82 | 6.99 | 7.15 | 6.32 | 8.05 | 9.01 | 1.61x | 1.32x |
| reasoning | 6.06 | 6.58 | 6.51 | 6.61 | 7.29 | 8.76 | 9.74 | 1.61x | 1.48x |

Tokens per verify pass of dflash2 (w2 / w3 / w4): chat 1.83 / 2.48 / 2.97, code 1.99 / 2.97 / 3.92, doc 1.95 /
2.83 / 3.54, reasoning 1.86 / 2.55 / 3.11. Width 4 passes keep all 3 drafts 45 of 86 times in chat, 61 of 65 in
code, 59 of 72 in doc, 46 of 82 in reasoning.

`oflm serve` (`--spec`, `OFLM_OPEN_SPEC_WIDTH`), decode tok/s from `usage.decoding_speed_tps`, median of 3
sessions of the four serve prompts (chat template; `chat_1`, `code_1`, `doc_4`, `reason_1`), 256 tokens; message
content identical to spec off in all 15 sessions:

| prompt | off | lookup w2 | dflash2 w2 | dflash2 w3 | dflash2 w4 | w4 vs off | tokens per pass (w4) |
|---|---|---|---|---|---|---|---|
| chat_1 | 6.05 | 6.22 | 6.51 | 7.30 | 7.59 | 1.26x | 2.48 |
| code_1 | 5.54 | 7.61 | 5.90 | 7.18 | 7.94 | 1.43x | 3.31 |
| doc_4 | 5.60 | 6.31 | 5.98 | 7.22 | 8.19 | 1.46x | 3.31 |
| reason_1 | 6.07 | 6.32 | 7.09 | 8.52 | 9.28 | 1.53x | 3.04 |

The serve prompts are not the CLI's token lists (system turn and chat template, different texts), so the two
tables are not one experiment; both say the same about the widths.

### Per pass, width 4, CLI

| | chat | code | doc | reasoning |
|---|---|---|---|---|
| plain step at that context | 165 ms | 179 | 178 | 165 |
| verify pass, 4 positions, taps on | 262.8 | 300.4 | 298.9 | 264.1 |
| drafter block pass | 46.8 | 69.5 | 68.0 | 47.7 |
| drafter cache rows (3 new rows a pass) | 7.2 | 7.4 | 7.1 | 6.8 |
| one-time: the prompt's rows into the drafter cache | 25 ms (28 rows) | 1493 (1765) | 1326 (1671) | 90 (78) |
| tokens per pass | 2.97 | 3.92 | 3.54 | 3.11 |
| decode tok/s, steady state (no one-time work) | 9.37 | 10.40 | 9.46 | 9.76 |
| decode tok/s, measured | 9.43 | 9.76 | 9.01 | 9.74 |
| steady state vs off | 1.55x | 1.86x | 1.68x | 1.61x |

Projection (cost model, b = 31 ms a position, drafter 15 ms): chat 1.45x, code 1.85x, doc 1.49x, reasoning
1.52x. The drafter costs 54-77 ms a pass, not 15, and still lands at or above it: the acceptance on these
prompts is higher than the 20-prompt corpus's (2.73 / 3.49 / 2.81 / 2.86 tokens a pass at k = 3).

- **Taps** cost 2-3 ms a verify pass (`--bench-verify`, 3 rounds interleaved, short prompt: verify 4 252.5 ->
  254.3-256.2 ms median; step 162.8 -> 164.5-165.4), about 1%, and 0.3 s on a 1700-token prefill.
- **The verify pass in the loop** is ~8 ms slower than `--bench-verify`'s (262.8 against 254): the drafter's
  hardware context runs between two verify passes, and the target's has to be switched back.
- **At a full window** (code, doc) the drafter's block pass grows from 47 to 69 ms, all of it host attention
  over 2048 cached rows (25 ms of the pass, 5 layers x ~3.8 ms at 24 threads; memory-bound on the f32 K/V
  rows). A band version that reads each key row once for several queries was bit-identical but slower (5.96
  ms a layer against 3.77); with AVX exp and intrinsics for the value sums, 2.63 ms: ~6 ms of a ~370 ms pass,
  not kept. An f16 cache (the model rounds to f16 anyway) would halve the traffic.
- **The prompt's rows** go into the drafter cache once, 8 rows to a pair of dispatches, ~0.85 ms a row: 1.3-1.5 s
  for a 1700-token prompt, on top of a 15.8 s prefill, 6% of a 256-token decode. Only the last 2048 positions go in.

### Where it stands against prompt lookup

Prompt lookup's drafts are free, and on a copy-heavy prompt they are right: code at width 4 gives 12.84 tok/s
(2.30x off) against dflash2's 9.76, with 3.75 tokens a pass and no drafter pass (lookup w4: chat 1.12, code
3.75, doc 1.70, reasoning 1.23 tokens a pass). Everywhere else dflash2 w4 wins by 1.3-1.5x. The cost model of
the stack (`dacc/combo.txt`: lookup when it has a draft, dflash2 otherwise) says 2.21x on code and 1.52-1.60x
elsewhere at width 4; the engine does not stack them yet.

### For reference: the 0.8B draft model (branch `feat/bonsai2-specdraft`, width 2, drafter 25-39 ms a token)

Measured on the 2-wide verify image, 3 clean rounds: chat 1.09x, reasoning 1.08x, doc 1.16x, code 1.16x. The
DFlash 2 drafter at the same width 2: 1.19x, 1.20x, 1.13x, 1.15x; at width 4 1.56x-1.75x.

### Gates (2026-10-12/13)

- CLI, spec off against `--spec-method dflash2` at widths 2, 3, 4: `short` (33 tokens, == its reference list),
  `prompt512 --gemm-block` (129, == reference), chat, code, doc, reasoning (128, then 256 tokens in the 60 timed
  runs): all identical.
- `oflm serve` through the `oflm add` entry: the four prompts at `temperature 0`, 128 tokens, spec off /
  `OFLM_OPEN_SPEC=1` / `--spec`: identical, and equal to the 2026-10-08 lookup-day content; a `temperature
  0.7` request logs no `speculative decode:` line. `oflm serve|run <undeclared model> --spec` and a model whose
  `draft` is not installed both exit 1 before any weight loads.
- `--verify-check 48` with taps, n cycling 2..4, `short.ids`: 47 logit rows and 47 tap rows identical.
- Unit tests: `spec_test` (availability, `resolve_draft`), `dflash2_test` (walk, tables, `plan_sync`, attention),
  `taps_test`, `verify_slots_test`; `pytest specs/open-engine` 632 passed, `oflm-add` 28, `q4nx-build` 90.

## Lookup, 2026-10-08 (moved from OPEN-DECODE-SPECULATIVE; `open_kernels_v6`, width 2)

Both gates PASS: identical tokens on all six CLI cases (`short` and `prompt512` also equal their reference
lists), identical content on all four serve prompts through the variable and two through `--spec`, and a
`temperature 0.7` request does not speculate. Speed, three rounds under the timing lock, all CLEAN, 256
generated tokens, median decode tok/s off -> on:

- CLI: code 5.85 -> 9.19 (1.57x), doc 5.91 -> 7.03 (1.19x), reasoning 6.41 -> 6.96 (1.09x), chat 6.34 -> 6.46 (1.02x).
- `oflm serve`: code 5.81 -> 8.74 (1.50x), doc 5.85 -> 6.54 (1.12x), reasoning 6.36 -> 6.66 (1.05x), chat 6.34 -> 6.55 (1.03x).

Chat sits near breakeven: its drafts are accepted about one time in three, and a rejected draft costs the
pass's extra ~32 ms, and an accepted one saves a step less that, ~125 ms. Earlier history:
`bonsai2-speculative.md`.
