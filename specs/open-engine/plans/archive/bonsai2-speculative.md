# Plan: speculative greedy decode for Bonsai 2 (prompt-lookup draft, 2-position verify pass)

**Status:** done (2026-10-08). Builds on `bonsai2-round3.md`.

## Spec impact

**New requirements**
- **OPEN-DECODE-SPECULATIVE-VERIFY:** for an all-t2 merged-image spec the export adds a verify image
  (`vlx`, `vax`, `vlm` in their own hardware context) that runs two positions in one pass,
  bit-identical to two decode steps; DeltaNet state lives in width + 1 slots re-pointed by an
  instruction patch (`stateslot`), attention takes one window per position (`attnposv`).
  Verification: `test` for the slot / position bookkeeping and the two patches
  (`src/open_qwen36/verify_slots_test.cpp`) and the recipe's emission (`tests/test_verify_image.py`);
  `manual` for bit-identity on the NPU (`--verify-check 40`).
- **OPEN-DECODE-SPECULATIVE:** greedy requests may decode speculatively (CLI `--spec`, app
  `OFLM_OPEN_SPEC=1` / `oflm serve|run --spec`), with a prompt-lookup drafter; the output is the
  same on and off. Verification: `test` for the drafter and the accept rule
  (`src/open_qwen36/spec_test.cpp`); `manual` for the CLI and serve gates and the speed.

**Modified requirements**
- **OPEN-DECODE-ONE-CONTEXT-DENSE:** with a verify image the decode stream `lx` carries the
  `stateslot` patch (`tests/test_one_context_dense.py`).

## Measurement history

### The verify pass

**2026-10-07 (kernel set `open_kernels_v1` = `open_kernels_r3` + vlx / vax / vlm; container
`Ternary-Bonsai-2-27B-NPU2-lmhead`).** The 2-token tile alone (`designs/gemv_t2` probe, `t2x2`): y
byte-identical to the 1-wide tile per token on the gate, og and down shapes at 1, 2 and 8 cores;
1.01x the 1-wide time at 8 cores, 1.45x where both are core-bound. Main cores 16,000 B (glue
15,584). `--verify-check 40`: PASS on short.ids (20 passes, 6 steps, 40 rows) and on prompt512
`--gemm-block` (40 rows). Tokens 33/33 and 129/129. The decode step unchanged: clean A/B, 3 rounds
of prompt512 `--gemm-block` + 128 steps against the round-2 CLI on `open_kernels_r3`: step minimum
155.7 / 155.8 / 155.7 -> 155.9 / 155.7 / 155.6 ms.

Cost (`--bench-verify 20`, clean, min / median ms), `open_kernels_v5` (the hoisted 2-token tile, the
glue's alpha / beta pass shared by both positions, earlier attention and glue starts, both
positions' DeltaNet rounds as one run): at ctx ~512 a step 156.1 / 156.7 and a 2-position pass
195.9 / 197.6 (1.25x; `open_kernels_v1` 211.1 / 214.0); at ctx ~20-60, 153.0 / 153.9 and
188.6 / 190.2 (1.23x). Alternating verify and step (a hardware-context change each way): the pass
+1-2 ms, the step +1-2 ms. Per dispatch: vlx 3.04 ms against lx 2.35, vax 2.63 against ax 2.15,
vlm 6.43 against lm 6.63 -- the head's second row is free; the linear layer's extra is DeltaNet,
the norm stages, the preps and the glue's conv for the second position.

**2026-10-08 (`open_kernels_v6` = v5 with vlx / vax / vlm rebuilt).** Timing-only knock-out builds
(each dropping position 1's share of one stage, proven by verify[1]'s logits changing and
verify[0]'s not; the knob was removed before review and lives in the feat/bonsai2-specserve
history) put the linear layer's extra 0.60 ms at: position 1's glue 0.39 (the stream finished both
positions' glue before any DeltaNet round), its DeltaNet + post 0.15, its norms 0.035, its preps
and the 2-token tile's compute 0. v6 issues position 1's glue under position 0's DeltaNet rounds and
sends each position's xn / xm to the main cores as soon as its norm is drained. Same bits:
`--verify-check 40` PASS. Clean, one window, ctx ~512, `--bench-verify 10` min / median ms: v5 pass
198.1-199.3 / 199.8-200.1, v6 190.4-190.8 / 191.2-192.7, step 158.2-159.9 on both (vlx 3.02 ->
2.85 ms). A 4-wide GEMV tile was also probed (not used: width 3 and 4 do not fit a main core's data
memory without slicing K).

### Speculative decode

**2026-10-07 (`open_kernels_r3`, reference verifier, width 4).** Both gates PASS: identical tokens
on all six CLI cases and identical content on all four serve prompts, through the variable and the
flag. Tokens per pass: prompt512 4.00, code 3.85-3.94, doc 1.50-1.61, reasoning 1.19-1.26, chat
1.07-1.09. On the reference verifier every class decodes slower than plain (a 4-position pass costs
~700 ms against ~160 ms a step): it exists for correctness before the verify program.

**2026-10-08 (`open_kernels_v5`: width 2, a 2-position pass ~196 ms against a 156 ms step).** Both
gates PASS. Three CLEAN rounds, 256 tokens, median decode tok/s off -> on:
- CLI: code 5.90 -> 8.88 (1.51x), doc 5.93 -> 6.87 (1.16x), reasoning 6.42 -> 6.89 (1.07x), chat
  6.43 -> 6.57 (1.02x). Tokens per pass 1.96 / 1.38 / 1.14 / 1.08.
- `oflm serve`: code 5.88 -> 8.49 (1.44x), doc 5.90 -> 6.42 (1.09x), reasoning 6.43 -> 6.64 (1.03x),
  chat 6.41 -> 6.55 (1.02x).

Stricter drafting (a suffix of >= 3, or >= 3 unless the last draft was accepted) projects lower in
every class on the prompt-lookup transcripts, so the rule stays at >= 2.

**2026-10-08 evening (`open_kernels_v6`).** Gates PASS (the same pass and accept counts as v5; serve
content equal to the v5-day spec-off content). Off / v5 / v6 in the same windows, median decode
tok/s, on v6 (v5 in brackets): CLI code 1.57x (1.51x), doc 1.19x (1.16x), reasoning 1.09x (1.07x),
chat 1.02x (1.03x; one slow v6 round, v6 ahead in the other two); serve code 1.50x (1.45x), doc
1.12x (1.09x), reasoning 1.05x (1.03x), chat 1.03x (1.02x).
