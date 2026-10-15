# Prompt-lookup drafts on the L-row verify pass, and an adaptive cycle (track D3, 2026-10-15)

Branch `perf/granite-spec-lookup` from `origin/perf/granite-decode-depth` (570f98b). No kernel change: the verify pass
(`dxl` + `lmhl`, L = 4) and the draft pass (`dxl_lora`) are used as built.

## Why

A Uno cycle is a draft pass plus a verify pass (~2.9 plain steps at short context) and commits ~3.0 tokens on Granite's
six prompts: 1.07x geomean, chat 0.85x. Prompt-lookup drafting (PLD) needs only the verify pass (~1.33 steps): its drafts
are the tokens that followed the latest earlier occurrence of the recent n-gram in the prompt or output. Neither wins
everywhere: lookup is free when the answer repeats the context (code edits, quoting), Uno when it does not.

## Requirements

New: **OPEN-SPEC-LOOKUP** (spec.md, next to OPEN-UNO-DECODE).
- The drafter: up to L - 1 drafts from the most recent earlier occurrence of the longest suffix n-gram, n = 4 .. 1,
  extended by looking up again when the continuation runs into the end. Deterministic: **test**
  (`tests/test_spec_lookup.py`, `src/open_qwen36/spec_lookup_test.cpp`, one shared case file).
- The per-cycle choice (lookup verify / Uno cycle / plain step) by expected committed tokens per ms, from running
  means of acceptance (lookup split by matched n) and of each pass's measured time. Speed only: **manual**.
- Output identical to plain greedy decode: `open_qwen36_cli --spec N` prints `SPEC IDENTICAL`. **manual**.
- Context capacity: a lookup pass needs pos + L <= max_ctx, a Uno cycle pos + 1 + L; else a plain step.

Modified: **OPEN-UNO-DECODE** served behaviour. With `OFLM_SPECULATIVE=1`, a greedy request on a family that opts in
decodes by the adaptive cycle (Uno is one of its three methods); `OFLM_SPEC_MODE=uno` restores the two-pass cycle every
cycle. A kernel set with the rows route but no `uno.q4nx` gets lookup + plain.

Results go into OPEN-FAMILY-GRANITE-UNO.

## Steps

1. CLI `--spec-probe N`: greedy by draft passes, dumping each position's Uno drafts (one sweep = the transcript and
   every position's Uno acceptance). `utilities/spec-lookup/spec_lookup.py sim` replays the transcripts.
2. CLI `--spec N --spec-mode adaptive|lookup|lookup-uno|uno|plain`, then plain decode; `SPEC IDENTICAL`.
3. CLEAN measurements (`dibs run --measure --tag d3`): plain / Uno / lookup / adaptive over the six uno-ref prompts,
   chat + code at 1024 / 2048, four lookup prompts (code edit, bug fix, quote, document Q&A).
4. Engine `spec_cycle` + AutoModel shared generate; serve check: greedy, cut and streamed equal plain decode.
