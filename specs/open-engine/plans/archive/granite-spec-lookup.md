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
- Context capacity: a lookup pass needs pos + L < max_ctx, a Uno cycle pos + 1 + L < max_ctx; else a plain step.

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

## Outcome

Landed as planned; the CLEAN numbers and the gates are in spec.md (OPEN-SPEC-LOOKUP's result, OPEN-FAMILY-GRANITE-UNO's
2026-10-15 table). Two things found on the way: the app's AVX2 greedy scan broke exact ties by lane rather than by id
(fixed, OPEN-UNO-DECODE), and a cycle may leave its next position at its last row + 1, so the capacity test is strict.
A third, about the measuring: dibs let sibling tracks' NPU runs (and one of this track's own) into CLEAN units, so the
matrix checks each unit's plain decode against known figures and the table sets slow runs aside (skill open-dxl-rows).

## Simulation (2026-10-15, offline, from `--spec-probe` transcripts)

Fourteen greedy transcripts (`spec_lookup.py probe`, 256 tokens; the lookup set 512): the six uno-ref prompts (thinking
on), chat / code at 1024 and 2048 (real repo text cut to length), and four lookup prompts (thinking off): `edit_py` (add
an option, whole file back), `fix_py` (an injected bug, whole file back), `quote_doc` (quote a spec section's procedures,
then explain), `qa_doc` (answer from a skill's traps section, quoting it).

Tokens per pass for lookup alone (a pass with no draft is a plain step, 1 token), first 128 tokens, pooled per group;
in brackets the share of passes that had a draft. Extension on or off gives the same figures on these transcripts.

| group | n 1 | n 1..2 | n 1..4 | n 2..4 | n 3..4 |
|---|---|---|---|---|---|
| short (6) | k1 1.13, k3 1.21 (44%) | k3 1.25 | k1 1.17, k2 1.22, k3 1.25 (42%) | k3 1.18 (15%) | k3 1.13 (9%) |
| ctx 1024 (2) | k3 1.30 | k3 1.32 | k1 1.20, k2 1.27, k3 1.32 (51%) | k3 1.24 (15%) | k3 1.16 (11%) |
| ctx 2048 (2) | k3 1.28 | k3 1.31 | k1 1.20, k2 1.26, k3 1.31 (56%) | k3 1.22 (19%) | k3 1.16 (10%) |
| lookup (4) | k3 1.78 | k3 2.26 | k1 1.63, k2 2.00, k3 2.32 (78%) | k3 2.16 (52%) | k3 2.01 (40%) |

Per position (every pending position of the transcripts, k = 3): drafts accepted by matched n, and Uno's at the same
positions (the probe's draft rows). Short: n1 0.30 (Uno 1.25), n2 0.86 (1.66), n3 0.90 (1.78), n4 2.18 (2.30), no
match: Uno 0.95. Lookup set: n4 2.49 at 393 of 512 positions (Uno 2.37). Where both draft, a 4-gram lookup is about as
good as Uno and anything shorter is worse; lookup wins on cost (one pass, not two).

Projected speedup over plain decode (128 tokens, the measured pass costs: step / verify / draft 48.7 / 64.8 / 76.1 ms
short, 65.1 / 82.3 / 93.5 at 1024, 81.9 / 101.4 / 112.3 at 2048, 71.1 / 88.4 / 100.7 on the lookup prompts), geomean:

| group | Uno | lookup | lookup, else Uno | adaptive | oracle (best method each cycle, in hindsight) |
|---|---|---|---|---|---|
| short | 1.07x | 1.10x | 1.09x | 1.13x | 1.35x |
| ctx 1024 | 1.11x | 1.17x | 1.24x | 1.18x | 1.41x |
| ctx 2048 | 1.16x | 1.16x | 1.18x | 1.21x | 1.43x |
| lookup | 1.48x | 2.03x | 2.02x | 2.00x | 2.16x |

Negative results, also from the transcripts:
- Uno cycles whose verify rows take lookup drafts (after c) when the lookup matches n >= 1..4: 0.93-1.05x short, against
  Uno's own 1.07x. Filling lookup's rows with Uno drafts needs the draft pass, which makes it a Uno cycle.
- n-gram sizes past 4 change nothing (nmax 6 / 8: lookup set 1.84x / 1.85x against 1.83x).
- Uno's draft rows started from the lookup's guess of their positions instead of noise (`--spec-probe-seed`, the same
  noise stream elsewhere, so a paired comparison): accepted drafts 1.22 / 1.67 / 1.76 / 2.31 against 1.25 / 1.66 / 1.78 /
  2.30 at matched n 1..4 on the short prompts, the same within 0.13 on edit / fix. The LoRA reads its rows as noise.
- Uno only where the lookup drafts nothing (else lookup or plain): 1.13 / 1.25 / 1.19 / 2.00x against adaptive's
  1.13 / 1.18 / 1.21 / 2.00x; the policy's parameters (weight 0.15-0.5, drift 0-0.1, priors) move the overall geomean
  between 1.31x and 1.36x, inside what one draw of Uno's noise moves it.
- A 2-row or 8-row verify, priced linearly in rows from the measured 4-row pass (an estimate; an 8-row pass also needs a
  different attention layout, 8 groups of 5 cores do not fit): lookup alone over the whole transcripts (cut at the
  reply's end) goes 1.08x / 1.06x / 1.13x / 1.87x (short / 1024 / 2048 / lookup set, 4 rows) to 1.10 / 1.09 / 1.13 /
  1.48x (2 rows) and 0.94 / 0.92 / 0.99 / 2.05x (8 rows; `fix_py` 3.03x -> 4.76x). Only whole-file rewrites want more
  rows.
- A verify (and draft) pass at 1.10 plain steps instead of 1.24-1.33 (estimate) would lift adaptive to ~1.30x at every
  depth on the reasoning prompts and 2.27x on the lookup set: the pass's fixed overhead is worth more than its row count.
