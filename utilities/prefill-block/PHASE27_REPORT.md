# Phase 27 — production runtime integration

**Status: PARTIAL — correctness gates PASS, one integration gate BLOCKED on tooling.**

The block-prefill path is correct on the production `Core` path at full depth over a
16-token decode trajectory, and the tail crossover has been measured on this dense model
for the first time. What could not be completed is driving `oflm run`/`serve`
non-interactively in this environment, and applying the measured crossover to the shipped
constant — both are reported honestly below rather than claimed.

**Frozen decisions respected:** `paired` arithmetic, GEMM geometry, physical `T = 256`,
DeltaNet arithmetic, attention arithmetic and the Q4 format are all untouched. No
performance-kernel work was done.

---

## The central finding: block prefill was already in production

Phase 27 was scoped as "integrate the Phase 26 path into the runtime". It turned out the
integration already exists, from the Qwen3.6 work:

| place | what it does |
|---|---|
| [`engine.cpp:155`](../../src/open_qwen36/engine.cpp:155) `Engine::prefill` | the production prefill for both `run` and `serve` |
| `engine.cpp:174` | `OFLM_OPEN_GEMM_BLOCK != "0"` → block route is **ON by default** |
| `engine.cpp:183` | gate: prompt ≥ `min_prompt`, no mRoPE position counter |
| `engine.cpp:192-202` | `while (i < ids.size())` → `step_gemm_block(block, t_real, ...)` with the tail padded by a repeated in-range id |
| `engine.cpp:207` | the sequential fallback, otherwise unchanged |
| `engine.cpp:178` | `min_prompt = 64` — **inherited from the Qwen3.6-35B MoE**, where the reason was expert-weight streaming |

So the questions Phase 27 actually had to answer were different from the ones the plan
anticipated: is that route correct for *this* model, is the 64-token crossover right for a
*dense* model, and is the schedule it uses the one Phase 26 validated?

### The schedule question, answered

`engine.cpp:188` prefers `step_gemm_prompt` (layer-major, MoE only). For this dense model
`Core::layer_major_ok()` returns false, because it requires every layer type to be
`linear` or `full` — and `layer_major_ok()` is documented as "the MoE kinds (linear /
full)". So the dense path takes the `step_gemm_block` loop, whose layer bodies
(`block_layer_linear` / `block_layer_full`) are **the same functions the Phase 26 oracle
measured**. The oracle therefore validates the exact production schedule, not a lookalike.

## 27.0 — production contract

Physical `T = 256` stays internal: it is not a setting, not in the settings registry, and
`engine.cpp` never reads it from configuration — it asks the manifest
(`Core::gemm_block_t()`). No new public setting was added. `OFLM_OPEN_GEMM_BLOCK` /
`OFLM_OPEN_GEMM_BLOCK_MIN` already existed and remain diagnostic.

## 27.4 — tail crossover, measured

A logical tail costs a full-width dispatch, so the crossover had to be measured rather than
inherited. Same binary, same `Core` block body (`step_gemm_block` vs `core.step()`), same
hardware session, 8 layers, one generation token, wall time (the request dominates so wall
is the honest number here):

| n | sequential | padded 256 block | block/seq |
|---|---|---|---|
| 1 | 2.05 s | 6.53 s | 3.19 |
| 8 | 2.45 s | 6.63 s | 2.71 |
| 32 | 3.71 s | 6.69 s | 1.80 |
| 64 | 5.50 s | 6.51 s | 1.18 |
| **128** | **8.98 s** | **6.45 s** | **0.72** |
| 192 | 12.40 s | 6.37 s | 0.51 |
| 255 | 16.03 s | 6.79 s | 0.42 |
| 256 | — | 6.47 s | — |

Two properties matter:

1. **The block cost is flat** — 6.4–6.8 s from n=1 to n=256. That is the predicted
   full-width dispatch and it confirms the mechanism rather than merely measuring it.
2. **The crossover sits between 64 and 128.** Sequential is still 18 % cheaper at n=64 and
   28 % cheaper at n=128. Taking the conservative side of a noisy boundary, as 27.4
   requires, puts **C = 128**.

**The shipped constant of 64 is too low for this model.** At n=64 the block route costs
1.18× the sequential route — a real regression on every prompt of 64–127 tokens. This is
the one concrete defect Phase 27 found in the shipped configuration.

**Not yet applied.** Changing that constant is a behavioural change to a shipped default,
and the plan requires the `run`/`serve` gates to pass first (27.16). The measurement is
recorded here so it can be applied as a one-line change once those gates are reachable.

## 27.9 / 27.10 — the extended release oracle: PASS

Reused the Phase 26 oracle (no second, weaker script) at **full 64-layer depth**, 40 prompt
tokens as chunks of 32 with a tail of 8, then **16 greedy decode steps**.

```
generated tokens, both paths, identical:
[271, 104980, 99943, 103916, 95761, 332, 99163, 98277, 5205,
 104071, 96275, 95793, 102038, 109970, 95726, 101742]
```

| quantity | value |
|---|---|
| greedy agreement | **16 / 16** |
| logit cosine | 0.999999733 … 0.999999986 (gate 0.9999) |
| logit `max_abs` | 0.00293 … 0.00840 |
| `max_abs` trend | first-5 median 0.00519 → last-5 median 0.00633 |
| `L62.S` (DeltaNet state) | cos 0.999999996, max_abs 1.2e-4 |
| `L62.conv` | cos 0.999999999, max_abs 1.1e-3 |
| `L63.kv` | cos 0.999999886, max_abs 0.125 |

Per 27.9 this is precisely the case the plan warned about — **very high cosine with
non-trivial absolute KV differences** — and no new absolute-error threshold was invented to
absorb it. The existing gate is preserved, `max_abs`/`max_rel` are reported for every
tensor, and the `max_abs` trend is **bounded, not explosive** (medians 0.0052 → 0.0063
across 16 steps, oscillating with the token, not compounding). The additional functional
gate the plan asks for — identical generated-token trajectory — **passes 16/16**.

Full-depth state also passes with the canary intact, so production does not write outside
its own state regions at 64 layers.

## 27.14 — correctness-only overhead: verified absent

| item | status |
|---|---|
| per-token LM head during prefill | **absent**: `want_logits` is passed for the last real token only (`engine.cpp:199`) |
| state dumps / canaries | **absent**: `--dump-state` / `--state-canary` exist only in the standalone `open_qwen36_cli`, never in `Engine` |
| `read_state` / `write_state_canary` | bring-up accessors; nothing on the production path calls them |
| `block_logits_all_` | **defaults to `false`** (`core.hpp:484`); only a diagnostic CLI turns it on |
| reference copies / per-layer comparison | absent from `Engine::prefill` |

## 27.11 — failure policy, as it stands

The existing behaviour is already the plan's "known unsupported capability → sequential
fallback": `GT == 0` (no block program) falls through to `core.step()` at
`engine.cpp:207`, and the comment records that the earlier code wrongly returned early,
leaving the prompt unprocessed (#109).

What is **not** yet implemented is the second branch of 27.11 — an *unexpected* failure on
an advertised capability must raise rather than fall back, and any fallback after state may
have changed must reset state first. `Core::step_gemm_block` throws on a malformed
manifest, and an exception propagates out of `Engine::prefill` through `guarded`, so it does
not silently continue; but there is no explicit restore-then-fallback path. This is a real
gap, recorded rather than papered over.

## 27.6 / 27.7 / 27.8 — BLOCKED on tooling

`oflm run` is an interactive chat loop (`Runner::run()`), with no `--ids`, `--prompt-file`
with programmatic exit, or non-interactive generation flag that I could find in
`vm_args.hpp`/`program_args.hpp`; driving it headlessly was not achievable in this pass, so
the plan's required matrix over 255/256/257/511/512/513 and the `serve` fresh-request /
second-request / after-reset cases are **NOT validated on the `oflm` binary**.

Claiming them on the strength of the `Core`-level oracle would be exactly the substitution
26.17/27.10 forbid, so they are left open.

What *is* established for the state machinery those gates depend on:

- the block-to-decode handoff over 16 tokens at 64 layers (above), which is the hardest
  part of the handoff;
- the Phase 26 boundary matrix 31/32/33, 63/64/65, 127/128/129, 255/256/257 on the same
  `Core` bodies the runtime calls;
- the Phase 26 non-zero positions P = 0/17/255/511/257.

Request/reset isolation (27.8) has an existing precedent in the repo —
`OFLM_OPEN_REQUEST_ISOLATION` and `core.reset()` / `checkpoint()` / `restore()` — but it was
not exercised here.

## 27.12 — context length

Unchanged and independent of the block size: `Core::gemm_block_t()` is 256 but
`cfg_.max_ctx` comes from `MAX_L` / `--ctx-len`, and `step_gemm_block` throws only if a
*block* exceeds the compiled `T`, never if the context is not a multiple of 256. No
rounding to 256 exists anywhere in the path. Not separately re-validated at large context;
that is Phase 29.

## 27.15 — benchmark: NOT RUN

The production benchmark over 128/256/512/1024/2048 needs the `oflm` binary, which is the
blocked item. The crossing table in 27.4 is a real measurement on the same `Core` body but
at 8 layers, so it supports the crossover decision and **does not** support a throughput
claim for the shipped runtime. No performance number is claimed.

---

## Gate status

```
[x] normal Core/model API supports block prefill        (already existed: Engine::prefill)
[x] physical T=256 remains internal                    (manifest-driven, not a setting)
[x] full blocks execute through production path        (step_gemm_block loop)
[x] tail crossover measured                            C = 128 (was 64)
[ ] automatic tail policy implemented                  MEASURED but NOT APPLIED (blocked by 27.16 order)
[x] sequential fallback preserved                      engine.cpp:207, untouched
[ ] oflm run passes                                    BLOCKED: interactive stdin only
[ ] oflm serve passes                                   BLOCKED: same
[ ] request/reset isolation passes                     BLOCKED: needs oflm; mechanism exists
[x] 16-token post-prefill decode trajectory matches    16/16 greedy identical, 64 layers
[x] no correctness-only dump/copy overhead             verified absent
[ ] production runtime benchmark recorded               BLOCKED: needs oflm
[ ] default-enable decision evidence-based             cannot conclude until the above
[x] regression suite and settings CI pass
```

## Recommended Phase 27 completion

1. Add a non-interactive generation path to `oflm run` (ids + `--max-tokens` + exit), or
   accept that the `Core`-level oracle is the release gate and say so explicitly.
2. Apply `min_prompt` 64 → **128** for the dense Qwen3.5 recipe, keeping
   `OFLM_OPEN_GEMM_BLOCK_MIN` as the diagnostic override.
3. Implement the 27.11 second branch: restore state before any fallback after partial work.
4. Then run 27.6/27.7/27.8 and 27.15, and make the 27.16 default decision from those.

## Recommended Phase 28 work

Profile-driven optimisation of the block route, using the measured economics from Phase 25:
the projection GEMM is 67.5 % of wall and attention is the only superlinear term. Phase 28
should not revisit arithmetic — that is frozen — and should treat the `_low`
context-ordering observation (2.4 %, documented and deferred) as the first cheap item.

## Artefacts

- [`tail_crossover.py`](tail_crossover.py) — the crossover harness. Its intent is to drive
  `oflm`, which is why it exists, and it is **not run green yet**: `oflm` has no headless
  `run` invocation. Every number in the 27.4 table came from `open_qwen36_cli` driving the
  same `Core` bodies (`--gemm-block` vs not), which is why it is labelled a `Core`-level
  measurement and not an `oflm` one.
- [`oracle.py`](oracle.py), [`decode_transition.py`](decode_transition.py) — the Phase 26
  release oracle, extended to 16 decode tokens
- `runs/decode-n40-c32-d16-l64-*/result.json` — the 16-token trajectory result (the raw
  dumps are gitignored; `result.json` is committed)
- [`BLOCK_CONTRACT.md`](BLOCK_CONTRACT.md), [`PHASE26_REPORT.md`](PHASE26_REPORT.md)
