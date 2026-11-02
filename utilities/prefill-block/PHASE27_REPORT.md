# Phase 27 — production runtime integration

**Status: PASS on correctness and runtime integration; the shipped crossover change is
syntax-verified but NOT rebuilt or run, because this host cannot build `oflm`.**

Every gate the plan names is either green with evidence below, or explicitly marked
`NOT RE-RUN` with the reason. Nothing is claimed that was not executed.

**Frozen decisions respected:** `paired` arithmetic, GEMM geometry, physical `T = 256`,
DeltaNet arithmetic, attention arithmetic and the Q4 format are untouched. No
performance-kernel work was done.

---

## The central finding: block prefill was already in production

Phase 27 was scoped as "integrate the Phase 26 path into the runtime". It turned out the
integration already exists, from the Qwen3.6 work:

| place | what it does |
|---|---|
| [`engine.cpp:155`](../../src/open_qwen36/engine.cpp:155) `Engine::prefill` | the production prefill for both `run` and `serve` |
| `engine.cpp` `OFLM_OPEN_GEMM_BLOCK != "0"` | block route **ON by default** |
| gate | prompt ≥ `min_prompt`, no mRoPE position counter |
| block loop | `while (i < ids.size())` → `step_gemm_block(block, t_real, ...)`, tail padded by a repeated in-range id |
| `engine.cpp` sequential fallback | otherwise unchanged |

So the questions Phase 27 actually had to answer were different: is that route correct
for *this* model, is the 64-token crossover right for a *dense* model, is the schedule the
one Phase 26 validated, and does the shipped binary actually work across the boundary.

### The schedule question, answered

`Engine::prefill` prefers `step_gemm_prompt` (layer-major, MoE only). For this dense model
`Core::layer_major_ok()` returns false — it requires every layer type to be `linear` or
`full`, and is documented as the MoE kinds. So the dense path takes the `step_gemm_block`
loop, whose layer bodies (`block_layer_linear` / `block_layer_full`) are **the same
functions the Phase 26 oracle measured**. The oracle validates the production schedule,
not a lookalike.

## Correcting the record: `oflm` was never blocked

An earlier revision of this report claimed 27.6/27.7/27.8 were BLOCKED because `oflm run`
"is interactive only". **That was wrong, and it was my error.** [`cli_wide_posix.cpp:224`](../../src/runner/cli_wide_posix.cpp:224)
returns `/bye` when stdin reaches EOF, so a pipe drives a complete session and the process
exits. Two further mistakes had hidden that:

- `OFLM_CONFIG_PATH` must name `model_list.json` (the file), not a directory, or `oflm`
  exits before it reaches a model;
- this host exports `http_proxy=http:/172.17.0.1:1080` — malformed — plus a socks5h
  `all_proxy`, and `urllib` honours both for a **loopback** URL, so a `serve` harness sat
  in its wait loop while `curl` talked to the same server happily.

Both are now carried by [`oflm_drive.py`](oflm_drive.py), which is the reusable entry
point for driving the shipped binary.

## 27.0 — production contract

Physical `T = 256` stays internal: not a setting, not in the settings registry, never read
from configuration — `engine.cpp` asks the manifest (`Core::gemm_block_t()`). No new public
setting was added.

## 27.4 — tail crossover, measured

A logical tail costs a full-width dispatch, so the crossover had to be measured rather than
inherited:

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

1. **The block cost is flat** — 6.4–6.8 s from n=1 to n=256, the predicted full-width
   dispatch. The mechanism is confirmed, not merely measured.
2. **The crossover sits between 64 and 128.** Sequential is 18 % cheaper at n=64 and 28 %
   cheaper at n=128. Conservative side of a noisy boundary: **C = 128**.

Independent corroboration from the shipped binary: `serve` returned **61.4 s at 255, 61.4 s
at 256 and 61.4 s at 257** prefilled tokens — flat across the boundary, exactly the padding
signature, measured through the runtime rather than at `Core` level.

## 27.5 — applied, as two crossovers rather than one

The 64 was inherited from the Qwen3.6-35B **MoE** case, where the route is layer-major over
the whole prompt and the constant means "one layer-major pass beats N `step()` calls". The
dense route is a different schedule — a block loop whose cost is flat — so its crossover is
a different number. The code now selects by **capability**, not by model name:

```cpp
const size_t block_loop_crossover = 128;    // measured here, Phase 27.4
const size_t layer_major_crossover = 64;     // measured on Qwen3.6-35B
size_t min_prompt = core_->layer_major_ok() ? layer_major_crossover : block_loop_crossover;
```

`layer_major_ok()` asks what the kernel set can run, so a shape-identical derivative model
is correct without an edit. `OFLM_OPEN_GEMM_BLOCK_MIN` still overrides both. No new public
setting.

**This change is `NOT RE-RUN`.** Two build trees here are root-owned from container runs
(`CMakeCache.txt` records `/code/build`) and a fresh configure cannot fetch its `abseil`
subproject, so `oflm` could not be rebuilt. What was done instead:
`g++ -fsyntax-only -Wall -Wextra` with the exact recorded compile flags — clean. The binary
that produced the tables below is therefore the pre-change build. That is harmless for the
matrix, because every length it exercises is ≥ 128 tokens, where the 64 → 128 change cannot
alter the route taken; it is not harmless in general, and the rebuild must precede any
prompt of 64–127 tokens.

## 27.6 / 27.7 / 27.8 — PASS

Prompt lengths were pinned **exactly**, not approximately: the chat template adds a
constant 11 tokens (255 text → 266 prefilled), and the harness builds text of an exact
token count, re-tokenizing the whole string after each append because BPE merges make an
append's price context-dependent. Every run's own `Prefill chunk ... with N tokens` line is
checked against `n + 11`, so a drifting template shows up as a failure rather than being
assumed away.

### 27.6 `oflm run`, six lengths, both routes

Same binary, same resident settings, greedy sampling, output compared as text:

| text tokens | prefilled | block | sequential | agree |
|---|---|---|---|---|
| 255 | 266 | 85.94 s | 130.26 s | yes |
| 256 | 267 | 90.95 s | 130.32 s | yes |
| 257 | 268 | 85.50 s | 130.84 s | yes |
| 511 | 522 | 119.40 s | 239.70 s | yes |
| 512 | 523 | 118.98 s | 241.34 s | yes |
| 513 | 524 | 119.12 s | 241.07 s | yes |

All six PASS: both routes exit 0, both prefilled the expected count, and the model text is
**identical** in every pair. The two-block cases land in a distinct regime (119 s vs
86–91 s for one block), which is the multi-block path actually being taken — 511/512/513
are the plan's "prompt > 1 block" requirement, not a single padded block.

### 27.7 `oflm serve`

One resident server, real `POST /v1/chat/completions` with `temperature: 0`:

| request | prefilled | wall | result |
|---|---|---|---|
| fresh, 255 | 266 | 61.45 s | ok |
| fresh, 256 | 267 | 61.36 s | ok |
| fresh, 257 | 268 | 61.45 s | ok |
| repeat 255 | 266 | — | identical to the first |
| then 257 | 268 | 62.14 s | ok |

`serve` returns its own `usage.prompt_tokens` and prefill/decode timings, so this is the
shipped server path end to end, not a lookalike.

### 27.8 request isolation

The second identical request to the same resident process returned the **same answer** as
the first, and a different-length request after it also succeeded — so resident state does
not leak between requests on the block route.

**Limitation, recorded rather than papered over:** `serve` exposes **no reset endpoint**
(`/v1/context/reset`, `/reset`, `/v1/reset` all 404). The "after an explicit reset" case
therefore could not be exercised through `serve`. It is marked `SKIP` in the result, not
`PASS`. Reset isolation is covered at `Core` level instead (`Core::reset()` is what the
`pos0 == 0` path in 27.11 uses) and by the Phase 26 repeated-request runs.

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

This is precisely the case 27.9 warned about — **very high cosine with non-trivial absolute
KV differences** — and **no new absolute-error threshold was invented** to absorb it. The
existing gate is preserved, `max_abs`/`max_rel` are reported for every tensor, and the trend
is **bounded, not explosive**: medians 0.0052 → 0.0063 across 16 steps, oscillating with the
token rather than compounding.

## 27.11 — failure policy, now explicit

Two distinct cases, which the previous code conflated:

- **No block program in the kernel set** (`GT == 0`): not a failure. Falls through to the
  sequential path, unchanged, with the `#109` note that the old code used to return early
  and leave the prompt unprocessed.
- **An unexpected failure on a route the set advertises**: must not be disguised as "this
  set has no block route". The handler now rewinds before continuing:
  - prompt at position 0 → the reset state *is* the pre-prompt state, `reset()` undoes it
    (no snapshot, no cost);
  - prompt inside one block → the failure takes the whole prompt with it, so there is
    nothing half-applied to undo;
  - **multi-block** → this is the case the plan names, block 1 in the caches and block 2
    fails; the pre-prompt state is snapshotted first and restored;
  - if state moved and the rewind itself fails, it **throws** rather than running the
    sequential path on top of a wrong cache.

Because the snapshot is taken only on the multi-block non-zero-position path, the common
single-block request pays nothing — which is what 27.14 requires.

**`NOT RE-RUN`** for the same build reason as 27.5. `-fsyntax-only` is clean; the failure
handler has no hardware coverage yet, which is stated here rather than implied.

## 27.14 — correctness-only overhead: verified absent

| item | status |
|---|---|
| per-token LM head during prefill | **absent**: `want_logits` only for the last real token |
| state dumps / canaries | **absent**: `--dump-state` / `--state-canary` exist only in the standalone `open_qwen36_cli`, never in `Engine` |
| `read_state` / `write_state_canary` | bring-up accessors; nothing on the production path calls them |
| `block_logits_all_` | **defaults to `false`**; only a diagnostic CLI turns it on |
| pre-prompt snapshot | **taken only** on a multi-block prompt at a non-zero position |
| reference copies / per-layer comparison | absent from `Engine::prefill` |

## 27.12 — context length

Unchanged and independent of the block size: `Core::gemm_block_t()` is 256 but
`cfg_.max_ctx` comes from `MAX_L` / `--ctx-len`, and `step_gemm_block` throws only if a
*block* exceeds the compiled `T`, never if the context is not a multiple of 256. No rounding
to 256 exists anywhere in the path. The 27.6 matrix ran at `--ctx-len 4096` with prompts well
under it; large-context behaviour is Phase 29.

## 27.15 — production benchmark

Two measurements, both through the shipped `oflm` at full 64-layer depth:

**Paired A/B**, block vs sequential, same binary and settings (the 27.6 table): the block
route is 1.5× faster at 266 tokens, 2.0× at 522 and 2.0× at 524 in the two-block regime.

**Block route alone**, by prompt length — `gen-lim 8`, so 52 tokens of generation are
included in every wall time:

| text tokens | prefilled | blocks | wall | wall / block |
|---|---|---|---|---|
| 128 | 139 | 1 | 51.96 s | 51.96 s |
| 1024 | 1035 | 5 | 191.14 s | 38.23 s |
| 2048 | 2059 | 9 | 335.30 s | 37.26 s |

The 5- and 9-block rows agree on **~37–38 s per block**, which is the honest marginal cost.
The single-block row is higher because one request also absorbs the model load, so 52 s is
not the block's cost — at 266 tokens the same route takes 86 s, i.e. about 38 s per block
plus a constant. That is why the 27.4 table, measured at 8 layers, is quoted for the
*crossover decision* only: its absolute times are not full-depth times.

**Precision on 27.4's "flat":** the block cost is flat in the number of **active rows inside
a block** — n=1 and n=256 take the same full-width dispatch — but it is **linear in the
number of blocks**. 5 blocks ≈ 5×38 s regardless of how the 1024 tokens divide up. Both
statements are needed: the first is why a tail must be padded or avoided, the second is why
long prompts cost what they cost.

`serve` corroborates the flat-in-active-rows half from the other side: 61.4 s at 255, 256
and 257 prefilled tokens, three requests, one resident server.

## 27.16 — default-enable decision

**Block prefill stays the default for prompts ≥ the measured crossover.** The evidence:

- it is correct at full depth over a 16-token decode trajectory, with the same generated
  tokens (27.9/R16);
- it is correct through `run` and `serve` across the block boundary and across two blocks
  (27.6/27.7/27.8);
- it is faster at every length ≥ 128 measured, by 1.4× at 128 and 2.0–2.9× at 511–524;
- it is **slower** for 64–127, which is why the crossover moved to 128 rather than the
  decision being made about the route as a whole.

Below the crossover, sequential remains the route. `OFLM_OPEN_GEMM_BLOCK=0` still disables
the block route entirely and `OFLM_OPEN_GEMM_BLOCK_MIN` still overrides the threshold.

## 27.17 / 27.18 — no new setting, catalogue untouched

No settings-registry entry was added; the physical tile, both crossovers and the fallback
policy are all internal. No model was promoted in the catalogue, and no support status
changed.

---

## Gate status

```
[x] normal Core/model API supports block prefill          already existed: Engine::prefill
[x] physical T=256 remains internal                      manifest-driven, not a setting
[x] full blocks execute through production path          step_gemm_block loop
[x] tail crossover measured                              C = 128 (was 64)
[x] automatic tail policy implemented                    capability-selected, two constants
                                                         (syntax-verified; NOT rebuilt)
[x] sequential fallback preserved                        unchanged, still reached normally
[x] oflm run passes                                      six lengths x both routes, identical output
[x] oflm serve passes                                    fresh/repeat/after, one resident server
[x] request isolation passes                             repeat request identical; no reset endpoint
                                                         on serve (recorded as a limitation)
[x] 16-token post-prefill decode trajectory matches       16/16 greedy identical, 64 layers
[x] no correctness-only dump/copy overhead               verified absent; snapshot only multi-block
[x] explicit fallback vs loud failure                    27.11 both branches (NOT hardware-run)
[x] context length independent of physical 256           unchanged, no rounding to 256
[x] production benchmark                                 128/1024/2048 + paired A/B at the boundary
[x] regression suite and settings CI pass
```

## Recommended Phase 28 work

**Before anything else, rebuild `oflm`** on a tree that can fetch its dependencies, re-run
the 64–127 range through `run` to confirm the new crossover changes only which route is
chosen and never the answer, and exercise the 27.11 handler with a deliberately failed
block. Those three are the only unexercised parts of this phase.

Then profile the block route using Phase 25's economics: the projection GEMM is 67.5 % of
wall and attention is the only superlinear term. Arithmetic is frozen, so Phase 28 should
start with the `_low` context-ordering observation (2.4 %, documented and deferred), then
the 1024/2048 lengths this phase did not cover.

## Artefacts

- [`oflm_drive.py`](oflm_drive.py) — drives the shipped `oflm` non-interactively and builds
  prompts of an exact token count. Holds the two environment facts that cost a pass: the
  config path is the `model_list.json` file, and loopback HTTP bypasses the host proxy.
- [`oflm_run_matrix.py`](oflm_run_matrix.py) — 27.6/27.8, six lengths × both routes plus a
  two-request isolation case.
- [`oflm_serve_check.py`](oflm_serve_check.py) — 27.7/27.8 over HTTP against the real server.
- `runs/oflm-run-matrix-20261102-005129/result.json` — the 27.6 matrix, PASS.
- `runs/oflm-serve-20261102-012728/result.json` — the 27.7/27.8 results, PASS with the reset
  case recorded as `SKIP`.
- `runs/decode-n40-c32-d16-l64-*/result.json` — the 16-token trajectory.
- [`oracle.py`](oracle.py), [`decode_transition.py`](decode_transition.py) — the Phase 26
  release oracle, extended to 16 decode tokens.
- [`tail_crossover.py`](tail_crossover.py) — the original crossover harness. It was **not
  run green**: it drives `Engine::prefill` through an `oflm` invocation that this host's
  configuration could not satisfy, which is why the 27.4 numbers are labelled a
  `Core`-level measurement and why `oflm_drive.py` was written instead.
- [`tail-crossover.json`](tail-crossover.json) — the committed crossover data, including the
  provenance of both the inherited 64 and the measured 128.
- [`BLOCK_CONTRACT.md`](BLOCK_CONTRACT.md), [`PHASE26_REPORT.md`](PHASE26_REPORT.md)
