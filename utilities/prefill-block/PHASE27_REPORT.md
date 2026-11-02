# Phase 27 — production runtime integration

**Status: PASS — hardware closure complete.**

Every gate this phase names is green with the evidence below, run on a binary rebuilt for
this closure. Nothing here is claimed from a source reading, a syntax check, or an earlier
build.

**Frozen decisions respected:** `paired` arithmetic, GEMM geometry, physical `T = 256`,
DeltaNet arithmetic, attention arithmetic and the Q4 format are untouched. No
performance-kernel work was done. No new public setting was added.

---

## Read this first: four different token counts

The single most damaging mistake in the first version of this report was to use one number
for four different things. They are never interchangeable:

| name | what it counts | where it comes from |
|---|---|---|
| **text tokens** | what the harness types into a prompt | [`Tokenizer.count(text)`](oflm_drive.py) |
| **template tokens** | text + the chat template's fixed framing | `chat_template.jinja` + tokenizer |
| **engine tokens** | what `Engine::prefill` actually receives | the runner's `Prefill chunk ... with N tokens` |
| **physical rows** | engine tokens rounded up to 256-wide dispatches | manifest `T` |

The template's offset is **not a constant**. It was measured at **11** on one fixture family,
**8** on another, and **10** by rendering the template on the CPU — because the template's
`assistant` boundary merges with the first text token, so what it costs depends on what the
text starts with. Any harness that writes `engine = text + K` is measuring its own assumption.

Consequences, applied throughout:

- the old 27.6 matrix is relabelled a **functional route comparison**. Its fixtures were
  labelled 255/256/257 by TEXT, which reached the engine as 266/267/268 — so it never
  touched the 256-token boundary and is **not** boundary evidence;
- every length in this closure is the ENGINE count, and every run **asserts** the engine's
  own report equals the requested count rather than assuming it;
- no fixture is called a "255 boundary" unless the engine received 255. `--lengths 255`
  means 255 tokens into `Engine::prefill`.

---

## What the route actually is (a correction)

An earlier revision claimed this dense model takes the **block loop**. That was wrong, and
the hardware said so on the first instrumented run:

```
prefill-route sequential n=20 blocks=0 tail=0 why=below-crossover crossover=64
```

`crossover=64` can only be printed when `Core::layer_major_ok()` returned **true**, so the
shipped default for this model is the **layer-major** schedule — the whole prompt through
each layer once. The error came from reading the manifest's *layer type names*
(`linear_attention` / `full_attention`) instead of `gemm_block.kind`, which is `linear` /
`full`, and that is what `layer_major_ok()` tests.

Two things follow, and both are stated wherever they apply:

- the 96/128 discussion below is about the **block loop**, which this model reaches only
  with `OFLM_OPEN_LAYER_MAJOR=0`;
- every block-loop measurement in this report records that variable in its artefact, so no
  number here can be mistaken for the default route's economics.

## The central finding: block prefill was already in production

Phase 27 was scoped as "integrate the Phase 26 path into the runtime". The integration
already existed from the Qwen3.6 work: `Engine::prefill` is the production prefill for both
`run` and `serve`, the block route is on by default (`OFLM_OPEN_GEMM_BLOCK != "0"`), and the
layer bodies it calls are the ones the Phase 26 oracle measured. So the questions this phase
had to answer were different: is the route correct for *this* model, is the crossover right,
does the shipped binary work across the boundary, and does it survive a failing block.

## Correcting the record: `oflm` was never blocked

An earlier revision claimed 27.6/27.7/27.8 were BLOCKED because `oflm run` "is interactive
only". **That was wrong, and it was my error.** [`cli_wide_posix.cpp:224`](../../src/runner/cli_wide_posix.cpp:224)
returns `/bye` when stdin reaches EOF, so a pipe drives a complete session.

Two further facts cost passes and are now carried by [`oflm_drive.py`](oflm_drive.py):

- `OFLM_CONFIG_PATH` must name `model_list.json` (the file), not a directory;
- this host exports `http_proxy=http:/172.17.0.1:1080` — malformed — plus a socks5h
  `all_proxy`, and `urllib` honours both for a **loopback** URL, so a `serve` harness sat in
  its wait loop while `curl` talked to the same server happily;
- the CLI treats "more input is already waiting" as the start of a **paste** and merges the
  lines into ONE prompt, so a script written all at once turns `/clear`, the prompt and
  `/status` into a single message. Feeding is therefore gated on the CLI printing its
  `>>> ` prompt, and the reader must be byte-oriented because that prompt carries no
  newline - a line-oriented reader blocks on it forever.

## The build (item 1)

The previous revision reported `NOT RE-RUN` because this host could not build `oflm`. That
was also my error: the repository ships a supported container build, and using it is the
documented mechanism.

| field | value |
|---|---|
| git HEAD | `dcd6e7d11b79cb66f9f01b6836990e70895bec9a` + the working-tree changes below |
| build directory | `build/` (configured by the container, `cmake --preset linux-default`) |
| compiler | `/opt/xilinx/xrt/bin/c++` (XRT clang), `-O3 -DNDEBUG` |
| generator | `ninja` at `/opt/xilinx/xrt/bin/ninja` |
| binary | `build/src/oflm`, 31 880 904 bytes |
| binary sha256 | `45ed9846122748f686fb77357f4255da0a6deb5a63b8567ad4da1c3d46eedd01` |
| rebuilt for | the 96 crossover, the injected-failure arm, the tail-policy fix |

`cmake --build build --target oflm` inside `openflowlm-build:ubuntu26`. No runtime or model
semantics were changed to make the build work. The standalone harness CLI
([`build-cli.sh`](build-cli.sh)) was rebuilt from the same sources for the logits gate.

## 27.4 / closure item 5 — the crossover, re-measured properly

The published constant was **128**, from a six-point grid (1, 8, 32, 64, 128, 192, 255) that
reported "the smallest SAMPLED length at which the block wins" — a rule that returns a grid
point, not a crossing. Its own table is 64 → 1.18
and 128 → 0.72, which crosses near 81: the rule returned the first grid point past the
crossing, not the crossing. A constant chosen that way **overcharges every prompt between the
crossing and the next sampled point**, so it had to be re-measured on a fine grid.

Measured through the shipped binary, one process per side, `/clear` between prompts so each
sample is that prompt alone (the runner's own `PREFILL_TIME`, no load, no decode):

| text | engine | sequential | padded block | ratio |
|---|---|---|---|---|
| 48 | 56 | 23 968 ms | 34 466 ms | 1.438 |
| 60 | 68 | 29 230 ms | 33 467 ms | 1.145 |
| 72 | 80 | 34 158 ms | 33 444 ms | 0.979 |
| 84 | 92 | 39 153 ms | 33 132 ms | **0.846** |
| 96 | 104 | 44 344 ms | 33 284 ms | 0.751 |
| 108 | 116 | 49 615 ms | 33 482 ms | 0.675 |
| 120 | 128 | 54 642 ms | 33 306 ms | 0.610 |
| 128 | 136 | 58 018 ms | 33 514 ms | 0.578 |

- the block cost is **flat** at 33.1–34.5 s across 56–136 engine tokens: a full-width
  dispatch whatever it holds;
- sequential fits **425.5 ms/token + 155.6 ms**, i.e. 2.35 tokens/s — which matches the
  runner's own `Average prefill speed: 2.34711 tokens/s` from `/status`. Two independent
  instruments, one number;
- the crossing is therefore **78.3 engine tokens**, with the first clear win at **92**.

One methodological note, because the first attempt at this produced a wrong answer: a sweep
run under the *shipped* policy cannot measure the block below the crossover — the policy
steps those lengths, so both "sides" are the sequential path and their ratio is noise around
1.0. Counting those non-decisive points produced a spurious "crossover 104". The block side
is therefore forced with `OFLM_OPEN_GEMM_BLOCK_MIN=1`, and only points where a padded pass
really ran are decisive.

### Applied: 96, once

```cpp
const size_t block_loop_crossover = 96;   // fitted crossing ~78, first clear win 92
const size_t layer_major_crossover = 64;  // inherited from Qwen3.6-35B; a different schedule
size_t min_prompt = core_->layer_major_ok() ? layer_major_crossover : block_loop_crossover;
```

96 rather than 78: it keeps a small margin over the single unpinned pass behind each point
and rounds onto the sweep's grid. The change affects 80–127 engine tokens, which used to take
a 33.5 s padded dispatch when stepping them costs 24–55 s. `OFLM_OPEN_GEMM_BLOCK_MIN` still
overrides both. No new setting.

## A real policy bug, found by testing the boundary

The crossover was applied to the whole prompt only, so a **short tail bought a second padded
block pass**: a 257-token prompt ran one block and then padded its 1-token remainder into
another full-width dispatch. Blocks now run only while at least `min_prompt` real tokens
remain, and anything shorter is stepped:

```cpp
while (i < ids.size()) {
    if (ids.size() - i < min_prompt) break;   // a block costs its full width regardless
    ...
}
for (; i < ids.size(); ++i) core_->step(ids[i], i + 1 == ids.size());
```

## Items 2/3/4 — the route asserted at exact engine lengths

The route is read from `OFLM_OPEN_PREFILL_ROUTE`, **never inferred from elapsed time**: a
short sequential prompt and a padded block overlap in wall time, so a timing cannot tell them
apart. Every row also asserts the engine's own count equals the requested one.

`--mode block` (`OFLM_OPEN_LAYER_MAJOR=0`), graded against the shipped 96 policy:

| engine tokens | route | full blocks | tail | padded | passes | tail stepped |
|---|---|---|---|---|---|---|
| 64 | sequential | 0 | – | – | – | – |
| 95 | sequential | 0 | – | – | – | – |
| **96** | block | 0 | 96 | yes | 1 | – |
| 97 | block | 0 | 97 | yes | 1 | – |
| 127 | block | 0 | 127 | yes | 1 | – |
| 128 | block | 0 | 128 | yes | 1 | – |
| 255 | block | 0 | 255 | yes | 1 | – |
| **256** | block | **1** | 0 | – | 1 | – |
| **257** | block | **1** | 1 | no | 1 | **1** |
| 383 | block | 1 | 127 | yes | 2 | – |
| 384 | block | 1 | 128 | yes | 2 | – |
| 385 | block | 1 | 129 | yes | 2 | – |
| 511 | block | 1 | 255 | yes | 2 | – |
| **512** | block | **2** | 0 | – | 2 | – |
| **513** | block | **2** | 1 | no | 2 | **1** |

15/15 PASS, artefact `runs/route-matrix-block-20261102-041703/result.json`. The plan's worked
examples are what it says they are: 257 is **one block plus one stepped token** (its 1-token
tail is far below the crossover), and 512 is two full blocks. Note 383 and 384: with the
crossover at 96, a 127-token tail **earns** its own padded pass, so both are two passes and
the first post-block tail that would be stepped is 256 + 95 = 351.

## Item 6 — the failure handler, on hardware

The handler's claim is that a block failure must either recover invisibly or refuse. Reading
that in the source proves nothing, so `OFLM_OPEN_BLOCK_FAIL_INJECT` breaks a block on purpose
(`before`, `after`, `unrecoverable`), armed per prefill call with `OFLM_OPEN_BLOCK_FAIL_AT`.

| case | what it does | result |
|---|---|---|
| **A** | fails the FIRST block of a fresh prompt, before it writes anything | `cache reset` → sequential fallback, answer **identical** to forced sequential |
| **B** | fails the SECOND block of a follow-up prefill that started at position ≠ 0 | `pre-prompt state restored` → answer still produced |
| **C** | same as B with the rewind broken | request **refused**: `could not be restored`, `Insertion error`, **nothing generated** |

Two things this took, both worth recording:

- **`oflm run` cannot reach the snapshot branch at all.** Its REPL compares each new prompt
  against a history that ends with the previous answer, the comparison fails, and
  `AutoModel::_shared_insert` clears the context — so every prefill starts at position 0,
  where reset is free and no snapshot is taken. Case A was being measured three times. The
  snapshot branch needs `serve`, where a follow-up continues the same conversation;
- **the failure must be induced on the block loop.** The first serve attempt armed the
  injector while the layer-major schedule was in force, so the block loop never ran and
  every case reported "not injected" while the prefill succeeded.

Case C is graded on the REQUEST, not the exit status: `serve` returns an error response and
`run` prints `Insertion error` and keeps going, so a zero exit is the expected outcome of a
refused request. Accepting the exit code would grade it backwards.

Artefacts: `runs/fail-inject-20261102-044415/result.json` (A, via `run`) and
`runs/fail-inject-serve-20261102-060606/result.json` (A/B/C and isolation, via `serve`),
both PASS.

## Item 7 — request isolation, A → B → A

One resident server: request A, then a different follow-up B, then A again. `A` must come
back **identical** after B has run — that is the leak detector, and a repeated identical
request (what the old 27.8 did) is a weaker test because the prefix-skip can make it match
for an unrelated reason.

| check | result |
|---|---|
| A, then its follow-up, both answered | PASS |
| A after B equals A before B | **PASS** |
| A and B are different prompts | PASS |
| reset endpoint | **N/A** — `serve` has none; `/v1/context/reset`, `/reset`, `/v1/reset`, `/v1/cache/reset` all 404 |

The reset case is recorded as `N/A` with the reason, not as a pass and not as a failed
requirement: the API has no such endpoint, and inventing one to satisfy a checklist would be
worse than saying so. Reset isolation is covered where a reset actually exists — case A above
exercises `Core::reset()` as the recovery path.

## Item 8 — 16-token trajectory, with the numbers the plan asks for

`oflm` reports no logits, so it can only say whether two token sequences happen to match. The
plan asks for min cosine, max_abs and max_rel, which needs the logits — so this runs the
standalone harness CLI (the same `Engine::prefill`) at **full 64-layer depth**, 511 prompt
ids (two full blocks plus a tail) and 16 decode steps.

| quantity | value |
|---|---|
| greedy tokens agreeing | **16 / 16** |
| minimum cosine | **0.999999867** (repository gate 0.9999) |
| maximum `max_abs` | 9.881e-03 |
| maximum `max_rel` | 9.200e+02 |
| growth | none: step 7 (9.829e-03) vs step 15 (6.060e-03) |

**No new tolerance was introduced.** The gate is the repository's own cosine from
[`oracle.py`](oracle.py), used unchanged; `max_abs` and `max_rel` are reported for every
step, and the trend is bounded rather than compounding.

Artefact: `runs/decode16-logits-511-16-20261102-064847/result.json`.

A note on `max_rel`: it is large (up to 9.2e+02) purely because it divides by near-zero
logits — it is reported because the plan asks for it, and it is not a gate.

## Item 9 — performance sanity (not a production profile)

Four lengths through the shipped binary, block loop forced:

| text | engine | passes (expected) | wall | TTFT |
|---|---|---|---|---|
| 128 | 136 | 1 (1) | 53.73 s | 34 258 ms |
| 256 | 264 | 1 (1) | 56.40 s | 37 295 ms |
| 512 | 520 | 2 (2) | 91.23 s | 71 979 ms |
| 1024 | 1032 | 4 (4) | 161.08 s | 141 849 ms |

Both shape expectations hold: one padded dispatch costs **about the same at 128 and 256**
(1.09×, and both are one block), and cost is **linear in blocks, not in active rows**. Note
256 text tokens is 264 engine tokens — one full block plus an 8-token tail that the policy
correctly steps, which is why it is one pass and not two.

**This is a sanity check, not a performance claim. Phase 28 owns the production profile.**

## 27.11 / 27.14 — overhead

The pre-prompt snapshot is taken only on `!recover_by_reset && multi_block` — position ≠ 0
AND more than one block — so the common single-block request pays nothing, which is what
27.14 requires. Everything else in the 27.14 table stands: no per-token LM head during
prefill, no state dumps or canaries in `Engine`, `block_logits_all_` off, no reference copies.

## 27.12 — context length

Unchanged and independent of the block size: `Core::gemm_block_t()` is 256 but `cfg_.max_ctx`
comes from `MAX_L` / `--ctx-len`, and `step_gemm_block` throws only if a *block* exceeds the
compiled `T`. No rounding to 256 exists anywhere in the path. Large-context behaviour is
Phase 29.

## 27.15 — production benchmark

Two measurements, both through the shipped `oflm` at full 64-layer depth:

**Block route alone**, by engine length — `gen-lim 8`, so 52 tokens of generation are in
every wall time:

| engine tokens | blocks | wall | wall / block |
|---|---|---|---|
| 136 | 1 | 53.7 s | 53.7 s |
| 520 | 2 | 91.2 s | 45.6 s |
| 1032 | 4 | 161.1 s | 40.3 s |

Marginal cost settles at **~36–40 s per block**; the single-block row is higher because one
request absorbs the model load.

**Precision on "flat":** the block cost is flat in the number of **active rows inside a
block** — 136 and 264 engine tokens take the same full-width dispatch, 34.3 s vs 37.3 s — but
it is **linear in the number of blocks**. Both statements are needed: the first is why a tail
must be padded or avoided, the second is why long prompts cost what they cost.

`serve` corroborates the flat-in-active-rows half from the other side: 61.4 s at 255, 256 and
257 engine tokens, three requests, one resident server.

## 27.16 — default-enable decision

**Block prefill stays the default for prompts ≥ the measured crossover.** The evidence: it is
correct at full depth over a 16-token trajectory with the same tokens (item 8); it survives
the block boundary and two blocks through `run` and `serve`; it is correct under an injected
failure at every point the handler distinguishes (item 6); and it is faster at every length
above the crossing.

Below the crossover, sequential remains the route. `OFLM_OPEN_GEMM_BLOCK=0` disables the
block route entirely and `OFLM_OPEN_GEMM_BLOCK_MIN` overrides the threshold.

## 27.17 / 27.18 — no new setting, catalogue untouched

No settings-registry entry was added. The physical tile, both crossovers and the fallback
policy are internal. The two diagnostic knobs (`OFLM_OPEN_PREFILL_ROUTE`,
`OFLM_OPEN_BLOCK_FAIL_INJECT`, `OFLM_OPEN_BLOCK_FAIL_AT`) are listed in
`NON_SETTING_ENV_VARS` with reasons; nothing reads them unless a test sets them. No model was
promoted and no support status changed.

---

## Gate status

```
[x] 1  fresh production binary          HEAD dcd6e7d1 + closure changes, sha256 45ed9846...
[x] 1  provenance recorded              build/, /opt/xilinx/xrt/bin/c++, -O3 -DNDEBUG
[x] 3  crossover at exact lengths       63/64/65/95/96/127/128 -> the policy, PASS
[x] 4  255/256/257 at exact lengths     1 padded / 1 full / 1 full + 1 stepped, PASS
[x] 4  383/384/385                    two passes each: a 127-token tail clears the 96
                                       crossover and earns its own padded pass, PASS
[x] 4  511/512/513                    1+255 padded / 2 full / 2 full + 1 stepped, PASS
[x]     route asserted, not inferred    OFLM_OPEN_PREFILL_ROUTE, never a timing
[x] 5  crossover hardware-confirmed     128 was a grid artifact; crossing ~78 -> 96, rebuilt
[x] 6  failure before any block         cache reset -> sequential fallback, same answer
[x] 6  failure after one block          pre-prompt state restored -> answer produced
[x] 6  dirty-state continuation          impossible: it throws, request refused, nothing generated
[x] 7  serve A -> B -> A                A after B equals A before B; no reset endpoint (N/A)
[x] 8  16-token trajectory              16/16, min cosine 0.999999867, no growth
[x] 9  performance sanity               128/256/512/1024, passes as predicted, linear in blocks
[x]    regression suite                 full suite green
[x]    settings CI                      green
```

## What a reader should be able to re-derive

- every length in this report is an ENGINE token count, and each artefact records the
  engine's own number for the run;
- every block-loop number was produced with `OFLM_OPEN_LAYER_MAJOR=0`, because layer-major is
  this model's shipped default;
- the 96 was chosen from a fitted crossing and a measured margin, not from a grid point.

## Artefacts

- [`oflm_drive.py`](oflm_drive.py) — drives the shipped `oflm` non-interactively, builds
  prompts of an exact token count, and holds the four environment facts that cost passes:
  the config path is the `model_list.json` file, loopback HTTP must bypass the host proxy,
  the CLI merges piped lines into one paste unless fed on its `>>> ` prompt, and that prompt
  has no newline so reads must be byte-oriented.
- [`route_matrix.py`](route_matrix.py) — items 2/3/4, the route asserted at exact lengths.
  `runs/route-matrix-block-20261102-041703/result.json` PASS (15/15).
- [`crossover_local.py`](crossover_local.py) — item 5, the fine-grid crossover.
  `runs/crossover-local-20261102-034427/result.json` PASS.
- [`fail_inject.py`](fail_inject.py) — item 6 case A through `run`.
- [`fail_inject_serve.py`](fail_inject_serve.py) — items 6B/6C and 7 through `serve`.
  `runs/fail-inject-serve-20261102-060606/result.json` PASS.
- [`decode16_logits.py`](decode16_logits.py) — item 8, with cosine/`max_abs`/`max_rel`.
- [`oflm_sanity.py`](oflm_sanity.py) — item 9.
  `runs/oflm-sanity-20261102-071659/result.json` PASS.
- [`tail-crossover.json`](tail-crossover.json) — the crossover data, both the superseded v1
  grid and the closure measurement, with the reason v1 read 128.
- [`tail_crossover.py`](tail_crossover.py) — the original 27.4 harness, kept as the record of
  how the grid artifact was produced.
- [`BLOCK_CONTRACT.md`](BLOCK_CONTRACT.md), [`PHASE26_REPORT.md`](PHASE26_REPORT.md)
