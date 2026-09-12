# `oflm bench` configs

```
oflm bench <tag> -i <config.json>
```

It is `-i` / `--prompt`, not a positional argument -- `vm_args.hpp:110` binds
that option to `parsed_args.input_file_name`, which `main.cpp:615` passes to
`run_benchmarks` as `bench_config_file`, replacing the built-in default config
wholesale. Passing the path positionally gives
`Error parsing arguments: too many positional options`. The flag is named
`prompt` because `run` and `serve` use it for one; for `bench` it is this file.

Every file here carries the **same `input_text` as the built-in default**, copied
verbatim out of `src/src/benchmarking.hpp`, so a run with any of them is
comparable to a bare `oflm bench` at the stages it shares. Only `max_length` and
`iterations` differ. The `_comment` key is ignored — the parser reads three keys
and no more.

**`iterations` has to be in the file.** The CLI's `--iterations` only reaches the
default config (`benchmarking.hpp:253`); a file-supplied config that omits the
key will not run.

## What one stage does

From `benchmarking.hpp:313-338`:

```
stages      = floor(log2(max_length / 1024)) + 1
per stage n = prefill (input_text repeated 2^n times), then generate EXACTLY 32 tokens
order       = hardest first: 32k, 16k, ... 1k
```

**`max_length` does not size the KV allocation.** `run_benchmarks` clamps the
value it passes to `load_model` -- `if (max_len < 8192) max_len = 8192;` -- so
`bench-1k.json` and `bench-2k.json` still load an **8k** context. Startup time
and resident memory are the same for every file here; only the work per stage
differs. A small config makes the benchmark shorter, not lighter.

Two consequences worth knowing before you start one:

- `max_length` picks the **largest** stage and every smaller power of two runs
  after it. One file gives a curve, not a point.
- Because the hardest stage runs **first**, a run you kill early has measured
  nothing at all.

## Why the default (32k) may not finish

Prefill costs roughly what a decode step costs *at that position*, so it is
**quadratic in the prompt length**, not linear. On a design whose decode step is
`step(n) = base + s*n`, prefilling N tokens costs about `N * (base + s*N/2)`.

That is measured, not assumed. Granite 4.2 3B on this machine, `bench-1k.json`,
one stage, 1005 prompt tokens, on two builds of the same model that differ only
in the attention kernels:

| | TTFT | prefill | decode |
|---|---:|---:|---:|
| before | 1216.24 s | 0.826 tok/s | 0.415 tok/s |
| after | **58.93 s** | **17.05 tok/s** | **13.33 tok/s** |
| | 20.6x | 20.6x | 32.1x |

The quadratic model is what connects those two rows, and it was checked twice
before the second one existed. On the `before` run, fitting the slope from the
**prefill** alone gives 2.289 ms/position, which predicts a decode rate of
0.41719 tok/s against the 0.41529 measured on that independent series -- 0.5%
apart. Carrying the same arithmetic to the second build predicted TTFT ~58 s,
prefill ~17.3 tok/s and decode ~14.3 tok/s, written down before the run;
measured 58.93, 17.05 and 13.33, i.e. within 1.6%, 1.4% and 7%.

A flat 52 ms/token prefill -- which is what the per-token cost looks like on a
short prompt -- would have put those 1005 tokens at 52 seconds rather than 1216.
That is the trap this section exists for.

Fitted slopes, and what they project for the larger files:

| | slope | step @ 1k | 1k stage | 8k stage | 32k stage |
|---|---:|---:|---:|---:|---:|
| before | 2.289 ms/pos | 2.40 s | 20 min *(measured)* | ~28 h | **~342 h** |
| after | 0.0247 ms/pos | 0.075 s | 59 s *(measured)* | ~20 min | ~4.1 h |

92x flatter. A 32k run on the first was ~1% into its first stage after three
hours, NPU at 100% throughout. It was not hung.

## Estimating it for your own model

Two decode steps at two positions give the slope, and the rest follows:

```
s          = (step(P) - step(0)) / P
prefill(N) ~= N * (step(0) + s*N/2)
stage(N)   ~= prefill(N) + 32 * step(N)
```

Rough cost of each file at the two slopes above:

| file | stages | at 2.289 ms/pos | at 0.0247 ms/pos |
|---|---|---:|---:|
| `bench-1k.json` | 1k | 20 min *(measured)* | 59 s *(measured)* |
| `bench-2k.json` | 2k, 1k | ~1.7 h | ~4 min |
| `bench-4k.json` | 4k, 2k, 1k | ~7 h | ~9 min |
| `bench-8k.json` | 8k … 1k | ~28 h | ~30 min |
| `bench-32k.json` | 32k … 1k | ~14 days | ~4.5 h |
| `bench-1k-x5.json` | 1k, five times | ~1.7 h | ~5 min |

## Reading the result

`write_bench_csv(results, tag, ".")` writes a CSV into the working directory, so
run from a directory you can name. Three series per stage — `TTFT`,
`prefill_speed`, `decoding_speed`.

**None of them is a control.** Prefill pays the same per-position cost decode
does, so anything that changes the decode curve changes all three. If you are
comparing two builds and expecting `prefill_speed` to hold still, it will not,
and that is not a sign something else moved.

## The embedding sibling: `oflm bench-embed`

```
oflm bench-embed <tag> [-i <config.json>] [--max-batch N] [--bench-iterations N]
                       [--prompt-name query|document|...]
```

Encoders get their own command rather than a flag on this one, because **not one
of TTFT, prefill or decode exists for them**: there is no first token, no
prefill/decode split, and the sequence length is fixed by the compiled design
rather than by the request. The axis that costs is the **batch**, so that is
what it sweeps -- 1, 2, 4 ... `max_batch`, the same doubling shape, hardest
stage first for the same reason.

Its config file reads four keys:

```json
{ "max_batch": 32, "iterations": 3, "task": "query", "texts": ["...", "..."] }
```

**Two deliberate differences from `oflm bench`.**

1. **The CLI flags still work when a config file is given.** `max_batch` and
   `iterations` in the file win when present, and fall back to `--max-batch` /
   `--bench-iterations` when absent. `oflm bench` ignores `--bench-iterations`
   outright once `-i` is used -- the trap documented above. Same file shape,
   better rule.
2. **There is a discarded warm-up, and it doubles as the identity gate.**
   `oflm bench` has neither.

   An earlier version of this section claimed the warm-up kept `.npue`
   **packing** out of iteration 1. That was **wrong**, and worth recording:
   `load_model()` runs before the warm-up and calls `find_container()`, which
   is what packs, so packing was already outside the timed loop. What the
   discarded call actually excludes is first-call *runtime* cost -- faulting
   in the mmapped container, the tokenizer's first use, the lanes' first
   dispatch.

   The same call also runs the identity gate: every row of the largest batch
   is compared against the same text embedded alone, and the footer prints how
   many vectors that was. The first version compared only the first row, which
   is a probe whose coverage nothing checked -- an ordering or truncation
   error in any later row would still have printed `BIT-IDENTICAL`.

`texts` is optional; omitted, a built-in corpus of 16 sentences is cycled to
fill each batch. The cycling is printed rather than implied, because a reader
who assumes 128 unique documents is reading a different experiment.

### What the columns mean

Every stage times **two paths over the same texts**: one `embed_batch()` call,
and the same texts one `embed()` call at a time -- the baseline for a caller
that sends one request per text. (`/v1/embeddings` itself batches now, and this
benchmark is what decided that it should.) `Speedup` is the second over the
first: what a
caller gains by sending one request with N inputs instead of N requests.

**That ratio is the reason the command exists.** Batching is a scheduling
choice, not an arithmetic one, so **both paths return the same vectors** and no
accuracy gate, cosine or bit-identity check can see the slow one. The only
symptom is time, and nothing measured time. The benchmark compares one vector
from each path anyway and prints `BIT-IDENTICAL` or the difference, because a
pure stopwatch would not have noticed if the fast path were wrong.

**Reading the curve: the flat stretches are the design's batch tiers.** A
request is right-sized to a tier, so several batch sizes can cost the same wall
clock and then step. You do not have to infer which -- the engine prints them
in its own banner a few lines above the table (`tiers  4, 16, 32, 128` for
bge-base), so the curve can be read against the stated values rather than
guessed at.

Be careful about how far that explanation reaches. On bge-base the flat stretch
across batch 1, 2 and 4 matches the smallest tier exactly. Beyond it the cost
is **not** a simple round-up to the next tier: batch 8 and batch 16 both fit
tier 16 and do not cost the same, and batch 64 is the best per-text point on
all six models while 128 is worse. The tier list alone does not account for
that, and this benchmark does not claim to know what does.

### Output

A table, then `bench_embed_<tag>_<YYYYMMDD>[_<cpu>].csv` in the working
directory. The prefix is `bench_embed_`, not `bench_`: benchmarking both sides
of one model on one day would otherwise have the second run silently overwrite
the first.

`Tokens/s` reads `not reported` for a backend that does not return a token
count. Empty CSV fields rather than zeros, for the same reason -- a zero reads
as a real count.

### What it refuses, and why each one is there

Every one of these used to be accepted, and every one of them produced a
plausible number rather than an error.

| you asked for | what happens |
|---|---|
| a chat tag (`llama3.2:1b`) | refused, naming the seven embedding tags -- **before** any download. It used to pull the model first and fail afterwards. |
| a shorthand tag (`bge-base`) | **accepted**, and canonicalised to `bge-base:en-v1.5`. It used to clear every check and then fail as an unknown embedding model. |
| `--max-batch 3` | refused. The sweep doubles from 1, so a limit it cannot land on would run 1 and 2 and never 3 -- a result depending on an undocumented rounding rule. |
| no model tag | refused, naming an example. The pre-existing check tests `.empty()`, and an omitted positional is `model-faker`, not empty. |
| `--prompt-name` on bge/MiniLM/gte | refused: they have no task-prompt concept and `/v1/embeddings` refuses a prompt for them. |
| no `--prompt-name` on nomic | refused: it declares prompts and the endpoint **requires** one, so timing it without one would measure a request no client can send. |
| `--prompt-name tullball` | refused, listing the valid names. |
| `-i` a file whose root is `[]`, `null` or a scalar | refused. nlohmann's `contains()` is `is_object() && ...`, so every key would have read as absent and the sweep would have run on the CLI defaults with nothing to say the file was ignored. |
| `-i` a file holding `{}` | **accepted** -- an empty object means "use the CLI values", said explicitly. |
| `--port`, `--cors`, `--socket`, `--q-len`, `--host` | refused as serve-only |

**The two guards have different reach, and it is worth knowing which.**
`--max-batch` and `--prompt-name` are checked **above** the early exits in
`parse_options`, so they are refused for every other command, `bench`
included. The serve-only five are checked **below** them, so they are refused
for `run`, `pull`, `remove`, `check` and `bench-embed` and NOT for `bench`,
`list`, `version`, `port` or `validate`. Measured, not assumed.

That is a pre-existing hole, and it is left alone because it is not the
one-line fix it looks like: `oflm port --port 8123` prints
`Server Port: 8123`, so the `port` command really does consume `--port`, and
hoisting the guard above the early exits would break it.

The task policy is decided by `openai_compat::task_policy()` -- the endpoint's
own predicate -- so a task this benchmark accepts is one a client could also
have asked for, by construction rather than by intention.

`src/src/benchmark_embed_test.cpp` holds all of the above plus the config
parsing, with no device, no weights and no network:
`ctest --test-dir src/build -R bench_embed`. It prints its own assertion count;
quoting one here was a drift source, and drifted (104 against 108) within a day.

### Not measured

No NPU-side breakdown (dispatch count, occupancy, per-bucket host time): the
engine's counters are behind a PIMPL that exists to stop an ODR/ISA hazard, and
`oflm bench` has no such breakdown either. No contention guard -- quiesce the
NPU yourself before comparing two runs, because wall clock measures how busy
the machine was as much as how good the kernels are. No energy, and no CPU arm.
