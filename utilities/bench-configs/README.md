# `flm bench` configs

```
flm bench <tag> -i <config.json>
```

It is `-i` / `--prompt`, not a positional argument -- `vm_args.hpp:110` binds
that option to `parsed_args.input_file_name`, which `main.cpp:615` passes to
`run_benchmarks` as `bench_config_file`, replacing the built-in default config
wholesale. Passing the path positionally gives
`Error parsing arguments: too many positional options`. The flag is named
`prompt` because `run` and `serve` use it for one; for `bench` it is this file.

Every file here carries the **same `input_text` as the built-in default**, copied
verbatim out of `src/src/benchmarking.hpp`, so a run with any of them is
comparable to a bare `flm bench` at the stages it shares. Only `max_length` and
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
one stage, 1005 prompt tokens:

```
TTFT: 1216.24s, Prefill Speed: 0.82632 tokens/s, Decoding Speed: 0.41529 tokens/s
```

Fitting `s` from the **prefill** alone gives 2.289 ms/position, which predicts a
decode rate of **0.41719 tok/s** against the **0.41529** measured in the same
run -- a 0.5% error on an independent series. A flat 52 ms/token prefill, which
is what the per-token cost looks like on a short prompt, would have put those
1005 tokens at 52 seconds rather than 1216.

Projected from that fit, and from the same fit taken on a build whose attention
work is done (0.023 ms/position):

| decode slope | step @ 1k | step @ 32k | 1k stage | 32k stage |
|---:|---:|---:|---:|---:|
| 2.289 ms/position | 2.40 s | 75 s | 20 min *(measured)* | **~342 h** |
| 0.023 ms/position | 0.070 s | 0.81 s | ~1 min | ~3.9 h |

A 32k run on the first was ~1% into its first stage after three hours, NPU at
100% throughout. It was not hung.

## Estimating it for your own model

Two decode steps at two positions give the slope, and the rest follows:

```
s          = (step(P) - step(0)) / P
prefill(N) ~= N * (step(0) + s*N/2)
stage(N)   ~= prefill(N) + 32 * step(N)
```

Rough cost of each file at the two slopes above:

| file | stages | at 2.289 ms/pos | at 0.023 ms/pos |
|---|---|---:|---:|
| `bench-1k.json` | 1k | 20 min | ~1 min |
| `bench-2k.json` | 2k, 1k | ~1.7 h | ~4 min |
| `bench-4k.json` | 4k, 2k, 1k | ~7 h | ~11 min |
| `bench-8k.json` | 8k … 1k | ~28 h | ~40 min |
| `bench-32k.json` | 32k … 1k | ~14 days | ~5 h |
| `bench-1k-x5.json` | 1k, five times | ~1.7 h | ~5 min |

## Reading the result

`write_bench_csv(results, tag, ".")` writes a CSV into the working directory, so
run from a directory you can name. Three series per stage — `TTFT`,
`prefill_speed`, `decoding_speed`.

**None of them is a control.** Prefill pays the same per-position cost decode
does, so anything that changes the decode curve changes all three. If you are
comparing two builds and expecting `prefill_speed` to hold still, it will not,
and that is not a sign something else moved.
