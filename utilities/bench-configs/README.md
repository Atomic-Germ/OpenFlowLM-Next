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
