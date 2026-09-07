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
**quadratic in the prompt length**, not linear. On a design whose decode step
grows by `s` ms per position, prefilling N tokens costs about `N * (base + s*N/2)`.

Worked example, measured on Granite 4.2 3B on this machine:

| decode slope | step @ 32k | 32k prefill | the 32 generated tokens |
|---:|---:|---:|---:|
| 1.98 ms/position | 65 s | **~295 h** | 35 min |
| 0.023 ms/position | 0.81 s | ~3.9 h | 26 s |

A 32k run on the first of those was ~1% into its first stage after three hours,
with the NPU at 100% throughout. It was not hung; it was doing what it was told.

**Measure your model's slope before picking a file.** Two decode steps at two
positions give it: `s = (step(P) - step(0)) / P`. Then

```
prefill(N) ~= N * (step(0) + s*N/2)
stage(N)   ~= prefill(N) + 32 * step(N)
```

For the 1.98 ms/position case above that puts `bench-1k` at ~18 minutes and
`bench-8k` at about a day; for the 0.023 ms/position case, ~1 minute and
~35 minutes.

## Reading the result

`write_bench_csv(results, tag, ".")` writes a CSV into the working directory, so
run from a directory you can name. Three series per stage — `TTFT`,
`prefill_speed`, `decoding_speed`.

**None of them is a control.** Prefill pays the same per-position cost decode
does, so anything that changes the decode curve changes all three. If you are
comparing two builds and expecting `prefill_speed` to hold still, it will not,
and that is not a sign something else moved.
