# PR4B final acceptance protocol

This is the operator-requested final acceptance campaign, fixed before its timing
samples. It does not change the optimization in `c5d20c85` or the original report.
Earlier incomplete measurements remain separate and are not pooled with this set.

## Preconditions

The validated archived context probe must create the manifest's 11 required
contexts. The immutable unchanged control (`ad867144`) and candidate (`c5d20c85`)
must initialize, map the same XRT core/coreutil/xdna libraries and pass the pending
numerical and application gates. No competing NPU client is allowed; all runs
hold the existing model lock. No automatic reset, reload, reboot, firmware change
or power-policy write is authorized. Control failure blocks the campaign.

Distinct-prompt A/B/A, fresh-reference equality, exact full-model logits/greedy
continuation, sampled recurrent/convolution/KV state, repeated requests,
prefill-to-decode and relevant boundary checks retain byte-exact requirements.
Actual candidate `oflm run` and candidate-specific `oflm serve` inference are
mandatory for MERGE CANDIDATE. Missing gates are not passes.

## Fixed timing selection

Use fresh output directories and the independently attested unchanged control and
candidate executables, identical model/kernel files, thread policy, runtime,
power policy and optional instrumentation settings. Candidate enables only the
existing split-reuse switch. The measured baseline is not a historical timing.

- 256-real-token prefill: exactly **14 measured matched pairs**, alternating
  order (seven control-first, seven candidate-first), after one excluded warmup
  pair. All 14 pairs are retained regardless of their result.
- 512 and 1024 real prompt tokens: exactly **six measured pairs each**, with
  alternating order, after one excluded warmup pair per scenario.
- Short decode: 32 real prompt tokens plus 32 generated steps, exactly **six
  measured pairs**, after one excluded warmup pair.
- Approximately-4K decode: if the genuine populated-context route is supported
  and practical, 4000 real prompt tokens plus 32 generated steps, **two measured
  pairs** after one excluded warmup pair. This is exploratory long-context
  evidence with limited statistical power, not a full-context capacity claim.

Record direct fold windows in both prefill arms, process sampled RSS/HWM, exposed
NPU memory/context indicators, per-capture runtime mappings, commands/environment,
hashes, thermal/load/power conditions and every sample. State which measurements
include profiling. Outer CLI wall remains the primary prefill metric; do not
subtract serialization or substitute inner timers. No synthetic extrapolation.

## Fixed decision rule

Primary reduction is the median of the 14 paired percentage wall reductions.
Require at least **1.00%**, at least **11/14 improving pairs**, and a two-sided
95% percentile bootstrap interval for paired-median reduction with lower endpoint
above zero (100,000 resamples, fixed seed 26026). Report the individual pairs,
median absolute difference, percentage reduction, range, MAD and interval. No
optional stopping, additional repetitions, selective omission or pooling with
the earlier two exploratory pairs is permitted.

The paired-median regression limit remains 1% for 512/1024 prefill and short
decode. Report any long-context regression with its limited sample count; an
unexplained material regression blocks promotion. The RSS limit remains 1% or
32 MiB (whichever is larger); RSS is not total NPU memory. Correctness failure
or an unresolved integration gate blocks MERGE CANDIDATE regardless of timing.

MERGE CANDIDATE requires all mandatory gates and these performance/memory checks.
KEEP DRAFT denotes an external blocker or bounded missing acceptance evidence.
NO-GO denotes a complete planned confirmation that misses the threshold or remains
statistically inconclusive, or an unacceptable regression. Do not change the
algorithm or criteria to chase a result. Defaults remain OFF; no push or merge.
