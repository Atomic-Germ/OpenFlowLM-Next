# Qwen3.8 host-fold hardware profiling

## Decision: NEEDS MORE EVIDENCE

The production paired-prefill fold is directly measurable, but the split-only
candidate has only a conditional roughly 2.22% prefill-wall opportunity. There is
no measured production-candidate gain. Allocation/reuse and phase-specific CPU
versus waiting remain unresolved. This report alone does not establish acceptance
for a duplicate-split optimization. No inference arithmetic, kernel, precision, scheduling default
or oracle threshold changes are part of this work.

## Machine, identity and conditions

Measurements: 2026-11-23, EVO14-AI9-STP, AMD Ryzen AI 9 365, 20 logical CPUs,
approximately 91 GiB RAM, Strix XDNA2 6x8, Ubuntu 26.04.1,
Linux 7.0.0-44-generic, firmware 1.1.2.64, amdxdna 2.26.0_20260817.
The system utility reports XRT 2.25.00; the measured process actually maps the
installed 2.26.0 core/coreutil/xdna libraries. Their full SHA-256 values are in
each selected sample in [the numerical report](qwen38-deltanet-pipeline-performance.json).
Do not substitute the utility's version string for loaded-library provenance.

CPU governor is powersave. Captures request no power-mode change; the actual NPU
governor is not independently verified. CPU thermal-zone readings at measured-run
boundaries span 56–72 C; these are not NPU die-temperature measurements. The
machine is not isolated: boundary load readings and memory availability are
retained per run, and the observed paired spread must not be attributed entirely
to instrumentation. Initial available RAM was 67 GiB; no competing model process
was observed. The advisory NPU lock is held for each execution; it does not
exclude unrelated programs that ignore that lock.

The executable is the instrumentation-gate build, not a rebuild from the current
documentation revision:

- source revision: `f494ccdbb4040394020e2217a25ec4645b06cb02`;
- executable SHA-256: `76ecff0ffd796895dfe28e6a580de85a1877f60183cfa7c1457273bec9268c47`;
- model SHA-256: `5f5c282ae81b63df34e5f9ce42c616d73b454691c4b8d253027b2e08a1fe3d84`;
- manifest SHA-256: `744879192c49d2399b18cc6a50692ded1d6c86208387b3307ee5b172a54a3cee`.

The model is the existing Q4NX fixture, not new Q4NX model support. All 95 regular
kernel-tree files are byte-identical to the production-v2 tree. The capture tree
omits its unrelated self-referential directory symlink; the original is untouched.
The copy inventory and build attestation are retained with the local evidence.
Source/build/model/kernel inputs match before and after execution. No old capture
is assigned this executable's identity.

## Smoke gate

- A 257-real-token prefill emits two independent fold rows: real/physical counts
  256/256 and 1/256, each with 256 completed folds. Strict capture-v2 audit passes.
- A short decode smoke records 16 actual generated steps, with nonzero LM-head
  timing, after an eight-token prompt. Prompt steps are excluded from decode.
- Two resident 257-token requests reproduce the output token and emit four
  independent fold rows with the same expected counts. The CLI exercises its
  checkpoint/restore and request reset paths.
- Profiling on/off produces byte-identical complete logit dumps (993,280 bytes
  each) and 4 KiB state slices at offset zero for layers 0, 47 and 63. This is
  sampled state evidence, not full-state equality or a new independent model oracle.
- All measured executions exit successfully; logs contain no context failure,
  timeout, NOAVAIL or trajectory mismatch. System-wide driver-log auditing is
  **NOT MEASURED**; successful runs do not establish absence of every driver event.

## Matched profiling overhead

Each phase has one excluded warm-up pair and five measured pairs. Pair order
alternates; with an odd number of pairs, three start OFF and two start ON. Both
arms use the same executable, model, kernels, prompt fixture, inherited non-OFLM
environment and runtime libraries. Inherited OFLM overrides are removed. Enabled
prefill includes both layer and fold profiling; enabled decode includes layer
profiling for exactly the generated steps. Disabled arms have neither profiler.

| Phase | OFF median (range) | ON median (range) | Median paired delta | Paired delta range |
|---|---:|---:|---:|---:|
| 256-token prefill, ms | 29,683 (29,525–30,323) | 29,722 (29,507–30,720) | +39 ms, +0.131% | −379 to +984 ms |
| Generated decode, ms/token | 451 (448–456) | 450 (449–455) | 0 ms, 0.000% | −6 to +1 ms/token |

These are descriptive paired results, not proof of zero overhead or confidence
intervals. A difference of medians is not the median paired difference. Decode
summary timing is rounded to integer milliseconds; detailed step records remain
available. All 33 output IDs match within every decode pair, and the prefill
output matches within every prefill pair.

The ON timings include hook execution and serialization performed during the
request. Fold operation windows exclude dispatch callbacks and do not include
fold-row serialization. Serialization-only time and profiling-disabled regression
against a different pre-instrumentation build are **NOT MEASURED**. The comparison
measures enabled instrumentation cost, not a candidate optimization.

## Direct fold measurements

Five enabled 256-real-token captures, one physical block each, 256 completed
four-product folds per capture:

| Exclusive host operation | Median ms | Range ms |
|---|---:|---:|
| Split, finite validation and input workspace resize | 1,319.487 | 1,243.148–1,613.149 |
| Three retained output copies and associated assign/resize | 610.624 | 605.872–624.345 |
| Grouped FP32 accumulation | 637.963 | 634.271–666.755 |
| Total fold | 2,568.053 | 2,496.981–2,898.583 |

Component medians need not sum to the median total. Each raw row's total equals
its component sum within printing precision. Dispatch residence is excluded from
these windows. Allocation and reuse are included where resize/assign executes,
but separate allocation counts/time or warmed-reuse savings are **NOT MEASURED**.
No CPU fixture's zero-allocation result is promoted to a full-model measurement.

| Real tokens | Physical blocks | Repetitions | Prefill ms | Fold ms | Fold / wall |
|---|---:|---:|---:|---:|---:|
| 256 | 1 | 5 | 29,722 median | 2,568.053 median | 8.640% median of shares |
| 512 | 2 | 1 | 59,282 | 5,113.193 | 8.625% |
| 1024 | 4 | 1 | 119,890 | 10,194.602 | 8.503% |

512/1024 are single-run observations, not statistically established scaling.
Their rows contain 512/1024 completed folds respectively. The historical
2,522.5 ms capture is not used anywhere in these calculations.

## Prefill accounting and ceilings

The following table uses arithmetic means across the same five captures so the
bookkeeping sum is meaningful. It is not a claim of exclusive device occupancy.

| Recorded bucket | Mean ms | Interpretation |
|---|---:|---|
| Request wall | 29,897.776 | Profile request window |
| GEMM bucket | 24,883.401 | Existing dispatch/GEMM accounting, not device-active time |
| Host bucket | 2,170.193 | Existing host stages; includes the DeltaNet figures below |
| LM head | 25.888 | Final logits |
| Direct exclusive fold | 2,610.698 | Split + copy + accumulation |
| Remaining accounting residual | 207.597 | Unassigned; not relabeled as fold or allocation |

Within the host bucket, mean DeltaNet convolution plus recurrence is 521.707 ms
(median per-run share 1.720% of request wall). Other host stages remain in their
own recorded buckets: attention, norm, tiling/transposition, sync, tail and state.
They are not additive to their parent host total. The layer-boundary residual
field is a different quantity from the bookkeeping residual above.

Direct fold occupies 8.417–9.435% of prefill wall. Removing **all** fold cost with
no replacement would at most save that wall fraction (median 8.640%, corresponding
to approximately 1.095x in the idealized model). This is not implementable by the
split-only candidate, which retains all three output copies and grouped sums.

Measured split cost is 4.439% of wall at the median. Under the explicit assumption
that the two splits cost equally, removing only the duplicate saves at most
2.220% of wall (observed conditional range 2.076–2.626%, idealized 1.023x).
The two split halves are not separately timed; cache, allocation, finite checking,
and reuse effects can invalidate that equal-cost assumption. Actual savings can
be smaller or zero. No synthetic 21% result is transferred into this model.

## Genuine decode

Five ON captures each contain 32 generated steps after a 32-token prompt: 160
measured generated-step profiles total. Five matched OFF captures contain the
same trajectories. Loading and prompt processing are excluded from decode rate.

- ON median summary: 450 ms/token, approximately 2.22 generated steps/s.
- Median across capture means: step wall 449.220 ms, LM head 25.634 ms,
  linear-layer wall sum 333.566 ms and full-attention-layer wall sum 90.209 ms.
- Median recorded dispatch-residence sum: 675.060 ms per generated step. It
  exceeds step wall because commands overlap. It is **not** exclusive NPU time.
- The existing host bucket is 421.827 ms at the median of capture means. It
  includes dispatch/waiting work and must not be called CPU compute time.
- Block fold does not execute on this sequential NPU decode route. Zero host
  DeltaNet substage fields mean those host hooks do not time the fused NPU
  internals; they do not mean DeltaNet costs nothing. Per-operation device
  DeltaNet time, CPU/wait fractions and NPU utilization are **NOT MEASURED**.

Approximately-4K decode is **NOT MEASURED**. These short-context observations do
not establish long-context behavior, utilization or a wait-bound bottleneck.

## Evidence and reproduction

Raw local evidence is retained beneath the hardware-campaign directory in Models.
The numerical report lists every selected capture path, exact argument vector and
environment, capture-metadata hash, raw hashes, runtime hashes and conditions.
One shared attestation and full artifact snapshot avoid duplicating hundreds of
file entries per sample. Original commands are retained in each invocation;
missing raw phase labels are never supplied after capture.

Enabled samples use the strict version-2 capture contract. OFF samples use a
separate disabled schema and retain their own raw logs, live runtime observations,
before/after snapshots and invocation hashes; they never fabricate layer rows.
The summary revalidates the raw OFF phase ordering, geometry, generated positions,
LM-head and output labels as well as hashes. Supplemental condition sidecars live
outside sealed enabled-capture directories, with their hashes in the report.

Run from the repository root. Set CONTROL_CLI, BUILD_IDENTITY, KERNELS and
CAPTURE_ROOT from the selected capture metadata, and choose a fresh output
directory for a new campaign:

```sh
ironvenv/bin/python utilities/prefill-block/pipeline_campaign.py \
  --cli "$CONTROL_CLI" \
  --build-identity "$BUILD_IDENTITY" \
  --kernels "$KERNELS" \
  --out "$CAPTURE_ROOT/campaign-new"
ironvenv/bin/python utilities/prefill-block/pipeline_summary.py \
  --root "$CAPTURE_ROOT" \
  --out specs/open-engine/reports/qwen38-deltanet-pipeline-performance.json
```

The summary selects the retained campaign and the two explicit single-run scaling
captures, not directory-name guesses or historical results. The campaign driver
does not rerun smoke or create the scaling captures; their exact invocations are
in the retained capture metadata. For a new campaign, complete the same smoke
gate before timing and use the standard capture tool for the scaling samples.

[Historical JSON](qwen38-deltanet-pipeline-historical.json) is a byte-preserved
archive, not accepted input. The [instrumentation gate](qwen38-pipeline-instrumentation.md)
documents CPU-only lifecycle and numerical checks separately.

## Verification and remaining evidence

| Check on this Linux host | Result |
|---|---|
| Focused fold, capture, layer-profile and old/new report suite | 339 passed |
| Complete portable specification suite | 2,903 passed, 172 skipped |
| Final selected enabled-capture audits and disabled raw-log/snapshot checks | Pass |
| Settings registry, clean committed-source scratch merge against fetched main | Pass |
| Historical JSON SHA-256 versus the previously committed file | Identical |
| Whitespace/diff check | Pass |
| Production source, kernel and settings changes | None |

Logs are recorded with the local campaign. The settings check uses the clean
committed source gate because this profiling work changes no production source,
settings schema or registry; it is not a claim that an uncommitted working tree
was merged. Skipped capabilities are not acceptance passes. No kernel build,
new model support claim, speculative implementation, push or merge is part of
this work.

The profiling prerequisites for duplicate-split acceptance are: resolve split-half
versus allocation/reuse attribution, quantify serialization separately if a sub-percent effect matters,
and tighten environmental/power controls enough to discriminate the roughly 2%
conditional opportunity. A CPU-versus-wait claim additionally requires directly
verified phase accounting. No optimization is implemented by this report.
