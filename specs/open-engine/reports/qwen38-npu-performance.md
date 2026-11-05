# Qwen3.8-27B profiling baseline

**Status: review fixes validated on the NPU; controlled performance acceptance remains pending.**
This change adds instrumentation, without changing kernel arithmetic, model weights or
routing policy. It does not claim an inference speedup.

Machine-readable evidence: [baseline JSON](../../../utilities/prefill-block/qwen38-performance-baseline.json).
The [production correctness baseline](qwen38-27b.md) remains separate.

## Review corrections

- `layer_profile.py --route layer-major` explicitly forces `OFLM_OPEN_LAYER_MAJOR=1`
  and verifies that the profile contains `gemm-prompt`. Previously, the labelled
  layer-major capture actually executed `gemm-block`: the reported 2.3% route
  comparison is invalidated, not evidence of a scheduling improvement.
- Per-token calls are labelled `step`. `want_logits` cannot distinguish prefill
  from decode: both the final prompt token and generated-token steps need logits.
- The CLI finishes the prefill dispatch trace before starting the decode timer.
  The previous attribution contained 32 prefill and 3 decode steps; its 429.8 ms
  average and LM-head share must not be presented as decode-only measurements.
- `dispatch_ms` measures host-observed submit-to-completion intervals, including
  queue residence. Those intervals overlap under submit-ahead. They are not device
  execution time, and same-context elapsed-time differences do not establish
  kernel variability or an optimization opportunity.
- The overhead driver profiles the entire prompt plus every measured decode step,
  verifies coverage and generated-token identity, discards warm-up pairs and uses
  at least five measured pairs in alternating order. The former two-step budget
  expired during prefill, invalidating the old enabled-overhead PASS.
- The parser rejects missing/duplicate/out-of-order layers, misplaced records,
  mismatched phases and nonfinite timings. Every request must cover all declared
  layers. Multi-block token/block counts are summed; post-prefill step samples
  remain in raw records but do not contaminate block-prefill summaries.

## What the retained measurements establish

These observations predate the corrections; they are not repeated measurements
of the corrected tree. The machine was a Ryzen AI 9 365, 91 GiB RAM, XDNA2 Strix,
XRT 2.26.0, amdxdna 2.26.0_20260817, firmware 1.1.2.64, NPU performance mode and
CPU powersave governor. Load average was recorded as 0.42, available RAM 62 GiB.
Temperature and repeat-based variability were not recorded.

| Observation | Recorded value | Limitation |
|---|---:|---|
| TTFT at 256 / 512 / 1024 prompt tokens | 38.37 / 73.36 / 141.93 s | Single observation per length |
| Decode loop at 40 / 300 prompt tokens | 2.074 / 2.056 tok/s | Separate from the invalid mixed-phase trace |
| 256-token block, total | 35007.2 ms | One complete 64-layer capture |
| GEMM dispatch accounting | 25020.1 ms | Host-observed dispatch intervals |
| Host stage accounting | 6160.0 ms | Named windows, some nested |
| Activation tiling / output transpose | 3047.7 / 956.4 ms | Plausible host optimization targets |
| Not assigned to named top-level stages | approximately 3800.1 ms | Must remain visible in attribution |

The 96.5 ms outside layer walls **includes** the 27.1 ms LM head. It is not the
3.8 s unassigned to stage counters, and must not be added to the LM head again.

An earlier slower pass under high load and low free memory is retained as rejected
measurement evidence. The correlation does not prove page faults caused the slowdown.

## Profiling and reproduction

Build a fresh standalone CLI from this tree, then select its path explicitly:

```bash
source utilities/env.sh
utilities/prefill-block/build-cli.sh Models/qwen38-27b/profile-review-fix/build

ironvenv/bin/python utilities/prefill-block/layer_profile.py \
  --cli Models/qwen38-27b/profile-review-fix/build/open_qwen36_cli \
  --model oflm-data/models/Qwen3.8-27B-NPU2 \
  --route block layer-major --tokens 256 --max-tokens 16 --steps 15

ironvenv/bin/python utilities/prefill-block/profile_overhead.py \
  --cli Models/qwen38-27b/profile-review-fix/build/open_qwen36_cli \
  --model oflm-data/models/Qwen3.8-27B-NPU2 \
  --what decode --reps 5 --warmup 1
```

`OFLM_OPEN_PROFILE=<path>` writes a header and request/layer groups as JSONL.
`OFLM_OPEN_PROFILE_STEPS` caps per-token calls across the entire Core lifetime;
zero profiles block routes only. The API deliberately labels these calls `step`.
The overhead driver knows the CLI's prompt length and checks the decode suffix.

Both drivers retain full CLI logs and hash the binary, model and kernel artifacts.
Files under `Models/` and benchmark `runs/` are local evidence, not shipping assets.
Do not compare runs with different input IDs, weights, kernels or runtime options.

The optional `--baseline-cli /path/to/parent/cli` adds parent-OFF/new-OFF timing
and greedy comparisons. Without it, **disabled-path regression is NOT MEASURED**;
an enabled-overhead PASS must never be substituted for that check.

## Remaining acceptance work

- Repeated production `oflm run` TTFT measurements at 256/512/1024 tokens, with
  warm-up, at least five measured repetitions, variability and thermal conditions.
- Parent-OFF/new-OFF and new-OFF/new-ON comparisons for both prefill and decode.
- Updated full-model correctness evidence and production CLI/server regression coverage.
- Device execution versus queue/idle attribution remains NOT MEASURED without
  device timestamps or a separately validated measurement method.

Host tiling and transpose remain candidate targets for the next optimization PR,
not demonstrated speedups. Complete profiling acceptance before advancing that PR.

## Review-fix validation — 2026-11-05

The full `pytest specs/` suite passed **2460 tests, 172 skipped**, including
22 profiling regression tests and a host-compiled harness of the real C++
profiling methods. Fresh standalone and production builds passed. Standalone CTest passed 4/4;
production CTest passed 29 tests with 2 skipped. The settings scanner/generator
produced a byte-identical registry. The merged-target CI helper refuses a dirty
worktree, so that check is not claimed for these uncommitted changes.

Both routes ran 256 prompt tokens through all 64 layers and generated the same
16 token IDs. Their profiles now identify `gemm-block` and `gemm-prompt`
respectively, each followed by 15 complete step groups. This validates the
instrumentation/routing checks, not an independent numerical oracle.

The decode experiment executed one warm-up triplet followed by five measured
triplets (parent-OFF/new-OFF/new-ON), alternating order. All eight generated IDs
agreed in every run. Each ON profile has 39 requests and 2496 layer records:
32 prompt steps plus all seven timed decode steps.

| Mode | Measured ms/token | Median |
|---|---|---:|
| Parent OFF | 445, 449, 448, 451, 449 | 449 |
| New OFF | 466, 451, 448, 449, 446 | 449 |
| New ON | 463, 448, 448, 449, 450 | 449 |

**These timings are exploratory, not controlled performance-mode acceptance.**
Every CLI log reports a failed request to configure `performance`; a subsequent
`xrt-smi examine --report platform` observed `default`. NPU temperature was N/A;
host ACPI readings changed from 75°C to 67°C and are not NPU die temperatures.
The drivers now reject a failed requested power setting instead of issuing a
benchmark PASS. Use `--pmode none` only when deliberately retaining an externally
observed mode, and record that observation. Production settings are unchanged.

Local evidence: `Models/qwen38-27b/profile-review-fix/{routes,overhead}/result.json`
and `build-evidence.json`. The machine-readable baseline pins their hashes and
retains superseded measurements separately with the reason for invalidation.

A separate trace check with `--pmode none` confirmed **32 prefill steps / 2050
dispatches** and **3 decode steps / 198 dispatches** in distinct tables. This
checks phase isolation; their submit-to-completion intervals are not device
execution-time measurements. Evidence: `profile-review-fix/trace.log`.
