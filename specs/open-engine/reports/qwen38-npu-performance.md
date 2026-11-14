# Qwen3.8-27B layer profiler: columns, drivers and how to run them

`OFLM_OPEN_PROFILE` writes one JSONL row per layer, so a prefill or decode run can be
attributed stage by stage instead of only as a total. It changes no arithmetic, weights or
routing policy, and it is silent when unset.

Machine-readable evidence: [baseline JSON](../../../utilities/prefill-block/qwen38-performance-baseline.json).
The [production correctness baseline](qwen38-27b.md) remains separate.

## How the profiler works

| variable | effect |
|---|---|
| `OFLM_OPEN_PROFILE=<path>` | write a JSONL profile to `<path>` |
| `OFLM_OPEN_PROFILE_STEPS=<n>` | how many per-token `step()` calls to profile (default 8; `0` = block routes only) |

Rows are differences of the engine's existing `StepTiming` sampled at the layer
boundaries, plus one steady-clock read per boundary (`wall_ms` — `StepTiming::total_ms` is
written once per block, so a per-layer duration is the only column it cannot provide).
Hooks sit at the top of each schedule's layer loop, so the layer calls keep their source
shape.

Records: `header` (version, family, spec_hash, layers), `request` (what, tokens, blocks and
the totals), `layer` (what, kind, layer, `wall_ms` and every `StepTiming` column that layer
moved). `what` is `gemm-block`, `gemm-prompt` or the phase-neutral `step`.

## What each column means

| column | measured window |
|---|---|
| `wall_ms` | steady clock across the layer's own call — the column that ranks layers |
| `gemm_ms` | the layer's GEMM dispatches (`part0`) |
| `host_ms` | the layer's host stages (`part1`) |
| `per_token_ms` | per-token dispatches on the block routes |
| `lmhead_ms` | the tail (final norm + lm_head) |
| `prenorm_ms` | input RMSNorm |
| `gemm_tile_ms` | `tile_x` into the GEMM's tiled bf16 layout |
| `gemm_tr_ms` | `[N, T]` → `[T, N]` transpose of the GEMM output |
| `mid_ms` | the host half of attention or DeltaNet |
| `dn_conv_ms`, `dn_rule_ms` | inside `mid`: the DeltaNet conv/norm and the delta rule on S |
| `attn_ms` | inside `mid`: the attention host half |
| `tail_ms` | residual, post-attention norm, router |
| `shared_ms` | the shared expert (silu·up, gate, add) |
| `state_ms` | the state BO syncs |
| `sync_ms` | the GEMM globals' host↔device syncs |
| `dispatch_ms` | host-observed submit-to-completion intervals, **including queue residence** — these overlap under submit-ahead and are not device execution time |

`mid_ms` is inside `host_ms`; `dn_conv_ms`, `dn_rule_ms` and `attn_ms` are inside
`mid_ms`. Do not add a nested column to its parent.

## Drivers

```bash
source utilities/env.sh
utilities/prefill-block/build-cli.sh <outdir>

ironvenv/bin/python utilities/prefill-block/layer_profile.py \
  --cli <outdir>/open_qwen36_cli --route block layer-major --tokens 256

ironvenv/bin/python utilities/prefill-block/profile_overhead.py \
  --cli <outdir>/open_qwen36_cli --what decode --reps 5 --warmup 1
```

`layer_profile.py` validates the capture (missing/duplicate/out-of-order layers, misplaced
records, mismatched phases, nonfinite timings; every request must cover all declared
layers), forces and verifies the requested route, writes `result.json` + `layers.csv`, and
asserts three accounting consistencies. `profile_overhead.py` compares ON against OFF of the
same binary, alternating, excluding warm-up pairs, and requires every measured decode step
to have a complete 64-layer profile with identical generated tokens.

Both drivers hash the binary, model and kernel artefacts into the result. Files under
`Models/` and benchmark `runs/` are local evidence, not shipping assets.

## Measurements on one machine

The numbers below come from one box and describe that box. Any comparison is only valid
against another run of the same inputs, weights, kernels and runtime options.

Host: Ryzen AI 9 365, 91 GiB RAM, XDNA2 Strix. Runtime: XRT 2.26.0, amdxdna
2.26.0_20260817, firmware 1.1.2.64. NPU performance mode, CPU powersave governor.
Load average 0.42, 62 GiB available. Temperature and repeat-based variability not recorded.

| Observation | Recorded value | Limitation |
|---|---:|---|
| TTFT at 256 / 512 / 1024 prompt tokens | 38.37 / 73.36 / 141.93 s | Single observation per length |
| Decode loop at 40 / 300 prompt tokens | 2.074 / 2.056 tok/s | Per-token steps, prefill excluded |
| 256-token block, total | 35007.2 ms | One complete 64-layer capture |
| GEMM dispatch accounting | 25020.1 ms | Host-observed dispatch intervals |
| Host stage accounting | 6160.0 ms | Named windows, some nested |
| Activation tiling / output transpose | 3047.7 / 956.4 ms | Plausible host optimization targets |
| Not assigned to named top-level stages | approximately 3800.1 ms | Must remain visible in attribution |

The 96.5 ms outside layer walls **includes** the 27.1 ms LM head. It is not the
3.8 s unassigned to stage counters, and must not be added to the LM head again.

## Limits of these numbers

Recorded from one machine in one session: temperature and repeat-based variability were not
recorded, so an absolute figure here is evidence for this box, not a specification. Where a
comparison is quoted it is between two runs of the same binary with the same inputs.

Host tiling and output transpose are candidate targets for the next optimization PR, not
demonstrated speedups.
