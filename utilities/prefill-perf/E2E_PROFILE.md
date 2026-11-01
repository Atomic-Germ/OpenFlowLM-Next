# End-to-end prefill profile, v1 (Phase 24)

**Scope: attribution of the CURRENT implementation. Not a published benchmark. Not an
acceptance decision. `reps = 1`, so run-to-run dispersion is NOT measured.**

| | |
|---|---|
| commit | `252a161c` (Phase 23 STOP) |
| precision mode | `paired` (the frozen production arithmetic), no role override |
| model | Qwen3.8-27B-NPU2, 64 layers (48 DeltaNet + 16 full attention) |
| kernels | `Models/qwen38-27b/production-v2/kernels-model` |
| hardware | AMD Ryzen AI 9 365 w/ Radeon 880M, NPU Strix `aie2p` 6x8 (`EVO14-AI9-STP`) |
| driver | amdxdna 2.26.0_20260817 (`2b71d02b`) |
| firmware | NPU 1.1.2.64, BIOS N.1.22GOS12 |
| XRT | 2.25.00 |
| toolchain | mlir-aie 1.4.2, Python 3.14.4, 20 host CPUs |
| power mode | **not set**: `xrt-smi configure --pmode` returns EACCES as a non-root user, so the NPU ran in whatever mode the system had. This is a known limitation below. |

Raw values: [`e2e-profile-v1.json`](e2e-profile-v1.json).

## Fixtures

| tokens | construction | sha256 (first 16) |
|---|---|---|
| 128 | prefix of the 513-token fixture | `bdd9863e5689cc41` |
| 512 | prefix of the 513-token fixture | `9d0e089f33f8f04f` |
| 1024 | the 513-token fixture repeated to length, then truncated | `b5e6b29855da8aad` |
| 2048 | the 513-token fixture repeated to length, then truncated | `5d1d92c98a071b8c` |

128 and 512 are prefixes of the same documented prompt, so for those two the only
variable is length. 1024 and 2048 are the same text repeated, because **no fixture in
the tree is longer than 513 tokens** — that is a fixture limitation, not a property of
the model, and it matters for interpreting the attention column (see below).

```bash
source utilities/env.sh
ironvenv/bin/python utilities/prefill-perf/e2e_profile.py \
  --lengths 128 512 1024 2048 --reps 1 \
  --out utilities/prefill-perf/e2e-profile-v1.json
```

## Headline

| tokens | wall s | GEMM ms | host ms | prompt tok/s | NPU dispatches | GEMM calls |
|---|---|---|---|---|---|---|
| 128 | 44.26 | 24940.9 | 5519.2 | 2.89 | 1026 | 1026 |
| 512 | 79.11 | 49830.0 | 12272.4 | 6.47 | 2050 | 2050 |
| 1024 | 149.73 | 99475.5 | 26349.3 | 6.84 | 4098 | 4098 |
| 2048 | 294.31 | 198619.2 | 58144.3 | 6.96 | 8194 | 8194 |

Prompt throughput saturates at ~7 tok/s. Wall time grows linearly: 2048 tokens cost
6.65× the 128-token run for 16× the tokens, because fixed per-request cost is amortised.

Dispatches follow `1024·blocks + 2` exactly (blocks = ⌈T/256⌉), i.e. **1024 dispatches per
256-token block**. That 1024 is `64 layers × 2 GEMMs × 8 array columns`: every array
column is dispatched separately, so a block costs 8 dispatches per projection instead of
1. It scales with work, not with overhead, but it is worth naming — at ~1000 dispatches
per block the per-dispatch cost is a constant multiplier on all GEMM work, and it is one
of the reasons the host side is measurable at all.

## Experimental role switch is NOT a public interface

`--prefill-roles` / `--prefill-role-mode` (and `OFLM_OPEN_PREFILL_ROLES` /
`OFLM_OPEN_PREFILL_ROLE_MODE`) exist only for the Phase 23 attribution and for
re-measuring closed decisions. They are:

- absent from the shipped `oflm` CLI and from `src/config/settings.json` — they are not
  settings and no application can depend on them;
- present only in the standalone `open_qwen36_cli` harness;
- defaulting to nothing, so the production arithmetic is `paired` unless a diagnostic run
  names a role explicitly.

Since Phase 23 closed as STOP, nothing in production uses them, and the default path —
no roles named — is the frozen `paired` route.

## Attribution

Absolute milliseconds, and share at 2048. Every row is a disjoint `StepTiming` field
group, so nothing is double counted; `residue` is the engine's own
`total − GEMM − host − route − lm_head`.

| category | 128 | 512 | 1024 | 2048 | share @2048 | scaling |
|---|---|---|---|---|---|---|
| projection GEMM | 24940.9 | 49830.0 | 99475.5 | 198619.2 | **69.8 %** | O(T) |
| FFN/GEMM host (tile + transpose) | 3938.0 | 7822.7 | 15913.6 | 31534.9 | **11.1 %** | O(T) |
| residue (unattributed) | 3458.1 | 7025.5 | 13938.5 | 27643.7 | **9.7 %** | O(T) |
| full attention | 136.0 | 1046.4 | 3609.0 | 13600.6 | **4.8 %** | ~O(T²) |
| DeltaNet (conv + rule) | 563.7 | 1519.1 | 3030.6 | 5944.6 | 2.1 % | O(T) |
| norm/elementwise | 585.0 | 1290.7 | 2617.0 | 4688.1 | 1.6 % | O(T) |
| host setup | 529.4 | 1252.6 | 2483.9 | 4814.9 | 1.7 % | O(T) |
| BO sync | 293.5 | 586.6 | 1165.7 | 2349.1 | 0.8 % | O(T) |
| state movement | 5.0 | 5.4 | 6.3 | 8.1 | 0.0 % | constant |
| LM head | 26.9 | 25.5 | 30.5 | 25.6 | 0.0 % | constant |
| MoE / qkv scatter | 0 | 0 | 0 | 0 | 0.0 % | not used by this dense model |

### What the numbers say

**1. The projection GEMM is 69.8 % and is pure O(T).** It is the same 2.0 TFLOP/s
measured in Phase 21/22 (198619.2 ms over 2048 tokens × 64 layers), now confirmed as the
end-to-end dominant term rather than a microbenchmark artefact. Any change to it moves
the total directly.

**2. The host half of the GEMM path is a real 11.1 %,** not noise: `gemm tile` +
`gemm tr` scale exactly with T. Together with `residue` (9.7 %) and `BO sync` (0.8 %),
**~21 % of wall is host-side work around the GEMM**, which is the same order as the GEMM
arithmetic gain any datapath change could buy.

**3. Attention is the only superlinear term.** 136 ms at 128 tokens → 13600.6 ms at
2048, i.e. 100× for 16× the tokens: `log(100)/log(16) = 1.66`, superlinear and heading
for O(T²). Its share rises 0.3 % → 4.8 %. It is NOT yet a >=20 % candidate, but it is
the only component whose share grows, and at 8K it would pass 20 % by this trend. The
caveat is in the fixture table: the 1024/2048 prompts repeat the 513-token text, so
attention sees duplicated content. Repetition does not change attention's asymptotic
cost, and the trend already holds between 128 and 512 where the prompts are genuine
prefixes; but the absolute 2048 figure should be re-measured on a distinct long prompt
before it is quoted.

**4. DeltaNet is not a problem yet, at 2.1 %.** The conv and the delta rule together cost
5944.6 ms at 2048, growing linearly. This closes a plausible-sounding hypothesis cheaply:
even a perfect DeltaNet would return 2 %.

**5. `residue` at 9.7 % is the honest gap.** It is 27643.7 ms the engine cannot place. It
scales linearly, so it is not one fixed cost. Phase 21 attributed 10.2 % the same way; two
independent runs now agree on the magnitude, which makes it a real category rather than a
timer bug — but it is still unattributed, and it is large enough that the next profile
must split it before any optimisation is credited.

## Selection rule applied

The plan's rule: a component is a candidate only at >= 20 % of wall, or with clear
superlinear growth that will dominate later.

| verdict | component | why |
|---|---|---|
| **candidate** | projection GEMM (69.8 %) | dominant, O(T) |
| **candidate** | host side around the GEMM (11.1 % + 9.7 % + 0.8 % ≈ 21 %) | over 20 % combined, O(T) |
| watch | full attention (4.8 %, ~O(T²)) | only superlinear term; not yet 20 %, but its share grows monotonically |
| no | DeltaNet (2.1 %), norm (1.6 %), setup (1.7 %), sync (0.8 %), state (0.0 %), LM head (0.0 %) | below 10 %, no trivial fix exists |

## 4096 decision gate

The rule lists five conditions for spending the extra ~2 h. Checked:

1. superlinear component needing 4096 to estimate — attention qualifies in *kind*, but it
   is 4.8 % of wall at 2048 and 4× the prompt would at most double its share to ~9 %; it
   still would not clear 20 %.
2. dominant bottleneck changing between 1024 and 2048 — no: GEMM stays ~70 %.
3. extrapolation ambiguous — no: every non-attention term is clean O(T).
4. runtime/cache threshold above 2048 — no evidence; dispatches scale exactly `8·T+2`.
5. needed to choose between two implementation directions — **no**: the next target is
   already decided by the >=20 % rule.

**Decision: skip 4096 for now.** Proceed to Phase 25/26. Attention gets a dedicated
long-prompt measurement later, on a distinct (not repeated) fixture.

## Known measurement limitations

- `reps = 1`: no dispersion. A single pass supports attribution, not performance claims.
- Power mode unset and unreadable as non-root (`xrt-smi configure --pmode` → EACCES), so
  the NPU clock state is unknown and unlabelled absolute numbers are not portable.
- 1024/2048 fixtures are repeated text (no longer prompt exists in the tree), which
  affects attention's absolute cost.
- `residue` is 9.7 % unattributed and linear; the total is trustworthy, the split inside
  it is not.
- Categories are the engine's own timers; they are disjoint by construction but their
  boundaries are the engine's, not an external profiler's.
