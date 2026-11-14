# End-to-end prefill profile, v2 (Phase 24, regrouping; corrected by the Phase 25 pass)

**Scope: attribution of the CURRENT implementation. Not a published benchmark. Not an
acceptance decision. `reps = 1`, so run-to-run dispersion is NOT measured.**

v1 of this document divided wall time by categories that **overlapped**, so they summed
to 4796.0 ms more than a whole 2048-token run. The measurement was fine; the grouping was
not. v2 keeps every raw number from the same run and recomputes the allocation from the
committed JSON — **no hardware was re-run** — and now proves it adds up.

| | |
|---|---|
| commit | `252a161c` for the v1 measurement; the v2 regrouping is a pure re-derivation |
| precision mode | `paired` (the frozen production arithmetic), no role override |
| model | Qwen3.8-27B-NPU2, 64 layers (48 DeltaNet + 16 full attention) |
| kernels | `Models/qwen38-27b/production-v2/kernels-model` (manifest v4, 11 hardware contexts) |
| hardware | AMD Ryzen AI 9 365 w/ Radeon 880M, NPU Strix `aie2p` 6x8 (`EVO14-AI9-STP`) |
| driver | amdxdna 2.26.0_20260817 (`2b71d02b`) |
| firmware | NPU 1.1.2.64, BIOS N.1.22GOS12 |
| XRT | 2.25.00 |
| toolchain | mlir-aie 1.4.2, Python 3.14.4, 20 host CPUs |
| power mode | **not set**: `xrt-smi configure --pmode` returns EACCES as a non-root user, so the NPU ran in whatever mode the system had. This is a known limitation below. |

Raw values: [`e2e-profile-v1.json`](e2e-profile-v1.json) (the original measurement) and
[`e2e-profile-v2.json`](e2e-profile-v2.json) (the corrected allocation of the same run).

## Two denominators, never interchanged

Every percentage below names its denominator, because the two are **seven points apart**
on the dominant term and reading one as the other is a silent error:

- **wall share** = category / process wall (the whole `open_qwen36_cli` process: model
  load, xclib creation, the request, teardown). This is what a user experiences.
- **engine share** = category / `StepTiming.total_ms` (only the request the engine timed).

The difference between them is `outside_engine_ms` — the process gap — reported as its
own row and never folded into a category.

## The accounting model

Taken from [`core.cpp`](../../src/open_qwen36/core.cpp) rather than assumed:

```
total   = part0 + part1 + route + lm_head + residue
part1   = prenorm + gemm_tile + gemm_tr + tail + setup + sync + qkv_scatter + mid
mid     = dn_conv + dn_rule + attn
residue >= shared + state
```

That last line is the trap v1 fell into. `ffn_block()` adds `shared_ms` to its own field
and **not** to `part1`, so `shared` lives *inside* `residue`. v1 put `shared` in
`host_setup` and then also counted `residue` whole, counting the FFN activation twice.
The overshoot at each length (531.5 / 1251.1 / 2476.8 / 4796.0 ms) matches `shared`
(526.4 / 1246.1 / 2470.7 / 4788.6 ms) to within the 0.1 ms resolution of the printed
stage lines, which is what identified it.

`allocate()` now checks, on every run and every length:

| check | 2048 | meaning |
|---|---|---|
| `categories_sum_ms` | 284432.1 | sum of the exclusive categories |
| `categories_minus_engine_ms` | −0.7 | must be ~0: the categories *are* the engine total |
| `categories_minus_wall_ms` | −9877.9 | **must be ≤ 0**: no category set may exceed wall |
| `within_wall` | `True` | the explicitly requested sanity check |
| `part1_identity_delta_ms` | 0.1 | `part1` really is the sum beneath it on this route |
| `total_identity_delta_ms` | 0.0 | `total` really is the printed columns’ sum |

Nested engine fields are reported as **diagnostics only** and are never added in:
`attention_in_mid` (attn sits inside mid inside part1) and `moe_in_route` (the MoE
dispatches sit inside route on a MoE family). This model emits no MoE, so
`moe_in_route = 0`.

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
the model, and it matters for the attention column (see below).

The tool refuses to run a fixture whose token count differs from the length the row
claims: the first version took a prefix of the 513-token file for *every* length, so two
rows measured the same prompt and the table would have read "cost stops growing".

## Headline

| tokens | wall s | engine total ms | GEMM ms | prompt tok/s | GEMM dispatches |
|---|---|---|---|---|---|
| 128 | 44.26 | 33945.0 | 24940.9 | 2.89 | 1024 |
| 512 | 79.11 | 69153.4 | 49830.0 | 6.47 | 2048 |
| 1024 | 149.73 | 139793.8 | 99475.5 | 6.84 | 4096 |
| 2048 | 294.31 | 284432.8 | 198619.2 | 6.96 | 8192 |

Prompt throughput saturates at ~7 tok/s. Wall time is linear above the fixed cost:
2048 tokens cost 6.65× the 128-token run for 16× the tokens.

## Attribution — exclusive categories, both denominators

`wall%` = / process wall, `engine%` = / engine total. `unattributed_engine` is the
engine's own remainder with `shared` and `state` taken back out; `outside_engine` is the
process gap.

| category | 128 | 512 | 1024 | 2048 | wall% | engine% | scaling |
|---|---|---|---|---|---|---|---|
| projection GEMM (device) | 24940.9 | 49830.0 | 99475.5 | 198619.2 | **67.5 %** | **69.8 %** | O(T) |
| host GEMM tiling | 3041.5 | 5940.2 | 12166.3 | 24124.2 | **8.2 %** | 8.5 % | O(T) |
| host GEMM transpose | 896.5 | 1882.5 | 3747.3 | 7410.7 | **2.5 %** | 2.6 % | O(T) |
| DeltaNet (conv + rule) | 563.7 | 1519.1 | 3030.6 | 5944.6 | 2.0 % | 2.1 % | O(T) |
| full attention | 136.0 | 1046.4 | 3609.0 | 13600.6 | **4.6 %** | 4.8 % | ~O(T²) |
| residual/norm/router (host) | 431.1 | 935.4 | 1916.3 | 3423.2 | 1.2 % | 1.2 % | O(T) |
| FFN activation (host) | 526.4 | 1246.1 | 2470.7 | 4788.6 | 1.6 % | 1.7 % | O(T) |
| dispatch BO sync | 293.5 | 586.6 | 1165.7 | 2349.1 | 0.8 % | 0.8 % | O(T) |
| norm entry | 153.9 | 355.3 | 700.7 | 1264.9 | 0.4 % | 0.4 % | O(T) |
| request setup | 3.0 | 6.5 | 13.2 | 26.3 | 0.0 % | 0.0 % | O(T) |
| BO state (KV) | 5.0 | 5.4 | 6.3 | 8.1 | 0.0 % | 0.0 % | constant |
| LM head | 26.9 | 25.5 | 30.5 | 25.6 | 0.0 % | 0.0 % | constant |
| qkv scatter | 0 | 0 | 0 | 0 | 0 | 0 | not used on this route |
| **unattributed (engine)** | 2926.7 | 5774.0 | 11461.5 | **22847.0** | **7.8 %** | 8.0 % | O(T) |
| **outside engine (process)** | 10315.0 | 9956.6 | 9936.2 | **9877.2** | **3.4 %** | — | constant |

The sum of the exclusive categories **is** the engine total (−0.7 ms at 2048), and it is
**less** than wall time by the process gap (9877.9 ms). Nothing is counted twice, and
every microsecond of wall time is in exactly one place — a category, the engine
remainder, or the process gap.

### What the numbers say

**1. The projection GEMM is 67.5 % of wall / 69.8 % of engine and is pure O(T).** It is
the same 2.0 TFLOP/s measured in Phase 21/22 (198619.2 ms over 2048 tokens × 64 layers),
now confirmed end-to-end rather than as a microbenchmark artefact.

**2. The host half of the GEMM path is 10.7 % of wall** (tiling 8.2 % + transpose 2.5 %),
not noise: both scale exactly with T. This is a *different number* from v1's 11.1 %,
because v1's 11.1 % was "tile + transpose" measured against the engine total while
`shared` was wrongly folded in beside it.

**3. Attention is the only superlinear term.** 136 ms at 128 tokens → 13600.6 ms at 2048:
100× for 16× the tokens, `log(100)/log(16) = 1.66`. Its wall share rises 0.3 % → 4.6 %.
It is NOT yet a ≥ 20 % candidate, but it is the only component whose share grows, and at
8K it would pass 20 % by this trend. Caveat: the 1024/2048 prompts repeat the 513-token
text, so attention sees duplicated content. Repetition does not change attention's
asymptotic cost and the trend holds between 128 and 512 where the prompts are genuine
prefixes; but the absolute 2048 figure should be re-measured on a distinct long prompt
before it is quoted.

**4. DeltaNet is not a problem, at 2.0 %.** Conv and delta rule together cost 5944.6 ms
at 2048, growing linearly. Even a perfect DeltaNet returns 2 %.

**5. `unattributed` is 7.8 % of wall and does not fit "fixed cost".** It grows linearly
at ~2856 ms **per 256-token block** (2926.7 → 2887.0 → 2865.4 → 2855.9 ms/block), so it
is per-block work, not one-off setup. Phase 21 attributed 10.2 % the same way; two
independent runs agree on the magnitude. It is the one honest gap left, and Phase 25
bounded it (see below) rather than closing it — see
[`PHASE25_UPSTREAM.md`](PHASE25_UPSTREAM.md).

**6. `outside_engine` is 3.4 % of wall and genuinely constant** (10315.0 / 9956.6 /
9936.2 / 9877.2 ms — it does not grow with the prompt). That is model load, xclbin
registration and context creation, and teardown: a ~9.9 s fixed cost per process,
amortised. It cannot be reduced by changing prefill, and it is the reason the 128-token
run has a poor tok/s.

## The 1024 dispatches per block — corrected

Phase 24 claimed this was `64 layers × 2 GEMMs × 8 array columns`, i.e. "every array
column is dispatched separately". **That was wrong**, and Phase 25 refuted it from the
dataflow (see [`PHASE25_UPSTREAM.md`](PHASE25_UPSTREAM.md)). The correct decomposition,
from the measured dispatch table, is:

```
1024 dispatches per 256-token block
  = 64 layers × 2 projections per layer × 8
  where 8 = 2 weight components (hi, low) × 2 input components (x_hi, x_lo)
```

i.e. the paired four-product fold: `(W_hi X_hi + W_hi X_lo) + (W_lo X_hi + W_lo X_lo)`
issues **four** whole-array launches per projection, and each launch covers all eight
array columns at once. The tail of a request adds exactly 2 dispatches (`ln` + `lm`),
which is why the count is `1024·blocks + 2`, and why the non-GEMM share of the table is 2
at every length.

The eight columns are inside a single launch: `gemm_q4_prefill.py` builds
`Worker.grid(N_AIE_ROWS=4, N_AIE_COLS=8, ...)` over **one** `Program(dev, rt, workers=...)`
whose `Runtime` has **one** `sequence`, and the host issues **one** `xrt::run` per
`Step` in `gemm_dispatch`. **The host does not submit eight runs.**

## Selection rule applied

The plan's rule: a component is a candidate only at ≥ 20 % of wall, or with clear
superlinear growth that will dominate later.

| verdict | component | why |
|---|---|---|
| **candidate** | projection GEMM (67.5 % wall) | dominant, O(T) |
| watch | host GEMM tiling + transpose (10.7 % wall) | single-digit on its own, but it is the only large *host* term and it is addressable |
| watch | full attention (4.6 %, ~O(T²)) | only superlinear term; not yet 20 %, share grows monotonically |
| no | DeltaNet (2.0 %), FFN activation (1.6 %), residual/norm (1.2 %), sync (0.8 %), entry norm (0.4 %), setup (0.0 %), state (0.0 %), LM head (0.0 %) | below 10 %, no trivial fix exists |
| not yet | `unattributed` (7.8 %) | linear per block; Phase 25 bounded what it can and cannot be |

## 4096 decision gate

The rule lists five conditions for spending the extra ~2 h. Checked:

1. superlinear component needing 4096 to estimate — attention qualifies in *kind*, but it
   is 4.6 % of wall at 2048 and 4× the prompt would at most double its share to ~9 %; it
   still would not clear 20 %.
2. dominant bottleneck changing between 1024 and 2048 — no: GEMM stays ~68–70 %.
3. extrapolation ambiguous — no: every non-attention term is clean O(T).
4. runtime/cache threshold above 2048 — no evidence; dispatches scale exactly `1024·blocks+2`.
5. needed to choose between two implementation directions — **no**: the next target is
   already decided by the ≥ 20 % rule.

**Decision: skip 4096 for now.** Attention gets a dedicated long-prompt measurement
later, on a distinct (not repeated) fixture.

## Known measurement limitations

- `reps = 1`: no dispersion. A single pass supports attribution, not performance claims.
- Power mode unset and unreadable as non-root (`xrt-smi configure --pmode` → EACCES), so
  the NPU clock state is unknown and unlabelled absolute numbers are not portable.
- 1024/2048 fixtures are repeated text (no longer prompt exists in the tree), which
  affects attention's absolute cost.
- `unattributed` is 7.8 % of wall and linear per block; the total is trustworthy, the
  split inside it is not.
- Categories are the engine's own timers; they are exclusive and sum to the engine total,
  but their boundaries are the engine's, not an external profiler's.

## Reproducing without hardware

The correction of a grouping must never cost another multi-minute sweep, so the tool
re-derives everything from the committed JSON:

```bash
python utilities/prefill-perf/e2e_profile.py \
  --from-json utilities/prefill-perf/e2e-profile-v1.json \
  --out utilities/prefill-perf/e2e-profile-v2.json
```

It exits non-zero if any accounting check fails, so a future grouping error is a test
failure rather than a plausible-looking table.
