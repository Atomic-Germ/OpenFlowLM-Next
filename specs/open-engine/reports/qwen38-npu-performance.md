# Qwen3.8-27B on open XDNA2 — performance baseline and profiling (PR #1)

**Verdict: instrumentation PASS; no speedup claimed.** This is performance-plan PR #1: it
establishes a reproducible baseline, adds an opt-in per-layer profiler, and ranks the
bottlenecks. It changes no arithmetic and no schedule.

Machine-readable companion: [`qwen38-performance-baseline.json`](../../../utilities/prefill-block/qwen38-performance-baseline.json).
Correctness baseline (unchanged): [`qwen38-27b.md`](qwen38-27b.md) and
[`qwen38-production-baseline.json`](../../../utilities/prefill-block/qwen38-production-baseline.json).

---

## 1. Conditions

| | |
|---|---|
| host | Ryzen AI 9 365 / Radeon 880M (EVO14-AI9-STP), 20 CPUs, 91 GiB |
| NPU | Strix `aie2p` 6x8 |
| XRT / driver / firmware | 2.26.0 / `amdxdna` 2.26.0_20260817 / 1.1.2.64 |
| NPU power mode | `performance` (set by the CLI's `--pmode`) |
| CPU governor | `powersave` — unchanged for these runs |
| **load average** | **0.42** at the start (1.4 over the preceding 5 min) |
| **available memory** | **62 GiB** |
| kernel set | `Models/qwen38-27b/production-v2/kernels-model`, manifest v4, physical block 256 |

### A first pass was rejected for a measurement reason

The first measurement pass ran while the box carried unrelated work (load average 6.4–10.8)
**and** was short of memory (17 GiB free, 12 GiB swapped out). Prefill TTFT came out at
62.2 / 118.9 / 234.4 s, 1.58–1.64× the Phase 31 baseline, while decode reproduced exactly at
2.076 tok/s.

The split is the diagnosis: prefill's host stages touch hundreds of megabytes of scratch per
block and fault under memory pressure, while decode is device-bound and does not. So the
inflation was an environment artefact, not a property of the code. The pass was discarded and
every number below is a re-measurement on the idle box.

**What survived the change of conditions** — and is therefore the part worth acting on — is
the ratio structure: GEMM 72 % / host 17 % before, 71.5 % / 17.6 % after; activation tiling
50.5 % of host before, 49.5 % after.

---

## 2. Baseline reproduced

| metric | Phase 31 | idle box | Δ |
|---|---|---|---|
| TTFT, 256 prompt | 39.49 s | **38.37 s** | −2.8 % |
| TTFT, 512 prompt | 73.21 s | **73.36 s** | +0.2 % |
| TTFT, 1024 prompt | 143.02 s | **141.93 s** | −0.8 % |
| prompt tok/s, 256 / 512 / 1024 | 6.68 / 7.10 / 7.22 | **6.88 / 7.09 / 7.27** | — |
| decode tok/s (loop rate) | 2.087 | **2.074** (2.056 at a 300-token prompt) | −0.6 % |

The schedule's own shape checks pass: cost is linear in blocks, flat in the active rows inside
a block, and the pass counts match the policy's prediction at all three lengths.

---

## 3. What was added

One opt-in instrument, in `src/open_qwen36/core.cpp` / `core.hpp`:

| variable | meaning |
|---|---|
| `OFLM_OPEN_PROFILE=<path>` | write a JSONL profile to `<path>`: one header, one request record, one record per layer |
| `OFLM_OPEN_PROFILE_STEPS=<n>` | how many sequential `step()` calls to profile (default 8; `0` = block routes only) |

The rows are **differences of the accounting the engine already keeps** (`StepTiming`), taken
at the layer boundaries — the engine maintains every column already, but only as a running
sum over a whole prompt or block, so "which layer costs what" could not be read off it. The
one column `StepTiming` cannot provide is a per-layer duration (its `total_ms` is written once
per block), so each row also carries `wall_ms`, one steady-clock read per layer boundary.

Hooks sit at the **top** of each schedule's layer loop ([`profile_switch()`](../../../src/open_qwen36/core.cpp:664)),
so the layer calls themselves keep the exact source shape they had — the private-patch tools
under `utilities/prefill-*/build.py` anchor on that text, and moving a hook between a call and
its loop's closing brace breaks them. `profile_close()` flushes the last row.

Schedules covered: `step_block_moe` (the Qwen3.8 path), `step_gemm_block` (the dense block
loop), `step_gemm_prompt` (layer-major) and the per-token `step()`.

```text
{"record":"header",  "version":1,"family":"qwen35","spec_hash":"...","layers":64}
{"record":"request", "what":"gemm-block","tokens":256,"blocks":1,"total_ms":35007.2,
                     "gemm_ms":25020.0,"host_ms":6160.0,"per_token_ms":0,
                     "lmhead_ms":27.0,"setup_ms":...,"embed_ms":...,"patch_ms":...,
                     "dispatch_ms":...,"layer_wall_ms":34911.0,"residue_ms":96.0,"layers":64}
{"record":"layer",   "what":"gemm-block","kind":"linear_attention","layer":1,"wall_ms":...,
                     "gemm_ms":...,"host_ms":...,"mid_ms":...,"dn_conv_ms":...,"dn_rule_ms":...,
                     "attn_ms":...,"gemm_tile_ms":...,"gemm_tr_ms":...,"tail_ms":...,
                     "shared_ms":...,"state_ms":...,"sync_ms":...,"dispatch_ms":..., ...}
```

Two drivers, in `utilities/prefill-block/`:

```bash
python utilities/prefill-block/layer_profile.py --route block layer-major --tokens 256
python utilities/prefill-block/profile_overhead.py --what decode --reps 3
```

`layer_profile.py` aggregates a run into totals, the host-stage split, a per-kind table and
the widest layers, writes `result.json` + `layers.csv`, and asserts three consistencies that
keep the report honest (§6). `profile_overhead.py` is the "the profiler must be free when it
is off" gate.

---

## 4. Where a 256-token block's time goes

One block, 64 layers, block route, `OFLM_OPEN_PROFILE` on, idle box. The request total is the
engine's own figure for the block; the script's `wall` also carries the model load.

| | ms | share of the block |
|---|---|---|
| **GEMM dispatches** (`part0`) | **25 020** | **71.5 %** |
| host stages (`part1`) | 6 160 | 17.6 % |
| lm_head | 27 | 0.1 % |
| residue (unaccounted) | 96 | 0.3 % |
| **layer walls, summed** | **34 911** | 99.7 % of `total_ms` |
| request total | 35 007 | — |

Inside the host stages — note that some of these **nest**, so they must not be added:

| stage | ms | of host | nests in |
|---|---|---|---|
| **activation tiling** (`gemm_tile`) | **3 048** | **49.5 %** | — |
| output transpose (`gemm_tr`) | 956 | 15.5 % | — |
| mid (attention / DeltaNet host half) | 1 117 | 18.1 % | — |
| · DeltaNet conv+norm | 404 | 6.6 % | mid |
| · DeltaNet delta rule on S | 388 | 6.3 % | mid |
| · attention host half | 325 | 5.3 % | mid |
| shared expert (silu·up, gate, add) | 684 | 11.1 % | — |
| tail (residual, post-norm, router) | 450 | 7.3 % | — |
| GEMM buffer syncs | 325 | 5.3 % | — |
| input RMSNorm | 263 | 4.3 % | — |
| state BO syncs | 8 | 0.1 % | — |

**The two routes agree.** Layer-major is 2.3 % faster on this exact 256-token multiple
(34 216 vs 35 007 ms) — the parity the route policy already predicts at an exact multiple,
not a reason to change the default. Its host split is the same: tiling 50.7 %, transpose
16.3 %, mid 18.1 %.

**No layer stands out.** Layer 0 (first off an idle array) is 40–45 % above its neighbours;
from layer 1 on, the spread across the other 63 layers is under 8 %:

| layer | kind | wall ms |
|---|---|---|
| 0 | linear_attention | 786.4 |
| 52 | linear_attention | 603.0 |
| 9 | linear_attention | 575.7 |
| 5 | linear_attention | 575.0 |
| 56 | linear_attention | 569.5 |

Per kind the mean is 549 ms for a `linear_attention` layer and 534 ms for a `full_attention`
one, and the FFN dominates both (the 48 linear layers total 26.4 s, the 16 full ones 8.5 s).

---

## 5. Where a decode token's time goes

The per-layer profiler cannot answer this: the fused decode chain books blocked time into
`StepTiming::part1_ms` (there are no host stages inside a layer to attribute it to). The
existing per-dispatch trace can:

| kernel | calls/token | min ms | mean ms | p90 ms |
|---|---|---|---|---|
| `lx` (48 linear layers) | 48.0 | 6.67 | 11.72 | 14.12 |
| `ax` (16 full layers) | 16.0 | 5.38 | 5.58 | 5.85 |
| `ln` (final norm) | 1 | 0.22 | 0.30 | 0.37 |
| `lm` (lm_head) | 1 | 25.03 | 25.21 | 25.33 |

Per token: **430 ms wall = 409 ms of dispatch (sum of minima) + ~2.9 ms of lm read**, and
0.03 / 0.01 / 0.00 ms of embed / patch / route. So:

- the **host is not the decode bottleneck** — the dispatches' own times are the step;
- **66 dispatches per token** (16 `ax` + 48 `lx` + `ln` + `lm`);
- **lm_head is 25 ms of 430 ms = 5.8 %** per token — real, but not the target;
- `lx` is the one to look at: its mean is 1.76× its min, and the split by preceding context
  shows the whole gap is same-context work (13.82 ms mean when the previous dispatch was in
  the same context vs 7.52 ms after a switch). That is same-kernel, same-context variance —
  the shape of a weight-streaming or scheduling effect inside `lx`, not a context-switch cost.

Reproduced rate: **2.074 tok/s** loop rate, against 2.087 historical.

---

## 6. Bottleneck ranking

Ordered by expected end-to-end impact per unit of risk. "Share" is of the 256-token block.

| # | bottleneck | evidence | share | impact | complexity | numerical risk |
|---|---|---|---|---|---|---|
| 1 | **Q4 GEMM dispatch time** | 25 020 ms of a 35 007 ms block; 391 ms per layer for ~239 MB of streamed weights (≈0.60 GB/s implied) | **71.5 %** | high | high (kernel/AIE) | medium — must keep the paired four-product sum |
| 2 | **activation tiling + output transpose** | 3 048 + 956 ms = 4.00 s of a 35.0 s block, 65 % of all host time | 11.4 % | medium-high | **low** (pure host data movement, bit-exact by construction) | none if the layout contract is preserved |
| 3 | **`lx` variance at decode** | mean 11.72 ms vs min 6.67 ms; the gap is same-context | decode: 48 of 66 dispatches | medium | medium | none (no arithmetic change) |
| 4 | **DeltaNet host half** | conv 404 + rule 388 = 792 ms | 2.3 % | low-medium | medium-high (state semantics) | high — FP32 recurrence, causal order |
| 5 | **attention at long context** | 325 ms at 256 tokens and growing with the window; Phase 31 fitted 1.63 ms/position | 0.9 % here, ≈20 % of prefill by ≈7.6K | grows with ctx | high | medium |
| 6 | **lm_head at decode** | 25 ms of a 430 ms token | decode: 5.8 % | low-medium | medium | low |
| 7 | **dispatch count / context switches** | 66 per token; the trace's `mean(=)` vs `mean(sw)` split shows switches are cheap | — | low on present evidence | medium | low |

**What is explicitly *not* the problem**, from this data: host RMSNorm (263 ms, 4.3 % of
host), state BO syncs (8 ms), the router/tail (450 ms), and the number of layers per se — no
single layer is disproportionately slow.

---

## 7. Gates

| gate | result |
|---|---|
| baseline reproduced | **PASS** — TTFT −2.8 / +0.2 / −0.8 % and decode −0.6 % against Phase 31 |
| instrumentation free when off | **PASS** — decode 455.0 ms median off vs 456.0 ms on (+0.22 %), print resolution 1 ms (`profile_overhead.py`) |
| instrumentation changes no output | **PASS** — 64-token prompt through the block route, 8 greedy ids, byte-identical with the variable unset and set |
| layer accounting balances | **PASS** — layer walls + lm_head inside the request (34 937.8 / 35 007.2 ms); named host stages inside `host_ms` (6 160.0 / 6 160.0); dn conv+rule inside mid (791.8 / 1 117.3) |
| unit tests, engine | **PASS** — `ctest` 4/4 in `src/open_qwen36` |
| regression suite | **PASS** — `pytest specs/` 2438 passed, 172 skipped, 0 failed |
| settings CI classification | **PASS** — both variables are in `NON_SETTING_ENV_VARS` (`utilities/generate-registry.py`) with a reason, and `utilities/scan-settings.py` finds both; `check-settings-ci.sh` OK |
| shipped `oflm` builds | **PASS** — the repository's own Docker path, `cmake --build build --target oflm` |

---

## 8. Reproducing

```bash
source utilities/env.sh

# baseline through the shipped binary
python utilities/prefill-block/oflm_sanity.py --lengths 256 512 1024
python utilities/prefill-block/decode_rate.py --tokens 32 --prompt 40 300

# per-layer attribution (the standalone CLI is rebuilt by build-cli.sh)
./utilities/prefill-block/build-cli.sh
python utilities/prefill-block/layer_profile.py --route block layer-major --tokens 256

# the "free when off" gate
python utilities/prefill-block/profile_overhead.py --what decode --reps 3

# decode attribution: the engine's own dispatch table
OFLM_OPEN_STEP_TRACE=1 <the CLI> ... --max-tokens 4 --quiet
```

`runs/` is scratch and is not committed; each driver writes `result.json` (and
`layers.csv`) under `runs/<tool>-<stamp>/`.

---

## 9. Limitations

- **The weight-stream and TFLOPS figures in §6 row 1 are derived**, from wall time and an
  assumed 4.5 bit/packed weight — an order-of-magnitude check, not a kernel measurement. A
  kernel-side figure (the dispatch trace's `stream_mb` / `gbps` helpers already exist per
  dispatch) belongs to PR #3.
- **Per-layer device/host split does not exist on the sequential route**, because the fused
  chain has no host stages to attribute (§5). The dispatch trace covers it.
- **Prefill was profiled at 256 tokens only.** The block loop's cost is linear in blocks, so a
  1024-token prompt is four of these plus a stepped 8-token tail — an inference from the shape
  check, not a measured row.
- **The first pass's numbers are kept only as a rejected experiment** (§1). They are not a
  second baseline and are not to be compared against these.
- **No speedup is claimed.** PR #1 is instrumentation.

---

## 10. Recommended first target for PR #2

**The host data movement in front of every GEMM** (`host::tile_x` into the kernel's tiled bf16
layout, and the `[N, T]` → `[T, N]` transpose behind it): 4.00 s of a 35.0 s block, 65 % of
all host time, zero numerical risk by construction (it is a permutation, and the kernel's
layout contract is what it must reproduce), and it needs no kernel rebuild.

A second, independent candidate is the **`lx` same-context variance at decode** (mean 11.72 ms
vs min 6.67 ms over 48 dispatches a token): it is the largest single term in the 430 ms token,
and it is a scheduling/streaming question rather than an arithmetic one — but it belongs to
PR #5, not #2.

Both are hypotheses to be measured, not conclusions: PR #2 begins by measuring its chosen
change against these exact numbers.
