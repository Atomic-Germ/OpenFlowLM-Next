# Phase 28 — Which prefill schedule should production use?

**Status: PASS.** The production default changed, on hardware evidence, and was revalidated.

```
Phase:                28
Status:               PASS
Commit:               the run-order and preference change below (see Artefacts)
Binary SHA256:        45ed9846… (provenance check, HEAD d83effbd) -> 5d50e0dd… (after the policy change)
Default route before: layer-major
Default route after:  block-loop, for kernel sets with no routed experts
```

---

## 28.0 — Binary provenance

The A/B had to run on a binary whose source is unambiguous, so before any benchmark:

| field | value |
|---|---|
| git HEAD | `d83effbd4e773221870f8ba5ff76fe38ad99e518` (the Phase 27 closure commit) |
| `git status --short` | clean for every tracked file; no untracked `.cpp`/`.hpp` in `src/` |
| rebuild | `cmake --build build --target oflm` in `openflowlm-build:ubuntu26` → **`ninja: no work to do`** |
| binary sha256 | `45ed9846122748f686fb77357f4255da0a6deb5a63b8567ad4da1c3d46eedd01`, **byte-identical before and after that rebuild** |

So the A/B ran on exactly `d83effbd`; no ambiguous binary was profiled. The later policy
change produced `5d50e0ddbeb0e33c26f996525aba011961c327505e397797e5e7e00e7f0f2c53`, and every
revalidation number below is from that one.

Toolchain: `/opt/xilinx/xrt/bin/c++` (XRT clang), `-O3 -DNDEBUG`, `ninja` at
`/opt/xilinx/xrt/bin/ninja`, built in `build/` configured by `cmake --preset linux-default`.

## 28.1 — Support is not preference

`Core::layer_major_ok()` is a mechanical test, nothing else:

```cpp
bool Core::layer_major_ok() const {
    if (!layer_major_on_ || !gemm_block_t_) return false;
    for (int l = 0; l < nl_; ++l) {
        const std::string& kind = types_[l]->gemm_block.kind;
        if (kind != "linear" && kind != "full") return false;
    }
    return nl_ > 0;
}
```

This kernel set satisfies it because its manifest carries:

| layer type | `gemm_block.kind` | `t` | routed expert kernel |
|---|---|---|---|
| `linear_attention` | `linear` | 256 | none |
| `full_attention` | `full` | 256 | none |

Neither kind is `dense`, so the loop passes and `Engine::prefill` sent the whole prompt to
`step_gemm_prompt`. **Nothing in that decision measured anything.** Layer-major was selected
because the implementation was mechanically capable of executing it:

```
layer_major_supported     = true   (kind is linear/full for every layer)
layer_major_selected      = true   (that was the ONLY test applied)
block_loop_supported      = true   (gemm_block.t = 256 for every layer)
block_loop_selected       = false
sequential_selected       = false  (above the crossover)
```

The Phase 27 report's 64 came from the Qwen3.6-35B **MoE** case, where layer-major runs each
layer once over the whole prompt so one routed expert's weights stream once per layer instead
of once per block. That distinction is preserved in the code: the MoE case keeps layer-major
and the reason lives in `layer_major_preferred()`.

## 28.2/28.3 — The matched A/B

Identical binary, weights, prompt ids, engine-visible token count, `paired` precision, context
4096, one machine. Each point is its own process (a route cannot be switched inside one
process), and the number compared is the runner's own `PREFILL_TIME` from `/status`, with
`/clear` before the measured prompt so the profilers hold that prompt alone — model load and
decode are excluded rather than hidden.

Run order was interleaved by length with the direction alternating, because
`layer-major all lengths, then block-loop all lengths` would let sustained device load land on
one half:

```
layer-major@96, block@96, block@128, layer-major@128, layer-major@256, block@256,
block@512, layer-major@512, layer-major@1024, block@1024, block@2048, layer-major@2048
```

| request | engine tokens | layer-major | block-loop | winner | delta |
|---|---|---|---|---|---|
| 96 | 104 | 37 362 ms | 33 844 ms | block | 1.104× |
| 128 | 136 | 33 661 ms | 33 791 ms | layer-major | 1.004× (tie) |
| 256 | 264 | 67 578 ms | 37 619 ms | **block** | **1.796×** |
| 512 | 520 | 101 475 ms | 71 442 ms | **block** | **1.420×** |
| 1024 | 1032 | 170 395 ms | 140 912 ms | **block** | **1.209×** |
| 2048 | 2056 | 333 272 ms | 301 168 ms | **block** | **1.107×** |

Drift check: at 2048 both sides are ~10 % above their own per-block rate elsewhere
(301 s and 333 s against 8 × 34 s and 8 × 34 s), so the last pair is the thermally loaded one.
That biases the two routes *equally* and cannot manufacture the 256–1024 result, which lands
in the middle of the session where the other points are clean at ~34 s/block.

## 28.4/28.5 — What layer-major is actually buying, and what it costs

A decaying speedup (1.80 → 1.42 → 1.21 → 1.11) is the signature of a fixed per-block cost, and
that is what it is. Both schedules cost about the same per 256-wide block:

```cpp
// Core::step_gemm_prompt
const size_t N = ids.size(), B = (N + T - 1) / T, TOT = B * T;   // pads to whole blocks
```

so layer-major runs `ceil(N/256)` full-width passes for any N, while the block loop steps a
short tail token by token. The gap is exactly `ceil(N/256)` over the block loop's pass count:

| engine tokens | layer-major blocks | block-loop passes | predicted ratio |
|---|---|---|---|
| 264 | 2 | 1 (+8 stepped) | 2.00 |
| 520 | 3 | 2 | 1.50 |
| 1032 | 5 | 4 | 1.25 |
| 2056 | 9 | 8 | 1.125 |

Against the measured 1.80 / 1.42 / 1.21 / 1.11 — the same curve, with the residual being the
stepped tail's own cost.

### The exact-multiple control

At a prompt that **is** a multiple of 256 there is no tail to pad, so both schedules run the
same number of full-width dispatches and any difference is schedule efficiency rather than
waste. `align_probe.py` found the text lengths that land exactly on 256 and 512 engine tokens
(the template offset is 8 for this fixture family) and measured both routes there:

| engine tokens | layer-major | block-loop | ratio |
|---|---|---|---|
| 256 | 33 820 ms | 33 808 ms | **1.000** |
| 512 | 67 728 ms | 69 366 ms | **1.024** |

**Layer-major has no intrinsic advantage on this geometry.** Its `stage_major` overlap and
its resident-weights-per-layer do not beat the block loop's own pipelining by anything outside
noise, and it has nothing to amortise because there are no routed experts.

### Engine-internal accounting at 512 tokens (`route_phases.py`)

| stage | block-loop | layer-major | share of block-loop |
|---|---|---|---|
| total | 68 859 ms | 68 667 ms | — |
| **GEMM dispatches** | **50 277 ms** | 50 068 ms | **73.0 %** |
| host stages | 11 468 ms | 11 431 ms | 16.7 % |
| of which gemm tile | 5 496 ms | 5 405 ms | 8.0 % |
| of which gemm transpose | 1 813 ms | 1 760 ms | 2.6 % |
| of which mid (attn + DeltaNet) | 2 385 ms | 2 294 ms | 3.5 % |
| of which prenorm | 290 ms | 268 ms | 0.4 % |
| of which tail | 793 ms | 782 ms | 1.2 % |
| of which sync | 690 ms | 690 ms | 1.0 % |
| shared expert | 1 064 ms | 1 112 ms | 1.5 % |
| state BO syncs | 11 ms | 5 ms | 0.0 % |

Both assertions hold: `GEMM + host + per-token + lm_head ≤ total` for every schedule, and the
named host stages fit inside `host` (5 375.9 of 5 376.0 ms — the denominator is fully
accounted). `state_ms` and `shared_ms` are outside `part1` by construction and are excluded
from that sum with the reason recorded in the harness.

### The two questions, answered

**What does layer-major eliminate or amortise relative to block-loop on Qwen3.8-27B?**
Nothing. For a kernel set with no routed experts, `step_gemm_prompt`'s `layer_*_staged`
bodies run the same GEMM kernels and the same host stages the block loop calls; the accounting
above shows the two within 0.3 % at 512 tokens. The amortisation layer-major exists for — one
routed expert's weights streamed once per layer rather than once per block — has no routed
expert to apply to here.

**What extra work does layer-major introduce?** A full-width padded block for a short tail:
`TOT = B * T`. That is the entire measured loss, and it is exactly one block's worth of work
for any prompt whose length is not a multiple of 256.

## 28.6 — Decision

Block-loop is **≥10 % faster at 256, 512 and 1024** (1.80×, 1.42×, 1.21×), the advantage is
stable across those lengths, and the mechanism is understood rather than inferred — Case A of
the gate. So layer-major does not remain the production default for this geometry.

The change separates the two questions the code previously conflated, and derives the answer
from what the schedule exists to amortise:

```cpp
bool Core::layer_major_preferred() const {
    if (!layer_major_ok()) return false;
    for (int l = 0; l < nl_; ++l)
        if (!types_[l]->gemm_block.moe_kernel.empty() ||
            !types_[l]->gemm_block.moe_batch.kernels.empty())
            return true;
    return false;
}
```

`Engine::prefill` now asks `layer_major_preferred()`, not `layer_major_ok()`. A kernel set
that carries routed experts keeps layer-major and its measured win; a set that carries none
takes the block loop. There is no model-name, family or tag anywhere in the rule, and
`OFLM_OPEN_LAYER_MAJOR=1` still forces layer-major for a comparison.

The startup line says which of the three situations applies, so the log never asserts a
schedule the engine will not use:

```
prefill schedule: block-major (layer-major is runnable here but amortises nothing: no
routed experts in this kernel set; OFLM_OPEN_LAYER_MAJOR=1 forces it)
```

The route reporter gained `not-preferred` as a distinct reason from `fallback`, because one is
a decision and the other is a failure.

## 28.7 — Two crossovers, kept apart

| crossover | value | status |
|---|---|---|
| sequential vs padded block-loop | **96 engine tokens** | measured in Phase 27, unchanged |
| block-loop vs layer-major | **none needed** | layer-major is not preferred at any length for this geometry, so there is no crossover to place |

The 96 was not reused as the second crossover and no token-by-token sweep was run: the
exact-multiple control showed parity at a multiple of the block, which means there is no
length at which layer-major becomes worth choosing.

## 28.8 — Revalidation on the new default

All on `5d50e0dd`:

| gate | result |
|---|---|
| boundary matrix through the shipped default (63/95/96/97/127/255/256/257/383/384/511/512/513) | **13/13 PASS** — every case is the block-loop policy, including 257 = one block + one stepped token |
| serve A → B → A isolation | **PASS** — A after B equals A before B |
| failure case A (before any block) | **PASS** — reset, fallback, identical answer |
| failure case B (after one block, non-zero position) | **PASS** — `pre-prompt state restored` |
| failure case C (rewind broken) | **PASS** — refused, nothing generated |
| 16-token trajectory, 64 layers, 511 prompt ids | **16/16 tokens**, min cosine **0.999999867**, max_abs 9.881e-03, no growth |
| regression suite | **green** |
| settings CI | **green** |

No numerical criterion was weakened: the cosine gate is still the repository's 0.9999 and no
new absolute tolerance was introduced.

## 28.9 — Production profile, on the selected route

Shipped `oflm`, block loop (now the default), 64 layers, `/clear` before each measured prompt:

| text | engine | passes | TTFT | prompt tok/s |
|---|---|---|---|---|
| 128 | 136 | 1 | 33 858 ms | 4.02 |
| 256 | 264 | 1 | 38 089 ms | 6.93 |
| 512 | 520 | 2 | 71 711 ms | 7.25 |
| 1024 | 1032 | 4 | 141 492 ms | 7.29 |
| 2048 | 2056 | 8 | 285 675 ms | 7.20 |

The 256 row is one full block plus an 8-token stepped tail, which is why it costs more than
128 for the same number of passes: the stepped tail is ~4.4 s of token-by-token work. Above
512 the rate settles at **7.2-7.3 prompt tok/s** and the cost is linear in blocks.

The wall time at 128 (159 s) is the first run of the session and carries warm-up; its TTFT is
33.9 s, in line with every other row, so TTFT is the figure quoted. That is the same
first-run effect Phase 27 recorded.

Compared against the corrected Phase 24 profile only as history: the shape is unchanged —
GEMM dominates — but the route the numbers describe is now the block loop, so no Phase 24
number is merged with a Phase 28 number.

## 28.10/28.11 — Bottleneck hierarchy and the one next target

Eligibility rule: **≥20 % of production prefill wall**, or clearly superlinear.

| rank | category | share of prefill | eligible |
|---|---|---|---|
| 1 | **projection GEMM datapath** | **73.0 %** | yes |
| 2 | host GEMM preparation (tile 8.0 % + transpose 2.6 %) | 10.6 % | no (below 20 %, and linear in tokens) |
| 3 | attention + DeltaNet (`mid`) | 3.5 % | no (linear at these lengths; Phase 29 owns long context) |
| 4 | shared expert | 1.5 % | no |
| 5 | norm / tail / sync | 2.6 % | no |

**Selected next optimization target: NONE opened in this phase.**

The one eligible category is the projection GEMM, and Phase 25 closed that datapath by
decision — the current `gemm_q4_prefill` lineage is upstream-derived, the arithmetic is
frozen, and `weight_stationary`, `tile_n=64` and `three_term` promotion are all STOP. A large
GEMM share is therefore **not** permission to reopen it.

Two things are worth separating before anything is opened, and only the second would be a new
mechanism:

- **same datapath arithmetic ceiling.** 73 % of prefill is the same q4_1 → bf16-on-core GEMM
  the Phase 22-25 work already priced. Nothing in this phase's accounting says that ceiling
  moved.
- **avoidable runtime/schedule duplication.** The host preparation (10.6 %) is per-block tiling
  and transposing of activations, not arithmetic. It is the only part of the profile that a
  schedule change could remove, and at 10.6 % it is below the eligibility rule. It is
  recorded for a future proposal, not selected.

## 28.12 — GEMM research stays closed

Nothing above reinterprets the GEMM share as new evidence about the datapath. Reopening it
needs a separate proposal with an explicit expected end-to-end gain, not work inside Phase 28.

## 28.13 — 4096

**SKIPPED.** The gate asks for 4096 only if the 128-2048 profile shows attention rising enough
to matter, an unresolved layer-major/block crossover, a new memory threshold near 2 K, or an
ambiguous superlinear term. None holds:

- attention + DeltaNet is 3.5 % of prefill at 512 tokens and the 2048 row adds no superlinear
  term (285 675 ms for 8 blocks against 141 492 ms for 4 — 2.02×, linear);
- the layer-major/block question is settled at exact multiples, not deferred;
- no threshold appeared between 1024 and 2048.

Long context is Phase 29's, where it has its own schedule.

## 28.14 — Nothing was optimized on the losing route

The losing route is layer-major. No work was spent on it: the change narrows when it is used,
and its own machinery (`stage_major`, the resident-weights-per-layer pass) is untouched. It
remains the schedule for the kernel sets that carry routed experts, and the diagnostic route
here via `OFLM_OPEN_LAYER_MAJOR=1`.

## Artefacts

- [`route_ab.py`](route_ab.py) — 28.2/28.3, the interleaved A/B.
  `runs/route-ab-20261102-092720/result.json`.
- [`align_probe.py`](align_probe.py) — the exact-multiple control.
  `runs/align-probe-20261102-120724/result.json`.
- [`route_phases.py`](route_phases.py) — 28.4/28.5, engine-internal accounting with the Phase
  25 assertions. `runs/route-phases-512-20261102-125018/result.json`.
- `runs/route-matrix-default-20261102-125952/result.json` — 28.8 boundary matrix on the new
  default.
- `runs/fail-inject-serve-20261102-131418/result.json` — 28.8 isolation and failure cases.
- `runs/decode16-logits-511-16-20261102-133802/result.json` — 28.8 trajectory.
- `runs/oflm-sanity-20261102-135121/result.json` — 28.9 production profile.
- `runs/route-matrix-block-20261102-041703/…`, `runs/crossover-local-20261102-034427/…` —
  Phase 27's crossover evidence, unchanged.
- [`PHASE27_REPORT.md`](PHASE27_REPORT.md), [`BLOCK_CONTRACT.md`](BLOCK_CONTRACT.md)

## Recommended next work

1. **Phase 29 (long context)** owns the attention term: it is 3.5 % here and the only
   component this profile shows growing faster than the block count as position rises.
2. If the host GEMM preparation is ever taken up, it needs its own proposal: 10.6 % is real
   but below the rule, and it is a schedule/host change rather than an arithmetic one.
3. A model with routed experts should be re-measured once on this binary to confirm
   `layer_major_preferred()` keeps it on the schedule it was measured to want. That is a
   check of the preference rule, not a new optimization.
