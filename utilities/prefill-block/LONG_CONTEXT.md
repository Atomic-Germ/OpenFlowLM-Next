# Phase 29 — How far does this runtime scale, and what stops it?

**Status: PASS. Outcome B — attention becomes the dominant long-context cost, so the
full-context escalation stopped at 4K and attention is the selected next target.**

Three different claims, kept apart throughout:

| claim | meaning |
|---|---|
| **capacity validated** | the buffers allocate, the context object accepts the length |
| **addressing validated** | the RoPE / KV / position arithmetic is right at that absolute range |
| **end-to-end validated** | a real prompt of that length went through the real model and decoded |

A capacity PASS is not an end-to-end claim, and this report never collapses them.

---

## 29.0 — Route-policy compatibility closure

Phase 28's `layer_major_preferred()` is a capability rule, so before any long-context work I
enumerated every manifest on the machine and in the shipped xclbin bundle (40 in total) and
classified each against it.

The first pass — the rule as Phase 28 shipped it — **changed the default route for five
pre-existing kernel sets**: the Qwen3.5 0.8B/2B/4B/9B and the 27B itself. `layer_major_ok()`
passes for all of them because their layer kinds are `linear`/`full`, and none carries routed
experts. But only the 27B had ever been measured on the block loop, and **there are no
weights for the other four on this machine to smoke-test**.

That is exactly the "do not silently change it globally" case, so the preference was gated on
a **manifest capability** a recipe must declare:

```cpp
bool Core::layer_major_preferred() const {
    if (!layer_major_ok()) return false;
    for (int l = 0; l < nl_; ++l)
        if (!types_[l]->gemm_block.moe_kernel.empty() ||
            !types_[l]->gemm_block.moe_batch.kernels.empty())
            return true;            // routed experts: layer-major is what it was built for
    return !man_.block_loop_preferred;  // dense: only where the block loop was MEASURED
}
```

`block_loop_preferred` is written by the recipe for the shape Phase 28 validated (hidden
5120, 64 layers, FFN 17408 — checked against the model's own `config.json`), and the
manifest Phase 28 measured carries it. The audit then reads:

| family | kinds | routed | old | new | changed |
|---|---|---|---|---|---|
| `qwen36moe` (35B-A3B) | linear,full | **yes** | layer-major | **layer-major** | **no** |
| `qwen35` (0.8B…9B) | linear,full | no | layer-major | layer-major | **no** |
| `qwen35` (27B, declared) | linear,full | no | layer-major | block-loop | yes, and validated |
| `gemma3`, `granite`, `hunyuan`, `k2`, `llama3`, `phi3`, `qwen3`, `qwen2`, `lfm2` | dense/other | no | block-loop | block-loop | no |

**40 manifests, 0 pre-existing defaults moved.** The routed model keeps the schedule its win
was measured on; the four unmeasured dense sets keep theirs; only the measured 27B declares
and takes the block loop. No model-name test exists anywhere in the rule.

## 29.1 — The context contract

| stage | what it does | limit it imposes |
|---|---|---|
| CLI / API | `--ctx-len`, or `default_context_length` from the registry | the **total** context |
| `Engine` | `cfg_.max_ctx = MAX_L > 0 ? MAX_L : 4096` | rejects `MAX_L > core_->max_ctx()` and says "capacity stays N" |
| `Core` | `state_[l] = (cfg_.max_ctx + kv_slack) * state_row` for a `kv` layer | KV is allocated **from the requested length** |
| `_chunked_insert` | `meta_info.max_prefill_len`, rounded up to a power of two | one prefill operation, **not** the total |
| positions | `int pos_`, `int64_t mrope_pos_`; the ptab record stores `int32 valid` and `double` angles | 262 143 is representable |
| block loop | `step_gemm_block` refuses `pos_ + T > max_ctx`; the tail is stepped | per-block arithmetic is `size_t` |

The eight questions:

1. **What limits total context?** `cfg_.max_ctx`, from `--ctx-len` / `default_context_length`,
   enforced by `Core::step` and `step_gemm_block`.
2. **What limits one prefill request?** `max_prefill_len` (the chunk size), 4096 in the
   registry, rounded to a power of two.
3. **Independent?** Yes. `max_prefill_len` chunks a prompt; `max_ctx` bounds the total.
4. **Is KV allocated from the requested length?** Yes — `(max_ctx + kv_slack) * state_row`.
5. **Any buffer fixed to a bring-up size?** The fixtures were (a 512-id pool); that is a
   harness constant, raised deliberately to 8192 for this phase, not a runtime limit.
6. **Any 16-bit/narrow position?** No. `valid` is `int32`, angles are `double`, offsets are
   `size_t`; the largest KV byte offset at 262 143 is 2 147 475 456, inside `int32` anyway.
7. **Does `max_prefill_len` limit total context?** No, only one prefill operation.
8. **Does `--ctx-len 262144` reach the allocator as 262144?** Yes — the capacity probe loaded
   at 262 144 and the engine reported `context capacity 262144`.

## 29.2 — The memory model, from the manifest

`layout.kv_row` is **8192 bytes**, which is exactly `4 kv heads × 256 head dim × 2 bytes ×
(K + V)`. With 16 full-attention layers:

```
KV = 8192 × 16 × ctx = 131072 bytes/token
```

| ctx | KV | predicted total (container 18.48 GiB + KV + ptab + globals + state) |
|---|---|---|
| 4 096 | 0.50 GiB | 21.48 GiB |
| 8 192 | 1.00 GiB | 21.99 GiB |
| 16 384 | 2.00 GiB | 23.01 GiB |
| 32 768 | 4.00 GiB | 25.04 GiB |
| 65 536 | 8.00 GiB | 29.10 GiB |
| 131 072 | 16.00 GiB | 37.23 GiB |
| 262 144 | 32.00 GiB | 53.48 GiB |

Host has **91 GiB total, 58 GiB available**. The 256K allocation fits by arithmetic, so the
probe was worth running — and nothing larger was attempted.

## 29.3 — Capacity: 4K → 256K, all PASS

Each row loads the engine at the requested length and prefills nothing:

| ctx | capacity | reported | predicted |
|---|---|---|---|
| 4 096 | **PASS** | 4096 | 21.48 GiB |
| 8 192 | **PASS** | 8192 | 21.99 GiB |
| 16 384 | **PASS** | 16384 | 23.01 GiB |
| 32 768 | **PASS** | 32768 | 25.04 GiB |
| 65 536 | **PASS** | 65536 | 29.10 GiB |
| 131 072 | **PASS** | 131072 | 37.23 GiB |
| 262 144 | **PASS** | **262144** | 53.48 GiB |

No failure, and no silent reduction: the engine reported the full requested length every
time. Artefact `runs/capacity-probe-20261102-143728/`.

## 29.4 — Addressing: 19 positions, all PASS

Seek to a position, step **one** token, then read back what the engine wrote. Three
independent checks, and the reference values computed here in float64:

1. **RoPE** — the cos/sin in the position record equals `cos/sin(pos · inv_freq[i])`.
   Max error across all 19 positions: **2.97e-08**, which is the float32 the engine stores.
   A position truncated to 16 bits misses by order 1.
2. **The record's own position** — `valid == pos`. `attn_window` computes `valid = pos −
   start`, and with a full causal window `start = 0`; anything wrapped would show as ≤ 65535.
   It did not: `valid` tracked `pos` through 65 536, 131 072 and 262 143.
3. **The KV write** — the row at `pos · kv_row` is non-zero after the step, and the row after
   it is still zero wherever that row is inside the allocation.

| position | valid | RoPE err | verdict |
|---|---|---|---|
| 4095 / 4096 / 4097 | = pos | 2.9e-08 | PASS |
| 8191 / 8192 / 8193 | = pos | 2.9e-08 | PASS |
| 16383 / 16384 / 16385 | = pos | 2.9e-08 | PASS |
| 32767 / 32768 / 32769 | = pos | 2.9e-08 | PASS |
| 65535 / 65536 / 65537 | = pos | 2.7e-08 | PASS |
| 131071 / 131072 / 131073 | = pos | 2.9e-08 | PASS |
| **262143** | = pos | 2.9e-08 | **PASS** |

Artefact `runs/addressing-probe-20261102-145709/`. This proves **addressing**, not model
behaviour at those lengths.

Two harness errors of mine, both caught by the engine disagreeing with my expectation rather
than the reverse: `valid` is `pos − start` (I first asserted `pos + 1`), and I stepped **two**
ids so the second token legitimately wrote the row I wanted to be zero.

## 29.5 / 29.14 — The context boundary

| case | observed |
|---|---|
| engine 4095 at a 4096 capacity | **runs** — 15 full blocks + a padded 255 tail, 16 passes |
| engine 4096 at a 4096 capacity | **refused**: `[WARNING] Max length reached, stopping prefilling...`, **0 tokens prefilled** |
| silent truncation | **none** — nothing is prefilled and the request does not proceed |
| position wrap | **none** — addressing proves it at 262 143 |
| KV overwrite | none observed; `step_gemm_block` refuses `pos + T > max_ctx` |

The exact-capacity refusal is a **one-token off-by-one in `_shared_insert`**:
`if (total_tokens + tokens.size() >= MAX_L)` refuses a prompt that would exactly fill the
context, so the usable maximum is `MAX_L − 1`. It fails loudly and safely, which is the
behaviour the plan requires, but it is a real (and reportable) defect in limit accounting.
It was **not** changed in this phase: 29.15 forbids changing limits during the measurement
pass.

## 29.6 — End-to-end: 4096

Two routes to the same number, both real prompts through the full 64-layer model:

| route | engine tokens | passes | result |
|---|---|---|---|
| shipped `oflm` (block loop, now the default) | **4095** | 16 | answer produced, **664 s** wall, decode starts |
| standalone CLI, layer-major | 4000 | 1 | 590 205 ms |
| standalone CLI, block loop | 4000 | 16 | 592 000 ms |

Prefill rate ≈ **7.2 prompt tok/s**, consistent with Phase 28's 7.2-7.3 at 2K — the per-block
cost has not moved yet.

`TTFT` is not quoted for the 4095 run: `/status` resets its profilers on `/clear`, and the
`/clear` that precedes a 4095-token prompt reports `0`. The wall time and the CLI's own
internal `total_ms` are what the numbers above use. That is a reporting limitation of the
shipped CLI at this length, not a missing measurement.

## 29.8 — The attention gate fires

Engine-internal accounting, block route, 4000 ids, per 256-block:

| block start | total ms | GEMM ms | attention ms |
|---|---|---|---|
| 0 | 34 336 | 25 202 | 323 |
| 1792 | 36 723 | 25 084 | 3 024 |
| 3584 | 40 144 | 25 027 | 6 196 |

GEMM is **flat** — 25.0-25.4 s per block regardless of position, as Phase 28 established.
Attention is **not**: it grows by **1.628 ms per position** (R² = 0.99768 over 15 full
blocks), i.e. from 417 ms at the start of a block to 6 668 ms by the end of a 4K prompt.

That is the whole finding. **GEMM is linear in blocks; attention is linear in position, so
attention's share grows quadratically in context length:**

| ctx | GEMM | attention | attention share |
|---|---|---|---|
| 4 096 | 400.8 s | 56.3 s | **12.3 %** |
| 8 192 | 801.6 s | 219.4 s | **21.5 %** |
| 16 384 | 1 603 s | 865 s | **35.1 %** |
| 32 768 | 3 206 s | 3 438 s | 51.7 % |

The 20 % gate is crossed at **≈ 7.6K context**, and a 16K prompt is already a third
attention by the measured fit. The whole-prompt share at 4 000 ids is 8.6 %, lower than the
12.3 % per-full-block share only because the last block is a 160-token tail; both are quoted
in the artefact.

**So the ladder stops here.** 8K and 16K were **not run end-to-end**: the fit puts 8K at
21.5 % attention, past the gate, and 16K is ~0.7 h of prefill for a path already selected as
bottleneck-bound. 29.9 says to proceed past 16K only when attention is below the gate, and it
is not below it at 8K. Running them would have bought a confirmation of a curve already
fitted at R² = 0.998 on hardware data.

### Why this matters and why it is not the whole story

The attention here is the **host half** of the full-attention layers (`mid` minus DeltaNet's
`dn_conv`/`dn_rule`), which is what the block route spends on softmax and the output
projection across the window. DeltaNet's own two columns stay flat — `dn_conv` 294-379 ms and
`dn_rule` 199-353 ms per block, no trend with position, because its state is fixed-size.

That is a **scheduling and datapath** observation, not a kernel-arithmetic one, and it is the
first time the repository has a measured reason to open attention work.

## 29.7 — Correctness at 4K

Not run as a separate oracle pass, and the reason is recorded rather than implied: the 4K
end-to-end run had already produced a complete answer through 16 block passes, and the
Phase 26/28 release oracle (boundary matrix, A/B/A isolation, 16-token trajectory, cosine
0.9999) was revalidated on this binary line in Phase 28. What this phase adds is **addressing
correctness at 262 143**, which is the part that is new at long context and which no earlier
oracle covered.

The gap this leaves: no sequential-vs-block logits comparison at 4K. A sequential 4K prefill
is 16× the block route's dispatch count at one token per step — hours — which 29.7 explicitly
authorises skipping when it "would consume excessive hardware time".

## 29.13 — Server

Not run at 4K, and recorded as such: the server path shares `Engine::prefill` with the CLI,
and 29.13 asks for a representative case "unless it exposes a discrepancy". The 4K CLI run
reached the same code (block route, 16 passes, logits, decode), so the remaining risk was the
request-level plumbing, which Phase 27's A→B→A isolation already covered at 400 tokens. A 4K
serve request would be ~11 minutes per request for no new information.

## 29.11 / 29.12 — The two length settings

| setting | value | decision |
|---|---|---|
| `max_prefill_len` | 4096 | **unchanged.** 8K/16K did not pass end-to-end, so there is no evidence to raise it |
| `default_context_length` | 32768 | **unchanged.** KV at 256K is 32 GiB; a default that size would allocate it for every request. The 256K capacity is available to a request that asks for it with `--ctx-len` |

## 29.15 — Nothing was optimized

No kernel, arithmetic, route policy, block size or KV precision was touched during the
measurement. The only runtime change in this phase is the Phase 29.0 *preference gate*, which
moves **no** pre-existing model and was made before the measurements.

---

## Phase 29 decision tree: Outcome B

```
Phase 29 long-context capacity:      262144 (256K) -- buffers, indices, context object
Phase 29 addressing validated:      4K .. 262144 at 19 boundary positions
Phase 29 end-to-end validated:      4095 engine tokens (4K) on the shipped runtime
performance blocker:                attention
```

Attention is **12.3 %** of per-block prefill at 4K and the fitted share crosses **20 % at
≈ 7.6K**, so per 29.8/29.18 the expensive ladder stops and attention becomes the next target.
**GEMM is not reopened** — it is flat at 25 s/block and remains the largest single share at
4K (68 %), but it is the same frozen datapath Phase 25 closed.

### The minimum attention research phase, proposed

Not started, and nothing in the kernel is touched until it is agreed:

1. **Establish what the host attention half actually costs** at 4K by splitting `mid` into
   softmax, the output projection and the KV walk — the columns exist (`attn_ms` is the sum).
   The question is whether 6.2 s per block at position 3 584 is bandwidth on the KV window or
   arithmetic on the scores.
2. **Compare against the NPU attention path** already in the manifest (`attn_block`, used by
   `attn_block_on_`), which the block route can use for full layers. If the NPU path is flat
   in position, the host half is a scheduling choice, not a ceiling.
3. Only then consider a kernel change, with an explicit expected end-to-end gain.

Expected benefit if the host half were removed entirely: 4K prefill −8.6 %, 8K −18 %, 16K
−26 %. That is the size of the prize, and it is why this is worth one focused phase rather
than a datapath reopening.

## Artefacts

- [`route_policy_audit.py`](route_policy_audit.py) — 29.0, the compatibility closure over 40
  manifests. `runs/route-policy-audit-20261102-143330/result.json` (0 changed).
- [`capacity_probe.py`](capacity_probe.py) — 29.3.
  `runs/capacity-probe-20261102-143728/result.json`.
- [`addressing_probe.py`](addressing_probe.py) — 29.4.
  `runs/addressing-probe-20261102-145709/result.json`.
- [`route_phases.py`](route_phases.py) — 29.8, the accounting the attention fit is built on.
- [`long-context-v1.json`](long-context-v1.json) — the machine-readable table, with
  capacity / addressing / end-to-end as separate fields per length.
- [`PHASE28_REPORT.md`](PHASE28_REPORT.md), [`PHASE27_REPORT.md`](PHASE27_REPORT.md)
