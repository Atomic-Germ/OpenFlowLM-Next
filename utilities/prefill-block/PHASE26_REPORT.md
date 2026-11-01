# Phase 26 — production block-prefill correctness

**Status: PASS.**

The block route and the sequential route now produce the same logical state and the same
next token, measured from the production per-token open-NPU path as the reference rather
than from a replica. No kernel, no arithmetic and no shipped `oflm` runtime default was
changed.

| | |
|---|---|
| logical block size | **32** (and 31/33, 63/64/65, 127/128/129, 255/256/257 at the boundaries) |
| physical block size | **256** — the only legal value, see below |
| tail strategy | pad the physical block to 256 with a repeated in-range id; `t_real` bounds the state transition and the KV write, `pos[]` masks causality |
| NPU contexts / designs added | **zero** |
| changed files | [`core.hpp`](../../src/open_qwen36/core.hpp), [`core.cpp`](../../src/open_qwen36/core.cpp), [`cli.cpp`](../../src/open_qwen36/cli.cpp) — one accessor trio and two repeatable CLI flags, all bring-up only |

## 26.0 — the block contract, and why the physical tile is 256

Full derivation: [`BLOCK_CONTRACT.md`](BLOCK_CONTRACT.md). The decisive line, which the
plan's `T=32` suggestion could not survive:

```
attention_npu() throws unless M == ab.m
M = (nh/kvh) · T = (24/4) · T = 6T      ab.m = 1536
⟹ T = 256 exactly
```

This is not a preference. A `full` layer has exactly one legal physical `T`. The plan's own
escape clause applies, so **Phase 26 did not build a T=32 kernel**: it keeps
`physical_t = 256` and executes `active_t = 32` through the existing padding path, which
the manifest and `softmax_chunk` already support. No new GEMM geometry was authorised and
none was needed.

The three mechanisms that make a padded block safe, all pre-existing:

1. **attention** — `softmax_chunk` computes `valid = pos[r] - c0 + 1` per row and zeroes
   the rest, so a padded row is outside every real row's window;
2. **DeltaNet** — `deltanet_block` sets `R = t_real`, zeroes `og[t_real..]` and stops the
   recurrence at the last real token, so padding cannot reach `S` or the conv history;
3. **padding rows are still computed** through the projections and FFN, but the only
   cross-row operations are (1) and (2), so they can only affect themselves.

## 26.1 — the oracle

[`oracle.py`](oracle.py). Path A is `core.step()` — the production per-token open-NPU
route — and Path B is `--gemm-block`. **The same binary, the same weights, one flag apart**,
so the two cannot drift in build or load. Each runs `Core::reset()` and a deterministic
`seek`, so both start from identical state, conv history, KV and position.

## 26.2 — the fixture

A single deterministic 512-id draw (`seed=26026`, `replace=False`); a shorter fixture is a
**prefix** of a longer one, so length is the only variable between two runs. Distinct ids by
construction: T copies of one id cannot catch a row swap, a cross-token mix or a wrong block
position, because every row would expect the same answer.

## 26.3 / 26.4 — what is compared

Per tensor: cosine, max absolute, max relative, worst index, PASS/FAIL. Gate is the
repository's existing `cosine ≥ 0.9999`, unchanged — 26.2 forbids inventing a weaker
tolerance because the accumulation order changed. Logical state only: the KV slice covers
exactly the rows the run wrote, and the region beyond it is covered by a **canary**
(`--state-canary`, read back through `--dump-state`) rather than by meaningless equality.

## Results

21 runs, **all PASS**. Worst cosine across every run and tensor: **0.999999322** (gate
0.9999). Artefacts under [`runs/`](runs/).

| composition | layers | position | tokens | tensors | min cosine | verdict |
|---|---|---|---|---|---|---|
| DeltaNet + attention | 8 | 0 | 32 | 22 | 0.999999513 | PASS |
| DeltaNet + attention | 8 | 17 | 32 | 22 | 0.999999561 | PASS |
| DeltaNet + attention | 8 | 255 | 32 | 22 | 0.999999506 | PASS |
| DeltaNet + attention | 8 | 511 | 32 | 22 | 0.999999541 | PASS |
| boundary B-1/B/B+1 | 8 | 17 | 31 / 32 / 33 | 22 each | 0.999999557 | PASS |
| boundary 2B-1/2B/2B+1 | 8 | 17 | 63 / 64 / 65 | 22 each | ≥ 0.999999581 | PASS |
| boundary 4B±1 | 8 | 17 | 127 / 128 / 129 | 22 each | ≥ 0.999999560 | PASS |
| boundary at the physical 256 | 8 | 17 | 255 / 256 / 257 | 22 each | ≥ 0.999999557 | PASS |
| **full depth** | **64** | 17 | 32 | 120 | 0.999999522 | PASS |
| **full depth** | **64** | 255 | 32 | 120 | 0.999999322 | PASS |
| **block → decode** | 8 | 17 | 40 + 2 | 10 | 0.999999601 | PASS |
| **block → decode** | 64 | 17 | 40 + 2 | 10 | 0.999999897 | PASS |
| **block → decode** | 64 | 511 | 40 + 2 | 10 | 0.999999826 | PASS |

Full depth at 64 layers covers all 48 DeltaNet layers (conv + `S`, 96 tensors) and all 16
attention layers (KV cache, 16 tensors) plus logits, greedy and both canaries.

### 26.12 — the hard gate

40 prompt tokens as **two chunks** (32 + a tail of 8), then two real decode steps, in one
process each. Compared after each decode token: greedy, logits, the last DeltaNet layer's
conv and `S`, and the last attention layer's KV. At 64 layers the greedy path is
`[271, 1919]` on both sides — the generated-token path does not know how the prompt
arrived.

### 26.7 — non-zero positions

P = 0, 17, 255, 511, 257 all pass, which is what catches a RoPE offset, a KV offset or a
conv-history offset. P = 511 matters because the KV read crosses the 512-byte row grid at a
different alignment than P = 255.

## Six harness defects found and fixed

Every one of these produced a **clean PASS while testing almost nothing**, which is why
each is pinned by a test in [`test_prefill_block.py`](../../specs/open-engine/tests/test_prefill_block.py).

1. `--dump-state` held **one** spec, so an 8-layer comparison compared one tensor. Now a
   repeatable flag.
2. The state dump was read from the wrong path, so **no** state was compared at all.
3. The state **layout** was taken from the stage instead of the layer type, so "conv"/"S"
   were reported for the attention layers at index 3 and 7.
4. The canary region was **never dumped**, so the canary check never ran. A canary that is
   not read back proves nothing.
5. The logits dump was looked up by **absolute position** when `cli.cpp` keys it by the
   index inside `--ids`, so **every non-zero-position run silently skipped the logits
   comparison**.
6. A regex without `re.M` matched nothing and the greedy parse silently returned `[]` — the
   same class of mistake that cost Phase 23 a full sweep.

## A limitation recorded rather than hidden

`cosine` is **scale-invariant**: a state tensor that is uniformly 2× too large has cosine
1.0 and passes the gate. The gate is the repository's existing acceptance rule, and 26.2
forbids substituting a different one here, so this is stated as a limitation. `max_abs` and
`max_rel` are reported for every tensor and carry the magnitude information; the worst
`max_abs` seen (0.5012 on a KV cache at P=255, 64 layers) is visible in the artefacts.

## Inherited numerical limits, kept separate

No run needed the "both paths share the same model-vs-reference limitation" carve-out: the
comparison here is block-vs-sequential, both on the NPU, and it passes. The existing
paired-arithmetic model-vs-reference bound is a different question and was not re-opened.

## Regression and scope

- suite: see the commit; `test_prefill_block.py` adds 20 tests.
- **no** context or design was added (26.13) — the work reused the existing prefill
  contexts, runtime sequence, state BOs and KV BOs. The `_low` context-ordering observation
  from Phase 25 was **not** touched.
- **no** performance work (26.14): paired arithmetic, Q4 decomposition, tile geometry,
  attention and the DeltaNet scan are unchanged. The block route is not faster; that was
  never the goal.
- the shipped `oflm` runtime is **untouched** (26.15). The new flags exist only in the
  standalone `open_qwen36_cli` harness.

## Recommended Phase 27 runtime integration

1. Switch `oflm run` / `serve` to the block route for prompts once the state-equivalence
   below is the gate, not throughput.
2. Expose the logical block size as a setting; **physical T stays 256** and cannot be
   configured, because attention's `M = 6T` has one solution.
3. Expect chunks smaller than 256 tokens to cost a full 256-column dispatch each — a
   throughput property, not a correctness one, and the reason a minimum chunk of 256 is the
   natural production default.
4. Keep `utilities/prefill-block/oracle.py` as the release gate: the 64-layer,
   non-zero-position, block→decode composition is what must pass before the default flips.

## Reproducing

```bash
./utilities/prefill-block/build-cli.sh          # the standalone harness
source utilities/env.sh
./ironvenv/bin/python utilities/prefill-block/oracle.py \
    --stage both --tokens 32 --layers 64 --position 17
./ironvenv/bin/python utilities/prefill-block/decode_transition.py \
    --prompt 40 --chunk 32 --decode 2 --layers 64 --position 17
```
