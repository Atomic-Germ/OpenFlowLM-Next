# Opt-in Q4 GEMM DMA batching for Qwen3.8 prefill

## Base: stacked on PR #2

**Do not compare this PR against `upstream/main`.** PR #2 has not landed upstream.

| Item | Value |
|---|---|
| PR #3 base | `836072db` (`perf/npu-prefill-scheduling` tip) |
| Upstream main at review | `9b1409d3` |
| `836072db` ancestor of upstream? | **no** (merge-base check exits 1) |
| PR #3 commits | `34419a6b` … `0f15a67a`, plus review commits |
| Diff vs base | the GEMM change plus its tests, tools, evidence and PR description |
| Diff vs upstream main | much larger, because it also contains PR #2 and its merges |

Land PR #2 first and re-target, or review this PR against `836072db`. No rebase
or history rewrite was performed.

## What changes

Bounded DMA submission batches in the Q4 prefill GEMM design: up to four
consecutive output visits before waiting and reclaiming descriptors. Traversal
order, arithmetic, FP32 accumulation, paired high/low products, FIFO depths,
weights, manifest format and kernel ABI are unchanged.

**Opt-in only.** `GQP_DMA_BATCH` defaults to **1**; no production recipe selects
it. `open_kernels/export_qwen36_kernels.py` clears ambient `GQP_DMA_BATCH` so a
shell experiment cannot change a recipe build or its cache key.

## Results

- Ten projection cases: exact outputs, identical compute-core images, existing
  numerical and resource gates pass, including a partial final batch.
- Full depth: 147 checks pass with exact captured logits/state and identical
  16-token greedy continuation; all nine four-layer boundary cases match.
- Performance, one Ryzen AI 9 365 / Strix XDNA2 Linux host, 14 fixed pairs at
  1024 tokens: **median paired prefill −2.43%**, 14/14 faster, 95% bootstrap CI
  **−3683…−1984 ms**. Decode and process RSS within the ±3% gates.
- The earlier five-pair 1024-token experiment remains recorded as **FAIL**
  (−0.60%, 4/5) and is not overwritten.
- Candidate-specific `oflm serve`: both arms select their own 10 GEMM
  artifacts, pass A→B→A with byte-identical A, match a fresh-process B, and
  record XRT 2.25.00 mappings. 11/11 graded checks per arm.

## Runtime provenance and limitations

- The historical 14-pair run preserved no `/proc/<pid>/maps`. Re-running the
  benchmark shape under `utilities/env.sh` shows **both** arms load XRT **2.26.0**
  from `/opt/xilinx/xrt/lib`. The comparison is valid (same runtime, same
  binary) but is a 2.26 result; the earlier "2.25 verified" claim was wrong.
- `utilities/prefill-block/omp_policy.py` now records `runtime_libraries` for
  each timed process so future campaigns carry this proof.
- Not claimed: Windows, driver-owned NPU memory, rebuilt tiled exports, HTTP
  streaming TTFT, an independent per-layer residual oracle, or a 2.25
  end-to-end re-timing.

## Opt in manually

```bash
# build both weight companions of every projection with batch 4
GQP_DMA_BATCH=4 python open_kernels/export_qwen36_kernels.py ...   # explicit knob
```

A promoted default requires a recipe change on both high and low companions and
separate approval; this PR does not make one. Evidence and reproduction steps:
[`utilities/q4-gemm/README.md`](../utilities/q4-gemm/README.md),
[acceptance review](../specs/open-engine/reports/qwen38-q4-gemm-acceptance.md),
[performance confirmation](../specs/open-engine/reports/qwen38-q4-gemm-1024-confirmation.md).
