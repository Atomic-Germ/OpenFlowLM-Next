# Refined L3 attention through native output projection and FFN

This private full-ax design replaces only the DMA source of the FP32 attention
output consumed by the output projection. It adds an external OG vector as
a seventh data-buffer argument; production attention, output-projection, residual,
post-attention normalization and FFN arithmetic are unchanged.

It is a **forced-boundary experiment**, not an integrated correction. The
original attention computation and KV update still execute. The candidate
consumes the precomputed output of the independently audited 240-position
refined-attention trajectory, which includes two NPU projection passes and one
host FP32 addition. Scratch `AA_OG` still contains the original attention result;
the actual forced input is the bound 24,576-byte `*-og_*.bin` buffer.

## Reproduce

Use fresh output directories. The prerequisite attention run and its independent
audit must both pass, including all 240 native output and KV-prefix controls.

```bash
source utilities/env.sh
export OPENBLAS_NUM_THREADS=1
ironvenv/bin/python utilities/l3-tail-replay/build.py \
  --spec Models/qwen38-27b/production-post-fp32/kernels-v1/spec.json \
  --out Models/qwen38-27b/post-prefill-drift/l3-tail-build-v3
ironvenv/bin/python utilities/l3-tail-replay/replay.py prepare \
  --source Models/qwen38-27b/post-prefill-drift/l3-attention-v1 \
  --build Models/qwen38-27b/post-prefill-drift/l3-tail-build-v3 \
  --out Models/qwen38-27b/post-prefill-drift/l3-tail-v2
ironvenv/bin/python utilities/l3-tail-replay/replay.py run \
  --out Models/qwen38-27b/post-prefill-drift/l3-tail-v2 \
  --harness Models/qwen38-27b/prefill-sequential-drift/harness-l0-v1/run_kernel
ironvenv/bin/python utilities/l3-tail-replay/replay.py audit \
  --out Models/qwen38-27b/post-prefill-drift/l3-tail-v2
```

## Controls and references

- Run control and candidate at P125/P205/P239. Each dispatch restores the
  original preceding KV prefix and scratch, then loads the original layer input.
- The control must reproduce **all nine original captures bytewise**: output
  residual, complete scratch and active KV prefix at each position. Candidate
  upstream Q/gate/K/V/OG and KV must remain byte-exact to the original captures.
- Primary residual reference: independent FP64 causal attention derived from
  unrounded normalization of the native layer-input history, followed by FP64
  output projection, residual, post-norm and FFN. It uses no captured FFN norm.
- Also compare with the original full-model CPU residual, clearly separating
  propagated input effects from the local layer error.
- Conditioned checks separately cover the actual forced OG-to-output projection,
  composed tail, FFN norm/activation, each split-down partial and closing residual.
  All existing 0.005 maxrel gates remain unchanged. Improvement is reported
  separately from tolerance PASS; retain P239 even when it regresses.
- Independent audit recomputes numerical arrays and compares their dtype, shape
  and bytes. Inputs, captures, report, log and audited harness are hash-bound.

## Build and memory

The builder generates private recipe-specific translation units, uses the
production helper headers and pins every compiler dependency. It refuses changed
source anchors or stale generated/compiled units. The initial build attempt
failed because the generated units lacked the production helper include path;
it was preserved and superseded by a fresh build.

The position-table argument and its single runtime-patched transfer are
unchanged. An earlier attempt appended OG to the same buffer, but `attnpos`
correctly rejected its second position-table transfer before any dispatch.
The corrected ABI gives OG its own argument (index 6). Keep the original
harness checks: never suppress the duplicate-transfer guard to run a probe.

Validated build: 15 cores; maximum .text **13,760 / 16,384 B**; maximum L1
allocation end **59,920 / 65,536 B**; all stacks **6144 B**. Compiler provenance
binds 858 inputs. Only the L3 weight pool (232,783,872 B) is resident, with the
native scratch/KV/ptab and separate OG buffers. CPU verification dequantizes the
four L3 output/FFN matrices; it does not load all model weights.

No production kernel, model packing, registry or installed service changes.
Three positions do not establish full-prefix or full-model precision, request
isolation or production performance acceptance.

## Measured result (2026-10-28)

All six NPU dispatches complete. The nine full native control captures match
bytewise; candidate upstream attention and KV also match. All three primary,
model-reference, conditioned-tail, output-projection, FFN-front and split-down
checks pass at unchanged tolerances.

| Position | Control primary maxabs | Candidate primary maxabs | Reduction |
|---|---:|---:|---:|
| 125 | 3.298774181e-4 | 1.516489372e-5 | 21.75× |
| 205 | 4.871181614e-4 | 6.029089626e-5 | 8.08× |
| 239 | 2.350416033e-4 | 1.911725848e-4 | 1.23× |

Candidate primary maxrel is 7.29925e-7 / 2.30097e-6 / 1.08730e-5.
Maximum conditioned-tail maxrel is 1.90043e-6. Against the retained full-model
CPU residual, candidate maxabs is 7.05719e-5 / 2.59399e-4 / 4.63128e-5; all
three improve. These two references answer different questions and must remain
separate. The earlier P239 attention-output regression remains recorded; the
complete layer residual improves in this forced-boundary test.

Separate audit recomputes all measurements and byte-checks 13 arrays at each
position (39 arrays total), binding 18 raw captures and three NPZ files.
TDD: eight initial failures, ten final new tests. Full host suite:
**2036 passed, 47 skipped**. Logs retain both unsuccessful attempts and the
successful build/preparation/run/audit under `l3-tail-*`.

Next expand the tail check to all 240 causal positions before integrating the
projection correction into a private full-ax implementation. Full-model
numerical acceptance remains FAIL; this is not a performance result.
