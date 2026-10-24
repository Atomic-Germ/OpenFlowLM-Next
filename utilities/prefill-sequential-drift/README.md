# Sequential long-prefix drift diagnosis

CPU-only causal attribution at L60/P37–39 and L33/P240–242 using the verified
expanded 257-token native run. These include the first temporal residual failure
and the shallowest failing layer, plus neighboring positions. Compare sequential
and the passing hybrid route; the retained audit already requires both hybrid
schedules to match exactly. No model acceptance or NPU dispatch occurs here.

```bash
OPENBLAS_NUM_THREADS=1 ironvenv/bin/python utilities/prefill-sequential-drift/check.py run \
  --baseline Models/qwen38-27b/prefill-expanded-cli/boundaries-native-v1 \
  --out Models/qwen38-27b/prefill-sequential-drift/run-v1
OPENBLAS_NUM_THREADS=1 ironvenv/bin/python utilities/prefill-sequential-drift/check.py audit \
  --out Models/qwen38-27b/prefill-sequential-drift/run-v1
```

Use a fresh output directory. Validate the complete retained audit, preparation,
source/build/model identity and all native output hashes at all three lengths.
Remeasure the selected length from raw captures before interpreting it. Keep
its completed overall numerical FAIL; the diagnostic does not upgrade it.

Decode and cache effective Q4NX weights for one layer at a time. Replay every
causal row from zero conv/recurrent/KV state separately for independent-reference
inputs and each route's native inputs. Compute FFN only at selected positions:
other FFN outputs do not update the same layer's state, and preceding-layer
residuals already supply its future inputs. First reproduce every selected
independent reference bytewise after its original FP32 output rounding.

At each selected position also evaluate its native input on a copy of the
reference trajectory's state immediately before that position. State copies
are essential because the independent oracle mutates recurrent state. This
counterfactual must never alter the reference or another selected row.

Each NPZ contains native output, ideal output driven by native inputs/history,
reference-history output with the current native input, unrounded reference
output, stored reference, input vectors and these signed components:

- `reference_rounding`: unrounded CPU reference minus stored FP32 reference.
- `current_input`: reference-history output with native input minus unrounded reference.
- `input_history`: ideal native-history output minus the reference-history counterfactual.
- `implementation_gap`: native output minus ideal native-history output.
- `propagated`: current-input plus input-history effects.
- `total`: native output minus stored independent reference.

The first four components must sum to total within FP64 rounding error. All
output errors use the same reference scale; their scalar maxima cannot be
added because coordinates differ and signed errors may cancel. Input error
uses its own input scale and is reported separately. Zero-reference/nonzero
errors remain explicit JSON-safe FAIL values.

The implementation gap includes local arithmetic and the native state built
up over the prefix. Actual native sublayer/state captures are not available
in these files: do not attribute the gap to a particular kernel or treat the
ideal-history state as native state. This is not a corrected model execution.

The CPU audit rereads provenance and native captures, replays the oracle again,
requires byte-exact agreement for every saved array and recomputes all metrics.
Exit zero means the diagnostic completed, not that selected numerical gates
or the model passed. Preserve incomplete reports and never overwrite an
existing run. `model_acceptance` remains false in both run and audit.

Tests cover conv/recurrent/KV carry, omitted output positions, independent
history copies, reset between routes, signed cancellation/reference rounding,
invalid/nonfinite inputs, complete case inventory and persisted failures.

## Validated checkpoint — 2026-10-24

`run-v1` and its independent causal CPU re-audit complete: 6222 bound input
files, 6/6 exact reference reconstructions and 12 byte-exact NPZ replays.
Peak decoded-weight cache is 1.42578125 GiB. Twelve RED-to-GREEN tests; host
suite **1756 passed / 47 skipped**. Original native captures are unchanged.

| Sequential target | Current input | Input history | Implementation gap | Total maxrel |
|---|---:|---:|---:|---:|
| L33/P241 | 0.0050773486 | 0.0000522907 | 0.0000162030 | 0.0050229750 |
| L60/P38 | 0.0076863502 | 0.0000208493 | 0.0013556628 | 0.0090391466 |

Reference-rounding components are zero in these fixtures. Propagated maxrel is
0.0050250579 / 0.0076834838: the independent layer on native causal inputs
still fails at both crossings. Current input is the largest component, while
L60 also has a measurable implementation gap. Scalars must not be summed.
Sequential passes 4/6 selected output gates; hybrid passes 6/6 with worst
maxrel 1.8360597e-5 and implementation gap <=4.384963e-7.

Follow incoming drift at L32/P241 and L59/P38, and examine L0/P241, where
retained sequential residual error is already about 0.0012928 on the common
embedding input. Use that evidence to choose actual native sublayer/state
captures. Overall long-prefix sequential accuracy remains FAIL; these results
do not prove a particular arithmetic or state operation is responsible.
