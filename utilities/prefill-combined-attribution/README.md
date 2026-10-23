# Combined prefill weight-rounding attribution

CPU-only analysis of retained combined sublayers at L63/P1, L42/P17 and
L52/P17. No model residency, kernel rebuild or production edit. Read weights
one layer at a time and reuse the previously audited native captures.

```sh
OPENBLAS_NUM_THREADS=1 ironvenv/bin/python utilities/prefill-combined-attribution/check.py \
  --run Models/qwen38-27b/prefill-combined-sublayers/native-v1 \
  --audit Models/qwen38-27b/prefill-combined-sublayers/native-v1/sublayer-audit-v2.json \
  --out Models/qwen38-27b/prefill-combined-attribution/v1
```

Require unchanged audited sources, build dependencies, prepared model/reference
provenance, all baseline outputs and all 83 sublayer captures. Verify 40 hybrid
pairs before using hybrid as representative of both schedules. Refuse existing
output directories; persist incomplete failure reports. Do not rewrite historical
reference metadata or weaken accuracy bounds.

Dequantize each manifest packing operation and require equality with the source
reader's effective weights. Attention Q and gate share a tensor: use `chunk0`
to select their distinct row ranges. Check all eight operations (three FFN down,
Q/K/V/gate and attention output), retaining packed/effective hashes.

Compare three FP64 predictions: FP32 input with effective weights, paired-BF16
input with effective weights, and paired-BF16 input with BF16-rounded weights.
Retain separate per-head reference scales. A close fit of the last prediction
to native output supports weight-rounding attribution; it is not byte-exact
replay of native accumulation. Report native error and prediction fit separately.

Reconstruct three selected independent CPU outputs bytewise. Preserve signed
vectors in NPZ files: propagated error, FFN front, paired input, weight rounding,
remaining native difference and residual addition. Their vector sum must close
against the full error; magnitudes do not add. For L63 also decompose the
pre-FFN residual across entry, host attention and output projection. Causal
substitutions use only positions 0..P, fresh paired KV and FP32 projection/og
storage boundaries. Evaluate one-role and joint ideal attention substitutions
followed by ideal FFN. These conditioned suffixes are diagnostics, never
replacement model references or evidence of fresh full-depth acceptance.

## Measured results (2026-10-23)

All **8/8 manifest/source identities**, **1124 projection/head comparisons**,
three independent CPU output reconstructions and **4 signed decompositions**
complete. Every local native projection passes the original 0.005 bound.
Full-depth acceptance remains **4000/4032 FAIL** from the unchanged native run.

| Role | Comparisons | Worst native maxrel | Worst BF16 prediction gap | Worst remaining L2 error fraction |
|---|---:|---:|---:|---:|
| Q | 480 | 0.0023476238 | 2.8980e-6 | 0.2925% |
| K | 80 | 0.0018736759 | 2.4349e-6 | 0.2321% |
| V | 80 | 0.0010985654 | 3.0213e-6 | 0.3126% |
| Gate | 480 | 0.0015700262 | 2.4409e-6 | 0.1506% |
| Attention out | 1 | 0.0006512213 | 1.7816e-6 | 0.0723% |
| FFN down | 3 | 0.0039805268 | 3.3603e-6 | 0.1205% |

The remaining L2 fraction is the norm of native-minus-rounded prediction divided
by native-minus-effective prediction, per head/position. It is not a percentage
of full-model error explained.

L63/P1 output error **0.0096485275** includes propagated/ideal-FFN error
**0.0070258073** and down-weight rounding component **0.0026236663**, measured
on the same independent output scale. The remaining down implementation
component is **2.2129e-6**. L42/P17 and L52/P17 weight-rounding components are
**0.0005789814 / 0.0002769049**; ideal FFN still leaves **0.0052107169 /
0.0123487787**. Fixing down alone therefore cannot close these retained errors.

At L63/P1, joint ideal Q/K/V/gate, attention output and FFN on the actual
incoming prefix gives **0.0040534510**. Replacing gate alone while using ideal
attention output and FFN gives **0.0049558575**; Q/K/V substitutions separately
remain **0.0068323553 / 0.0070201505 / 0.0059709254**. These experiments preserve
the accumulated incoming error and do not establish causal propagation after
correction across the full model.

Next isolate paired-weight FFN down at N5120/K17408/T256 on all three captured
H rows. Require a byte-exact current native control, two slot placements,
inactive-column checks, unchanged conditioned gates and composed residual gates.
Then validate full-attention fused entry N14336/K5120 and output N5120/K6144
with exact native controls and causal prefixes before expanding the combined
CLI policy. Neither isolated acceptance nor this CPU result permits production
promotion or closes long-prefix failures.

Nine new RED-to-GREEN tests; full host suite **1699 passed / 47 skipped**.
Evidence: `Models/qwen38-27b/prefill-combined-attribution/v1/`.
