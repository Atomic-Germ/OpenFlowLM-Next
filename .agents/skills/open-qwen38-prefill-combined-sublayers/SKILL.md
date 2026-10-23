---
name: open-qwen38-prefill-combined-sublayers
description: Capture combined Qwen3.8 L63/P1 attention and L42/P17, L52/P17 DeltaNet boundaries with exact native controls and conditioned independent audits.
---

# Combined causal sublayer diagnosis

Read [the runbook](../../../utilities/prefill-combined-sublayers/README.md).
Copy the verified combined private stage; change only read-only capture plumbing.
Pin the parent, helper/source/staged/compiler dependencies and reused kernels.
Keep the original GEMM wrapper, role guards and host arithmetic. Register this
workflow before building; refuse existing output directories.

Run three routes serially under the shared resident lock, with >=35 GiB RAM
available and exclusive NPU access. Require all 261 baseline files byte-exact,
83 complete captures and 40 hybrid pairs before interpreting the audit.
L63/P1 uses full attention; L42/P17 and L52/P17 use DeltaNet. Capture complete
20-token projection prefixes and block-end state at P19, not a synthetic state
at the selected position. Paired KV includes only the 20 initialized real rows.

Reconstruct selected independent CPU outputs bytewise. Separate conditioned
FFN, original-input/captured-norm projections, per-head gates, actual conv/state
and captured-prefix FP64 host diagnostics from global residual accuracy.
Check pre-FFN residual and layer output reconstruction. Never substitute
conditioned history for primary references or relax bounds. Preserve numerical
failures even with exact controls. Reuse captures for CPU-only re-audits;
no new resident run merely to fix a checker. Long-prefix acceptance and
production promotion remain separate gates.

## Validated checkpoint (2026-10-23)

GCC 15.2, CTest 4/4, 592 compiler dependencies. CLI SHA256:
`e6e53ce672ff447655cb0345736a2b6e10a9dcbdcdb980c9e77b47a4b57aaa20`.
Artifacts: `Models/qwen38-27b/prefill-combined-sublayers/`. All three routes
match 261 baseline files; 83 captures and 40 hybrid pairs validate. Three
independent CPU outputs reconstruct bytewise. 54/54 FFN and 30654/30654
projection/host gates pass (29760 per-head). Full-depth remains 4000/4032 FAIL.

Use `native-v1/sublayer-audit-v2.json`; the first failed audit is retained.
Convert ModelSpec's RoPE frequency list to np.float64 before replica_block:
Python list multiplication by position 0 empties it. Regression coverage also
checks causal independence from future values. Do not rerun NPU for this fix.
Ten new tests; full host suite 1690 passed / 47 skipped.

Ideal FFN on actual pre-FFN inputs still fails at all targets: L63/P1
0.00702581, L42/P17 0.00521072, L52/P17 0.01234878. L52's incoming residual
already fails. L63 down-from-H differs by 0.00398053, up/gate by ~1–2e-6.
Next compare effective/BF16 weight predictions for uncorrected FFN down and
full-attention projections, keeping per-head and signed error decompositions.
This is a hypothesis to test, not a proven cause or permission to widen policy.
Read weights per layer; no resident model needed for that CPU attribution.
