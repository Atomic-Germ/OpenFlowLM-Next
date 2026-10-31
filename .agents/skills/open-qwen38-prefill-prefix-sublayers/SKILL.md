---
name: open-qwen38-prefill-prefix-sublayers
description: Capture early Qwen3.8 drift boundaries at L34/P17, L37/P7 and L53/P17 with full projection prefixes, actual block state and exact weighted native controls.
---

# Early causal prefill boundaries

Read [the runbook](../../../utilities/prefill-prefix-sublayers/README.md) before
building. Compose the existing weighted CLI and sublayer patchers in a private
copy. Redirect targets and add read-only captures; do not edit pinned production
sources or change the four-product FFN arithmetic. Record compiler dependencies,
source/staged/cache/CLI and reused kernel hashes. Refuse existing build/output
directories. Register this workflow before the first native build.

Use >=35 GiB available RAM and exclusive NPU access. Three serial routes run
the retained 20-token prompt plus independent continuation. Require 66 baseline
logits, 192 residual files, 81 captures and 39 hybrid pairs byte-exact. Capture
all 20 real rows of norm/QKV/Z/og and complete raw state around the host block.
State-after is P19, not the selected row; refuse another prefix/block geometry.

Reconstruct all three selected independent CPU outputs bytewise. Verify zero
initial state, exact conv carry, state padding and selected residual reconstruction.
Compare original-input and captured-norm projections separately. Use captured
projection prefixes with independent FP64 causal history for conditioned host
og/state checks, never as replacement model references. Report FFN/global
propagation separately. Preserve failed audits and numerical gates; CPU re-audit
must reuse native captures. Full model/long-prefix acceptance remains open.

Validated 2026-10-21: GCC 15.2.0, CTest 4/4, 587 compiler dependencies.
CLI SHA256 is in the runbook. Three native routes complete; all 66 logits,
192 residual files, 81 captures and 39 hybrid pairs satisfy exact controls.
Three independent causal CPU references reproduce bytewise. Conv carry and
pre-FFN residual reconstruction are exact; state-after is explicitly P19.

All 54 FFN and 732 projection/host gates pass. Worst conditioned og/state
maxrel is 1.4542305e-6 / 9.2116065e-7; aggregate QKV worst 0.0022380252.
QKV uses a whole-vector scale, not separate Q/K/V or per-head acceptance.
Global L53/P17 still fails: pre-FFN 0.0075342533, output 0.0078214144,
ideal FFN on actual input 0.0079466232. Full residual acceptance remains
3974/4032 FAIL. No arithmetic change or new XCLBIN. Ten RED-to-GREEN tests;
full host suite 1581 passed, 47 skipped. Artifacts are under
Models/qwen38-27b/prefill-prefix-sublayers/.

Next perform CPU attribution of effective versus BF16-rounded QKV/Z/out
weights on the captured complete prefixes. Separate Q/K/V and head scales;
keep conditioned host history distinct from primary references. Test the
rounding hypothesis against actual captures before selecting a new NPU replay.
No resident run is required for that attribution step.

CPU attribution follow-up (2026-10-21): use
[the rounding runbook](../../../utilities/prefill-projection-attribution/README.md).
No build/NPU execution. Nine manifest-packed/source weight identities pass;
7683 conditioned projection comparisons preserve individual Q/K/V/Z head scales.
Two failures emerge: Q L34/P6/H13 0.0050724904 and V L37/P0/H46 0.0071224278.
BF16-weight predictions fit within 2.9830263e-6 maxrel, with remaining L2
error <=0.2262% of native-minus-effective-weight error. This strongly supports
weight rounding but is not byte-exact native replay. Three signed causal
residual decompositions close; joint ideal suffix still leaves L53/P17 pre-FFN
at 0.0066083261 FAIL. Its largest local component is out projection 0.0008264166.
Eleven RED-to-GREEN tests; host suite 1592 passed, 47 skipped. Evidence:
Models/qwen38-27b/prefill-projection-attribution/.

Next isolate the fused entry N16384/K5120/T256 on L34/P6 and L37/P0, using
rows from the existing L34/P17 and L37/P7 prefix files. Require byte-exact
paired-input native controls before a private paired-weight kernel, unchanged
per-head gates, exact packing and two placements with zero inactive columns.
No full model residency is needed. Keep L53/P17 out as a follow-up precision
target; aggregate/isolated PASS cannot replace full-depth or long-prefix gates.
