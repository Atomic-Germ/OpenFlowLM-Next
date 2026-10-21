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
