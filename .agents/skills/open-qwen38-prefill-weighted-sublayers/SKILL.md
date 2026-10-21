---
name: open-qwen38-prefill-weighted-sublayers
description: Capture weighted-candidate L55/P7 and L51/P17 sublayers, require exact native baselines and separate propagated drift from conditioned FFN precision.
---

# Weighted FFN causal sublayers

Read [the runbook](../../../utilities/prefill-weighted-sublayers/README.md)
before building. Wrap the unchanged private weighted CLI builder and existing
sublayer patcher; redirect only the capture header to targets L55/P7 and L51/P17.
Keep the validated four-product up/gate and original down arithmetic intact.
Record source/staged/binary/cache and all compiler dependencies; refuse existing
build directories before mutating evidence. No new XCLBIN or production edits.

Use the retained weighted v2 candidate as the exact baseline. Run all three
routes serially with at least 35 GiB available RAM and exclusive NPU access.
Require 66 exact logits, 192 exact residual files, 26 complete selected captures
and 12 identical hybrid pairs. Native state remains causal and independent.

Reuse independent FP64 causal and conditioned-FFN helpers; reconstruct selected
CPU outputs bytewise before interpreting metrics. Keep global error, local FFN
error and ideal-FFN diagnostics separate. An ideal CPU state driven by captured
native inputs is diagnostic only. Save provenance and failures; do not promote
conditioned FFN PASS to model acceptance.

Validated 2026-10-21: GCC 15.2.0, CTest 4/4, 587 compiler dependencies;
CLI hash and commands are in the runbook. Three resident routes complete,
66 logits and 192 residual files are exact; all 26 captures and 12 hybrid pairs
validate. Both selected independent CPU outputs reconstruct retained refs
bytewise. All 36 conditioned FFN gates pass, worst 0.0020749792.

Hybrid pre-FFN already fails: 0.0061193443 (L55/P7), 0.0092505118 (L51/P17).
Ideal FFN on actual inputs still fails output at 0.0060717536 / 0.0138246489,
versus native 0.0062761194 / 0.0138594086. Local FFN output differences are
0.0002031324 / 0.0003275429. Ideal causal state driven by native inputs differs
from native pre-FFN by 0.0006651352 / 0.0004594719. This points to propagated
input/history drift, not a unique faulty earlier operation. Sequential selected
outputs pass; global residual accuracy remains FAIL (3974/4032 passing).

The first audit rejected selected residual captures as extra layer residuals.
Exclude only exact declared capture names from layer inventory; never ignore
arbitrary prefixes. Preserve inventory-failure.json and native-audit.log; the
corrected CPU-only re-audit uses the same native captures. Eight RED-to-GREEN
tests; host suite 1558 passed, 47 skipped. Evidence is under
Models/qwen38-27b/prefill-weighted-sublayers/.

Next scan preceding layers using retained native/independent causal prefixes
without a new resident NPU run. Keep ideal-history diagnostics explicitly
separate from primary acceptance and actual native state.
