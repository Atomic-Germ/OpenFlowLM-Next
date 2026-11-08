---
name: open-qwen38-prefill-combined-validation
description: Validate combined Qwen3.8 prefill with fresh exact paired/weighted controls, independent all-layer gates, and separate repeated resident captures after Core::reset.
---

# Combined prefill native validation

Start with the prepared triple of CLIs documented in
[the runbook](../../../utilities/prefill-combined-cli/README.md).
Use run.py run/audit with prepared-v1; keep references and historical source
metadata byte-identical. Verify recorded historical/current identities and
native build/compiler dependencies. No historical exception for artifacts.

Require >=35 GiB available RAM and exclusive /dev/accel/accel0. Run processes
serially under the shared resident-validation lock. Compare paired control
against retained paired logits/scratch, weighted against retained weighted
logits/scratch/all residuals, and sequential captures across fresh variants.
Never run combined before both controls match. Require all 64-layer residual
and independent logit gates; numerical failure is a valid completed experiment,
not permission to weaken limits. Keep model_acceptance=false.

For repeat validation, repeat.py build copies the verified combined stage to a
fresh directory and changes only CLI capture plumbing. Require --twice; keep
request() and its Core::reset() unchanged. Use separate residual writers and
logit/scratch prefixes per request. Pin the parent build, all sources, compiler
dependencies, CMake cache and new executable. No new XCLBIN.

Run repeat.py run only after the single-request report has a complete independent
audit. Reproduce every original combined capture in the first request, then
compare all logits, all residuals and scratch bytewise after the second request
on the same resident engine. Extract two ordered marked log sections from the
retained raw log and independently re-audit with repeat.py audit. This tests
normal request reset, not clearing every scratch/KV buffer or independent
checkpoint/restore correctness. Preserve numerical failures even if repeats
are exact. Long-prefix and production promotion remain separate gates.

## Validated evidence (2026-10-23)

Artifacts live under `Models/qwen38-27b/prefill-combined-cli/`:
`native-v1/`, `repeat-build-v1/`, `repeat-native-v1/`.
The nine single-request runs reproduce 69/69 paired and 261/261 weighted
controls. Sequential captures match across variants; hybrid schedules match.
Combined logits pass 63/63, residuals 4000/4032: 16 failures per hybrid route,
worst 0.0121972127 against 0.005. Numerical audit completes with FAIL.
Three new failures relative to weighted FFN occur at L53/P18, L56/P17, L63/P1;
L63/P1 is 0.0096485275. Do not claim uniformly improved accuracy.

Repeat build: GCC 15.2, 591 compiler dependencies, CTest 4/4; CLI SHA256
`f2f9ca67238e44e0f73c7bcb6030087a71785a87fb3a17228fbf78002490186e`.
Three processes each run two requests. All 261 first captures equal the
single-request candidate; all 261 repeat captures equal the first request.
Independent repeat audit passes; numerical_passed and model_acceptance remain
false. Eight new RED-to-GREEN tests; host suite 1680 passed / 47 skipped.
Use the runbook's exact build/run/audit commands and fresh output directories.
Next diagnose conditioned L63/P1, L42/P17 and L52/P17 after exact native identity.
