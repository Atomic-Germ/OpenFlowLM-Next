---
name: open-qwen38-prefill-expanded-cli
description: Build private combined control and expanded attention/down prefill CLIs with exact roles, pinned replay evidence and repeat capture support.
---

# Expanded prefill preparation

Read [the runbook](../../../utilities/prefill-expanded-cli/README.md).
Register this workflow before building. Build fresh combined, expanded and
expanded-repeat CLIs using the existing verified low kernels for FFN up/gate,
linear entry, output, attention entry and FFN down. No new XCLBIN or production
change is needed. Preserve the exact old combined control and the original
single-dispatch body, paired workspaces and host FP32 sum order.

Match roles and buffer arguments as well as shapes. Full and linear attention
share output geometry but must not accept one another's weight role. Reject
unsupported split, norm/activation/epsilon, state or paired-KV ABI. Hash kernel
build/generated/compiler/resource evidence before reuse, and all CLI sources,
staged sources, seven translation-unit dependencies and binaries after CTest.
Never overwrite previous directories or mutate historical evidence.

Prepare unchanged independent references only after verifying all builds,
accepted attention replay/raw captures and retained full-depth measurements.
Repeat engine arithmetic must match the single-request candidate; only CLI
request plumbing differs. Keep native_validated/model_acceptance false.

This stage requires no model residency. The next resident step needs at least
35 GiB free RAM, exclusive NPU access, exact fresh combined controls, unchanged
sequential captures, independent all-layer/logit gates, hybrid schedule identity
and two separately captured requests after Core::reset. Do not equate build,
local suffix acceptance or reset repeatability with model accuracy. Preserve
full-depth numerical failures; long-prefix and production acceptance stay open.

## Validated preparation — 2026-10-23

All three v1 builds pass CTest 4/4, with 591/592/592 compiler dependencies.
Combined control is byte-identical to the retained combined CLI. Candidate and
repeat bind the same five low kernels, with identical recursive engine source
inventories excluding cli.cpp. GCC 15.2.0; CLI hashes are in the runbook.

`Models/qwen38-27b/prefill-expanded-cli/prepared-v1/` pins 2006 inputs and
151 unchanged reference files (149 arrays plus IDs/metadata). Current manifest,
accepted replay and retained full-depth measurements revalidate. Seven
RED-to-GREEN tests; 1730 host tests pass, 47 skip. No resident run occurred.

At that preparation checkpoint the runner/audit still needed adaptation to
the new combined/expanded names and the separate repeat build: the old three-variant runner cannot consume this preparation
directly. About 22 GiB RAM was available at preparation time; verify at least
35 GiB before loading all layers. Full-depth 4000/4032 remains FAIL and both
native_validated/model_acceptance remain false until the required native gates.

## Expanded resident runner — 2026-10-24

Use `utilities/prefill-expanded-cli/run.py preflight/run/audit` and
`repeat.py run/audit` as documented in the runbook. The runner now handles the
new variant inventory directly. It revalidates immutable preparation/builds,
requires exact fresh combined captures before expanded, and requires candidate
sequential identity plus independent all-layer/logit and hybrid schedule gates.

The repeat runner requires a complete single-request CPU audit, checks first
captures against expanded, second against first, and independently scores both
requests. Preserve raw marked logs and separate numerical_passed from identity.
Reports bind exact commands, CLI/source/preparation hashes and never promote
model_acceptance. Recheck RAM and exclusive device ownership before each
process; an earlier free-memory reading is not sufficient.

Seven RED-to-GREEN runner tests; 1737 host tests pass, 47 skip. The first
preflight validates evidence but reports only 24.3 GiB available RAM. Do not
interpret preflight completion as native accuracy or memory readiness.

## Resident acceptance — 2026-10-24

`native-v1` and its independent CPU audit pass **4032/4032 residual gates and
63/63 logits** for expanded (64 layers, 20 prompt tokens plus continuation).
Combined reproduces all 261 retained captures and preserves 4000/4032 FAIL.
Expanded sequential matches 87 captures; hybrid schedules match 86 logit/residual
files. Hybrid worst residual maxrel improves 0.0121972127 to 0.0012128028,
minimum correlation 0.9999979147. Sequential worst remains 0.0048820188. Gates
and continuation IDs `[17,478]` are unchanged.

`repeat-native-v1` and its separate CPU audit pass: 261 first captures match the
single-request baseline; 261 second captures match the first after normal reset.
Both requests independently pass 4032 residual and 63 logit gates. Raw marked
logs and extracted logs are hash-bound. This does not establish checkpoint or
full-buffer-clearing correctness. Reports keep model_acceptance false.

RAM reached about 61 GiB before native execution; retain the earlier failed
memory preflight. No new libraries or kernels were built for this validation.
Next validate expanded 255/256/257-token full-depth propagation with retained
independent references, fresh exact controls and causal prefix/schedule identity.
Do not infer long-prefix or production acceptance from this short-prefix PASS.

## Long-prefix workflow

Use `boundaries.py prepare/run/audit` from the runbook. Reuse accepted short-run
builds and immutable 255/256/257 references; no new binaries are needed. The
control is the fresh paired-input CLI from the combined preparation, so its
long-prefix logits and final activation captures can be checked against retained
paired-input native runs before expanded starts. Do not substitute the combined
short-prompt control: it has no retained long-prefix baseline.

Preparation validates historical/current source identity, all reference hashes,
finite sizes, CPU continuation selection and causal reference prefixes. Verify
all 64-layer geometry fields against the accepted short run. Native validation
runs 18 serial processes, with exclusive NPU access and >=35 GiB RAM checked
before each. Compare expanded sequential to fresh paired captures, all hybrid
schedule files, and cross-length logit/residual prefixes excluding continuation.

Numerical failures remain completed FAIL results, not runner errors. Audit raw
captures independently and preserve failed experiments. Short-reference metadata
and previous preparations must never be edited to make long-prefix reuse pass.
Long-prefix PASS alone does not establish production or serving acceptance.

### Long-prefix result — 2026-10-24

`boundaries-prepared-v1` verifies 1155 references, 3504 inputs and 172 historical
entries. `boundaries-native-v1` completes all 18 processes and its independent
CPU re-audit. Both exit 1 for completed numerical failure, not process error.
Expanded hybrid passes 98,688/98,688 residual gates and 1542/1542 logits, worst
maxrel 0.0025616438, minimum correlation 0.9999961799, no argmax mismatch.
All 2331 historical controls, 969 sequential captures, 1917 causal prefixes per
variant and 966 expanded hybrid schedule files match exactly.

Sequential remains FAIL: 46,357/49,344 residuals and 550/771 logits pass. First
temporal residual crossing L60/P38; shallowest failing layer L33/P241; first
logit failure P80. Worst residual 0.5389499720 at L58/P220, minimum correlation
0.9351669042; argmax still matches. These crossings do not identify a faulty
operation. Next diagnose causal sequential state/arithmetic using saved traces.
Keep overall model_acceptance false and long-prefix repeat/production open.
Seven new RED-to-GREEN tests; 1744 host tests pass, 47 skip. No new builds.
