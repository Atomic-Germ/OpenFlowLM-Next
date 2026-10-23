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

Next adapt runner/audit to the new combined/expanded names and the separate
repeat build: the old three-variant runner cannot consume this preparation
directly. About 22 GiB RAM was available at preparation time; verify at least
35 GiB before loading all layers. Full-depth 4000/4032 remains FAIL and both
native_validated/model_acceptance remain false until the required native gates.
