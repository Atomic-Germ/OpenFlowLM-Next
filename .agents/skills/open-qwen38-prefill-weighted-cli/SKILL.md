---
name: open-qwen38-prefill-weighted-cli
description: Build private paired-input control and paired-weight FFN CLIs with residual tracing, then validate fresh causal propagation against retained independent references.
---

# Private paired-weight FFN integration

Read [the runbook](../../../utilities/prefill-weighted-cli/README.md) before
building. Copy standalone sources into fresh directories. Reuse the existing
paired-input wrapper and serial residual tracer; leave the original single
GEMM body untouched. Control has paired inputs only. Candidate uses the low
weight kernel only for unsplit dense FFN up/gate N34816/K5120/T256, with matching
manifest role/args and FF17408. Reject stage-major and incompatible geometry.

Use separate high/low input workspaces and copy high-weight output before low
weight dispatch. Sum in FP32 host memory, never mapped device output. Give the
low kernel its own context; pin its absolute paths, build and binary hashes.
Reuse validated prefill-paired-weights/v2/low. No production kernel/ABI changes.

Record staged/source hashes, all CLI compiler dependencies, cache, CLI binary,
commands, compiler version and native CTest results. Before full-depth execution,
require 35 GiB available RAM and exclusive NPU use. Run processes serially with
fresh retained independent reference copies. Require exact control logits,
unchanged sequential outputs, hybrid schedule identity and original numerical
gates. Preserve failed reports and never promote a short-prompt pass to
long-prefix model acceptance.

Validated 2026-10-21 with fresh v2 control/candidate builds, GCC 15.2.0,
CTest 4/4 each and 585/586 compiler dependencies. XRT's xclbin constructors
are ambiguous for a string literal: use std::string{path}. Preserve the failed
first candidate build; use v2 evidence for reuse. CLI hashes are in the runbook.

Six resident short-prompt runs complete (64 layers, 20 prompt tokens and one
continuation, three routes per variant). Control logits reproduce 66/66
baseline files; candidate sequential logits/scratch/residuals are exact in
87/87 checks. Both hybrid schedules match at all logits and layer residuals.
Hybrid minimum correlation improves to 0.9999742814 and continuation residual
to 0.0004262400; all 63 logit gates pass.

However candidate intermediate residuals pass only 3974/4032 gates: 29 failures
remain per hybrid route (control 45). First L55/P7 is 0.0062761194; worst L51/P17
is 0.0138594086, above the unchanged 0.005 limit. The aggregate audit must return
1 with capture_valid=true and passed=false. This is expected numerical failure,
not an execution failure. Sequential residuals pass 1344/1344 unchanged.

Five new RED-to-GREEN tests; full host suite 1550 passed, 47 skipped. Evidence:
Models/qwen38-27b/prefill-weighted-cli/v2/. Next capture conditioned sublayers at
L55/P7 and L51/P17 with exact baseline identity before longer propagation tests
or production promotion. All production sources and kernel binaries remain
unchanged.
