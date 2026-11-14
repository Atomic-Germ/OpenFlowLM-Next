---
name: open-qwen38-prefill-production-validation
description: Build capture-only shared-runtime CLIs and audit all-layer paired-prefill precision and separate resident reset captures against the standard manifest-v4 export.
---

# Integrated paired-prefill all-layer and reset validation

This workflow requires unchanged arithmetic and the recorded BF16 controls.
For the standard FP32 post-output export, use
[post-prefill validation](../open-qwen38-post-prefill-validation/SKILL.md),
which runs fresh unpatched controls against the retained independent CPU oracle.

Use the existing 46-set standard export and a passing unpatched-CLI smoke from
open-qwen38-paired-prefill-production. Do not patch kernel manifests or insert
private paired/weighted projection implementations into Core.

`utilities/prefill-production/build.py` stages current shared sources and applies
only the existing residual trace patch (serial completion/readback) and optional
repeat capture plumbing. `request()` and Core::reset() stay unchanged. Build two
fresh directories, with and without --repeat; keep the normal runtime untouched.
Record original/staged source hashes, compiler dependencies, CMake cache and CLI
hash. `verify` reconstructs allowed instrumentation from current shared sources.

```bash
OPENBLAS_NUM_THREADS=1 ironvenv/bin/python -m pytest specs/open-engine/tests -q
ironvenv/bin/python utilities/prefill-production/build.py \
  --out Models/qwen38-27b/prefill-production/trace-v1 \
  --xrt-include /home/prihlop/sources/xdna-driver/xrt/build/Release/opt/xilinx/xrt/include \
  --xrt-lib /home/prihlop/sources/xdna-driver/xrt/build/Release/opt/xilinx/xrt/lib
# Repeat with a new repeat-v1 output directory and --repeat.
```

Use installed XRT for execution, not the development library directory. The
runner sets that environment, takes the shared resident-validation lock, checks
exclusive device access and requires >=35 GiB available RAM before each run.
No new xclbins or weight packing are needed.

Run `validate.py run` with --build trace-v1, --smoke smoke-v2,
--baseline Models/qwen38-27b/prefill-expanded-cli/native-v1,
--kernels kernels-model-v1 and a fresh --out all-layer-v1 (short names here are
relative to Models/qwen38-27b/prefill-production). Then `validate.py audit --out`.
Only after that passes, use --build repeat-v1, --repeat-baseline all-layer-v1,
the same fixture arguments and a fresh --out reset-v1. Audit separately again.

Both flows cover sequential, layer-major hybrid and block-major hybrid, all
64 layers, 20 prompt tokens and one continuation. First reproduce every retained
capture, then enforce independent per-layer residual maxrel <0.005, logit
correlation >0.9999 and matching argmax. Both hybrid schedules must agree.
Repeated requests use distinct writers/logit/scratch files on one resident Core;
first request must match audited single capture, second must match first. Parse
ordered request markers and preserve raw logs. Missing/extra files, wrong
shapes, non-finite values, changed inputs and inaccurate numerical results fail.

Historical reference metadata is bound to its audited preparation; model and
artifact hashes have no historical exception. Historical sources are not rerun
or rewritten. Current manifest semantics must match the model-derived recipe.
Keep model_acceptance=false: this is a short-prompt integration gate, not
long-prefix, complete buffer zeroing, independent checkpoint/restore or serving
acceptance. Hybrid recurrence/norms still run on the host.

## Build, single-request and resident-reset evidence, 2026-10-26

trace-v1 and repeat-v1 compile with GCC 15.2; each records 590 dependencies and
passes CTest 4/4. All-layer-v1 and its separate audit pass: 4032/4032 residual
rows, 63/63 logits and 261/261 exact historical captures. The two hybrid
schedules agree on 86/86 residual/logit files. Worst residual maxrel is
0.0048820188 for sequential (L63/P18) and 0.0012128028 for hybrid (L0/P20);
minimum correlations are 0.9999943778 and 0.9999979147 respectively. Bounds
remain 0.005 and 0.9999. The sequential short case is close to the residual
bound; it does not remove the known long-prefix failure.

Nine new RED-to-GREEN tests pass; full suite: 1857 passed, 47 skipped.
No production arithmetic, model bytes or existing kernel export was changed.

Reset-v1 and its independent audit pass: 261/261 first-request captures match
the single-request baseline, and 261/261 separate repeat captures match the
first request. Across all six requests, 8064/8064 residual rows and 126/126
independent logits pass. Do not use serialized capture timings as production
benchmarks; long-prefix and performance acceptance remain open.
