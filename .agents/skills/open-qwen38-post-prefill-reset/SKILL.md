---
name: open-qwen38-post-prefill-reset
description: Build a capture-only repeated-request CLI and audit ordinary resident reset against the fresh FP32 post-output short-prefill baseline.
---

# Resident short-prefill reset on the standard FP32 export

Prerequisite: a passing `post_validate.py` run and separate audit, using the
standard 46-set FP32 post-output export. Follow open-qwen38-post-prefill-validation
first. Do not substitute historical BF16 output hashes or edit old evidence.
No kernel rebuild or model repacking is required.

Run host tests, then build a fresh capture CLI. On this host compiler and XRT
commands require execution-tool escalation. The builder adds synchronized
residual captures and separate repeat filenames, preserving shared arithmetic
and the body of `request()` (including `Core::reset()`). It pins current sources,
compiler dependencies, staged instrumentation, CMake cache and executable.

```bash
ironvenv/bin/python utilities/prefill-production/build.py \
  --out Models/qwen38-27b/post-prefill-production/repeat-v1 --repeat --jobs 2 \
  --xrt-include /home/prihlop/sources/xdna-driver/xrt/build/Release/opt/xilinx/xrt/include \
  --xrt-lib /home/prihlop/sources/xdna-driver/xrt/build/Release/opt/xilinx/xrt/lib
OPENBLAS_NUM_THREADS=1 ironvenv/bin/python utilities/prefill-production/post_reset.py run \
  --build Models/qwen38-27b/post-prefill-production/repeat-v1 \
  --baseline Models/qwen38-27b/post-prefill-production/short-v1 \
  --out Models/qwen38-27b/post-prefill-production/reset-v1
OPENBLAS_NUM_THREADS=1 ironvenv/bin/python utilities/prefill-production/post_reset.py audit \
  --out Models/qwen38-27b/post-prefill-production/reset-v1
```

Use a new output directory for every build/run. Retain failed evidence. Baseline
report and audit hashes are bound before execution; current source/build/model,
reference and export identities are rechecked. Baseline measurements are freshly
recomputed, including plain/trace and hybrid schedule identity. No old build
source exemptions or record relabelling are permitted.

Each of the three routes (sequential, hybrid layer-major, hybrid block-major)
loads one model and runs two identical 20-token prompts plus continuation.
The runner takes the shared resident-validation lock, requires exclusive NPU
access and at least 35 GiB available RAM before each process. Do not run other
NPU jobs or CPU weight-loading checks alongside it. Each process has a 30-minute
timeout. Request markers and separately named files bind first and second runs.

Require all 87 first-request files per route to reproduce the audited baseline
and all 87 second-request files to reproduce the first. Independently score
both requests: residual maxrel <0.005, logit correlation >0.9999 with matching
argmax, expected generated tokens and exact last-prompt alias. All 64 layers
and 21 positions must pass. The two hybrid schedules must agree on all 86
residual/logit files; final scratch identity is required within each route.
A separate audit reconstructs measurements and invalidates stale success on
failure. Keep `model_acceptance: false`.

This covers ordinary reset for repeated identical short prompts, with the
existing checkpoint/restore before request two. It does not establish full-buffer
zeroing, different-prompt isolation, 255/256/257-token boundaries, long-prefix
precision, serving or performance acceptance. Hybrid host stages remain explicit;
capture timings are not production benchmarks. No installed bundle is replaced.

## Measured checkpoint — 2026-10-26

`repeat-v1` uses GCC 15.2, pins 590 compiler dependencies and passes CTest 4/4.
`reset-v1` and its separate audit pass: 8064/8064 residual rows, 126/126 logits,
261/261 baseline identities, 261/261 repeat identities and 86/86 hybrid
schedule identities. Worst residuals are unchanged between requests:
sequential 0.0008187178358 (L63/P17), hybrid 0.0000779138524 (L61/P16).
Minimum logit correlation is 0.9999978969689. Generated IDs are 17, 478 for
every request. Host suite: 1912 passed, 47 skipped. See `reset-summary.json`;
next validate block boundaries and long-prefix propagation on this export.
