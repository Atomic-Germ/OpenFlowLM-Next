---
name: open-qwen38-post-prefill-validation
description: Validate the standard FP32 DeltaNet post-output export over full-depth short prefill with fresh unpatched CLI controls and independent residual/logit gates.
---

# Short prefill after FP32 post-output promotion

Use the existing standard 46-set export from open-qwen38-post-output-integration.
No kernel rebuild or model repacking is needed. Historical BF16 output hashes
are not the expected results for this arithmetic. Do not rewrite old evidence.

The shared-runtime builder supports --plain, which stages byte-unmodified core
and CLI sources. Its default mode adds only synchronized per-layer residual
captures. Both build records pin sources, reconstructed allowed changes, compiler
dependencies, CMake cache and executable. --plain and --repeat are incompatible.
Build two fresh directories under Models/qwen38-27b/post-prefill-production:

```bash
ironvenv/bin/python utilities/prefill-production/build.py \
  --out Models/qwen38-27b/post-prefill-production/plain-v1 --plain --jobs 2 \
  --xrt-include /home/prihlop/sources/xdna-driver/xrt/build/Release/opt/xilinx/xrt/include \
  --xrt-lib /home/prihlop/sources/xdna-driver/xrt/build/Release/opt/xilinx/xrt/lib
# Repeat with --out .../trace-v1 and without --plain.
```

Run host tests first. On this host compilation and XRT execution need execution-
tool escalation. Existing shared runtime arithmetic must remain unchanged in the
capture build. New source changes require fresh builds, not relabelled records.

```bash
OPENBLAS_NUM_THREADS=1 ironvenv/bin/python utilities/prefill-production/post_validate.py run \
  --plain-build Models/qwen38-27b/post-prefill-production/plain-v1 \
  --trace-build Models/qwen38-27b/post-prefill-production/trace-v1 \
  --baseline Models/qwen38-27b/prefill-expanded-cli/native-v1 \
  --kernels Models/qwen38-27b/production-post-fp32/kernels-v1 \
  --out Models/qwen38-27b/post-prefill-production/short-v1
ironvenv/bin/python utilities/prefill-production/post_validate.py audit \
  --out Models/qwen38-27b/post-prefill-production/short-v1
```

The retained reference is bound to its passing audit and preparation metadata.
Require all 149 CPU reference files, exact prompt IDs, unchanged oracle source
hashes and model bytes. Recipe changes are checked against the current export;
never silently substitute old native output hashes for a CPU reference.

The runner uses installed XRT, the shared NPU validation lock, exclusive device
access and at least 35 GiB available RAM before every resident run. It executes
sequential NPU and both hybrid schedules first with the plain CLI, then with the
capture CLI. Each trace must match all plain logits and the final scratch dump
byte-for-byte. Require every residual row maxrel <0.005, every logit correlation
>0.9999 with matching argmax, exact generated tokens and the last-prompt alias.
The two hybrid schedules must reproduce all residual/logit captures exactly.
Separate audit reconstructs measurements and checks source/build/input identity.

This is a 20-token prompt plus one continuation through all 64 layers. It does
not validate full-depth long-prefix, 255/256/257-token boundaries, resident reset
for this prompt, serving isolation or performance. Hybrid host stages remain
explicit. model_acceptance stays false; instrumented timings are not benchmarks.

## Measured short-prefill checkpoint — 2026-10-26

plain-v1 and trace-v1 build with GCC 15.2, pin 589/590 compiler dependencies,
and pass native CTest 4/4 each. The model and unchanged oracle bind all 149
reference files; 257 input hashes are verified. The standard export build key
remains 67f3c4395334a0bc3aa0dcb2bcff398cfa01147451cddd3b022afca4215c7a13.

short-v1 and its separate audit pass. Across all three routes, 4032 residual
rows and 63 traced logits pass; the plain CLI independently passes 63 logits
and its continuation residuals. All 69 fresh trace/plain files (22 logits plus
scratch per route) match exactly. Both hybrid schedules agree on 86 residual
and logit files. Worst residual maxrel: sequential 0.0008187178358 at L63/P17,
hybrid 0.0000779138524 at L61/P16. Minimum correlations:
0.9999978969689 / 0.9999979146652 / 0.9999979146652. All thresholds are unchanged.
Host suite: 1899 passed, 47 skipped.

These six resident runs are single requests in separate processes. Do not claim
repeated-request/reset acceptance from them. Next validate separate captures
of repeated short requests against short-v1, then block boundaries and the
long prefix. Preserve the old BF16 reset evidence as historical data only.
