---
name: open-qwen38-post-prefill-validation
description: Validate the standard FP32 DeltaNet post-output export over full-depth short prefill with fresh unpatched CLI controls and independent residual/logit gates.
---

# Short prefill after FP32 post-output promotion

The [final model report](../../../specs/open-engine/reports/qwen38-27b.md)
summarizes production validation, measured performance and context limits.

## Profiling controls

For timing-only investigations, use `utilities/prefill-block/layer_profile.py`
and `profile_overhead.py`; see the [profiling report](../../../specs/open-engine/reports/qwen38-npu-performance.md).
Force `OFLM_OPEN_LAYER_MAJOR=1` for a layer-major experiment and require
`gemm-prompt` in the observed profile. Unsetting the override selects the dense
block-loop default. Validate complete layer groups for every request.

Per-token profile records use the neutral label `step`: `want_logits` is not a
phase indicator. The CLI closes the prefill dispatch trace before decode. An
ON overhead measurement must cover every measured decode step, excluding the
warm-up pairs. Compare parent-OFF/new-OFF separately from new-OFF/new-ON; the
latter cannot establish disabled-path regression. Use at least five measured
pairs, retain artifact hashes, conditions, full logs and generated-token checks.
Treat a failed CLI power-mode configuration as a failed benchmark setup. Use
`--pmode none` only to intentionally retain a mode observed externally; record
that mode and do not label it as successfully configured performance mode.
Submit-to-completion includes queue residence and overlapping intervals; it
does not measure device execution or NPU idle time. Profiling validation alone
does not replace the independent numerical acceptance gates below.

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


## Shared block decode checker

The numerical block/decode checks now share
[decode_transition.py](../../../utilities/prefill-block/decode_transition.py).
Select `--check state` for exact greedy-token equality plus final conv/recurrent/KV
state and logits, or `--check logits` for the cosine-gated trajectory with argmax
agreement reported separately. The standalone 16-token logits script was removed;
use the latter mode instead. The shipped `oflm` text check remains separate.

```bash
OPENBLAS_NUM_THREADS=1 ironvenv/bin/python utilities/prefill-block/decode_transition.py \
  --check state --prompt 257 --decode 2 --layers 4 --out Models/qwen38-27b/decode-check-state
OPENBLAS_NUM_THREADS=1 ironvenv/bin/python utilities/prefill-block/decode_transition.py \
  --check logits --prompt 257 --decode 16 --layers 4 --out Models/qwen38-27b/decode-check-logits
```

Use fresh output directories and the shared NPU lock. The checker uses the CLI,
model and kernel paths declared in `utilities/prefill-block/oracle.py`; inspect
these before running. Rebuild the standalone CLI if its native sources changed.
XRT device access may require execution outside the sandbox: a sandbox-only
`0 devices found` result does not prove that the host lacks an NPU.

Only physical `--chunk 256` is accepted. Old arbitrary chunk values changed
metadata without configuring the CLI schedule. The block side now explicitly
sets `OFLM_OPEN_LAYER_MAJOR=0`; inspect `out_block.log` for `gemm-block` rather
than `gemm-prompt`. Prompt dump indices remain relative even with `--position`.
Missing/empty/malformed logits, incomplete token coverage in state mode and
wrong-sized state dumps fail. Neither mode overwrites an existing run directory.

Default prompt/decode/layers are 40/2/8 (state) and 700/16/64 (logits). Keep the
original oracle cosine bound 0.9999. State and logits retain distinct tool IDs
and acceptance rules; do not turn an argmax mismatch into a logits-only failure
or interpret near-zero-reference max-relative error as a new acceptance bound.

Validated 2026-11-03 on four layers, prompt 257: state PASS with two generated
tokens; logits PASS with 16/16 agreeing tokens and minimum cosine 0.999999993.
The log records physical blocks of 256 and 1 tokens. Existing source/binary
identities, raw outputs and both reports are retained under
`Models/qwen38-27b/prefill-block-consolidation-20261103/validation.json`.
This validates the checker and its route on a layer slice, not full-depth model
accuracy, serving or throughput. Host regression tests live in
`specs/open-engine/tests/test_prefill_decode_checks.py`; their subprocess double
is explicitly software-only evidence. Current suite: 2351 passed, 60 skipped;
native CTest: 22 passed, 2 skipped. No new XCLBIN/library or production change.


## Generation headroom metadata regression

After changing `Qwen3_5VL::generate`, run the executable host checks:

```bash
OPENBLAS_NUM_THREADS=1 ironvenv/bin/python -m pytest -q \
  specs/open-engine/tests/test_qwen35_generation_metadata.py
```

The test compiles the production method and preamble-row helper with host doubles;
no weights or NPU are needed. Both thinking branches must reject the first prompt
above `MAX_L - generation_preamble_rows() - 1` before any forward call, with
`MAX_LENGTH_REACHED`, zero generated tokens and zero decoding duration. The last
accepted prompt emits one token and finalizes its length-stop metadata. Stale
metadata, EOS, cancellation and ordinary token limits are covered separately.
With `length_limit = 0`, regular-loop context exhaustion must report
`MAX_LENGTH_REACHED`; explicit EOS on the last slot must retain `EOT_DETECTED`.
The output-limit matrix covers 1, 2, 0 and -1 in both branches, checking exact
forward, sample and emitted-token counts. A one-token cap allows only the preamble
forwards and one sample. Initial EOS retains EOT precedence, including at the
context boundary; its one cache-commit forward does not sample another token.
Keep this control-flow evidence separate from the numerical NPU gates above.

## GEMM8 merge regression (2026-11-04)

After merging upstream `3d5f2d78`, manifest v4 supports both paired Qwen3.8
prefill and the dense Qwen3-8B `bfp16_dit` route. Both generated manifests are
loaded without overrides by `test_prefill_precision_runtime.py`.
A freshly built standalone CLI passed `decode_transition.py --check state
--prompt 257 --decode 2 --layers 4`, with `oracle.CLI` set to that build and
`oracle.MODEL` set to `oflm-data/models/Qwen3.8-27B-NPU2` (19,844,633,616 bytes).
The historical default model path points to a different artifact; choose paths
explicitly when reproducing this run. Binary, source hashes, kernel manifest
hash, logs and all state/logit comparisons are retained under
`Models/qwen38-27b/main-merge-20261104/validation.json` and `state/result.json`.
All gates passed; this is a layer-slice regression, not full-model validation.

## Building the standalone CLI

`utilities/prefill-block/build-cli.sh` needs `-mavx2 -mfma` for `cli.cpp`: its
argmax is vectorised. `core.cpp` includes the intrinsics header for `_mm_mfence`
explicitly — do not rely on transitive includes from build flags.

Every `OFLM_*` switch the engine reads must be classified, or the settings CI
fails on the merged tree. For diagnostics that is `NON_SETTING_ENV_VARS` in
`utilities/generate-registry.py`.

For a merge-only Docker check, name the host build targets explicitly
(`cmake --build build --target oflm`). The default `all` target also runs
`utilities/export-kernels.py` for every configured family, which is not needed
to rebuild `oflm` and the CTest executables.

## What a route check establishes

A block-vs-layer-major comparison at 256 prompt tokens over all 64 layers with
identical generated tokens establishes that the two routes agree. It does not
establish controlled throughput, and it does not accept newly compiled kernels:
those need their own numerical gates. Pin the power mode explicitly (or record
the observed one when using `--pmode none`) before quoting any timing.
