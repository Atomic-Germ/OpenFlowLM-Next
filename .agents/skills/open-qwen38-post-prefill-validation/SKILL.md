# Qwen3.8 post-prefill validation

Use this skill for the Qwen3.8-27B open-kernel prefill/decode transition, state
comparisons and production request-isolation checks. It records the commands and
artifact conventions; it does not turn a layer-slice check into full-model
acceptance.

## Model and kernel artifacts

The validated model is `oflm-data/models/Qwen3.8-27B-NPU2` (19,844,633,616-byte
`model.q4nx`). The production paired-prefill export is under
`Models/qwen38-27b/production-v2/kernels-model`. Select both explicitly in
standalone checks; a historical default path can name a different artifact.

Build the standalone CLI with `utilities/prefill-block/build-cli.sh <outdir>`.
The script supplies AVX2/FMA for the CLI and block host code, disables FP
contraction for the host arithmetic and links the discovered XRT runtime. The
OpenMP startup helper is a separate `omp_policy.cpp` unit; keep it in both the
standalone and production source lists.

## State and transition checks

`utilities/prefill-block/decode_transition.py` compares sequential and block
routes using the existing oracle. It checks logits, greedy tokens, DeltaNet
recurrent state, convolution history, KV cache and position. Use a fresh output
directory and retain the complete logs and JSON result.

A four-layer check is a smoke test only. Full acceptance requires all 64 layers,
the prefill/decode transition, a 16-token greedy trajectory and repeated request
reset/isolation. Boundary lengths are 95/96/97, 255/256/257 and 511/512/513.

The existing numerical gates are not relaxed: logits correlation is at least
0.9999 and the established residual bounds remain in force. Exact byte equality
is appropriate for a scheduling-only before/after comparison, not as a reason to
weaken the independent route oracle.

## Production and server checks

`utilities/prefill-block/oflm_drive.py` drives the shipped `oflm run` through
stdin and measures the engine's actual prompt-token count. `/clear` separates
requests. `oflm_serve_check.py` provides server readiness and request helpers.

For repeated server requests, use distinct prompts and repeat them after both
prompt sizes have allocated their state. Compare complete responses, completion
counts and resident RSS. Process RSS is not total driver-owned NPU memory.

## Linux OpenMP startup policy

The Linux static constructor must leave `OMP_WAIT_POLICY` untouched. The early
`reexec_with_passive` helper in `omp_policy.cpp` supplies the default only when
the caller supplied no policy and did not opt out with `OFLM_OPEN_OMP_PASSIVE=0`.
Explicit ACTIVE/PASSIVE values take precedence. The shipped app calls it only
for a registry entry whose family/name identify Qwen3.8, before model/device
access; other models and HRX do not inherit the policy. The standalone CLI calls
it directly for controlled engine measurements. Exec failure restores the absent
policy/guard and reports the error. Windows keeps delay-loaded vcomp.

Run `test_omp_reexec.py` and `test_omp_policy_benchmark.py` under
`specs/open-engine/tests`. They compile and execute the production helper,
including model selection, an injected exec failure and an XRT-free HRX boundary.
They do not replace a native MSVC build.

Use `utilities/prefill-block/omp_policy.py --check benchmark` with explicit
`--before` and `--after` CLI binaries. It records artifact hashes, alternating
pairs, a warm-up pair, five measured pairs, token equality, power/host conditions
and peak process RSS. `--check production` drives the shipped app with the policy
off/on; `--check server` checks repeated distinct requests and resident RSS.
`--check boundaries` compares before/after logits and state with the existing
oracle bounds plus exact bytes. Use four layers for the length matrix and a
separate `--layers 64 --generate 16 --lengths 256` full-depth control.

Use fresh output directories and the shared NPU lock. The NPU checks retain
the existing kernel export. Report engine prefill time separately from network
TTFT; process RSS is not total driver-owned memory, and ACPI temperatures are
not NPU temperatures. Build/run instructions and evidence locations are in
`specs/open-engine/reports/qwen38-prefill-omp-policy.md`.

## Build and settings checks

For a merge-only Docker check, name the host build targets explicitly
(`cmake --build build --target oflm`). The default `all` target also runs
`utilities/export-kernels.py` for every configured family, which is not needed
to rebuild `oflm` and the CTest executables.

Every `OFLM_*` switch the engine reads must be classified, or settings CI fails
on the merged tree. Diagnostics belong in `NON_SETTING_ENV_VARS` in
`utilities/generate-registry.py` with a reason. Run
`utilities/check-settings-ci.sh` after committing; it refuses a dirty tracked
worktree and tests the merge with the target branch.

## What a route check establishes

A block-vs-layer-major comparison at 256 prompt tokens over all 64 layers with
identical generated tokens establishes that the two routes agree. It does not
establish controlled throughput, and it does not accept newly compiled kernels:
those need their own numerical gates. Pin the power mode explicitly (or record
the observed one when using `--pmode none`) before quoting any timing.
