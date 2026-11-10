# Qwen3.8-27B: apply the OpenMP wait policy before runtime initialization

Linux libgomp reads its environment before application static constructors.
Setting `OMP_WAIT_POLICY` from the old constructor therefore changed the value
printed by the engine without applying that default at runtime initialization.
PR #2 moves the Linux default to an early self-exec, before model/device access.

## Implementation and compatibility

- The backend-independent helper is `src/open_qwen36/omp_policy.cpp`; the app
  does not include the XRT-only `core.hpp` just to reach it. HRX builds compile
  the helper but do not select this optimization.
- The shipped app selects the policy only for a registry entry with family
  `qwen3.5` and a `Qwen3.8-` model name, for `run`, `serve` and `bench`. Other
  models, including Whisper, retain their existing OpenMP behavior. The
  standalone CLI calls the helper directly for controlled engine measurements.
- The Linux constructor leaves the environment untouched. An explicit
  `OMP_WAIT_POLICY`, including `ACTIVE`, takes precedence and skips re-exec.
- Otherwise startup sets `PASSIVE` and re-execs the same binary with its original
  arguments. `OFLM_OPEN_OMP_PASSIVE=0` opts out; `OFLM_OPEN_OMP_REEXEC` is a
  diagnostic guard. Successful exec does not return or change the process ID.
- Failed environment setup or exec prints a diagnostic. A failed exec restores
  the previously absent policy/guard instead of reporting an unapplied policy.
- Windows retains its delay-loaded vcomp initialization; the POSIX header is
  guarded. A host test covers that branch, not a native MSVC build.
- Kernel arithmetic, weights, ABI, routing thresholds and state layouts are
  unchanged. The selected policy is process-wide only for the Qwen3.8 command.

Executable tests compile and run the production helper and its test driver.
They cover explicit policies, opt-out, the guard, arguments, one re-exec, model
selection, an injected exec failure and the XRT-free HRX dependency boundary.
The benchmark runner imports its POSIX lock only in hardware modes, so its pure
summary/parser tests skip rather than fail where that capability is absent.

## Measurement method

The repeatable driver is `utilities/prefill-block/omp_policy.py`. Its modes share
existing profile provenance, CLI, server and state-comparison helpers.

The full-model benchmark compares the retained pre-PR binary from `7f227b15`
with the corrected implementation in `8ba65ae9`. It excludes one warm-up pair,
then runs five measured pairs with alternating order, a fixed 256-token prompt
and 16 generated tokens. Profiling is disabled. Engine prompt-ingestion time
excludes model loading; decode time excludes prefill. This is not a measurement
of HTTP time to first streamed token.

The runner retains both binary hashes, model/kernel hashes, commands, complete
logs, generated token IDs, per-run host conditions, observed NPU power mode and
peak process RSS. `--pmode none` deliberately retains the observed `default`
mode; it does not claim successful performance-mode configuration. Available
ACPI temperatures are recorded separately from the unavailable NPU temperature.

Host: AMD Ryzen AI 9 365 / NPU Strix, 20 OpenMP threads, Linux 7.0.0-38-generic,
glibc 2.43, CPU governor `powersave`. Both arms use the same 19,844,633,616-byte
Q4NX model and existing paired-prefill kernel export. SHA-256 identities:

| Artifact | SHA-256 |
|---|---|
| Before CLI | `c3bb978b16be18622ad030a00369f0752775d3af849cdd7c3c7a06665ef12b87` |
| After CLI | `7613619055503a5fe8578eee948a292c6972f381dd17377041ed0c7afd031c28` |
| Model | `1f78e62b723f66ad540e8a35a5cb31de67e2eae48c3a58e226873eefa10b349a9` |
| Kernel manifest | `744879192c49d2399b18cc6a50692ded1d6c86208387b3307ee5b172a54a3cee` |

Acceptance requires five faster prefill pairs with identical output IDs, and
median decode time and peak process RSS increases no greater than 3%. With five
wins, the one-sided paired sign-test probability under no improvement is 1/32.
This performance criterion does not alter any numerical oracle threshold.

## Validation results

The five measured full-model pairs passed on 2026-11-06:

| Metric | Before median (range) | After median (range) | Median change |
|---|---:|---:|---:|
| 256-token engine prefill | 34.581 s (34.255–34.755) | 29.241 s (29.187–29.388) | **−15.44%** |
| Decode per token | 453 ms (450–456) | 454 ms (453–455) | +0.22% |
| Peak process RSS | 17,007,900 KiB (17,007,816–17,008,208) | 17,007,788 KiB (17,007,480–17,008,596) | −0.001% |

All five prefill pairs improved; all 16 generated IDs matched in every pair.
There is no demonstrated decode or peak-RSS regression in this sample. This
gain is for the measured full-model workload, not a throughput guarantee for
other models or prompt lengths.

The shipped `oflm run` also completed two resident sessions, with the policy
disabled/enabled. Each session included one warm-up and three requests separated
by `/clear`. Engine prompt lengths were **72, 264, 520 and 1032** tokens (the chat
template adds eight to the user-text fixture). All four answers matched, with
16 text tokens each. These logs were re-audited after correcting the validation
driver: its original comparison accidentally included changing `/status` timings
and the next turn's streamed answer. Both processes exited successfully; inference
was not rerun for the parser correction. Single-session timings are retained as
a cross-check, not additional statistical performance evidence.

Host validation on the review-fixed, main-merged tree:

- `OPENBLAS_NUM_THREADS=1 ironvenv/bin/python -m pytest -q specs/`:
  **2532 passed, 172 skipped**.
- Production Docker build: `cmake --build build --target oflm -j 4` passed;
  `ctest --test-dir build --output-on-failure`: **29 passed, 2 skipped**.
- Standalone CLI build passed; standalone CTest: **6 passed**, including the
  OpenMP model-selection test. Direct settings scan and registry regeneration
  matched the committed registry.
- The earlier five-pair NPU result was measured with `8ba65ae9`; the review-fixed
  tree has not been re-benchmarked. Its performance claim remains historical
  evidence for the same startup mechanism, not a fresh measurement of this tree.

**Acceptance still pending:** the four-layer boundary matrix, full-depth state
comparison, and repeated server requests/RSS check. On resumption at 09:23 on
2026-11-06, even the pre-PR control failed before inference with
`DRM_IOCTL_AMDXDNA_CREATE_HWCTX ... EINVAL`. The corrected CLI failed identically.
The kernel log reported `aie2_xrs_load: create context failed, ret -22` and
`Allocate AIE resource failed`; no active hardware contexts were reported.
Binary/kernel hashes still match the successful benchmark. The underlying
driver/firmware cause is undetermined. No device reset was performed.
Only about 14 GiB RAM was available; full-model/server checks were explicitly
deferred by the user. This is not a completed hardware acceptance claim.

Retained evidence: `Models/qwen38-27b/pr2-final/`. The benchmark's `result.json`
contains each sample and artifact identity; `production/result.json` records the
corrected log audit, and `boundaries/` retains both context-creation failures.

## Reproduction

```bash
source utilities/env.sh
utilities/prefill-block/build-cli.sh Models/qwen38-27b/pr2-final/build
utilities/prefill-block/build-tile-x-bench.sh Models/qwen38-27b/pr2-final/tile-bench

OPENBLAS_NUM_THREADS=1 ironvenv/bin/python utilities/prefill-block/omp_policy.py \
  --before Models/qwen38-27b/main-merge-20261105/build/open_qwen36_cli \
  --after Models/qwen38-27b/pr2-final/build/open_qwen36_cli \
  --out Models/qwen38-27b/pr2-final/benchmark

OPENBLAS_NUM_THREADS=1 ironvenv/bin/python utilities/prefill-block/omp_policy.py \
  --check production --after build/src/oflm \
  --out Models/qwen38-27b/pr2-final/production

OPENBLAS_NUM_THREADS=1 ironvenv/bin/python utilities/prefill-block/omp_policy.py \
  --check boundaries \
  --before Models/qwen38-27b/main-merge-20261105/build/open_qwen36_cli \
  --after Models/qwen38-27b/pr2-final/build/open_qwen36_cli \
  --generate 2 --lengths 95 96 97 255 256 257 511 512 513 \
  --out Models/qwen38-27b/pr2-final/boundaries
# Full-depth state control: same command with --layers 64 --generate 16 --lengths 256.

OPENBLAS_NUM_THREADS=1 ironvenv/bin/python utilities/prefill-block/omp_policy.py \
  --check server --after build/src/oflm \
  --out Models/qwen38-27b/pr2-final/server
```

Use fresh output directories. Production checks use the registered `qwen3.8:27b`
model; standalone checks accept explicit model/kernel paths. A shared NPU lock
serializes resident runs. Peak RSS measures process memory, not all driver-owned
NPU allocations. No newly compiled kernel export is used for this change.
For a fresh checkout, build the control from `7f227b15` in a separate worktree
and substitute its CLI path. The example control path is a retained local build.

## Isolated tiling investigation

The original mapped-buffer experiment used `FROM_DEVICE` after CPU writes.
That direction was incorrect for preparing GEMM input; its transfer-cost
conclusions are superseded. The corrected probe uses `TO_DEVICE` and verifies
identical BF16 bits for host, mapped-buffer and AVX2 paths.

One corrected 20-repetition minimum-time probe (20 threads, externally supplied
`OMP_WAIT_POLICY=PASSIVE`) measured, in milliseconds:

| Input | Host scalar | Mapped buffer + sync | Host AVX2 |
|---|---:|---:|---:|
| K=5120, T=256 | 0.11 | 0.12 | 0.06 |
| K=17408, T=256 | 0.45 | 0.37 | 0.27 |

These isolated minima do not establish an end-to-end gain from a tiling rewrite.
The experimental AVX2 implementation remains in the probe only. Earlier single-run
13–15% TTFT estimates are historical observations, not the acceptance evidence
for the corrected change; the repeated benchmark above supersedes them.
