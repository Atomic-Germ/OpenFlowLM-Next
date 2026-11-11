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
selection and an injected exec failure. Linux/libgomp tests skip when that
capability is absent. The benchmark runner imports its POSIX lock only in
hardware modes, so its pure summary/parser tests remain portable.

## Measurement method

The repeatable driver is `utilities/prefill-block/omp_policy.py`. Its modes share
existing profile provenance, CLI, server and state-comparison helpers.

The full-model benchmark compares a fresh control build from `7f227b15` with the
PR #2 implementation at `78c41524` plus the review fixes in this working tree.
It excludes one warm-up pair, then runs five measured pairs in alternating order
at each of 256, 512 and 1024 prompt tokens, generating 16 tokens per run.
Profiling is disabled. Engine prompt-ingestion time excludes model loading;
decode time excludes prefill. This is not HTTP time to first streamed token.

The runner records the requested source commits, current source commit and dirty
state, both binary hashes, model/kernel hashes, commands, complete logs,
generated token IDs, per-run host conditions, observed NPU power mode and peak
process RSS. `--pmode none` deliberately retains the observed `default` mode; it
does not claim successful performance-mode configuration. Available ACPI
temperatures are recorded separately from the unavailable NPU temperature.

Host: AMD Ryzen AI 9 365 / NPU Strix, 20 OpenMP threads, Linux 7.0.0-42-generic,
glibc 2.43, CPU governor `powersave`. Both arms use the same 19,844,633,616-byte
Q4NX model and existing paired-prefill kernel export. Exact artifact hashes are
in each benchmark's `result.json`.

Acceptance requires five faster prefill pairs with identical output IDs, and
median decode time and peak process RSS increases no greater than 3%. With five
wins, the one-sided paired sign-test probability under no improvement is 1/32.
This performance criterion does not alter any numerical oracle threshold.

## Validation results

All three five-pair campaigns passed on 2026-11-11. Values are medians, with the
minimum–maximum range in parentheses.

| Prompt | Before prefill | After prefill | Change | Before decode | After decode |
|---|---:|---:|---:|---:|---:|
| 256 | 34.658 s (34.567–34.945) | 29.374 s (29.304–29.498) | **−15.25%** | 453 ms (451–454) | 455 ms (454–457) |
| 512 | 69.595 s (69.573–69.945) | 58.675 s (58.612–58.799) | **−15.69%** | 460 ms (459–461) | 460 ms (459–460) |
| 1024 | 139.888 s (139.254–140.838) | 119.087 s (118.889–119.250) | **−14.87%** | 474 ms (474) | 474 ms (471–475) |

Every prefill pair improved and all 16 generated IDs matched in every pair.
Median peak process RSS changed by −0.0019%, +0.0009% and +0.0026% respectively.
There is no demonstrated decode or peak-RSS regression in these samples.

The shipped `oflm run` completed policy-disabled/enabled resident sessions with
one warm-up and three requests separated by `/clear`. Engine prompt lengths were
72, 264, 520 and 1032 tokens. All four answers matched, with 16 text tokens each.
Single-session timings are a cross-check, not statistical performance evidence.

The four-layer boundary matrix passed at 95/96/97, 255/256/257 and 511/512/513
prompt tokens. The full 64-layer, 256-token prefill plus 16-token decode control
also passed. Both required the existing numerical oracle and exact before/after
logits and state bytes.

The production server completed seven alternating 272/528-engine-token requests.
Repeated responses were identical and complete. After both prompt sizes had
allocated once, maximum RSS growth from each prompt's warmed baseline was 0.44%
and 0.21%. Process RSS does not include all driver-owned NPU allocations.

Host validation on the acceptance tree:

- `OPENBLAS_NUM_THREADS=1 ironvenv/bin/python -m pytest -q specs/`:
  **2534 passed, 172 skipped**.
- Production Docker build: `cmake --build build --target oflm -j 4` passed;
  `ctest --test-dir build --output-on-failure`: **29 passed, 2 skipped**.
- Fresh standalone control and candidate CLI builds passed.
- Native Windows build: **NOT MEASURED**.

Retained evidence: `Models/qwen38-27b/pr2-acceptance/`. Each benchmark directory
contains its samples and artifact identities; `boundaries/`, `full-state/`,
`production/` and `server-v2/` contain the correctness and integration results.

## Reproduction

```bash
source utilities/env.sh
utilities/prefill-block/build-cli.sh Models/qwen38-27b/pr2-acceptance/after-build

OPENBLAS_NUM_THREADS=1 ironvenv/bin/python utilities/prefill-block/omp_policy.py \
  --before Models/qwen38-27b/pr2-acceptance/before-build/open_qwen36_cli \
  --after Models/qwen38-27b/pr2-acceptance/after-build/open_qwen36_cli \
  --before-commit 7f227b15 --after-commit 78c41524 \
  --model oflm-data/models/Qwen3.8-27B-NPU2 \
  --kernels Models/qwen38-27b/production-v2/kernels-model \
  --tokens 256 --out Models/qwen38-27b/pr2-acceptance/benchmark
# Repeat with fresh output directories and --tokens 512 / 1024.

OPENBLAS_NUM_THREADS=1 ironvenv/bin/python utilities/prefill-block/omp_policy.py \
  --check production --after build/src/oflm \
  --out Models/qwen38-27b/pr2-acceptance/production

OPENBLAS_NUM_THREADS=1 ironvenv/bin/python utilities/prefill-block/omp_policy.py \
  --check boundaries \
  --before Models/qwen38-27b/pr2-acceptance/before-build/open_qwen36_cli \
  --after Models/qwen38-27b/pr2-acceptance/after-build/open_qwen36_cli \
  --generate 2 --lengths 95 96 97 255 256 257 511 512 513 \
  --out Models/qwen38-27b/pr2-acceptance/boundaries
# Full-depth state control: same command with --layers 64 --generate 16 --lengths 256.

OPENBLAS_NUM_THREADS=1 ironvenv/bin/python utilities/prefill-block/omp_policy.py \
  --check server --after build/src/oflm \
  --out Models/qwen38-27b/pr2-acceptance/server-v2
```

Use fresh output directories. Production checks use the registered `qwen3.8:27b`
model; standalone checks accept explicit model/kernel paths. A shared NPU lock
serializes resident runs. No newly compiled kernel export is used for this change.
For a fresh checkout, build the control from `7f227b15` in a separate worktree
and substitute its CLI path. The example control path is a retained local build.
