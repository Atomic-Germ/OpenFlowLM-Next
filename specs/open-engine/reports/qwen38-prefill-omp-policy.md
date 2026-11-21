# Qwen3.8-27B: OpenMP startup policy

## Scope and export compatibility

The current merged runtime passes PR #2 acceptance on the existing export.
The source includes upstream `9b1409d3`, export-compatibility fix `1448fc65`,
and acceptance driver `4573739c`. No kernel rebuild or model repacking is used.

PR #2 installs the Linux default `OMP_WAIT_POLICY=PASSIVE` before libgomp reads
its environment. Explicit policies and `OFLM_OPEN_OMP_PASSIVE=0` take precedence.
The backend-independent helper re-execs once with the original arguments;
failed setup or exec restores the previously absent environment and reports the
error. Windows retains its delay-loaded vcomp path. Commands initially selecting
another model and HRX builds do not select this Linux optimization.

The policy is process-wide. A server started without Qwen3.8 does not acquire it
when a later request selects Qwen3.8; a server started with Qwen3.8 retains it when
switching models. This is not dynamic per-request OpenMP tuning.

The DeltaNet host/kernel interface declares `key_head_mapping` in the linear
`gemm_block`:

- Absent or `grouped`: value head `h` reads `h / (value_heads / key_heads)`,
  preserving existing compiled exports.
- `tiled`: value head `h` reads `h % key_heads`, matching current upstream kernels.
  Current recipes emit it in both Q4NX and GGUF views.
- Unknown mappings and invalid head ratios are refused. Prefill and rollback use
  the same field, and rollback refuses a tape recorded with a different mapping.

No arithmetic, precision, weights, buffer ABI, or route threshold changes for an
existing export. An unmarked export compiled with modulo-head kernels must be
re-exported with `tiled` declared, not silently relabelled as a grouped binary.
Host tests cover both mappings, scalar-reference agreement and every rollback
prefix. Hardware results here accept the retained grouped export, not a newly
compiled tiled export.

## Machine and measurement method

Measured on 2026-11-14: AMD Ryzen AI 9 365 / NPU Strix, 20 OpenMP threads,
Linux 7.0.0-43-generic, glibc 2.43, GCC 15.2.0, CPU governor `powersave`.
Host XRT 2.26.0 (`e9db9ab15f10173f8d2fc93ff92ab4c7eb09d2e6`), amdxdna
2.26.0_20260817, NPU firmware 1.1.2.64. NPU power mode is externally observed
`default`; CLI `--pmode none` leaves it unchanged. Background desktop/services
remain active. Observed one-minute host load ranges are 1.55–5.22, 1.30–11.07
and 0.58–8.85 for the 256/512/1024-token campaigns. Per-run load, available
memory and ACPI temperatures are retained. ACPI temperature is not NPU die
temperature. No competing NPU context is observed before the correctness campaign;
the shared validation lock serializes resident jobs.

Standalone and Docker production binaries are built fresh from the current
production sources. `build-identity.json` binds binaries and source hashes;
benchmark results bind model, manifest and kernel hashes. SHA-256 identities:

| Artifact | SHA-256 |
|---|---|
| Current standalone CLI | `c8c1ec5eebebc6e7e1d28114393870d386a6eeb545b3277dac651b0ff1aa66ad` |
| Production app | `577b925ab182e948c510e4bd4bdef8076510a7a47baf5078327bb13c71af4339` |
| Retained pre-merge accepted CLI | `7b5b5aa650279d144f45855b2f103d2fa7ee9c849e17c6fe181606d311df3b6f` |
| Existing Q4NX model (19,844,633,616 bytes) | `1f78e62b723f66ad540e8a35a5cb31de67e2ae48c3a58e226873eefa10b349a9` |
| Existing kernel manifest | `744879192c49d2399b18cc6a50692ded1d6c86208387b3307ee5b172a54a3cee` |

Performance compares **the same current CLI** with the policy opted out and
enabled, isolating the optimization from upstream runtime changes. Both arms
name source commit `4573739c`. Each prompt size uses one warm-up pair, followed
by five measured pairs in alternating order. Prompts have fixed IDs; every run
generates 16 tokens. Profiling is off. The driver records both source commits,
binary/model/kernel hashes, commands, arm-specific opt-out values, full logs and
peak process RSS.

Engine prefill excludes model loading; decode excludes prefill. These timings
are not HTTP time to first streamed token. Acceptance requires all five prefill
pairs to improve, identical generated IDs, and at most 3% median decode/RSS
regression. Five paired wins give a one-sided sign-test probability of 1/32
under no improvement; numerical oracle thresholds remain unchanged.

## Acceptance results

All three campaigns pass. Values are medians with min–max ranges; each row has
five measured pairs. Before/after mean policy off/on on the same binary.

| Prompt | Before prefill | After prefill | Change | Before decode | After decode |
|---|---:|---:|---:|---:|---:|
| 256 | 34.384 s (34.049–34.477) | 29.713 s (29.207–29.739) | **−13.58%** | 460 ms (453–464) | 461 ms (456–462) |
| 512 | 68.749 s (68.644–69.081) | 59.731 s (58.666–59.929) | **−13.12%** | 466 ms (462–472) | 466 ms (462–476) |
| 1024 | 138.536 s (138.188–138.704) | 120.869 s (120.858–120.974) | **−12.75%** | 481 ms (475–488) | 486 ms (479–488) |

Every measured prefill pair improves and every 16-token trajectory matches.
Median peak process RSS changes are +0.00006%, −0.00081% and +0.00091%, at
approximately 34,278,000 KiB in either arm. Median decode changes are +0.22%,
0% and +1.04%, within the unchanged 3% acceptance bound.

Correctness and integration on the current binaries:

- Four layers, 95/96/97, 255/256/257 and 511/512/513 prompt tokens plus two
  generated tokens: all nine cases pass the existing numerical gates and exact
  before/after logits and all requested layer-state bytes.
- All 64 layers, 256-token prompt plus 16-token greedy continuation: **147/147
  checks pass** against the retained pre-merge accepted CLI. All **112 state
  slices** (conv/recurrent state for 48 linear layers and KV for 16 attention
  layers) and prompt/decode logits are byte-identical. No old result is relabelled.
- Production CLI: policy-off/on sessions each have a warm-up and three requests
  separated by `/clear`; all four answers match. Actual engine prompts are 72,
  264, 520 and 1032 tokens. This is integration evidence, not repeated TTFT timing.
- Production server: seven alternating 272/528-token requests return complete,
  identical responses on repeat. Maximum per-prompt RSS growth from the warmed
  baseline is **0.6722% / 0.0064%**. RSS samples and responses are retained.

Host gates: **2586 Python tests passed, 172 skipped**; standalone CTest **5/5**;
production Docker build passes, Docker CTest **30 passed, 2 skipped**. The direct
host fixture also passes both head mappings and rollback tests. Settings registry
regeneration passes in a scratch merge with actual upstream commit `9b1409d3`.
Skipped tests are not counted as passes.

**NOT MEASURED:** native Windows build, HTTP streaming TTFT, approximately-4K
context decode, total driver-owned NPU memory, and freshly rebuilt tiled kernels.
These results establish the scheduling change on the existing grouped export;
they do not add model support or establish new context/numerical capabilities.

## Reproduction

Evidence root: `Models/qwen38-27b/pr2-ready-20261114/`. Raw artifacts remain local;
this report records their identities and measured summaries. Always use fresh
output directories and avoid other NPU jobs or memory-intensive host workloads.

```bash
source utilities/env.sh
ROOT=Models/qwen38-27b/pr2-ready-20261114
MODEL=oflm-data/models/Qwen3.8-27B-NPU2
KERNELS=Models/qwen38-27b/production-v2/kernels-model
utilities/prefill-block/build-cli.sh "$ROOT/build"
CLI="$ROOT/build/open_qwen36_cli"

for N in 256 512 1024; do
  OPENBLAS_NUM_THREADS=1 ironvenv/bin/python utilities/prefill-block/omp_policy.py \
    --before "$CLI" --after "$CLI" \
    --before-commit 4573739c --after-commit 4573739c \
    --model "$MODEL" --kernels "$KERNELS" \
    --tokens "$N" --out "$ROOT/benchmark-$N"
done

OPENBLAS_NUM_THREADS=1 ironvenv/bin/python utilities/prefill-block/omp_policy.py \
  --check boundaries --before /path/to/accepted/open_qwen36_cli --after "$CLI" \
  --model "$MODEL" --kernels "$KERNELS" --generate 2 \
  --lengths 95 96 97 255 256 257 511 512 513 --out "$ROOT/boundaries-all-state"
# Repeat with --layers 64 --generate 16 --lengths 256 --out "$ROOT/full-state".

OPENBLAS_NUM_THREADS=1 ironvenv/bin/python utilities/prefill-block/omp_policy.py \
  --check production --after build/src/oflm --out "$ROOT/production"
OPENBLAS_NUM_THREADS=1 ironvenv/bin/python utilities/prefill-block/omp_policy.py \
  --check server --after build/src/oflm --out "$ROOT/server"
```

The state control is the retained accepted pre-merge CLI identified above; build
that version (`9f8415c9`) in a separate worktree when reproducing without local
artifacts. The policy-only benchmark needs no historical binary. Production
checks use the registered `qwen3.8:27b` model; standalone modes accept explicit
paths. Process RSS is not total driver-owned NPU allocation. The startup opt-out
changes no arithmetic, weights, route or state ABI.
