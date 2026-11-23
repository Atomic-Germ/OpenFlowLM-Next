# Instrumentation correctness gate

## Scope

This gate covers instrumentation, the synthetic fold harness and capture
validation. It changes no inference arithmetic, kernel, precision, routing
default or oracle threshold. No NPU inference, new performance campaign, or long
synthetic timing campaign is included in this gate. Production duplicate-split
optimization is outside its scope.

## Instrumentation correctness requirements

| Requirement | Implementation | Executable gate |
|---|---|---|
| One record per eligible unit, dense and hybrid entry paths | A shared-entry scope covers one physical block, or the complete layer-major prompt; no nested route hooks | Multiple full/partial blocks, two route-body stubs and repeated requests through the production scoped profiler |
| Phase, real/physical token metadata and independent counters | Each enabled scope constructs one fresh record; incomplete scopes discard it | 256/256, 1/256, 255/256 and 257/512 metadata, exact counters/totals, no duplicates |
| Error and early-exit cleanup | Unwinding or missing completion suppresses a successful fold row; related request activity is cleared | Dispatch failure, early return, tail failure after completion, recovery, invalid metadata and nested-scope rejection |
| Exact FP32 association and complete output comparison | Cancellation/rounding-sensitive product fixtures; unchanged production grouped sums | Full buffers against an independent grouped-sum oracle; both reassociation mutants fail |
| Non-finite and corruption rejection | NaN, both infinities and BF16 conversion overflow; actual last-element corruption | Both arms reject invalid input; corrupted and truncated buffers fail; temporary corruption mutant fails |
| Actual lock and safe subprocess lifetime | Exclusive nonblocking advisory lock; output directly to disk; process-group cleanup and reaping | Separate handle/process contention, exception release, output beyond pipe capacity, timeout and descendant cleanup |
| Fail-closed capture | Strict profile/log/fold coverage, exact generated steps, metadata, manifest-derived fold counts and provenance validation | Missing, malformed, contradictory, cumulative, contaminated and tampered evidence fails |
| Disabled path | No record is constructed without a sink; no fold timestamp is read without an active timing sink | Injected counting clock reports zero reads; warmed disabled fold/scope reports zero allocations |
| Synthetic definition and shape | Split-only candidate retains all three output copies; full-attention output input width is 6144 | Description-only benchmark execution compared with the recipe-generated manifest; no timing loop |
| Inference preservation | Dispatch callbacks, grouped FP32 sums and non-finite checks stay unchanged | Instrumented/uninstrumented complete-buffer equality, overwritten-output-buffer fixture, existing portable suite and full host-CLI build |

The compiled behavioural fixtures are in
[test_fold_instrumentation.py](../tests/test_fold_instrumentation.py). Route bodies
are stubbed: the real scoped profiler and fold execute, but hardware-backed Core
routes, model states and end-to-end outputs are **NOT MEASURED** here. Source
integration places the scope before the dense/hybrid branch in each public
prefill entry. A complete host build checks those integrations compile.

The reassociation and corruption checks modify only temporary copies of the
synthetic candidate header and require the same numerical executable to reject
each compiled mutant. No production source is mutated. The reused mapped-output
fixture also verifies that host folding does not dirty a dispatch output buffer.

Capture tests are in [test_pipeline_capture.py](../tests/test_pipeline_capture.py).
Tests requiring POSIX locking, live process mappings or a compiler skip explicitly
when those capabilities are absent. Skips do not constitute acceptance passes.

## Capture and evidence contract

[pipeline_capture.py](../../../utilities/prefill-block/pipeline_capture.py)
requires an explicit phase, prompt/output lengths, exact profiling budget, and
caller-trusted build-time attestation. It never infers which source built an old
executable from the checkout's current revision.

The version-1 build attestation contains:

- executable SHA-256;
- an explicit source revision/build identity, not the literal current-HEAD alias;
- build-command argument vector;
- nonempty absolute-path/SHA-256 inventories of source files and build inputs.

Those files must remain available and match, for example in an archived source
and build tree. Hashes establish integrity, not that a compiler actually consumed
the asserted inputs; the caller must supply authentic and complete build records.

Each version-2 capture owns its executable/build identity, source/build/model and
kernel/instruction/manifest snapshots, measured-process XRT mappings and hashes,
phase and token counts, exact argument vector, working directory and environment,
invocation/token-input snapshots, and raw evidence hashes. Inputs are checked
before and after execution. All inherited OFLM overrides are removed and recorded;
only explicit supported thread/wait controls may be supplied as extra overrides.
There are no silent defaults for missing phase or build evidence.

Manifest block capability is recorded separately from executed physical blocks:
sequential decode has no physical prefill block. With N requested output tokens,
the prompt produces the first token and the driver requires exactly N-1 generated
step records, while preserving all N output IDs.

The driver is intentionally limited to the paired Qwen fixture geometry. It
requires Linux-style live process provenance and POSIX advisory locking; it fails
explicitly on unsupported hosts instead of running unlocked. Capture validation
is not a cross-platform inference support claim.

[pipeline_evidence.py](../../../utilities/prefill-block/pipeline_evidence.py)
audits retained directories read-only. Missing phase/token labels, build identities,
runtime mappings or replaced executables remain errors. Its audit inventory
preserves each capture's own recorded binary hash, command, phase/token evidence,
artifact/runtime identities and raw hashes; unavailable historical fields remain
missing with a diagnostic. It never reconstructs labels from a directory name or
turns an old capture into a valid new one.

Historical numerical JSON and raw files are unchanged. A new final performance
report is not generated from them. Different recorded executable hashes denote
different measured snapshots, even when the path is identical.

## CPU-only reproduction

From the repository root, set BUILD_DIR to a fresh native CLI build directory:

```sh
ironvenv/bin/python -m pytest -q \
  specs/open-engine/tests/test_fold_instrumentation.py \
  specs/open-engine/tests/test_pipeline_capture.py \
  specs/open-engine/tests/test_layer_profile.py
ironvenv/bin/python -m pytest -q specs/
bash -c '. utilities/env.sh >/dev/null; utilities/prefill-block/build-cli.sh "$1"' bash "$BUILD_DIR"
```

The portable tests compile and run only a description of benchmark shapes, not
the long timing campaign. They derive the reference shapes from the checked-in
recipe/spec and do not require model weights.

A read-only historical audit, expected to return nonzero for legacy evidence.
Set the three capture-directory variables from the retained evidence inventory:

```sh
ironvenv/bin/python utilities/prefill-block/pipeline_evidence.py \
  "$LEGACY_PREFILL_CAPTURE" \
  "$LEGACY_FOLD_CAPTURE" \
  "$LEGACY_DECODE_CAPTURE"
```

The audit does not run the model or change raw captures. Settings are checked
with the repository's scratch merged-tree
workflow, against the recorded target revision, after committing the source gate.

## Results and limits

Host verification: Linux 7.0.0-44-generic, GCC 15.2, Python 3.14. The standalone
CLI uses the existing build helper and its strict host-arithmetic flags, the
from-source XRT development headers and the installed XRT coreutil library.
No NPU inference is part of these checks. Log locations are retained in the
build and capture metadata.

| Check | Result |
|---|---|
| Focused fold/capture/historical-summary/layer-profiler suite | 317 passed, no skips |
| Complete portable specification suite | 2,881 passed, 172 skipped |
| Compiled scoped lifecycle fixture | 16 successful units with exact independent metadata/counters; failed/early-exit units emit no successful row |
| Temporary numerical mutations | Left association, right association and last-output corruption all rejected by the same compiled numerical gate |
| Disabled instrumentation | Zero injected fold-clock reads and zero warmed fold/scope allocations |
| Full native host CLI | Compiled and linked; not run on the NPU |
| Standalone host CTest | 5/5 passed |
| Historical evidence audit | 12 legacy directories rejected as expected; raw data unchanged |
| Merged-tree settings registry | PASS against fetched target `67ee44a6ddc2a311782929cc94fac3e96da030ff` |
| Kernel sources, settings/defaults, historical numerical JSON | Unchanged relative to `2ec3e9c2` |

Source commits: `9eac5fae` contains the scoped fold/numerical/shape gate;
`f494ccdb` contains capture validation and evidence tooling; `60c92809` separates
manifest block capability from sequential-decode execution metadata. The native build's
own attestation records full source revision
`f494ccdbb4040394020e2217a25ec4645b06cb02`, input hashes checked before/after the
build and executable SHA-256
`76ecff0ffd796895dfe28e6a580de85a1877f60183cfa7c1457273bec9268c47`.
That identity describes only this unmeasured verification executable, never the
historical prefill/decode binaries. The attestation is retained alongside it at
`native-cli/build-identity.json` under the log directory.

The settings check runs from a clean detached copy of the committed source gate
and performs its own target merge in another scratch worktree. Documentation-only
working changes are not used to fabricate a code snapshot. The 172 skips remain
unmeasured capabilities, not passes; no hardware acceptance is inferred from the
portable suite's pass count.

Phase-specific CPU/wait attribution is **NOT MEASURED**. CPU observations remain
whole-process samples, not a decode-only CPU fraction. Dispatch residence can
overlap under submit-ahead and does not establish a wait-bound bottleneck. No
residual timing is assigned to fold, allocation, copy or another operation without
direct windows.

The synthetic candidate eliminates only one duplicate input split and associated
validation; all output copies remain. Its persistent workspaces exclude repeated
allocation after warm-up, but shape-major cache reuse and independent stub products
do not reproduce the production memory/cache schedule. No production speedup is
predicted from it, and its changed fixtures/shape have no new timing measurements.

Zero fold timestamp reads is an instrumentation contract, not a measured claim of
zero end-to-end overhead. Matched NPU overhead measurements, model-state/reset and
prefill/decode acceptance, native Windows verification, and any new performance
report remain **NOT MEASURED** by this gate. Separate hardware profiling and
duplicate-split acceptance are required before a production speedup claim.
