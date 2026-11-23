# PR4B bounded host-fold deduplication hardware experiment

## Decision: KEEP DRAFT — hardware prerequisite blocked

The initial five-pair campaign is incomplete. A fresh-A native list request and
one bounded retry both fail before inference with hardware-context creation
`DRM_IOCTL_AMDXDNA_CREATE_HWCTX`, error -22 (Invalid argument). The kernel journal
records management opcode 0x2 status 0x2000003 and AIE resource allocation failure.
The device remains visible, runtime status active / D0, with no competing
accelerator client found. The underlying cause is not established; no driver
reset, power-policy change, runtime replacement or candidate tweak is performed.
This is not a numerical mismatch and is not a measured performance NO-GO.

Distinct-prompt correctness is **NOT MEASURED**, so no restart warmup or new timed
sample is collected. The original two accepted pairs remain accepted within
their original scope, including the negative-gain pair. No five-pair speedup,
confirmation eligibility, RSS acceptance or application acceptance is claimed.
All numerical and primary-performance thresholds remain unchanged.

## Machine and measurement conditions

2026-11-23; EVO14-AI9-STP, AMD Ryzen AI 9 365, Strix aie2p 6x8, 20 logical
CPUs, 97,889,218,560 bytes RAM; Ubuntu 26.04.1, Linux 7.0.0-44-generic,
glibc 2.43. XRT and amdxdna runtime 2.26.0, NPU firmware 1.1.2.64.
Live core/coreutil/xdna mappings are 2.26.0, not inferred from an older CLI label;
their exact paths/hashes appear in the JSON and each successful capture.

Power policy is predeclared `none unchanged`; NPU governor control is unavailable
and **unverifiable**. Observed CPU governor is powersave. OpenMP has eight threads
with PASSIVE waiting; BLAS has one. All timing arms use identical layer profiling
and both prefill arms use direct fold profiling. Primary wall observations include
profiling. No synthetic transfer or utilization inference is used.
Initial load averages are 2.12/2.14/1.91 and thermal readings are 66 C; individual
successful captures retain before/after load, temperature and memory conditions.
Conditions sidecars live outside the sealed raw directories. A relocation manifest
preserves their original/new paths and unchanged content hashes; capture metadata
and raw inventories are not rewritten.
No quiet-machine guarantee is made. Tests and application-build attempts finish
before timing. Every run holds the exclusive model NPU lock, checks >=35 GiB
available before load and requires no existing accelerator client.

## Immutable native builds

Control revision ad8671440c9c3ab3671a00aa1cf9bfe962a210dc and candidate revision
c5d20c85defced034d8f2912953f10a2acf97969 are fresh git archives under
`Models/qwen38-27b/pr4b/control/` and `candidate/`. Each archive's own native
build helper compiles the CLI. The known per-unit AVX2/FMA and block-host
`-ffp-contract=off` flags are unchanged. Source files are read-only after build.
Identity version 1 records full revision, executable digest, argv, nonempty
absolute source/build-input inventories, compiler version, XRT headers/libraries,
and equal before/after input snapshots. Native execution uses installed XRT.
Existing PR4A binaries/evidence and production kernels are untouched.

The model is the existing Qwen3.8-27B-NPU2 and the kernel tree is the PR4A
95-file production-equivalent tree excluding self-loop. Per-run snapshots bind
model/kernel contents, identities and executable; mappings attest live libraries.
The capture helper used for correctness is preserved by its exact recorded hash
in `tool-snapshots/correctness-pipeline_capture.py`. Timing adds only a locked
launch-preflight callback; both helper versions are retained. Raw metadata is not
rewritten to point at the preserved helper.


## Candidate and redundant split rationale

Each paired projection dispatches the same activation twice, against high and
low weight halves. The high-weight half already validates the FP32 input and
forms its BF16 residual. The candidate low-weight half borrows this same
host-owned residual through its two existing dispatches instead of repeating
the split/validation loop. The first validation, four GEMM products, three
output copies and FP32 `(a+b)+(c+d)` grouping remain unchanged. The opt-in stays
OFF by default. Product/sum arithmetic and workspace ownership are unchanged;
there is no separate allocation-strategy refactor.

The default-OFF candidate is committed as c5d20c85; PR4A evidence is committed
as ad867144. The narrow production changes are
[the prepared-input fold helper](../../../src/open_qwen36/prefill_precision.hpp:51),
[the Core opt-in and call site](../../../src/open_qwen36/core.cpp:622), and
[the default-false member](../../../src/open_qwen36/core.hpp:648).
The environment control is classified in
[the registry generator](../../../utilities/generate-registry.py).
The candidate changes duplicate host work, not product inputs, output layout,
dispatch order, grouped FP32 rounding, kernel code or production defaults.

## Accepted full-depth correctness scope

All 13 full-64-layer captures pass their scoped comparisons. Real prompt lengths
255, 256 and 257 each use max-tokens 17. Fresh candidate OFF/ON complete captured
logit buffers are byte-exact, including the padded vocabulary (248320 floats),
for the final prompt position and all 16 decode positions; all 17 greedy IDs match.
Candidate OFF also matches the original archived control at 256 tokens.

For each boundary and arm, a separate resident repeat-2 process reproduces both
greedy trajectories. The final repeated logits are byte-exact to fresh logits.
The existing CLI checkpoints/restores before the second request, whose request
path resets state. Dumps reuse logit filenames: only final-repeat logit files
remain, and no separate first-repeat logit retention is claimed.

State samples use manifest layout: layers 0 and 62 include the entire 3,317,760
byte linear allocation (FP32 convolution history followed by recurrence storage,
state_s_off 122880); full-attention layers 3 and 63 include all populated rows
from offset zero, row size 8192, through prompt+16 positions. State samples match
byte-exact OFF/ON and fresh/repeat processes. State dumps occur only after the
first request, not after each repeat. Per-repeat state equality, distinct-prompt
request isolation, every-layer state and independent checkpoint correctness are
**NOT MEASURED**. Thus this is a scoped reset check, not full serving acceptance.
All raw buffers and their sizes/digests are retained in the JSON inventory.


## Distinct-prompt reset/isolation prerequisite

The inspected native `run_list` takes one comma-separated prompt per line via
`--ids-list`, writes to an existing `--list-out` directory, and calls `Core.reset`
before each request. It writes complete real-vocabulary prompt logits at every
position and 17 generated IDs per request. The prescribed fixtures are a
20-token A (seed 26026), a different 13-token B (seed 26027), then A again.
For both OFF and ON the tooling requires p0/p2 complete logits and generated-ID
file hashes to match fresh A, plus complete OFF/ON outputs and runtime identity.
List execution does not expose per-request state dumps; those remain unmeasured.

Only OFF fresh A is attempted, twice, in separate new directories. Both processes
return 1 at hardware-context creation before residency or output. Their actual
argv/environment, token-list inputs, pre-launch artifact snapshots and logs are
retained, not reconstructed into successful capture metadata. No successful live
runtime mapping, RSS/HWM, logits or generated IDs is claimed for these failures.
The initial tooling version is archived outside the raw directories. The JSON
links both attempts and device/kernel-journal diagnostics. No A/B/A pass marker
exists, and the timing entry point refuses to run without that marker.

## Retained timing observations — not the initial five-pair result

Measurements below are the original single-run observations on the machine above,
not new measurements. The primary metric is outer CLI prefill wall, including
profiling and wrapper work, with no subtraction. RSS/HWM are whole-process peaks
sampled at nominal 20 ms, not phase-specific allocation measurements.

| Pair | Arm | Selection | Prefill ms | Fold split ms | Fold total ms | RSS MiB | HWM MiB |
|---|---|---|---:|---:|---:|---:|---:|
| -1 | off | excluded warmup | 30255 | 1398.211 | 2773.410 | 33446.484 | 33465.859 |
| -1 | on | excluded warmup | 29394 | 693.700 | 2086.183 | 33467.598 | 33467.598 |
| 0 | off | retained | 30264 | 1368.179 | 2721.600 | 33446.418 | 33465.613 |
| 0 | on | retained | 30491 | 738.366 | 2143.735 | 33459.613 | 33466.438 |
| 1 | off | retained | 30010 | 1302.455 | 2653.522 | 33437.000 | 33466.336 |
| 1 | on | retained | 29660 | 739.293 | 2130.376 | 33446.445 | 33466.156 |
| 2 | off | unpaired, excluded | 30219 | 1296.644 | 2672.822 | 33436.176 | 33466.242 |

Retained paired reductions: -0.750066%, 1.166278%.
Their descriptive two-pair median is 0.208106%,
spread 1.916344 percentage points and
1/2 wins. These do not establish the initial-five target or
authorize confirmation. Decode before/after samples are **NOT MEASURED**.

The explicit selection in JSON retains original pairs 0/1 and reserves exactly
three fresh pairs 2/3/4 under resume-prefill256 names, after one excluded restart
warmup. Original pair 2 OFF remains an excluded unpaired observation. Original
pair 2 ON remains invalid: final process metadata is absent and is never
reconstructed, even though its timer inequality is now structurally valid.
The original warmup, failed raw files, selection and conditions remain unchanged;
the pre-continuation inventory verifies their bytes. Prior report snapshots are
archived separately under Models before current reports are rewritten.

Five pairs each of 512/1024 prefill and 32-prompt plus 32 generated decode steps,
each after one excluded scenario warmup, remain pending behind correctness.
Around-4K decode is **NOT MEASURED**: genuinely populated context has additional
cost and is not practical while basic hardware-context creation is blocked.
No seek or empty-context substitute is used. No confirmation is collected.

## Structural timer validation

Core request timing ends before profile serialization, fold flush and local
deallocation. CLI prefill wall also includes printing/setup. Inner totals must
therefore fit inside outer wall with only the printed rounding allowance:
sum(inner) - outer <= 0.5 + 0.0005*N + 1e-9 ms. There is no upper bound on
legitimate excluded wrapper work. The integer outer-ms and outer-ms/token
rounding intervals must intersect. Block geometry lines must be after residency
and before the prefill boundary. Exact log/profile timing agreement, layer sums,
finite checks, labels, request/fold counts and generation checks are preserved.
These are scope checks, not relaxed numerical or performance thresholds.

### Rejected experiments / hypotheses

Symmetric timer equality is rejected as a structural hypothesis: the retained
invalid ON attempt has inner total 29490.778 ms and outer 29493 ms, but excluded
serialization/flush/deallocation/printing work has no 2 ms upper bound. The old
rejection does not establish a bad timing measurement. Its absent final process
metadata independently makes it unusable; correcting the inequality cannot
recover that metadata. No sample is relabelled or accepted retroactively.

## Application/server and verification

Application startup and server inference remain **NOT MEASURED / unmet**. The
prior Docker configure attempts encounter read-only SentencePiece/abseil
dependencies; their original command/log records remain linked in JSON.
No retained build container is found in the read-only container inventory.
The current hardware prerequisite prevents promotion regardless of application
build success, so no unrelated dependency modifications are attempted.
Full 64-layer CLI correctness is scoped evidence, not full-application acceptance.

Pre-hardware focused CPU tests: 322 passed. Final focused result:
329 passed in 2.22s. Final portable result:
2930 passed, 172 skipped in 66.97s (0:01:06). Skips are not passes. Settings CI:
NOT RUN: dirty-tree refusal; parent commit required. The earlier dirty-tree refusal is
retained separately and is not counted as a pass.
Exact test logs, full commands/environments, build IDs, runtime hashes, original
capture references and continuation tool hashes are in JSON and retained local
evidence. No further production optimization, default promotion, push, merge or
rebase is performed. The final evidence/tooling commit is separate from the
already committed candidate and PR4A snapshot.
