# Qwen3.8-27B validation report


## Current full-ax L3 replay and boundary checks — 2026-10-28

Added a one-layer native replay for current L3 using sequential L2 inputs from
the audited long-prefix trace. The standard ax binary, weights/manifest ABI and
arithmetic are unchanged. Native KV advances from zero through P239; RoPE and
KV offsets are patched for every position. No state reset or CPU state injection
occurs between tokens. CPU reference build metadata remains immutable.

All **240/240 native residuals match the full-model trace byte-for-byte**.
Separate native audit confirms **720 captures**, totaling **307,691,520 bytes**:
residual, complete activation scratch and active KV prefix at every position.
Logical NPU buffers total **254,072,832 bytes**, with one weight pool resident.
The existing harness and its 326 compiler dependencies are verified and reused;
no kernel, harness or library rebuild was needed.

A new boundary scorer requires that independently audited exact control before
interpretation. It checks unrounded FP64 entry/projection and causal attention
from actual native layer inputs, then conditioned attention-head and FFN gates.
The CPU run and a separate recalculation audit both pass and reproduce every
recorded boundary measurement.

| Boundary coverage | Passed |
|---|---:|
| Entry, output projection and causal attention sublayer, P0–239 | 240 / 240 |
| Attention heads, conditioned on captured Q/K/V/gate | 5760 / 5760 |
| FFN front and split down, P125/P205/P239 | 3 / 3 |

Worst entry projection maxrel is **0.0002980986** for V at P197; worst entry
norm maxrel is **0.0000203582** at P113. The conditioned attention-head maximum
absolute error is **0.0000566654** (P76/H20), below 0.02; all cosine gates pass.
Independent attention-sublayer residual maxrel peaks at **0.0000391476** at P144.
At P205, conditioned FFN activation maxrel is **0.0000374043**, while the closing
residual error conditioned on the captured FFN activation is **7.95406e-8**.
These metrics use different boundary scales and must not be added or ranked as
contributions to the final residual. No tolerance was relaxed.

The scorer does not expose separate raw up/gate projections or softmax
accumulators. KV snapshots are retained for subsequent attribution; independently
recomputed KV is not a measurement of native KV arithmetic. All local gates
passing does not remove the known full-model sequential numerical **FAIL**.
Next perform signed attribution at P125/P205/P239 on a common final-residual
scale, preserving causal histories and reproducing the retained mixed-oracle
outputs before replacing entry, attention/output or FFN boundaries. Include
native KV in the analysis before proposing a precision change.

TDD covers causal position order, persistent KV, active-prefix inventory,
current ABI validation, exact control binding, stale harness evidence and
complete boundary coverage. **25 new tests pass**; full host suite:
**1958 passed, 47 skipped**. Production runtime, installed models and services
are unchanged. Evidence is retained under
`Models/qwen38-27b/post-prefill-drift/native-l3-v1` and `l3-boundaries-v1`,
with RED/GREEN, preparation, native-run and audit logs alongside them.



## FP32 causal attribution across all layers — 2026-10-27

Extended the unchanged conditioned replay to every layer at P125/P205/P239.
Each trajectory starts from zero state and consumes every causal input through
P239; this is a three-position output sweep, not attribution of all prompt rows.
The run reconstructs **192/192 independent reference outputs exactly** and saves
**384** sequential/hybrid array sets. A separate audit repeats every calculation
and confirms all **384/384** saved array sets byte-for-byte.
All **18/18** overlapping array sets and measurements from the previous targeted
run match byte-for-byte, with identical input and source inventories.

| Route | Selected total-output gates | First failing layer at P125 / P205 / P239 | Worst total maxrel |
|---|---:|---|---:|
| Sequential | 168 / 192 | 54 / 50 / 41 | 0.0992221863 at L58/P205 |
| Hybrid | 192 / 192 | None | 0.0025616438 at L58/P205 |

In all 24 failing sequential cases, the current-input contribution has greater
maxrel than both input-history and implementation-gap contributions. The largest
sequential implementation gap is **0.0000631635 at L63/P239**, and the largest
input-history contribution is **0.0002090308 at L63/P239**. Both are below the
unchanged 0.005 output bound at every selected point. The corresponding hybrid
maxima are 0.0000107918 and 0.0000088873. These are conditioned diagnostics;
they do not override the native baseline's overall numerical **FAIL**.

| Sequential point | Current input | Input history | Implementation gap | Total maxrel |
|---|---:|---:|---:|---:|
| L3/P205 | 0.0000084441 | 0.0000008735 | 0.0000185624 | 0.0000109918 |
| L31/P125 | 0.0000241155 | 0.0000588564 | 0.0000019329 | 0.0000719554 |
| L50/P205 | 0.0282954522 | 0.0000431698 | 0.0000564713 | 0.0283734452 |
| L63/P239 | 0.0080316072 | 0.0002090308 | 0.0000631635 | 0.0080632683 |

L3 contains the three largest selected implementation gaps before L41, at all
three positions. This makes it a useful early capture target, not a proven root
cause. Its signed components partially cancel: decreasing one component alone
does not guarantee a smaller total error. The sweep also observes history effects
at L31/P125 and L39/P125, and predominantly input-driven failure at L50/P205.
Each layer uses its own reference scale; cross-layer maxrel ratios are not causal
amplification measurements. Unselected positions may contain other contributors.

Next prepare a current-standard full-ax L3 replay from the retained sequential
L2 residual prefix, advancing native KV from zero through P239. Require exact
native residual identity before interpreting scratch, Q/KV/attention, output
projection and FFN boundaries at P125/P205/P239. Reuse the current production ax
binary and manifest ABI; historical attention experiments are not valid controls
for this export. The implementation gap includes native accumulated state error,
which cannot be identified from ideal replay state alone. Do not change arithmetic
or relax thresholds before obtaining those boundary measurements.

The evaluator and production code are unchanged. The 35 existing focused tests
pass again; the last full host suite remains 1933 passed / 47 skipped from the
preceding implementation stage. The replay pins 6686 input files and caches one
decoded layer at a time (peak weight cache 1.42578125 GiB). No NPU execution,
kernel/library build, installed model or service change occurred. Evidence:
`Models/qwen38-27b/post-prefill-drift/all-layers-v1`, the run/audit logs,
`all-layers-prior-identity.json`, and `all-layers-components.csv`.



## FP32 sequential causal attribution — 2026-10-27

Added CPU conditioned replay for the audited standard FP32 boundary captures.
The adapter remeasures all three prompt lengths and ordinary/traced routes,
pins 6686 input files, and preserves the native baseline's numerical FAIL.
It caches only one decoded layer at a time (peak weight cache 1.42578125 GiB).
No production arithmetic, native captures or installed runtime was changed.

Run and separate audit complete: **9/9 exact reference reconstructions** and
**18/18 byte-exact NPZ replays**, covering both routes at L0/P125,205,239,
L40–41/P239, L53–54/P125 and L57–58/P205. L0 input/history effects are zero.
All signed decompositions close; reference-rounding effects are zero.

| Sequential target | Current input | Input history | Implementation gap | Total maxrel |
|---|---:|---:|---:|---:|
| L41/P239 | 0.0067798097 | 0.0000183318 | 0.0000143036 | 0.0067813607 |
| L54/P125 | 0.0056910580 | 0.0000078442 | 0.0000126243 | 0.0056958381 |
| L58/P205 | 0.0992276775 | 0.0000043239 | 0.0000081645 | 0.0992221863 |

Components share the output reference scale, but their scalar maxima cannot be
added: signed vectors can cancel. The ideal layer driven by native inputs
already reproduces the failing magnitude at these targets. The largest selected
sequential implementation gap is **1.43036e-5**. L57/P205 already has total
maxrel 0.0926540506. Selected total-output gates pass 5/9 sequential and 9/9
hybrid cases; hybrid worst maxrel is 0.0025616438.

This localizes the dominant effect to incoming errors at the selected targets,
not their original source. The implementation gap includes native arithmetic
and accumulated native state error; ideal replay state is not a native state
capture. Next extend conditioned replay through all layers at P125/P205/P239
to locate earlier contributions before choosing sublayer captures or a fix.
Overall numerical acceptance remains **FAIL**, with unchanged thresholds.

TDD: eight new tests fail before implementation and pass afterward; 35 focused
tests pass. Full host suite: **1933 passed, 47 skipped**. Evidence is retained
under `Models/qwen38-27b/post-prefill-drift/targets-v1` with a separate audit,
`targets-summary.json`, and test logs. No kernel/library build or NPU run was
needed for this diagnosis. Long-prefix reset, serving and performance remain open.


## FP32 full-depth block-boundary results — 2026-10-27

Completed all **18 resident processes** for 255/256/257-token prompts, all 64
layers and one continuation, using the verified ordinary and capture-only CLIs
with the standard FP32 post-output export. A separate audit remeasures every
capture and confirms current source/build/model/reference identity. Run and
audit both report **complete: true, passed: false**: execution is complete,
but sequential numerical accuracy remains a blocker. Exit status 1 records
that numerical failure, not a process or provenance failure.

| Route, all three lengths | Residual rows passed | Logits passed | Worst residual maxrel | Minimum logit correlation |
|---|---:|---:|---:|---:|
| Sequential | 49,058 / 49,344 | 759 / 771 | 0.0992221863 | 0.9963255072 |
| Hybrid, layer-major | 49,344 / 49,344 | 771 / 771 | 0.0025616438 | 0.9999961799 |
| Hybrid, block-major | 49,344 / 49,344 | 771 / 771 | 0.0025616438 | 0.9999961799 |

Sequential residual/logit passes by length are 16,290/16,384 and 252/256
(255 tokens), 16,352/16,448 and 253/257 (256), and 16,416/16,512 and 254/258
(257). Thresholds remain residual maxrel <0.005 and logit correlation >0.9999
with matching argmax. The ordinary CLI reproduces the traced logit results.
All argmax values and generated IDs (248046, 198) match the CPU reference.
All **576 continuation residual rows** pass; failures occur in prompt rows.

All **2331/2331 fresh plain/trace control files**, **966/966 hybrid schedule
identities**, and **1917/1917 cross-length prompt-prefix identities** pass.
The latter cover 1533 logits and 384 residual prefixes, excluding continuations.
The common-prefix error is therefore reproducible across lengths and ordinary
versus instrumented execution; it already appears before the block boundary.

The earliest temporal residual crossing is **L54/P125**, maxrel 0.0056958381.
The shallowest failing layer is **L41/P239**, maxrel 0.0067813607. Worst error
is **L58/P205**, maxrel 0.0992221863. Logit correlation fails at P205, P213,
P214 and P220 for each length. These are observed crossings, not attribution
to a particular kernel, arithmetic operation or native state representation.

Next replay the current causal inputs at these targets and their preceding
layers, with one decoded layer at a time, to separate propagated input/history
error from local implementation error. Preserve the failed overall acceptance
while selecting subsequent native sublayer/state captures. Long-prefix reset,
serving and performance remain unaccepted; hybrid host stages remain explicit.

Evidence: `Models/qwen38-27b/post-prefill-production/boundaries-native-v1`,
its separate `audit.json`, and `boundaries-native-summary.json`. All 38 focused
validator tests pass; the unchanged tooling's full host suite passed 1925 tests
with 47 skips at preparation. No runtime/kernel source, installed bundle or
service was changed, and no kernel/library rebuild was needed.


## FP32 block-boundary preparation — 2026-10-27

Added a preparation/runner/auditor for 255/256/257-token prefill with the standard
FP32 export and the verified plain/trace shared CLIs. Reference metadata remains
immutable. The preparation binds it to the passing reference audit, checks the
model and oracle against the fresh short-run baseline, and revalidates current
kernel and CLI identities. No production runtime or kernel changes were made.

Preparation and a separate recheck pass: **1155 independent reference files**,
**1162 input hashes**, **128 residual-array prefixes** and **511 logit prefixes**.
The CPU-selected continuation IDs are 248046, 198 for each length. Existing
builds are reused after verification; no new kernel or library build is needed.

The runner requires 18 resident processes: plain/traced controls for all three
lengths and routes. It checks fresh control identity, independent all-layer
precision, hybrid schedule identity and causal prefixes excluding continuations.
A completed numerical failure remains `complete: true, passed: false`, including
after separate audit. Partial or changed evidence cannot become a passing result.
TDD covers missing lengths/routes, numerical and identity failures, oracle and
geometry mismatches, metadata audit binding and prompt/continuation separation.
Host suite: **1925 passed, 47 skipped**.

**Native execution was deferred at this preparation checkpoint.** The memory preflight measured **34.1 GiB**
available, below its unchanged 35 GiB threshold. No model was loaded and no NPU
boundary-accuracy claim is made. Preparation evidence is retained under
`Models/qwen38-27b/post-prefill-production/boundaries-prepared-v1`, with its
separate preparation audit and `boundaries-memory-preflight.json`. Native
boundary/long-prefix validation, serving and performance gates remain open.


## Resident short-prefill reset on the FP32 export — 2026-10-26

Added a runner and separate auditor for repeated requests against the fresh
FP32 short-prefill baseline. Both baseline report and audit are pinned; the
runner recomputes their numerical measurements and verifies current builds,
model, CPU references and export inputs before and after execution. Production
runtime and kernels are unchanged. The new capture build preserves `request()`
and `Core::reset()`, with separate output files and ordered request log markers.

Three resident processes cover sequential NPU and both hybrid schedules. Each
executes two identical 20-token prompts plus one continuation through all 64
layers. Both requests independently pass the unchanged gates: **8064/8064
residual rows** and **126/126 logit cases**. All **261/261 first-request files**
match the audited single-request baseline; all **261/261 second-request files**
match the first. Hybrid schedules agree on **86/86 residual/logit files**.
Separate audit reproduces all measurements and verifies their provenance.

Worst residual maxrel is unchanged between requests: sequential **0.0008187178**
at L63/P17 and hybrid **0.0000779139** at L61/P16, below the 0.005 bound. Minimum
logit correlation is **0.9999978970**, above 0.9999; argmax and generated tokens
match. The fresh repeat CLI pins 590 compiler dependencies and passes **CTest
4/4**. TDD covers stale or failed baseline proof, changed inputs/measurements,
failed second-request gates and invalidation of prior audit success. Host suite:
**1912 passed, 47 skipped**.

Evidence: `Models/qwen38-27b/post-prefill-production/{repeat-v1,reset-v1}` and
`reset-summary.json`. This checks ordinary reset for identical short prompts,
including the existing checkpoint/restore before the second request. It does
not establish full-buffer zeroing, different-prompt isolation, block-boundary
or long-prefix precision, serving or performance acceptance. Hybrid host stages
remain explicit. No installed bundle, model configuration or service changed.


## Full-depth short prefill with FP32 post-output — 2026-10-26

Added a byte-unmodified shared-CLI build mode and a separate runner/auditor for
new-arithmetic prefill. The existing residual tracer retains shared runtime
arithmetic. Its output must reproduce a fresh ordinary CLI run of the same
standard export, rather than historical BF16 native output hashes. Production
runtime, model packing and kernel binaries are unchanged in this checkpoint.

The fixture binds 149 retained independent CPU reference files to their passing
historical audit and preparation. Model, prompt and oracle code hashes remain
unchanged; the current manifest and source key are checked separately. All 257
input hashes pass. Both fresh builds use GCC 15.2, record 589/590 compiler
dependencies and pass native CTest **4/4** each.

Six resident executions cover sequential NPU and both hybrid schedules using
ordinary and capture CLIs: 64 layers, 20 prompt tokens and one continuation.
All **4032/4032 residual rows** and **63/63 traced logit cases** pass the unchanged
bounds. The ordinary CLI also passes all **63/63 logit cases** and continuation
residuals. All **69/69 trace/control files** (logits and final scratch) match
byte-for-byte; both hybrid schedules agree on **86/86 residual/logit files**.
Separate audit reproduces measurements and verifies source/build/input identity.

| Route | Worst residual maxrel | Minimum logit correlation |
|---|---:|---:|
| Sequential | 0.0008187178 (L63/P17) | 0.9999978970 |
| Hybrid, layer-major | 0.0000779139 (L61/P16) | 0.9999979147 |
| Hybrid, block-major | 0.0000779139 (L61/P16) | 0.9999979147 |

The previous standard short-prompt maxima were 0.0048820188 and 0.0012128028.
The reference and bounds (residual <0.005, correlation >0.9999, matching argmax)
are unchanged. Fresh input validation checks all model and oracle hashes rather
than exempting artifacts because they are historical.

TDD covers plain-source identity, complete capture inventories, independent
numerical failures, logits/alias/scratch mismatch, changed input provenance and
invalidating a prior successful audit after a failed recheck. Full host suite:
**1899 passed, 47 skipped**. Evidence is retained under
`Models/qwen38-27b/post-prefill-production/{plain-v1,trace-v1,short-v1}`.

This validates short prefill on the standard FP32 export. Repeated short-prompt
requests, block-boundary and long-prefix runs remain required; the passing
three-position decode reset does not replace them. Hybrid host stages remain
explicit. No serving or performance acceptance is claimed, and no installed
bundle or running service was changed.


## Standard FP32 DeltaNet post-output — 2026-10-26

Promoted the shared FP32 post-output path into the standard wide dense Q4 recipe.
The 24576-byte region is appended at byte 393216; linear scratch is now 417792
bytes. Existing scratch offsets, recurrent state and weight packing stay intact.
Manifest v4 declares the post region and the native reader rejects invalid dtype,
alignment, state kind and bounds. The private BF16 control explicitly retains
its old layout. Other model geometries and mixed Q8 roles keep their storage.

A fresh model-derived export builds all **46 sets / 92 binary hashes**. Compiled
consistent-sum/factored headers match current production sources, including
isolated staging copies. Standard lx instructions match the audited private FP32
candidate exactly; the XCLBIN differs only in 80 build-stamp bytes. lx/ax maximum
text is **16000 / 13760 bytes**, assigned L1 end **62336 / 59920**, stacks **6144**.
No resource limits were raised. Build key:
`67f3c4395334a0bc3aa0dcb2bcff398cfa01147451cddd3b022afca4215c7a13`.

All **66 model/pool hashes** and **195 fresh CPU reference files** match retained
provenance. A new 64-layer, three-position NPU run completes 198 dispatches;
**192/192 residuals pass**, worst maxrel **0.0001072489** at L63/T1 against 0.005.
The previous standard three-position maximum was 0.0030399006. Logit correlations
are **0.9999993162 / 0.9999987908 / 0.9999988379**, with CPU/NPU argmax
**8678 / 198 / 2**. Acceptance thresholds and the independent oracle are unchanged.

All seven independent boundary checkers pass: DeltaNet input/state **144/144**,
strict recurrence heads **6912/6912**, attention heads **1152/1152**, attention
projections **48/48**, FFN front/down **192/192** each and LM-head **3/3**.
A fresh shared CLI runs two resident requests; all three second-request logits
match the harness bytewise. Full-buffer forced-token reset has **0/3** differing
replays. CPU weight checkers and resident NPU runs were serialized.

TDD includes recipe/manifest and native parser failures before implementation,
plus an audit regression for isolated staged headers. Host suite: **1886 passed,
47 skipped**; native CTest **4/4**. The manifest regression executable also
loads the actual new export successfully. Evidence is retained under
`Models/qwen38-27b/production-post-fp32/` with binary/resource, source, reference,
numerical and lifecycle reports. No installed bundle or running service changed.

This completes standard-export integration and short decode validation. Fresh
short prefill, block-boundary and long-prefix checks remain required for this
export. Serving and measured performance acceptance are still open; full model
acceptance is not claimed.


## Shared DeltaNet post-output checkpoint, 2026-10-26

Moved the private FP32 post-output specialization into the shared post kernel,
FIFO declarations, dense DMA and output-projection preparation. One explicit
`Linear.POST_F32` selection controls all four boundaries. The private builder
now selects that path rather than rewriting arithmetic/stream source strings;
its scratch append remains private until recipe/manifest promotion.

Fresh BF16 and FP32 full-lx builds reproduce their previously audited binaries:
`insts.bin` is exact; complete XCLBINs differ only in documented build stamps
(79 / 77 bytes). Comparison records bind the old passing causal audit, build
record, binary hashes and all 729 capture files for each mode. This preserves
historical evidence without treating its source tree as the current source.
Current sources, compiler dependencies and resource audits are checked separately.

Both builds pass all 11 core budgets. Maximum linked text is 15936 / 16000
bytes; maximum assigned L1 buffer end is 62336; stacks remain 6144 bytes.
Each build pins 849 compiler dependency entries. Toolchain: mlir-aie 1.4.2,
llvm-aie 21.0.0.2026080301+c9c5ecb7. Evidence is in
`Models/qwen38-27b/deltanet-post-fp32/shared-{control,fp32}-v1` and the matching
`shared-*-identity-v1.json` reports.

Eight new stream/store cases follow RED-to-GREEN. Thirteen additional cases
check archive integrity and preserve the published Qwen3.5/Qwen3.6 post ABI.
Full host suite: **1878 passed, 47 skipped**. Production recipes still select
BF16 post output. This is shared-code preparation and binary-equivalence
evidence, not a fresh numerical/performance run or full-model acceptance.
The next change must promote FP32 scratch and manifest semantics through the
standard export, then validate full-depth short and long-prefix propagation.


## Latest checkpoint — integrated all-layer validation, 2026-10-26

Added capture-only builders for the shared runtime and a runner/auditor for the
standard manifest-v4 export. The builders preserve projection arithmetic and
`request()`/`Core::reset()`, adding synchronized residual readback and optional
separate first/repeated capture writers. Verification reconstructs the allowed
instrumentation and pins original/staged sources, 590 compiler dependencies,
CMake cache and executable hashes. Both builds use GCC 15.2 and pass CTest 4/4.
No NPU kernels, weight packing or runtime arithmetic changed in this checkpoint.

The single-request integrated check passes sequential NPU and both hybrid
schedules over 64 layers, 20 prompt tokens and one continuation. All
**4032/4032 residual rows** and **63/63 independent logit cases** pass unchanged
bounds. All **261/261 files** reproduce the retained accepted captures exactly;
the two hybrid schedules agree on all **86/86 residual/logit files**. Separate
re-audit recomputes the numerical results and confirms hashes and identities.

| Route | Worst residual maxrel | Minimum logit correlation |
|---|---:|---:|
| Sequential | 0.0048820188 (L63/P18) | 0.9999943778 |
| Hybrid, layer-major | 0.0012128028 (L0/P20) | 0.9999979147 |
| Hybrid, block-major | 0.0012128028 (L0/P20) | 0.9999979147 |

Resident reset validation and its separate audit also pass. Three resident
processes execute two requests each with distinct capture writers: **261/261**
first-request files match the single baseline and **261/261** second-request
files match the first. Across six requests, **8064/8064 residual rows** and
**126/126 independent logit cases** pass.

Nine new RED-to-GREEN tests cover permitted instrumentation, complete capture
inventories, exact reset identity and aggregation that cannot hide a failed
route or numerical gate. Full host suite: **1857 passed, 47 skipped**.
Evidence is under `Models/qwen38-27b/prefill-production/`: `trace-v1`,
`repeat-v1`, `all-layer-v1` and `reset-v1`. The existing `kernels-model-v1` bundle
and audited short smoke remain pinned inputs. Historical metadata is preserved;
references are bound to the original preparation, and changed model or binary
artifacts have no exemption.

This is short-prompt integration evidence. It does not establish long-prefix
reset, full-buffer zeroing, independent checkpoint/restore semantics or serving
request isolation. Sequential long-prefix precision still fails its known gate;
hybrid recurrence and norms still run on the host. Full model acceptance remains
open, with FP32 DeltaNet post-output promotion a separate change. Performance
acceptance also remains open: paired projections currently disable stage-major
submission and paired KV uses host attention. These instrumented runs serialize
readback and are not production performance benchmarks.

## Latest checkpoint — standard paired-prefill export, 2026-10-25

The ordinary qwen35 recipe now exports five low-weight GEMM companions for
validated dense Q4 H5120/FF17408 geometry, producing **46 sets** and manifest
version 4. The shared GEMM design computes BF16 dequantization residuals from
the original Q4NX weights. High kernels retain their default specialization;
other model sizes and MoE exports are unchanged. Runtime paths remain relative
to the exported bundle. No weight conversion or manual manifest edits are needed.

A private build root now isolates source staging and compiler lock files as
well as outputs. Two initial attempts hit Docker-owned shared staging/locks;
regression tests preceded both fixes. The third complete export succeeds.
A second normal export reuses those binaries with the local model's exact spec:
`real_vocab=248077` instead of the geometry fixture's 248320. Padded vocabulary
and kernel geometry agree; `--model-dir` derives the same spec automatically.

All five low GEMMs pass compiler-dependency and **32-core** resource checks:
maximum linked text **5568/16384 bytes**, assigned L1 buffer end **61448/65536**,
stack **4096 bytes**. Each instruction stream is byte-identical to its retained
validated low-weight control; XCLBIN comparison differs only in build stamps.
The original memory limits and independent numerical thresholds are unchanged.

The unpatched shared CLI loads all 64 layers from the new bundle. For a
20-token prompt and one continuation, both hybrid schedules reproduce all
**44/44 logit files** exactly and pass **42/42 independent CPU logit cases**;
continuation IDs are 17 and 478. The smoke runner pins model/kernel/reference
hashes, binds reference metadata to the audited preparation, and rejects missing,
extra, truncated or non-finite outputs. Full host suite: **1848 passed,
47 skipped**. Seven new smoke-gate cases and nine recipe/export cases follow TDD.

Evidence: `Models/qwen38-27b/prefill-production/` contains `build-v3`,
`kernels-v3`, `kernels-model-v1`, `resources-v1.json`, `low-identity-v1.json`
and `smoke-v1`/`smoke-v2`. The first smoke passed; the second uses the final
reference-metadata binding guard. The normal application already includes the
shared runtime, but an executable rebuild alone does not replace an installed
older kernel bundle. This checkpoint does not activate a running service.

This is export and short-logit integration evidence. Fresh integrated all-layer
residual, long-prefix and resident reset checks remain open, as do serving and
full model acceptance. Hybrid norms/recurrence still execute on the host.
FP32 DeltaNet post-output promotion remains a separate change; sequential
long-prefix precision is still a known failing gate.

## Latest checkpoint — shared paired prefill runtime, 2026-10-25

Moved four-product prefill composition into the shared Core compiled by both
`oflm` and `open_qwen36_cli`. Manifest version 4 explicitly binds high/low
projection roles, validates wide dense geometry and state ABI, and rejects
missing, partial or swapped companions. Versions 1–3 retain their single
dispatch behavior. Stage-major direct submission is disabled on paired routes
so it cannot silently bypass input/weight corrections. Host sums preserve the
validated order and never modify mapped device outputs.

Full host suite: **1832 passed, 47 skipped**. Eighteen RED-to-GREEN CPU
regression cases cover cancellation-sensitive sum order, output
ownership, non-finite inputs, legacy compatibility and malformed declarations.
Standalone CLI builds with the existing XRT development tree and passes CTest
4/4. The standard `oflm` application builds in the existing Docker builder; with
`OFLM_CONFIG_PATH=/code/src/model_list.json`, `oflm version` exits 0.
The initial local build lacked installed XRT headers; the retry uses the same
development headers/libraries as earlier accepted CLI builds. The first Docker
retry lacked permission to write old artifacts; the normal builder user works.

This is runtime integration, not an activated kernel export or new resident
accuracy result. Production export of five low-weight companion streams and
fresh integrated numerical/reset checks remain open. No NPU execution, model
weight changes or FP32 post-output promotion occurred in this checkpoint.

## Latest checkpoint — private FP32 DeltaNet post-output, 2026-10-24

Added a private full-lx builder and causal replay that retain FP32 post-recurrence
norm/gate output through direct FP32 Q4 output-projection preparation. The fresh
BF16 control reproduces all **729 original activation/state/residual captures
byte-for-byte** over 243 positions. The candidate preserves entry norm, QKV/Z,
records, recurrence output and complete native state exactly at **243/243**
positions. Independent final residual gates pass **243/243**, and all six selected
post/output/FFN-front/down cases pass unchanged bounds.

| Position | Control residual maxrel | FP32 residual maxrel | Reduction |
|---|---:|---:|---:|
| P36 | 0.0008119891 | 0.0000015643 | 519x |
| P37 | 0.0011891532 | 0.0000263390 | 45x |
| P38 | 0.0000585909 | 0.0000129919 | 4.5x |
| P240 | 0.0010099215 | 0.0000018431 | 548x |
| P241 | 0.0012928460 | 0.0000038302 | 338x |
| P242 | 0.0000202869 | 0.0000123079 | 1.65x |

Across the prefix, worst residual maxrel drops from **0.0015055132 (P143)** to
**0.0000357078 (P116)**; median drops from **0.0003848893** to **0.0000083408**.
242 positions improve. P202 rises slightly from **0.0000257763** to
**0.0000260274**, still within the original bound. P242 improves despite the
prior frozen-error CPU prediction because this experiment changes both post
storage and projection preparation; the downstream error is not frozen.

All 11 cores pass the original resource limits. Maximum linked text is
**15,936 / 16,000 bytes** for control/candidate; maximum assigned L1 buffer end
is **62,336 bytes** for both. The candidate appends **24,576 bytes** to scratch
(393,216 -> 417,792), retains the 3,317,760-byte state ABI and streams six FP32
blocks through the existing depth-two FIFO. Each build pins 849 dependency
entries, including current factored recurrence and consistent Q4 helpers.
The initial control-v1 compiled but failed its dependency audit due to IRON's
`post_fn.cc` rename; a regression test fixes the mapping and control-v2 is fresh.

Thirteen RED-to-GREEN tests; full host suite **1814 passed, 47 skipped**.
Control and candidate execute **486 NPU dispatches** in total, with one L0 pool.
Separate numerical re-audits pass for both runs, reproducing every capture
hash and metric with pinned source/build/input provenance.
Artifacts: `Models/qwen38-27b/deltanet-post-fp32/`.
Workflow: `utilities/deltanet-post-fp32/README.md` and the registered
`open-qwen38-fp32-post-output` skill.

This establishes a local L0 precision improvement. Production sources/exports
are unchanged; full-model sequential long-prefix acceptance remains FAIL.
Broaden the private experiment to other linear layers with streamed weights,
fresh exact controls and actual causal prefixes before full-depth propagation.

## Latest checkpoint — signed L0 post-output attribution, 2026-10-24

Added a CPU-only attribution tool over the exact audited production L0 captures.
It reconstructs independent mixed-precision and FP64 histories through P242,
reproduces all **six stored L0 references byte-for-byte**, and propagates selected
post-output changes through output projection and the nonlinear FFN. A separate
causal re-audit reproduces every array in all six NPZ artifacts and all metrics.
Signed decomposition closure is exact for these fixtures (max absolute gap 0).

The replacement chain separates reference rounding/math, recurrence, Z,
post-output BF16 rounding, remaining post arithmetic, output projection,
residual addition and conditioned FFN arithmetic. Every output contribution uses
the same final-reference scale; scalar maxima must not be added.

| Position | Native residual maxrel | Post-rounding contribution | Output-projection contribution | CPU result without post rounding |
|---|---:|---:|---:|---:|
| P36 | 0.0008119891 | 0.0008331544 | 0.0000220021 | 0.0000211652 |
| P37 | 0.0011891532 | 0.0012200074 | 0.0000323789 | 0.0000308542 |
| P38 | 0.0000585909 | 0.0000417831 | 0.0000150979 | 0.0000168078 |
| P240 | 0.0010099215 | 0.0009798124 | 0.0000310685 | 0.0000301091 |
| P241 | 0.0012928460 | 0.0013051784 | 0.0000122293 | 0.0000123325 |
| P242 | 0.0000202869 | 0.0000688372 | 0.0000872370 | 0.0000891240 |

Post-output rounding dominates the selected P37/P241 drift. Removing its signed
contribution while freezing downstream error vectors predicts about **39x/105x**
lower residual error. This is a CPU counterfactual, not a corrected NPU run.
At P242 the prediction worsens because rounding partially cancels projection
error. The chain's nonlinear attribution depends on the stated replacement
order. All selected residuals remain below 0.005; full-model sequential
long-prefix accuracy remains FAIL.

The BF16 rounding prediction differs from captured post output in only **6/7
of 6144 values at P37/P241**, with maxrel **2.4414e-5 / 1.2207e-5**. Remaining
post arithmetic is measured separately rather than attributed to rounding.
Next build a private FP32 post-output/direct-FP32 output-preparation experiment
with a byte-exact production control, all 243 causal positions and explicit
scratch/stream/resource checks. Include P242 and projection gates to expose
cancellation regressions before considering production integration.

Nine RED-to-GREEN tests; full host suite **1801 passed, 47 skipped**. Materialized
mixed/FP64 weights: **4.282 GiB**, one layer; no new build or NPU dispatch.
Artifacts: `Models/qwen38-27b/prefill-sequential-drift/post-attribution-v1/`.
Tool: `utilities/prefill-sequential-drift/post_attribution.py`.
Production code, binaries, retained native captures and thresholds are unchanged.

## Latest checkpoint — exact production L0 replay, 2026-10-24

Added a one-pool production L0 replay over P0–242 with original model embeddings,
continuous native conv/recurrent state and full residual/activation/state capture.
The unchanged lx export completes **243 NPU dispatches**. All **243 residuals
match the retained full-model sequential CLI byte-for-byte**; a separate audit
rechecks all **729 captures (906,743,808 bytes)** and their provenance.

Independent entry/conv/state checks pass **243/243** positions; conditioned
recurrence passes **11,664/11,664 strict head cases**. All six selected
Z/post/output and FFN front/down cases at P36–38 and P240–242 pass unchanged
bounds. No CPU state is substituted into native execution or independent carry.

| Boundary | Worst maxrel over the full prefix |
|---|---:|
| Entry norm | 0.0000165581 |
| QKV / conv history | 0.0000128526 |
| Independent composed state | 0.0000251507 |
| Independent composed output | 0.0000212016 |

The selected post-recurrence norm/gate BF16 boundary has maxrel **0.0029327378
at P37** and **0.0029708429 at P241** against its unrounded conditioned
reference. Selected FFN activation errors at those positions are about 8.49e-6
and 7.30e-6. These local gates pass and use different reference scales; their
scalar errors cannot establish contribution to the final residual. This
nominates post-recurrence BF16 output for signed causal attribution, not a
proven correction or a recurrence defect.

Next quantify the post-output rounding vector through the output projection
and FFN against the independent L0 residual, separating incoming recurrence/Z
error from local rounding and downstream arithmetic. Require byte-exact native
controls before any FP32 post-output experiment. Overall sequential long-prefix
accuracy remains FAIL, and production kernels/defaults are unchanged.

Logical NPU buffers: **0.239906 GiB**, one resident weight pool. Fresh standard
harness: GCC 15.2.0, **326 pinned compiler dependencies**. Link with source-tree
XRT development libraries and run with installed `/opt/xilinx/xrt` plus the
XDNA driver. The initial build-tree-runtime attempt failed before dispatch;
its logs are retained separately. No full model residency is required.

TDD: 19 initial RED-to-GREEN cases plus two failure/overwrite regression tests;
full host suite **1792 passed, 47 skipped**. Evidence:
`Models/qwen38-27b/prefill-sequential-drift/{native-l0-v2,l0-boundaries-v1,harness-l0-v1}/`.
Tools: `utilities/prefill-sequential-drift/{native_l0.py,l0_boundaries.py}`.
Build/replay instructions are in the registered sequential-L0 skill.

## Latest checkpoint — preceding-layer causal attribution, 2026-10-24

Added configurable target selection to the CPU drift diagnostic without changing
its pinned predecessor. Targets bind exact layer/position coverage and refuse
ambiguous or out-of-range selections. Layer zero uses the shared FP32 model
embedding, with exact zero current-input, input-history and propagated vectors
required before interpreting its implementation gap.

Replayed L0/P240–242, L32/P240–242 and L59/P37–39, carrying every causal row.
All **9/9 independent references reconstruct bytewise**, all **18 signed
attributions** close, and selected output gates pass **9/9 for each route**.
A separate CPU audit reruns the histories and reproduces every array in all
18 NPZ artifacts and their metrics bytewise.
These selected gates remain below 0.005; the overall sequential long-prefix
FAIL is unchanged.

| Sequential target | Current input | Input-history effect | Implementation gap | Total maxrel |
|---|---:|---:|---:|---:|
| L0/P241 | 0 | 0 | 0.0012928460 | 0.0012928460 |
| L32/P241 | 0.0032113675 | 0.0001437686 | 0.0000152968 | 0.0030828957 |
| L59/P38 | 0.0041249896 | 0.0000384913 | 0.0000115911 | 0.0040749071 |

L32/L59 mainly transmit incoming error. At L0 the inputs and ideal histories
are exact, isolating the discrepancy to the first layer's arithmetic/native
history; residual-only captures still cannot identify the responsible operation.
Hybrid L0/P241 maxrel is **0.0000016314**, versus sequential **0.0012928460**.
The same embedding token (ID 1235) occurs at P37/88/139/190/241; retained L0
sequential maxrel is already **0.0011891532 at P37**, rising to 0.0012928460
at P241. This nominates two causal-history lengths, not a recurrence diagnosis.

Next prepare an isolated production L0 replay through P242 with one weight
pool, require byte-exact native residual controls for every position, and
capture available entry/conv/recurrence/output/FFN boundaries at P37 and P241
(with neighboring rows). Keep the same production arithmetic and independent
references until those controls identify a concrete precision change.

Fifteen RED-to-GREEN tests; full host suite **1771 passed / 47 skipped**.
Maximum cached decoded weights **1.426 GiB**; no NPU dispatch or native build.
Artifacts: `Models/qwen38-27b/prefill-sequential-drift/predecessors-v1/`;
workflow: `utilities/prefill-sequential-drift/targets.py`. Production integration,
long-prefix reset and model acceptance remain open.

## Latest checkpoint — sequential causal drift attribution, 2026-10-24

Added a CPU diagnostic for L33/P240–242 and L60/P37–39 on the retained
257-token run. It verifies the audited native inputs and immutable preparation,
then replays every preceding token with independent conv/recurrent/KV history.
Weights are decoded and cached one layer at a time (maximum **1.426 GiB** of
cached decoded weights); no NPU dispatch or full-model residency is required.

All **6/6 selected CPU references reconstruct bytewise** and all **12 signed
attributions** close. A separate CPU audit reruns all causal histories and
requires byte-exact equality for every saved NPZ array and metric. Counterfactual state copies separate current-input effects
from effects of earlier native inputs; reference output rounding is explicit
(and zero for these fixtures). Every component uses the same output-reference
scale, so signed vectors add but their scalar maxima generally do not.

| Sequential target | Current input | Input-history effect | Implementation gap | Total maxrel |
|---|---:|---:|---:|---:|
| L33/P241 | 0.0050773486 | 0.0000522907 | 0.0000162030 | 0.0050229750 |
| L60/P38 | 0.0076863502 | 0.0000208493 | 0.0013556628 | 0.0090391466 |

The main error at both threshold crossings is propagated through the current
input. Even the independent layer on native causal inputs fails at these rows:
propagated maxrel is **0.0050250579 / 0.0076834838**. Incoming residual error,
on its own input scale, is **0.0030828957 / 0.0040749071**. L60 additionally has
a measurable implementation gap. This gap includes local arithmetic and actual
native state accumulated over the prefix; these residual-only captures cannot
attribute it to one operation or prove native-state correctness.

Selected sequential output gates pass **4/6**; hybrid passes **6/6**, worst
maxrel **0.0000183606** and implementation gap <=**4.384963e-7**. Native outputs
and overall long-prefix FAIL are unchanged. This is diagnostic attribution,
not a corrected model execution or production acceptance.

Twelve RED-to-GREEN tests cover causal carry, state-copy isolation, route reset,
invalid inputs, signed cancellation/rounding and persisted failures. Full host
suite: **1756 passed / 47 skipped**. The diagnostic binds **6222 input files**
and **12 NPZ artifacts**. Workflow: `utilities/prefill-sequential-drift/`;
artifacts: `Models/qwen38-27b/prefill-sequential-drift/run-v1/`.

Next follow the incoming drift into preceding layers, starting at L32/P241 and
L59/P38 and examining L0/P241, where the retained sequential residual already
has maxrel 0.0012928 despite identical embedding input. Select actual native
sublayer/state capture targets from that evidence. Production integration,
long-prefix reset and final model acceptance remain open.

## Latest checkpoint — expanded long-prefix validation, 2026-10-24

Added preparation, resident runner and CPU audit for immutable 255/256/257-token
references. The runner verifies historical/current source identity and accepted
short-run builds, then requires all fresh paired-input controls before expanded.
Every process checks >=35 GiB available RAM and exclusive device access. It
scores every layer/position and separately checks causal prefixes, schedule
identity and continuation. Completed numerical failure remains distinct from an
incomplete run; acceptance thresholds are unchanged.

All **18 native processes complete**. The separate CPU audit recomputes all
measurements and confirms commands, provenance, controls and prefix identities.
Run/audit exit 1 means completed numerical FAIL, not a process failure.
Expanded hybrid layer-major and block-major
pass all **98,688/98,688 residual gates and 1542/1542 logits**, including one NPU
continuation for each prompt. Minimum correlation is **0.9999961799 >0.9999**,
with no argmax mismatches. Worst hybrid residual maxrel improves from
**0.4325418278 to 0.0025616438 <0.005** (L58/P205).

| Prompt tokens | Expanded residual gates, both hybrid routes | Logits | Final continuation maxrel |
|---|---:|---:|---:|
| 255 | 32,768/32,768 | 512/512 | 0.0001198608 |
| 256 | 32,896/32,896 | 514/514 | 0.0006099073 |
| 257 | 33,024/33,024 | 516/516 | 0.0002888666 |

All **2331 fresh paired control captures** match the historical baseline.
Expanded sequential matches all **969 fresh control captures**, including
all-layer residuals. Each variant passes **1917/1917 cross-length prefix
identities** (1533 logit files and 384 residual prefixes); expanded hybrid
schedules match **966/966 logit/residual files**. Continuations are scored
independently and excluded from comparisons between different prompt lengths.

**Overall acceptance remains FAIL because sequential is unchanged:**
**46,357/49,344 residuals and 550/771 logits** pass. The first temporal residual
crossing is **L60/P38**; the shallowest failing layer is **L33/P241**. Logit
correlation first fails at **P80**, reaches **0.9351669042** at its worst, and
argmax remains correct. Worst sequential residual is **0.5389499720** at L58/P220.
These are threshold crossings, not attribution to a specific faulty operation.
The aggregate expanded result is **145,045/148,032 residuals and 2092/2313 logits**.

Preparation verifies **1155 reference files**, **3504 inputs** and **172
historical/current dependency entries**, including CPU-selected continuations
and all reference prefixes. Seven RED-to-GREEN tests; full host suite **1744
passed / 47 skipped**. Existing private binaries and kernels are reused;
production sources, manifests and model packing are unchanged.

Artifacts: `Models/qwen38-27b/prefill-expanded-cli/`:
`boundaries-prepared-v1`, `boundaries-native-v1` and their preparation/run/audit
logs. Next localize sequential drift using the retained causal inputs and
residuals at L60/P38 and L33/P241, separating propagated error from local
arithmetic/state. Hybrid long-prefix precision now passes for these fixtures;
long-prefix repeated requests, production integration, serving and final model
acceptance remain open. `model_acceptance=false`.

## Latest checkpoint — expanded full-depth prefill validation, 2026-10-24

Added resident run/audit and two-request reset validators for the prepared
combined and expanded private CLIs. They bind build/source/preparation hashes,
exact commands and independent references, require fresh control identity
before candidate dispatch, and check RAM/device ownership before each process.
Completed numerical failure remains distinct from an incomplete run or identity
failure; neither report can promote overall model acceptance.

Fresh **64-layer, 20-token prompt plus continuation** measurements pass all
**4032/4032 residual gates and 63/63 logits** for expanded. Unchanged bounds:
residual maxrel <0.005 and logit correlation >0.9999 with matching argmax.

| Route | Residual gates | Worst residual maxrel | Minimum logit correlation |
|---|---:|---:|---:|
| Sequential | 1344/1344 | 0.0048820188 | 0.9999943778 |
| Hybrid layer-major | 1344/1344 | 0.0012128028 | 0.9999979147 |
| Hybrid block-major | 1344/1344 | 0.0012128028 | 0.9999979147 |

The fresh combined control reproduces all **261/261 captures** bytewise and
retains its **4000/4032 residual FAIL**. Expanded sequential preserves **87/87
captures**; hybrid schedules match in **86/86 logits/residual files**. Corrected
hybrid worst error falls from **0.0121972127 to 0.0012128028**. Continuation IDs
are `[17,478]`; expanded hybrid final continuation residual is **0.0007136553**.
The separate CPU audit recomputes metrics and confirms every control identity.

Two requests on each resident expanded engine also pass: **261/261 first-request
captures** match the single-request baseline and **261/261 second-request
captures** match the first after normal `Core::reset`. Each request independently
passes **4032/4032 residuals and 63/63 logits** across all three routes. Raw
request-marked logs and separate captures are retained. This checks normal
request reset, not checkpoint semantics or independent full-buffer clearing.
The separate repeat CPU audit also passes, remeasuring both requests and
verifying log extraction, commands and provenance.
Seven RED-to-GREEN runner tests; full host suite **1737 passed / 47 skipped**.
The first preflight retained insufficient RAM (24.3 GiB); subsequent resident
execution began after about 61 GiB became available. Existing binaries/kernels
were reused without production source, manifest or model changes.

Artifacts: `Models/qwen38-27b/prefill-expanded-cli/{native-v1,repeat-native-v1}`.
Short-prefix acceptance does not close 255/256/257-token full-depth accuracy,
production integration, serving or final model acceptance. Next validate the
expanded policy over the retained long-prefix references with fresh controls,
all-layer gates and causal prefix/schedule identity. `model_acceptance=false`.

## Latest checkpoint — expanded private prefill CLI preparation, 2026-10-23

Built fresh combined control, expanded candidate and two-request candidate CLIs.
The expanded policy adds paired-weight full-attention Q/K/V/gate, full-attention
output and dense FFN down, retaining corrected linear entry/output and FFN
up/gate. Exact role/argument/geometry checks distinguish the shared output shape
and reject incompatible split, epsilon, activation, norm or state/KV layouts.
Original GEMM dispatch and FP32 summation helpers remain unchanged.

All three builds pass **CTest 4/4 each**, pinning **591 / 592 / 592 compiler
dependencies**. The fresh control binary is byte-identical to the retained
combined CLI. Both candidate executables bind the same five verified low-weight
kernels; repeat engine sources match recursively, excluding only CLI request
capture plumbing. No new XCLBIN or production source/manifest/model change.

Preparation verifies **2006 input hashes**, accepted attention replay/raw
captures and retained full-depth measurements. It preserves **151 independent
reference files** unchanged (149 arrays plus IDs/metadata) and confirms the
complete current model-derived manifest. Seven RED-to-GREEN tests; full host
suite **1730 passed / 47 skipped**. Workflow: `utilities/prefill-expanded-cli/`;
artifacts: `Models/qwen38-27b/prefill-expanded-cli/{v1,prepared-v1}`.

This is build/preparation acceptance only, with no engine residency or NPU
model dispatch. `native_validated=false` and `model_acceptance=false`; retained
full-depth **4000/4032 residual FAIL** is preserved. Next adapt the resident
runner/audit to the new variants, require fresh control identity, independent
64-layer prompt/continuation gates, hybrid schedule identity and two resident
requests after reset. At least 35 GiB available RAM is required for that run.
Long-prefix and production acceptance remain open.

## Latest checkpoint — composed attention/FFN replay, 2026-10-23

Private N14336/K5120 entry control/low kernels and fresh production host bridges
replay the complete captured L63 attention prefix, then compose P1 through
output projection, residual, normalization, FFN up/gate, activation and down.
All **64/64 final NPU dispatches** complete across two placements. Individual
production/private products, native Q/K/V/gate, paired KV, attention output,
FFN and final residual controls are byte-exact. Corrected intermediate results
agree across placements; all inactive columns are zero. Host block/single-token
attention and KV agree exactly.

All **2240 entry head gates** strictly improve maxrel and rel_fro (worst
4.8850442e-6 / 3.0968432e-6). All **1280 independent host head/cache gates** pass
(worst maxrel 1.1655479e-5 <0.001), together with **46 output/upgate/down gates**.

| Gate | Control maxrel | Candidate maxrel | Candidate rel_fro |
|---|---:|---:|---:|
| Independent L63/P1 output | 0.0096485275 | **0.0040805495** | **0.0017167913** |
| Conditioned FFN | — | 2.5746182e-5 | 4.7519969e-6 |

The composed final output now passes both unchanged bounds (maxrel <0.005,
rel_fro <=0.005). The CPU re-audit recomputes the report from raw captures.
This is isolated suffix acceptance on retained incoming residuals, not fresh
cross-layer propagation: full-depth **4000/4032** remains FAIL and
`model_acceptance=false`. Production sources, kernels, manifest and model packing
are unchanged; there is no throughput claim.

Both new builds pass 32-core resource checks (text 4784/5568 bytes, L1 buffer
end 61448 bytes, stacks 4096 bytes, 801 compiler dependencies each). Fresh host
bridges pin 342 attention / 105 FFN dependencies. Eight packed/source weight
identities pass. No full-model residency is needed.

The initial 28-dispatch run stopped at a strict dtype check: the retained native
output used an FP64 container for exact FP32 values. A tested lossless FP32
conversion fixes preparation; failed v1 evidence remains retained, and fresh v2
preparation/replay succeeds. Fourteen RED-to-GREEN tests; **1723 passed / 47
skipped** overall. Workflow: `utilities/prefill-attention-replay/`; artifacts:
`Models/qwen38-27b/prefill-attention-replay/{prepared-v2,native-v2}`.

Next extend the private combined CLI to attention entry/out and FFN down,
then validate fresh full-depth propagation and repeat/reset. Long-prefix,
production promotion and end-to-end model acceptance remain open.

## Latest checkpoint — isolated paired-weight FFN down, 2026-10-23

Built private N5120/K17408/T256 control and low-weight kernels from copied
production sources. All **36/36 final NPU dispatches** complete on captured H
from L63/P1, L42/P17 and L52/P17, each in two slots. Individual control products
match production bytewise; their FP32 sum matches captured native down, and
residual addition reconstructs native layer output exactly. Candidate placements
are byte-identical and all inactive columns are zero.

| Target | Control down maxrel | Candidate down maxrel | Composed residual maxrel |
|---|---:|---:|---:|
| L63/P1 | 0.0039805268 | 3.8361005e-6 | 0.0070247892 |
| L42/P17 | 0.0020657191 | 2.0427415e-6 | 0.0052110173 |
| L52/P17 | 0.0012303268 | 3.4903344e-6 | 0.0123485917 |

All six primary projection gates pass, with both maxrel and rel_fro strictly
improved. **All six composed residual gates still fail** the unchanged 0.005
bound. L52's residual worsens from **0.0121972127** despite local improvement,
confirming that local error partly cancelled incoming drift. The independent
CPU re-audit preserves projection PASS and residual FAIL, with
`model_acceptance=false`. No fresh full-depth result or production promotion.

Both builds pass all 32-core resource checks: max text **4784/5568 bytes**,
max L1 buffer end **61448 bytes**, configured stacks **4096 bytes**, **801
compiler dependencies each**. The existing harness is verified against 326
dependencies. No full-model residency, production source/kernel/manifest or
model-packing change. Four products replace two; no throughput claim.

The first preparation stored an FP32 oracle output in a file named `.f64`.
Its 36 dispatches finished, but scoring correctly rejected the extent. That
failed experiment is retained. An explicit FP64 writer and regression test
fix the issue; fresh v2 preparation/replay and re-audit complete. Ten new
RED-to-GREEN tests; full host suite **1709 passed / 47 skipped**.

Workflow: `utilities/prefill-down-replay/`. Evidence:
`Models/qwen38-27b/prefill-down-replay/`, validated `prepared-v2/` and
`native-v2/`. Next replay full-attention fused entry/output with causal L63
prefixes and exact native controls, then compose with the down correction
before private CLI policy expansion. Retained full-depth acceptance remains
**4000/4032 FAIL**, with long-prefix and production acceptance still open.

## Latest checkpoint — combined weight-rounding attribution, 2026-10-23

CPU-only attribution on retained combined captures verifies **8/8 manifest-packed
versus source weight identities**, including separate Q/gate slices, and
**1124 projection/head comparisons**. Three independent causal outputs
reconstruct primary references bytewise. Four signed error decompositions
close exactly in saved FP64 vectors. All capture, control, build and reference
provenance checks remain mandatory; no NPU process or new kernel was used.

BF16-rounded weight predictions fit native outputs within **3.3603e-6 maxrel**.
The worst remaining L2 error fraction is **0.3126%** of native-minus-effective
projection error, across the tested heads/positions. This strongly supports
weight rounding as the dominant local projection error; it is not byte-exact
native replay or a fraction of full-model error explained.

| Target | Native output error | Ideal FFN on actual residual | Down-weight rounding component |
|---|---:|---:|---:|
| L63/P1 | 0.0096485275 | 0.0070258073 | 0.0026236663 |
| L42/P17 | 0.0057900192 | 0.0052107169 | 0.0005789814 |
| L52/P17 | 0.0121972127 | 0.0123487787 | 0.0002769049 |

Components use the independent output scale and retain signs; their magnitudes
must not be added. L63's remaining down implementation component is only
**2.2129e-6**. Down correction alone cannot close these captured failures.
Joint ideal attention Q/K/V/gate, output projection and FFN at L63/P1, driven
by the actual incoming prefix, reduce output error to **0.0040534510**.
The gate-only entry substitution, with ideal output projection and FFN, gives
**0.0049558575**; Q/K/V substitutions separately remain above 0.005. These are
conditioned suffix diagnostics, not fresh full-model propagation.

Next validate isolated paired-weight FFN down N5120/K17408/T256 on the three
captured activation rows, requiring byte-exact native controls and two slot
placements. Full-attention entry/output replay follows before any combined
policy expansion. Production promotion and long-prefix acceptance remain open;
the unchanged full-depth residual result is **4000/4032 FAIL**.

Nine new RED-to-GREEN tests; host suite **1699 passed / 47 skipped**. Workflow:
`utilities/prefill-combined-attribution/`. Evidence and prediction/component
arrays: `Models/qwen38-27b/prefill-combined-attribution/v1/`.

## Latest checkpoint — combined causal sublayers, 2026-10-23

Built a private read-only tracer from the verified combined CLI for L63/P1
(full attention), L42/P17 and L52/P17 (DeltaNet). Three resident routes retain
**261/261** baseline logits/residual/scratch files byte-exact. All **83** selected
captures validate, with **40/40** exact hybrid pairs. Linear captures include
complete projection prefixes and state before/after the block; attention captures
include Q/K/V/gate and paired KV. State-after is P19, not the selected position.

Three selected independent CPU outputs reconstruct the retained references
bytewise. All **54/54 conditioned FFN gates** and **30654/30654 projection/host
gates** pass, including **29760 per-head gates**. Worst FFN maxrel is
**0.0039841551** (L63); worst projection/head maxrel is **0.0023476249** (L63 Q).
Conv carry and pre-FFN residual reconstruction are exact. The first CPU audit
exposed a list-to-array conversion error for RoPE frequencies; a regression test
and corrected auditor pass on the same captures, with the failed audit retained.

| Hybrid target (both schedules) | Incoming residual error | Pre-FFN error | Output error | Ideal FFN on actual pre-FFN input |
|---|---:|---:|---:|---:|
| L63/P1 | 0.0017499814 | 0.0026927327 | 0.0096485275 | 0.0070258073 |
| L42/P17 | 0.0029677911 | 0.0047458357 | 0.0057900192 | 0.0052107169 |
| L52/P17 | 0.0119563497 | 0.0106894266 | 0.0121972127 | 0.0123487787 |

L52 already receives a failing residual. At all three targets, replacing the
local FFN with the independent ideal FFN still exceeds 0.005. L63 has an
additional local FFN output error of **0.0026250996**; its down projection from
captured activation differs by **0.0039805268**, while up/gate differ by only
**1.2174e-6 / 2.1872e-6**. These measurements separate propagated drift from
local arithmetic; they do not yet prove weight rounding is the cause. Next
compare effective versus BF16-rounded weights for the uncorrected FFN down and
full-attention projections using these captures, with signed error decomposition.

Full-depth residual acceptance remains **4000/4032 FAIL**; all original bounds
remain unchanged and `model_acceptance=false`. No production arithmetic,
manifest, model packing or XCLBIN changed. Build: **CTest 4/4**, **592 compiler
dependencies**. Ten new tests (including the RoPE adapter regression); full
host suite **1690 passed / 47 skipped**. Workflow and results are in
`utilities/prefill-combined-sublayers/`; artifacts are under
`Models/qwen38-27b/prefill-combined-sublayers/`, final CPU audit
`native-v1/sublayer-audit-v2.json`. Production promotion and long-prefix
acceptance remain open.

## Latest checkpoint — combined full-depth prefill validation, 2026-10-23

Nine full-resident runs complete: fresh paired-input, weighted-FFN and combined
linear QKV/Z/output + FFN up/gate CLIs, each on sequential, hybrid layer-major
and hybrid block-major routes. Every run uses the same independent 20-token
prompt and one continuation across all 64 layers. Historical source metadata
and reference arrays remain unchanged. The runner checks historical/current
identities, build dependencies, kernel hashes, available RAM and exclusive NPU
use before execution. An independent auditor recomputes every numerical gate.

Fresh controls reproduce **69/69 paired logits/scratch files** and **261/261
weighted logits/scratch/residual files**. Sequential outputs remain byte-exact
across variants, and hybrid schedules agree in all logits and layer residuals.

| Hybrid metric (each schedule) | Paired input | Weighted FFN | Combined |
|---|---:|---:|---:|
| Passing layer/position residuals | 1299/1344 | 1315/1344 | 1328/1344 |
| Worst normalized residual error | 0.0183013326 | 0.0138594086 | 0.0121972127 |
| Minimum logit correlation | 0.9999488108 | 0.9999742814 | 0.9999807010 |
| L63 continuation residual error | 0.0011956505 | 0.0004262400 | 0.0002258117 |

All **63/63 combined logit gates pass**, and sequential residuals pass
**1344/1344**, unchanged. Combined residual acceptance is **4000/4032**, with
**16 failures per hybrid schedule** at the unchanged 0.005 threshold. The
independent audit is complete and correctly returns numerical FAIL.

The improvement is not uniform. Compared with weighted FFN, 16 previously
failing hybrid rows pass, but three new failures appear per schedule:
L53/P18 **0.0040623432 → 0.0051799227**, L56/P17 **0.0041391664 → 0.0052288517**,
and L63/P1 **0.0049026924 → 0.0096485275**. Earliest failing depth is L42/P17
(**0.0057900192**); worst is L52/P17 (**0.0121972127**). These need conditioned
sublayer attribution before promotion, especially the new early-position L63
failure. A lower total failure count does not establish full accuracy.

Evidence: `Models/qwen38-27b/prefill-combined-cli/native-v1/`, including captures,
complete numerical FAIL report and independent audit. Production sources,
kernels, model packing and manifest are unchanged.

A separate two-request CLI completes three more resident processes. Across
sequential and both hybrid routes, **261/261 captures** match the original
single-request candidate, and **261/261** match again after `Core::reset()` on
the same resident engine. The independent repeat audit passes and preserves
the numerical failures: `numerical_passed=false`, `model_acceptance=false`.
This verifies normal request reset, not full-buffer zeroing or independent
checkpoint/restore correctness. Evidence: `repeat-native-v1/` alongside the
single-request run. The repeat build pins 591 compiler dependencies and passes
**CTest 4/4**. Eight new RED-to-GREEN tests; host suite **1680 passed / 47 skipped**.

Next capture conditioned sublayers at L63/P1 (new regression), L42/P17 (first
failing depth) and L52/P17 (worst error), requiring exact native baseline
identity before separating incoming residual drift from local arithmetic.
Production promotion and full-depth 255/256/257-token acceptance remain open.

## Latest checkpoint — combined prefill CLI preparation, 2026-10-23

Built three private CLIs from the merged source: paired-input control,
weighted-FFN control and combined linear QKV/Z/output + FFN up/gate candidate.
The original single-dispatch body and both control wrappers are preserved.
The candidate selects corrections by manifest role, arguments and geometry:
full-attention output has the same GEMM shape as linear output but retains its
existing arithmetic. Stage-major, split weights and incompatible roles fail.
The low-weight contexts reuse three verified kernel builds; no new XCLBIN or
production arithmetic/ABI change is introduced.

All three native builds pass **CTest 4/4 each**, with **589 / 590 / 591** compiler
dependencies pinned. Six new RED-to-GREEN tests include a compiled policy check
against the current 27B manifest, rejected roles/geometry, exact control
wrappers, preserved dispatch, kernel preflight and incomplete preparation.
Full host suite: **1672 passed / 47 skipped**.

Preparation verifies both historical control builds and capture inventories,
including **192 weighted-control residual files**. It copies **149 independent
reference arrays plus IDs and original metadata** unchanged, records **652
input hashes**, and confirms the retained export equals the complete current
model-derived manifest. Historical source differences are recorded separately
from current sources; artifact hashes are not relaxed. Evidence and all three
CLI hashes: `Models/qwen38-27b/prefill-combined-cli/`.

Preparation only, as requested: no model residency, fresh native captures or
accuracy result. `native_validated=false` and `model_acceptance=false` remain
explicit. Next require fresh paired/weighted baseline identity, combined
full-depth residual/logit gates and repeated-request/reset support. The existing
tracer accepts one request, so repeated-request/reset instrumentation remains
unfinished. Prior full-depth and long-prefix failures are not closed by builds.

## Main integration check — 2026-10-23

Merged main `1407484` while preserving the Qwen3.8 precision kernels, paired
KV/state ABI, replay tools and private `--build-root` export option. The exporter
now combines private workspaces with per-directory locks, source/knob caching,
per-spec staging and disambiguated default build directories. A regression test
first reproduced the external-workspace path failure, then passed after the
cache and lock paths were corrected. It also checks cache hits, source/knob
invalidation and rebuilding when an artifact is missing.

Regenerated the Qwen3-4B manifest fixture for the attention build directory's
`_m1024` suffix; its numerical geometry and binary layout are unchanged.
Validation: **1666 passed / 47 skipped** in the open-engine suite; converter
**136 passed / 42 subtests passed**; installer **55 passed / 1 skipped**.
The standalone native CLI builds, native CTest is **4/4**, and the host prefill/
rollback fixture passes. New settings registry/resolver/store/key-parser tests
pass **77/77** with assertions enabled; the generated settings registry matches
the merged source scan. Logs: `Models/qwen38-27b/main-merge-1407484/`.

This is host/build regression validation. No kernel set was rebuilt and no
full-model NPU or serving run was performed; retained numerical measurements
below remain tied to their original sources and artifacts.

## Latest checkpoint — paired-weight output replay, 2026-10-22

Built private N5120/K6144/T256 control/low output kernels and a fresh harness.
Validated retained prefixes as historical inputs using **2075 files / 25 evidence
records**; 23 changed tracked sources match the fixture revision in Git and
current HEAD separately. Artifact and historical build hashes were preserved.

All **72/72 NPU dispatches** complete. Individual production products match
retained outputs bytewise; new-control products match production. Inactive
columns are zero and candidate placements are byte-identical. All **240/240
projection checks** pass and strictly improve maxrel/rel_fro. Worst candidate
maxrel/rel_fro: **1.1842201e-5 / 4.8705972e-6**, versus worst control maxrel
**0.0022730411**. Independent pre-FFN residual results are:

| Row | Native | Entry correction only | Out correction only | Entry + out correction |
|---|---:|---:|---:|---:|
| L34/P17 | 0.0007177295 | 0.0006256207 | 0.0007008094 | 0.0006087006 |
| L37/P7 | 0.0010862168 | 0.0008810181 | 0.0010534831 | 0.0008481152 |
| L53/P17 | 0.0075342533 | 0.0074347548 | 0.0067071677 | 0.0066072119 |

L53 still fails the unchanged **0.005** bound. Correcting local projections
reduces error but leaves inherited drift. Native and entry-only values come
from the retained prefix measurement; both out-corrected variants use fresh
NPU dispatches on those gated prefixes. This does not establish corrected
cross-layer execution. Retained full residual acceptance remains **3974/4032
FAIL**. Next evaluate combined entry/out/up-gate precision in a private CLI
with a fresh post-merge control and independent full-depth references.

All 32 core budgets pass: text **4784/5568 bytes**, L1 buffer end **61448 bytes**,
801 compiler dependencies per kernel build; fresh harness has 326 dependencies.
**14 new RED-to-GREEN tests; 1665 passed / 47 skipped** overall. No full-model
residency or production arithmetic change; logical buffers **26.75 MiB** per
call excluding runtime overhead. Reproduction, provenance rules and hashes:
[output replay runbook](../../../utilities/prefill-out-replay/README.md).

## Main integration — 2026-10-22

Integrated main at `cb03fad`. Resolved the attention-route diagnostic to retain
paired-KV host fallback and the dense single-token route message. Preserved the
manifest route tests and their independently parsed move-only instances, with
all new manifest coverage retained. FP32 boundaries, paired-KV storage,
compensated DeltaNet and consistent Q4 activation sums remain intact.

Validation: **1651 passed / 47 skipped** in the open-engine Python suite;
converter **136 passed / 42 subtests passed** in a separate CPU environment;
standalone CLI builds, CTest **4/4**, host prefill fixture **PASS**, K2 chat
**1/1** and greedy sampler **1/1**. No model-resident NPU run or kernel rebuild
was performed. Evidence: `Models/qwen38-27b/main-merge-cb03fad/`.

Shared host/harness and recipe sources changed. Retained binary/capture evidence
still describes its original revision; do not rewrite those hashes or treat it
as validation of this merge. Before further hardware replay, verify or rebuild
the affected artifacts with fresh provenance. Full prefill accuracy remains
open; the prior L53/P17 residual failure is not resolved by this merge.

## Causal QKV/Z prefix replay — 2026-10-22

Composed paired-weight entry projections for complete 20-token prefixes at
L34, L37 and L53 with production host DeltaNet and original NPU out projection.
**60/60 dispatches complete** using verified existing binaries and one layer's
weights at a time. Production entry sums, individual control products, captured
host output/state and selected out/residual controls all match bytewise.
Block versus sequential host calls and both NPU placements are byte-identical;
inactive columns and host/state padding are zero.

All **15360/15360 primary entry head checks** pass and strictly improve both
errors (7680 distinct heads at two placements). Worst candidate maxrel/rel_fro:
**1.4844200e-5 / 9.4852176e-6**. All **6048/6048 conditioned host head/state
checks** pass, worst **6.0081719e-6**. All **240/240 conditioned output projection
rows** pass, worst maxrel **0.0022730411**. Independent residual acceptance is
still **FAIL**:

| Pre-FFN residual | Native control maxrel | Candidate maxrel | Candidate gate |
|---|---:|---:|---|
| L34/P17 | 0.0007177295 | 0.0006256207 | PASS |
| L37/P7 | 0.0010862168 | 0.0008810181 | PASS |
| L53/P17 | 0.0075342533 | 0.0074347548 | FAIL |

Each selected result agrees at both placements. This confirms a causal local
improvement while leaving L53 above the unchanged 0.005 bound. Incoming norm
and residual remain captured native values; no corrected cross-layer FFN or
logits are produced. Retained full-model residual acceptance stays
**3974/4032 FAIL**. Next evaluate paired-weight out projection against these
retained prefixes, keeping the independent residual gate and inherited drift
visible. Full-model and long-prefix acceptance remain open.

The replay derives alpha/beta stride from real tensor shapes (48), matching
native core, and rejects invalid constant extents before calling C++. The first
attempt used synthetic padded lanes (64); exact host control rejected it before
any NPU dispatch. Its evidence is preserved. **14 new RED-to-GREEN tests;
1618 passed, 47 skipped** overall. No binary rebuild or production arithmetic
change. Commands, evidence and limitations are in the
[causal prefix replay runbook](../../../utilities/prefill-prefix-replay/README.md).

## Isolated paired-weight QKV/Z replay — 2026-10-21

Built private control and low-weight fused entry kernels at N16384/K5120/T256.
Both preserve production packing and DMA; the low variant converts the BF16
weight remainder. Four GEMM products combine two weight and two input planes.
**24/24 NPU dispatches complete**, without full-model residency (~68.5 MiB
logical buffers per dispatch, excluding runtime overhead).

Fresh production sums reproduce native QKV/Z captures bytewise. Each individual
new-control product matches production exactly; inactive columns are zero.
Candidate placements 6/134 and 0/128 match bytewise. All **512/512 head checks**
pass the unchanged primary bounds (maxrel <0.005, rel_fro <=0.005), with strict
improvement of both errors in every head. These are 256 distinct heads measured
at two placements, against original-input/effective-weight FP64 references.

| Previously failing head | Control maxrel | Paired-weight maxrel |
|---|---:|---:|
| Q L34/P6/H13 | 0.0050724904 | 2.8237745e-6 |
| V L37/P0/H46 | 0.0071224278 | 1.4844200e-5 |

Worst candidate maxrel is **1.4844200e-5**; worst rel_fro is **7.8625458e-6**.
Both builds pass all 32 core budgets: maximum text **4784/5568 bytes**, L1 buffer
end **61448 bytes**, **801 pinned compiler dependencies per build**. Toolchain:
mlir-aie 1.4.2 / llvm-aie 21.0.0.2026080301+c9c5ecb7. **12 new RED-to-GREEN
tests; 1604 passed, 47 skipped** overall.

This validates two isolated input rows, not corrected causal state or full-model
propagation. Retained full residual acceptance remains **3974/4032, FAIL**.
Next replay complete captured QKV/Z prefixes and measure their causal effect
through host DeltaNet and the original output projection, retaining exact native
controls and independent references. L53/P17 output projection remains another
target. Production promotion and long-prefix acceptance remain open; doubling
GEMM products has not been benchmarked. Reproduction and scope are in the
[entry replay runbook](../../../utilities/prefill-entry-replay/README.md).

## Projection weight-rounding attribution — 2026-10-21

Added a CPU-only analysis of captured QKV/Z prefixes and selected out
projections. Nine manifest-packed/source effective-weight comparisons are exact.
Prior native/build/reference provenance validates. Each CPU prediction separates
paired-input approximation from BF16 weight rounding and accumulation residual.
The analysis covers **7683 comparisons**, using separate Q/K/V/Z head scales
and conditioned on captured norm/gated-output inputs.

**7681/7683 native projection gates pass**; two failures were hidden by aggregate
QKV scaling (zero-based indices):

| Role | Layer / position / head | Native maxrel | BF16-weight prediction gap maxrel |
|---|---|---:|---:|
| Q | 34 / 6 / 13 | 0.0050724904 | 7.3845882e-7 |
| V | 37 / 0 / 46 | 0.0071224278 | 1.4109131e-6 |

Worst BF16-weight prediction gap is **2.9830263e-6**; its remaining L2 error
is at most **0.2262%** of native-minus-effective-weight error. This strongly
supports weight rounding as the source of the measured local projection errors.
It is a CPU prediction, not byte-exact NPU replay or corrected model execution.

All three causal residual decompositions close (worst **2.2204460e-16**).
They preserve native input prefixes, independent state per variant, FP32
projection storage and signed QKV/Z interaction on a common reference scale.

| Pre-FFN row | Native maxrel | Joint ideal QKV/Z/host/out maxrel | Out-projection component maxrel |
|---|---:|---:|---:|
| L34/P17 | 0.0007177295 | 0.0006092193 | 0.0000179964 |
| L37/P7 | 0.0010862168 | 0.0008486373 | 0.0000323107 |
| L53/P17 | 0.0075342533 | 0.0066083261 | 0.0008264166 |

The ideal suffix still fails at L53/P17; inherited drift remains. Its largest
measured local component is out projection. Next prioritize isolated fused
QKV/Z replay at L34/P6 and L37/P0 with byte-exact native controls, then evaluate
paired-weight precision at unchanged per-head bounds. L53/P17 out projection
remains a follow-up target. Existing full residual accuracy is still
**3974/4032, FAIL**; no full-model or long-prefix acceptance is claimed.

**11 new RED-to-GREEN tests; 1592 passed, 47 skipped** overall. No native build
or NPU execution. Reproduction, signed-vector interpretation and scope are in
the [projection attribution runbook](../../../utilities/prefill-projection-attribution/README.md).
Production and candidate arithmetic remain unchanged.

## Early causal sublayer boundaries — 2026-10-21

A private instrumented weighted CLI captures L34/P17, L37/P7 and L53/P17,
including complete 20-row norm/QKV/Z/DeltaNet prefixes, selected output
projections, FFN boundaries and actual raw state around the host block.
All three full-resident routes complete: **66/66 logits and 192/192 residual
files are byte-exact** to the retained candidate. All **81 captures** validate,
with **39 identical hybrid pairs**. Three independent causal CPU outputs
reconstruct retained references bytewise. Conv carry and pre-FFN residual
reconstruction are exact. Block-end state belongs to **P19**, not the target row.

**54/54 conditioned FFN gates pass** (worst **0.0016358962**) and **732/732
projection/host boundary gates pass** (worst **0.0022380252**). Both hybrid
schedules have identical measurements:

| Layer | Worst prefix QKV maxrel | Worst prefix Z maxrel | Worst conditioned DeltaNet output maxrel | Selected out projection maxrel | Block-end state maxrel |
|---|---:|---:|---:|---:|---:|
| 34 | 0.0022380252 | 0.0014517693 | 1.0677252e-6 | 0.0009515609 | 9.2116065e-7 |
| 37 | 0.0009451065 | 0.0013125434 | 8.8183857e-7 | 0.0003931275 | 7.2322780e-7 |
| 53 | 0.0010144976 | 0.0011105445 | 1.4542305e-6 | 0.0012094501 | 4.8807962e-7 |

QKV/Z gates use actual inputs; captured-norm comparisons are similar. QKV
uses a whole-vector scale, not separate Q/K/V or per-head acceptance. The
FP64 host reference carries its own state driven by captured projections;
this is a conditioned diagnostic, never a replacement model reference.
The much smaller host-state/output differences direct the next investigation
toward QKV/Z/out projection rounding. They do not prove a unique faulty op.

Global hybrid output maxrel is **0.0009041704 / 0.0011400684 / 0.0078214144**.
L53/P17 already fails pre-FFN at **0.0075342533**; ideal FFN on actual inputs
still fails at **0.0079466232**. Full retained residual acceptance remains
**3974/4032, FAIL**. No model or long-prefix acceptance is claimed.

GCC 15.2.0, CTest **4/4**, **587 pinned compiler dependencies**. **10 new
RED-to-GREEN tests; 1581 passed, 47 skipped** overall. Reproduction and exact
capture scope are in the [early-boundary runbook](../../../utilities/prefill-prefix-sublayers/README.md).
Production sources, kernels, weights and candidate arithmetic are unchanged.

## Causal prefix drift scan — 2026-10-21

Added a CPU-only scanner over retained native residuals for layers 0–55 and
positions 7/17. Each route carries the complete causal prefix in fresh ideal
conv/recurrent/KV state. Weights are cached one layer at a time. **112/112
independent CPU outputs reconstruct bytewise; 224 diagnostic cases complete.**
Build/compiler/kernel/model provenance, 66 baseline logits and 192 residual
hashes validate; all 64 hybrid/block-major residual pairs are identical.

Sequential passes **112/112** selected output gates (worst **0.0032302929**,
L53/P17); hybrid passes **101/112**. First hybrid failures are **L40/P17** and
**L55/P7**. Signed propagated and ideal-history-gap vectors close to the total
error and use the same output reference scale:

| Hybrid row | Propagated maxrel | Ideal-history gap maxrel | Total maxrel |
|---|---:|---:|---:|
| L40/P17 | 0.0055866730 | 0.0003720786 | 0.0059587517 |
| L51/P17 | 0.0134731971 | 0.0004004110 | 0.0138594086 |
| L55/P7 | 0.0054887081 | 0.0007874112 | 0.0062761194 |

The propagated component already exceeds the unchanged <0.005 bound at both
first-failure rows. The largest hybrid ideal-history gap is **L53/P17,
0.0009166706**, also the largest sequential gap (**0.0010513228**). This gap
includes accumulated state and local arithmetic; it does not isolate a faulty
operation. Scalar maxima are not additive and cancellation is preserved in the
saved vectors. Early growth at L34/P17 and L37/P7, plus the L53/P17 gap, nominate
three causal sublayer captures for the next precision investigation.

**13 new tests; 1571 passed, 47 skipped** overall. Eleven initial tests failed
before implementation. No native build or NPU run was needed. Production and
candidate arithmetic remain unchanged; full retained residual acceptance is
still **3974/4032, FAIL**. This selected-prefix diagnostic does not establish
long-prefix or model acceptance. Reproduction and scope are in the
[CPU scan runbook](../../../utilities/prefill-prefix-scan/README.md).

## Weighted-candidate sublayer diagnosis — 2026-10-21

Instrumented copied weighted-candidate sources at L55/P7 and L51/P17 without
changing arithmetic. All three full-resident routes complete on the retained
20-token prompt plus continuation. **66/66 logits and 192/192 residual files**
match the previous candidate bytewise; all **26 selected captures** are complete
and **12 hybrid pairs** are identical. Independent causal CPU replay reproduces
both stored selected outputs bytewise.

All **36 conditioned FFN gates pass**, worst maxrel **0.0020749792**. Global
hybrid errors are already present before FFN:

| Row | Input maxrel | Pre-FFN maxrel | Native output maxrel | Ideal FFN on actual pre-FFN input |
|---|---:|---:|---:|---:|
| L55/P7 | 0.0029030925 | 0.0061193443 | 0.0062761194 | 0.0060717536 |
| L51/P17 | 0.0057009925 | 0.0092505118 | 0.0138594086 | 0.0138246489 |

Native-vs-ideal-FFN output error is **0.0002031324 / 0.0003275429**; ideal FFN
still fails the unchanged <0.005 global bound. Pre-FFN differences against
ideal causal state driven by native inputs are **0.0006651352 / 0.0004594719**.
This points to propagated input/history drift; it does not identify a single
faulty earlier operation or make ideal state a replacement for native history.
Sequential selected outputs pass at 0.0002325612 / 0.0008600592.

Full-model intermediate residual acceptance remains **FAIL (3974/4032)**.
Next localize drift growth in preceding layers using retained causal prefixes,
before further precision changes or longer-sequence acceptance.

GCC 15.2.0; CTest **4/4**, **587 compiler dependencies** pinned. **8 new
RED-to-GREEN tests; 1558 passed, 47 skipped** overall. The initial CPU audit
rejected sublayer residual files as extra full-layer traces; a regression fixes
that inventory distinction. The failed audit is retained and a CPU-only re-audit
validates the same native captures. Commands and limitations are in the
[weighted sublayer runbook](../../../utilities/prefill-weighted-sublayers/README.md).
Production code, kernels and model weights are unchanged.

## Private causal paired-weight FFN checkpoint — 2026-10-21

Built private paired-input control and paired-weight candidate CLIs with
synchronized residual capture. The candidate adds the verified low-weight
kernel only for unsplit dense FFN up/gate N34816/K5120/T256, preserving the
single-dispatch body, packing and FP32 sum order. Stage-major is rejected.
Production sources, manifests and kernel binaries remain unchanged.

Six full-resident requests complete on the retained **20-token prompt + one
NPU continuation** through 64 layers. Control reproduces **66/66 historical
paired-input logits bytewise**. Candidate sequential logits, continuation
scratch and residuals are exact (**87/87 checks**). Both hybrid schedules agree
in **22 logit files and 64 layer residual files** per variant.

| Hybrid metric | Control | Candidate |
|---|---:|---:|
| Minimum logit correlation | 0.9999488108 | 0.9999742814 |
| L63 continuation residual maxrel | 0.0011956505 | 0.0004262400 |
| Passing layer/position residuals per route | 1299/1344 | 1315/1344 |
| First failing residual, L55/P7 | 0.0088329705 | 0.0062761194 |
| Worst residual, L51/P17 | 0.0183013326 | 0.0138594086 |

**Accuracy remains FAIL.** Both variants pass 63/63 logits; candidate residuals
pass **3974/4032**, versus control **3942/4032**. Sequential residuals pass
1344/1344 unchanged (worst 0.0048820188), but **29 residual failures per hybrid
route remain** against the unchanged <0.005 gate. The aggregate audit correctly
returns numerical failure with valid captures. Improvement in final logits and
continuation does not override intermediate-layer failures or establish
long-prefix accuracy. Next capture conditioned sublayers at L55/P7 and L51/P17
before longer runs or production integration.

GCC 15.2.0; both fresh v2 builds pass CTest **4/4**, with **585/586 compiler
dependencies** pinned. The first candidate build failed on an ambiguous XRT
constructor; an explicit std::string fixes it, and the failed attempt is
retained. **5 new RED-to-GREEN tests; 1550 passed, 47 skipped** overall.
Commands, hashes and constraints are in the
[private CLI runbook](../../../utilities/prefill-weighted-cli/README.md).

## Selected-row FFN expansion checkpoint — 2026-10-21

Extended the composed FFN checker to retained hybrid L60/P38 and L35/P80.
Linear and attention pool offsets are selected independently from the manifest;
packed bytes and effective source weights match exactly for up/gate and down.
All existing bridge, harness and kernel binaries are reused unchanged.

**40/40 serial NPU dispatches complete.** Fresh production/private-control
high and low up/gate products agree individually. Control norm, activation,
down and output match the retained native captures bytewise. Candidate
up/gate/activation/down/output agree across T256 slots 38/166 and 80/208;
every individual output is finite, complete and zero in inactive columns.

| Retained hybrid row | Candidate up/gate maxrel | Control → candidate FFN maxrel | Control → candidate output maxrel |
|---|---|---|---|
| L60/P38 | 1.7964e-6 / 1.6948e-6 | 0.0009571130 → 0.0008939331 | 0.0014798680 → 0.0014800810 |
| L35/P80 | 2.2408e-6 / 1.8304e-6 | 0.0016492574 → 0.0005499310 | 0.0001922195 → 0.0001962610 |

All separate projection, conditioned FFN and global layer-output gates pass
with unchanged bounds. Output rel_fro improves to **0.0046959671 / 0.0025036066**,
but output maxrel slightly increases on both rows. Local arithmetic improvement
therefore does not imply monotonic global improvement. These hybrid outputs
already passed; the sequential route's failures at these positions remain open.
No new causal model run or production correction is claimed. The retained
long-prefix logit result remains **558/774 passing**.

TDD: **14 new RED-to-GREEN tests; 1545 passed, 47 skipped** overall. Reports
reject incomplete coverage and invalid controls and retain local/global gates
separately. The [replay runbook](../../../utilities/prefill-ffn-replay/README.md)
records commands and results. Next validate fresh causal propagation in a
private native candidate with byte-exact controls before production integration.

## Composed FFN replay checkpoint — 2026-10-20

Added a private CPU bridge extracting current host RMSNorm and activation,
then composed the retained paired-weight up/gate result with unchanged native
down GEMM at L63/P7. Production sources, recipes and binaries are unchanged.
No full-model residency or new XCLBIN is required.

**8/8 down dispatches complete.** Norm, control activation, down and final
residual reproduce the retained native bytes. Candidate slots 7/135 are
byte-identical; each high/low dispatch has zero inactive columns. Down packing
matches the production pool bytewise and effective source weights elementwise.

| Independent gate | Control maxrel | Candidate maxrel | Candidate rel_fro |
|---|---:|---:|---:|
| Conditioned FFN | 0.0005411159 | 0.0002390673 | 0.0010533278 |
| Final layer output | 0.0058400461 | 0.0045267063 | 0.0023137023 |

The candidate passes both gates at maxrel <0.005 and rel_fro <=0.005. The
original output maxrel bound is unchanged; the Frobenius gate is additional.
This verifies an actual recomputed down result, including changed cancellation,
not an estimate with the previous down error frozen. It establishes accuracy
for this composed FFN on one retained causal row. It does not replay attention
or propagate the correction through a new model run. Long-prefix model
accuracy remains unresolved (**558/774 retained logit gates** passing).

TDD: **9 new RED-to-GREEN tests; 1531 passed, 47 skipped** overall. GCC 15.2.0
builds the bridge with **105 pinned compiler dependencies**; source, generated
code, binary and fixture hashes are recorded. See the
[composed replay runbook](../../../utilities/prefill-ffn-replay/README.md).
Next expand the isolated correction to the other selected causal rows, then
validate propagation before production integration or model acceptance.

## Paired-weight GEMM candidate checkpoint — 2026-10-20

Added private control and low-weight GEMM builds with unchanged Q4NX packing
and DMA layout. The low kernel emits the BF16 residual of the dequantized
weight after its high BF16 part. Four native products are combined in FP32;
production sources, recipes and binaries remain unchanged.

**Eight NPU dispatches complete.** Fresh control high/low outputs individually
match retained production outputs bytewise. Candidate outputs at T256 slots
7 and 135 are byte-identical; inactive columns are zero. At L63/P7, separate
primary up/gate accuracy improves against the original FP32-input and exact-
effective-weight reference:

| Projection | Control maxrel | Candidate maxrel | Candidate rel_fro |
|---|---:|---:|---:|
| Up | 0.0019482680 | 0.0000015091 | 0.0000010450 |
| Gate | 0.0002586737 | 0.0000013845 | 0.0000012487 |

Both primary gates pass and improve strictly; bounds are unchanged. This
validates an isolated projection correction only. The candidate has not run
full FFN or propagated layer/logit gates. Long-prefix model accuracy remains
unresolved, with **558/774 retained logit gates** passing. Up/down cancellation
means projection improvement alone cannot establish model acceptance.

All 32 cores pass resource checks: maximum text **4784/5568 bytes** for control/
low, maximum buffer end **61448 bytes**, configured stack 4096 bytes. Both
builds pin **801 compiler dependencies**. An unsupported BF16-vector conversion
was corrected by widening through an FP32 accumulator; the failed attempt is
retained, and fresh v2 builds provide final provenance. Toolchain: mlir-aie
1.4.2, Peano 21.0.0.2026080301+c9c5ecb7.

TDD: **9 new RED-to-GREEN tests**; host suite **1522 passed, 47 skipped**.
The [paired-weight runbook](../../../utilities/prefill-paired-weights/README.md)
records build/run commands, hashes and limits. Next validate the complete FFN
and layer output on captured causal inputs with exact control before expanding
coverage or integrating. The extra weight pass doubles this projection's
GEMM dispatch count; performance remains unmeasured.

## Isolated up/gate replay checkpoint — 2026-10-20

Added an isolated L63/P7 paired-input GEMM replay using the existing production
kernel and captured norm. Only one projection's weight/input/output buffers
are resident per process (about **0.1395 GiB**, excluding XRT overhead).
Independently packed weights match the retained L63 pool bytewise, and their
dequantized values match the source container's effective weights elementwise.

**Four NPU dispatches complete.** High/low inputs in T256 slots 7 and 135 both
reproduce all **34816 captured up/gate values bytewise**. Inactive columns are
zero in each individual output. Thus sparse-row substitution preserves this
selected projection result; it does not establish dense-prompt accuracy.

| Independent comparison | Up maxrel | Gate maxrel |
|---|---:|---:|
| Exact effective weights, original FP32 norm | 0.0019482680 | 0.0002586737 |
| Exact weights, paired BF16 input | 0.0019495638 | 0.0002587289 |
| BF16-rounded weights, paired input (diagnostic) | 0.0000011273 | 0.0000015830 |

The CPU weight-rounding effect is **0.0019497818 / 0.0002588257**, closely
accounting for the measured up/gate discrepancy. This localizes most of the
tested projection error to the BF16 weight boundary, with correct effective
packing. The rounded-weight reference is diagnostic only; original bounds and
references remain unchanged. No production arithmetic or kernel was changed,
and global long-prefix accuracy remains **FAIL (558/774 retained logit gates)**.

Next test an isolated higher-precision dequantization/GEMM candidate against
the byte-exact control before integration. Up/down cancellation and propagated
state require renewed causal checks after any correction.

TDD: **13 new RED-to-GREEN tests**; full host suite **1513 passed, 47 skipped**.
The standard harness was built with GCC 15.2.0 and 326 pinned compiler
dependencies; all production kernel binaries were reused. Reproduction,
provenance and limitations are in the
[isolated replay runbook](../../../utilities/prefill-upgate-replay/README.md).

## Hybrid FFN vector attribution checkpoint — 2026-10-20

Added CPU-only attribution from retained hybrid sublayers at L63/P7, L60/P38
and L35/P80. It revalidates source/build/reference/capture provenance and exact
baseline identity, then streams one layer's FFN weights. Both hybrid schedules
share each result only after capture/output identity checks. Complete signed
vectors use the same independent final-output scale and sum to the actual
output error; worst closure discrepancy is **3.98e-14** across all three rows.

At **L63/P7**, up is the largest local FFN contribution (**0.0044931327** on the
common output scale), followed by down (**0.0013581369**) and gate
(**0.0002682804**). Down partially cancels up at the worst output coordinate
(3994). Up/gate interaction is explicit; activation, norm and rounding terms
are small. These maxima must not be summed.

| L63/P7 diagnostic | Output maxrel |
|---|---:|
| Captured native output | 0.0058400461 |
| Ideal down on captured activation | 0.0071981830 |
| Ideal up, activation and down; captured gate | 0.0046451266 |
| Ideal gate, activation and down; captured up | 0.0072075708 |
| Ideal complete FFN on captured pre-FFN input | 0.0046311444 |

The stored independent FP32 output is the common reference here, accounting for
the small difference from the earlier unrounded causal-output metric. CPU ideal
suffixes and algebraic removals with frozen downstream errors are separate
diagnostics; neither establishes what corrected NPU execution will produce.
No production arithmetic, kernels, weights or native state changed.
Long-prefix accuracy remains **FAIL (558/774 logits)**.

Next isolate the L63/P7 up/gate GEMM with production layout and a byte-exact
control. A replay must preserve T256/row 7 and prove any other-row substitution
does not change that result before testing a precision change. Weight packing
versus GEMM arithmetic remains unresolved; sequential propagated-history drift
is a separate open investigation.

TDD: **16 new tests**, 15 observed RED; full host suite **1500 passed, 47 skipped**.
No native build or NPU run was needed. Commands, formulas, signed contributions,
provenance and limitations are in the
[sublayer runbook](../../../utilities/prefill-sublayer-trace/README.md).

## Selected causal sublayers checkpoint — 2026-10-20

A private native CLI now captures selected prefill sublayers at L63/P7,
L60/P38 and L35/P80 (zero-based). Sequential NPU captures activation scratch;
both hybrid routes capture input, pre-FFN residual, norm, up/gate, activation
and down. Causal native state and all arithmetic are preserved.

All three full-depth 257-token runs complete. **777/777 logit files** and
**192/192 residual files** reproduce the controlled baseline bytewise.
All **39** selected captures pass coverage/shape/finite checks; **18/18** hybrid
capture pairs match. Each output reconstructs from captured residual/down within
FP32 summation bounds. Three independent causal CPU layer replays reproduce
the retained outputs exactly. **54/54 conditioned FFN gates pass**, worst
maxrel **0.0024681085**, across nine route/target cases.

| Failure (zero-based) | Pre-FFN maxrel | Native output maxrel | Ideal FFN on actual pre-FFN input: output maxrel | Local FFN error at layer output |
|---|---:|---:|---:|---:|
| Sequential L60/P38 | 0.0093615690 | 0.0090391466 | 0.0090472423 | 0.0000081696 |
| Sequential L35/P80 | 0.0084320701 | 0.0051683727 | 0.0051656855 | 0.0000027012 |
| Hybrid L63/P7 | 0.0008105224 | 0.0058400461 | 0.0046310385 | 0.0031273746 |

Sequential error is already present before FFN and is closely reproduced by
the ideal FFN. At hybrid L63/P7, the ideal FFN counterfactual passes the global
residual gate while the native result fails. Hybrid FFN arithmetic therefore
contributes to this crossing even though its conditioned gates pass. The scalar
errors have different reference scales and cannot be added as contributions.
This does not identify one faulty operation or validate full-model accuracy:
the same **558/774** logit gates pass, with unchanged thresholds.

Next isolate up/gate, activation and down contributions at hybrid L63/P7 before
selecting a precision change. Sequential upstream causal-input/state drift
remains a separate investigation. Ideal-state replay driven by native inputs is
explicitly diagnostic; it never substitutes CPU state in the native engine.

TDD: eight new tests, seven observed RED before implementation/fix; full host
suite **1484 passed, 47 skipped**. Both private builds pass CTest **4/4**.
A builder protection fix was rebuilt and proved byte-identical to the CLI that
produced the native captures; the initial stale-source refusal is retained and
the final audit uses current verified build evidence. Production sources,
kernels and weights remain unchanged. The
[sublayer runbook](../../../utilities/prefill-sublayer-trace/README.md) records
commands, source/compiler provenance, artifacts and interpretation limits.

## Residual localization checkpoint — 2026-10-20

The instrumented native CLI completed **64 layers × 257 prompt positions plus
one continuation** on sequential NPU, hybrid layer-major and hybrid block-major.
All **777/777** logits and aliases match the retained paired-input baseline
bytewise. All **64/64** hybrid residual files are identical across schedules.
The synchronization needed for capture preserved the measured outputs.

The **192 residual files** expose failures earlier than logits alone:
**46314/49536** layer/position gates pass, with the unchanged **maxrel <0.005**.
Indices are zero-based.

| Route | Residual PASS | First failure by position | Worst maxrel |
|---|---:|---|---|
| Sequential | 15510/16512 | P38/L60: 0.0090391466 | 0.5389499720 at P220/L58 |
| Hybrid layer-major | 15402/16512 | P7/L63: 0.0058400461 | 0.4325418278 at P205/L58 |
| Hybrid block-major | 15402/16512 | P7/L63: 0.0058400461 | 0.4325418278 at P205/L58 |

At hybrid P7 the incoming L62 residual maxrel is 0.0007421414; L63 is a full
attention layer. At sequential P38 the incoming L59 error is 0.0040749071; L60
is a linear layer. These are threshold crossings in propagated outputs, not
proof of a faulty operation. The shallowest failures are sequential P241/L33
and hybrid P162/L32. At the first failing logit positions, residual first fails
at sequential P80/L35 and hybrid P82/L58. Logit gates remain **558/774 PASS**;
matching generated IDs and passing continuation residuals do not override this.

Added chronological/depth localization with per-layer and per-position summaries
and complete coverage checks. Five tests failed before implementation; full host
suite **1476 passed, 47 skipped**, native CTest rerun **4/4**. Build/source and
reference hashes pass. The existing CLI and all production artifacts were reused;
no new compilation or numerical correction was required in this measurement step.

Evidence and commands are in the
[trace runbook](../../../utilities/prefill-residual-trace/README.md). Next inspect
causal sublayer boundaries at hybrid P7/L63, sequential P38/L60 and sequential
P80/L35, separating incoming error from attention/DeltaNet and FFN outputs.
Long-prefix numerical acceptance, production promotion and serving remain open.

## Previous preparation and boundary measurements — 2026-10-19

A private per-layer native prefill tracer is now built and CPU-tested. It
captures actual FP32 residuals after every layer, including continuation, on
sequential NPU and both hybrid schedules. Sequential queue-ahead is disabled
in this diagnostic copy to prevent the next layer overwriting `xres` during
readback. Production sources, kernel artifacts and paired-input arithmetic
are unchanged. The new auditor requires byte-exact baseline logits before
interpreting residuals and applies the unchanged <0.005 gate per layer/position.

Preparation adds **24 tests** (15 initially RED), including poisoned padding at
255/256/257 boundaries, coverage failures, per-row numerical errors and control
identity. Host suite: **1471 passed, 47 skipped**; native CTest **4/4**. Build
provenance includes **585 compiler dependencies**. The compiled CLI also rejects
unsupported host decode before loading the model. The
[trace runbook](../../../utilities/prefill-residual-trace/README.md) records the
build and deferred run commands. Full-model capture is **pending**: available
RAM was about 13 GiB, and this stage was limited to preparation. Native logit
identity and first-layer localization have not yet been measured.

Full-depth 255/256/257-token prefill and one NPU continuation have now run on
all three schedules. **Accuracy remains FAIL:** only **1668/2313** logit gates
pass. Sequential first fails at position **80**, minimum correlation
**0.9351669042** at **220**; both paired-input hybrid schedules first fail at
**82**, minimum **0.9422035563** at **205**. Hybrid argmax also differs at
**181/183/234**. At length 256, hybrid final continuation residual maxrel is
**0.0096012460 >0.005**. All other final continuation residuals pass. Matching
continuation IDs [248046,198] do not override the numerical failures.

The two hybrid schedules agree bytewise in **774/774** logit files, and all
**1533/1533** common-prefix comparisons across lengths are exact on all routes.
The capture audit completes with a numerical FAIL. Errors
already occur before the block boundary, so this is not solely a 256-token
transition issue. All **2331 native output files** are retained. The CLI used for those measurements only
captured L63 after continuation; identifying the first divergent layer still requires
running the new intermediate residual tracer. Full CPU residual references are ready.

Independent boundary references pass all **1155 files**, **128** common-prefix
residual arrays, **511** logit prefixes and **12** first-four-layer arrays against
the earlier slice. Source/export identity is verified. These are truncated chat
prefixes used for numerical validation, not an answer-quality test. The new
reference/native auditor adds **11 TDD tests**; host suite **1447 passed,
47 skipped**. Reproduction and measurements are documented in
[`utilities/prefill-slice`](../../../utilities/prefill-slice/README.md).

The preceding private paired-input experiment remains a useful short-prompt
result: **63/63** gates pass on 20 tokens plus one continuation. It reproduces
all **66** control logits, preserves **22/22** sequential outputs and improves
hybrid correlation from **0.9998908204** to **0.9999488108**. It confirms an
input-rounding contribution, but does not resolve the longer-sequence drift.
Its [runbook](../../../utilities/prefill-paired-input/README.md) describes the
two-pass GEMM, host-only output summation and build provenance. Production
engine sources, kernels and weight packing remain unchanged. The candidate
doubles projection dispatches and rejects stage-major. Precision correction,
production promotion, serving and final performance/resource acceptance remain
open.

## Earlier decode-entry checkpoint — 2026-10-13

Updated 2026-10-13. This branch adds strict decode/source validation, FP32
split-FFN and full-attention entry preparation, and paired-BF16 KV with FP32
gated attention/output-projection preparation. The complete FP32-entry export
passes **192/192 residual checks** at **64 layers × three positions**, worst
normalized error **0.0040853661 <0.005**. Logit correlation and argmax pass at
all three positions. The three residual failures seen before entry integration
are no longer reproduced on this fixture.

Independent attention gates pass: **1152/1152 heads**, worst absolute error
**0.0004209284 <0.02**; **48/48 composed attention branches**, worst normalized
error **0.0005515331 <0.005**. These use propagated inputs from the new export,
not forced inputs from the preceding isolated experiment.

The same export now passes both streamed-weight and resident execution:
all 387 activation/residual/logit captures match bytewise. Native-runtime logits
match the harness at all three positions; repeated requests and three full-buffer
reset replays pass. Independent final norm/Q8 LM-head checks pass at all three
positions, worst composed normalized error **0.0008821369 <0.005**.

DeltaNet recurrence now passes 144/144 whole-state/output checks with independent
FP64 carry from captured input records. Worst normalized errors are
**1.76881057e-5 / 3.68898003e-5 <1e-4**; conv history is bit-exact and padded
state rows remain zero. All 6912 head-state diagnostics pass, while
**99/6912 head-output diagnostics fail**: worst relative error **0.0493718151**
at reference magnitude 9.91052341e-8, absolute error 4.89300530e-9. Maximum
absolute error among failed head outputs is 1.80884340e-7. These failures remain
visible; whole-tensor acceptance does not imply per-head acceptance.

The checker covers all declared linear layers and positions, validates initial
state, sizes, finite values and export identity, and records 289 input hashes.
All 288 retained activation/state captures match the previous audit. The host
suite passes **1168 tests, with 47 skipped**, including 38 new recurrence tests.
No runtime/kernel change, new NPU run or full-model load was needed.

An independent input-record/composed-state checker now exposes additional
precision failures: **121/144 cases PASS**, state **133/144 PASS** with worst
error **0.0100545744**, output **134/144 PASS** with worst **0.0317001700**.
Its primary FP64 norm/projections, conv history and recurrent state are
unrounded, conditioned only on incoming layer residuals. Q/K/V fields have
2/5/5 failures under the 0.005 bound. Conditioned glue passes all 144 cases
at <1e-4 and cosine >0.99999; it cannot override the composed failures.

CPU diagnostics separately round entry norm, conv history, or both to BF16.
Both together reproduce the NPU state/output comparisons within 0.005 in all
144 cases, but do not constitute a hardware correction. Entry rounding alone
causes up to 0.0335971623 output error relative to the exact reference; history
rounding alone causes up to 0.0054568534 state error. The checker returns exit
status 1, pins 433 inputs and 336 effective weights, and uses one layer's
weights at a time. Its 30 new tests bring the suite to **1198 passed, 47 skipped**.

DeltaNet precision corrections and their hardware validation, head-output
diagnostics, remaining primitive/projection gates, prefill, serving and final
performance/resource acceptance remain open. This is not complete model support.

## Stage 1: complete decode comparison

- Added manifest-derived fixture metadata with the expected token/layer counts,
  hidden/logit sizes, context capacity, seed, spec/build identifiers and reference SHA256 hashes.
- The comparator requires every declared residual and logit capture, exact
  sizes and finite values. Reference hashes are verified before comparison.
- Enforced residual normalized maximum error < 0.005 alongside the existing
  logit correlation > 0.9999 and equal-argmax requirements. Top-5 remains a
  diagnostic. Zero reference residuals require exact zero outputs.
- Default comparison covers the whole fixture; explicit `--tokens` must match.
  Missing metadata requires reference regeneration. A cfg-only rewrite cannot
  relabel or rehash an existing reference, and interrupted generation invalidates
  the old metadata.
- Recorded and printed independent versus device-assisted reference routing.
  Metadata identifies the fixture; it does not hash all weights/kernel binaries.

TDD: the initial regression suite produced **38 failures and 4 passes** before
implementation. After the fix and generator integration checks, **51 tests
pass**. Cases cover truncated/missing files, missing complete layers, non-finite
values, modified references, residual bounds, zero residuals, token-count
mismatches, argmax/correlation failures and interrupted reference generation.
A further failing regression pinned context capacity for cfg-only rewrites.

## Stage 2: model preflight and production build

Added `q4nx-build --normalize-config -i DIR` for existing dense Qwen3.5-family
containers. It checks Q4NX header/ranges, layer indices and input-norm widths,
rejects conflicting metadata, normalizes through the converter's runtime
helper and preserves canonical `model_type=qwen3_5`. The original config is
backed up before atomic replacement. Weights/tokenizer files are not rewritten;
the command works offline and repeated invocation is a no-op.
Added `utilities/check-open-model.cpp` to run the production C++ manifest/config
check without loading XRT or model weights.

Source validation now fails for missing requested layers and non-finite tensor
comparisons. Finite equal scalar/constant tensors pass; nonconstant tensors
retain correlation > 0.99. TDD started with seven failing normalization tests
and six failing source-check tests. Additional regressions cover malformed
metadata/descriptors, canonical model type and backup preservation: **14
normalization tests and six source-check tests pass**.

Local model directory: `Models/qwen38-27b/Qwen3.8-27B-NPU2`. Weights are
hard-linked from the existing container; metadata assets are separate copies.
The normalized config passes C++ `Manifest::check_model`. It derives 64 layers,
248077 real tokenizer IDs and 248320 padded logits. The final export uses
`--model-dir` so those values, rather than a shape-only spec's vocabulary,
are recorded in its manifest.

Independent source check: text layers **0 and 3**, **25 tensors**, `ALL MATCH`;
minimum correlation **0.996756**. Source is the first safetensors shard of
`Qwen/Qwen3.8-27B` at revision `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`.
This checks representative linear/full-attention layers, not all model weights.

Production export: **41 sets** (four decode, 32 attention products, five GEMMs)
under `Models/qwen38-27b/production-kernels`. All **82 binary hashes**, every
manifest file reference and the model-derived spec/build keys were verified.
Spec hash: `ff08e8129933d27b3edad6830486bf757285805118972ffffb3d86ed92640ab0`.
Build key: `7424a5e3be78958995870b6b37a5551a07a0969de477310bf5d7599b09a0c092`.
Program memory maxima:

| Set | Maximum core .text, bytes | Limit |
|---|---:|---:|
| lx | 15584 | 16384 |
| ax | 14400 | 16384 |
| ln | 3424 | 16384 |
| lm_head_q8 | 2432 | 16384 |
| Attention products | 1984 | 16384 |
| GEMMs | 4784 | 16384 |

Linux toolchain: Python 3.14.4, mlir-aie 1.4.2,
llvm-aie `21.0.0.2026080301+c9c5ecb7`, XRT tools in `/opt/xilinx/xrt/bin`.
The exporter requires a local multiprocessing socket, so it was run outside
the sandbox after the restricted attempt failed at forkserver startup.

Reproduction and source/weight hashes are recorded in
[the production preflight skill](../../../.agents/skills/open-qwen38-production-preflight/SKILL.md).
The export's `toolchain.json` records binary hashes and build provenance.
Compilation was followed by the hardware validation below.

## Stage 3: eight-layer NPU decode and reset validation

Tested on AMD Ryzen AI 9 365 / Strix XDNA2, native Linux, with the production
export above. The harness and standalone C++ CLI were rebuilt from this branch;
CTest passed 3/3. Before execution, all 82 exported binary hashes were verified,
including equality of the eight decode artifacts in the harness's build
directories with their exported counterparts used by the runtime.

Regenerated the independent CPU fixture with `make_decode.py --layers 8
--tokens 3`. It includes six linear-attention and two full-attention layers,
seed token 248045, context capacity 4096, 5120 residual values per layer and
248320 logits per step. `decode_reference.json` pins all 27 reference captures
and the model-derived spec/build keys. The CPU replica mixes FP64 norm/state/
attention math with FP32 projections and LM-head accumulation; reference
captures are stored as FP32.
Harness execution completed **30/30 dispatches**, and the strict comparator
passed all three logits and all 24 layer residuals:

| Step | Full-logit correlation | Argmax, NPU / CPU | Harness / runtime logits |
|---|---:|---|---|
| 0 | 0.9999991073 | 220 / 220 | Byte-identical |
| 1 | 0.9999988538 | 220 / 220 | Byte-identical |
| 2 | 0.9999990533 | 220 / 220 | Byte-identical |

Worst normalized residual maximum error: **0.001397009**, layer 4 at step 0,
below the unchanged **0.005** bound. Every logit correlation exceeds **0.9999**;
dimensions, finite values and reference hashes also pass. The CPU reference
and runtime consume the same sequence: 248045, 220, 220.

The CLI's `--twice` run reproduced all three generated token IDs. Its second
request's logit dumps match the harness byte for byte. An additional
`--det-step 3 --det-full` run replayed the three input IDs from reset three
times: **0/3 replays differed** from the NPU reference run. Captured activation
buffers, current KV rows, residual/norm buffers and logits match byte for byte;
recurrent state is checked by hash. This tests repeatability, not independent
numerical correctness of the recurrent state or unused KV rows.

Reproduction commands are in the production preflight skill. Local captures,
reference metadata, logs and full-precision metrics are retained under ignored
`Models/qwen38-27b/production-slice/` (`results.json`, `logs/`).
This stage changes documentation only; no arithmetic fix or threshold change
was needed. Full 64-layer accuracy, per-head/state reference comparisons,
prompt coverage, serving and hybrid block-prefill parity remain unverified.
No performance claim: the power-mode request failed, and CPU reference
generation ran concurrently with the runtime checks.

## Stage 4: runtime comparison and full-depth checks

Added `compare_decode.py --runtime-prefix PREFIX` to require the CLI's
`PREFIX_tN.bin` logits to match the harness for every declared fixture step.
It rejects missing files, wrong sizes, partial float bytes, non-finite values,
one-ULP differences and signed-zero differences. CPU-reference residual/logit
gates still apply. A runtime run with more steps than the fixture does not
extend the fixture's coverage.

TDD: **18 new tests failed before implementation**, then all **69 comparator
tests passed**. The complete open-engine suite passes **845 tests, 47 skipped**.
The new option also passed on the retained eight-layer/three-token fixture.

The full-depth first-token fixture (seed 248045) passed the same strict gates:
**64/64 residual comparisons**, **248320 logits**, correlation
**0.9999973866**, and argmax **8678 / 8678**. Worst residual normalized maximum
error is **0.002713315**, layer 57, below **0.005**. Harness execution completed
**66 dispatches**. Its logits and the second runtime request's first-token
logits are byte-identical. All reference hashes, dimensions and finite-value
checks passed; no threshold or kernel arithmetic changes were needed.

The hardware capture ran while CPU reference generation continued, using the
same `build_cfg` function after all input pools were written. Its configuration
was verified byte-identical to `run_decode.cfg` published with the completed
reference contract. The 82 export hashes and eight harness decode artifact
hashes were checked before execution.

On the same Ryzen AI 9 365 / Strix NPU, the full **64-layer** runtime generated
8678, 198, 2 from seed 248045 and reproduced those IDs on a second request.
Three reset replays of inputs 248045, 8678, 198 reported **0/3 differing runs**
for observed buffers and recurrent-state hashes. These are repeatability
checks against an NPU reference run, not CPU state validation.

Two short text requests used the model's `chat_template.jinja` with thinking
disabled and sequential NPU prefill:

| Prompt | Prompt tokens | Answer before EOS | Repeated request |
|---|---:|---|---|
| Reply with only the result: 2 + 2 = ? | 25 | 4 | Identical |
| Name the capital of France. Reply with only the city. | 24 | Paris | Identical |

Both emitted `<|im_end|>` immediately after the answer. The diagnostic CLI
generated eight tokens per request; tokens after the first EOS are excluded
from the answer. Rendered prompts, exact input IDs, the template hash and raw
logs are retained under `Models/qwen38-27b/production-full/`. This is limited
chat smoke coverage, not serving or hybrid block-prefill validation.

Full-precision metrics and the pinned reference metadata are in
`Models/qwen38-27b/production-full/results.json`; the skill documents the run
commands. At stage 4, only the first full-depth token had an independent CPU
comparison; stage 5b below extends it to three. Independent state/head/FFN-partial
checks, block-boundary prefill parity, serving and resource/timing acceptance remain
open. The power-mode request still failed, so this is not a benchmark result.

## Stage 5 preparation: upstream compatibility and explicit kernel selection

The branch already contains upstream `c2ef0e8` through merge `9ace4fb`.
Relevant changes are exact Q8 splits in the block-prefill route (#172),
Windows OpenMP wait-policy initialization (#177), and AVX2 host DeltaNet
prefill loops (#179). They affect the shared runtime/prefill infrastructure;
they do not replace the decode comparator or config normalization. FLUX image
support is separate from this text-model validation.

For this Qwen3.8 container, the newly derived manifest equals the retained
production export apart from `build_key`; it still uses manifest version 1.
The source key changed from `7424a5e3...` to `04c6a214...`. All 82 retained
export hashes pass. Working build-directory xclbins differ from that export
(the inspected decode headers report XRT 2.21.75 versus 2.26.0); their instruction
streams match. The exporter's strict equivalence check rejected the working
build copies. They were not substituted for the validated export.

Added `make_decode.py --kernel-dir DIR` to use an explicit compatible export.
It checks the full manifest except its source build key, verifies matching
toolchain identifiers and all required binary hashes, and confines binary paths
to that directory. Both the generated harness paths and reference build identity
refer to the selected export. It never rewrites old provenance to claim a new
source build. TDD began with 14 failing tests; 15 export-selection tests now
pass, plus generator integration coverage for cfg-only and historical keys.

The updated native runtime builds and CTest passes 3/3. The block-host fixture
passes, including upstream's scalar/AVX2 checks. These CPU tests do not validate
the full hybrid route on the NPU.

On 2026-10-09, the 64-layer/three-token run exhausted available memory:
runtime loading reached 50 layers after 704 seconds, with available RAM briefly
around 260 MiB. It produced no completed decode result. The CPU fixture was also
incomplete and had no acceptance metadata. Hardware work was paused at the
user's request. This memory block was resolved on 2026-10-10, as recorded below.

## Stage 5b: full-depth three-token numerical comparison

Resumed on 2026-10-10 with about 62 GiB available RAM. CPU preparation and NPU
jobs ran serially, using the hash-verified production export explicitly via
`--kernel-dir`. Weights and packing were unchanged; retained input pools were
reused. No kernels or libraries were rebuilt for this stage.

CPU preparation exposed tensor-wide strides in Q4 dequantization's advanced
indexing. Contiguous gathers preserve the physical band mapping and FP32
multiply/add while avoiding those strides. TDD: four failing layout cases and
four passing cases before the change, then **8/8 passed**, covering empty,
single/multiple and strided input chunks against a scalar decoder. The complete
open-engine suite passes **870 tests, 47 skipped**.

`utilities/benchmark-q4-decode.py` compares the retained legacy decoder with the
new decoder on a real tensor. For layer 0's FFN up projection (10880 chunks),
outputs were byte-identical; dequantization took 7.94 s versus 0.236 s, about
33.6x faster in this single measurement. This is CPU reference preparation, not
NPU or end-to-end inference performance. All 65 first-token reference captures
(64 residuals and logits) also match the earlier full-depth reference bytewise.

The completed fixture pins **195 reference captures** for 64 layers, three
steps, 5120 residual values and 248320 logits, with seed 248045. The harness
completed **198 dispatches**. Strict comparison passed all **192 residuals**
and all three full-logit arrays:

| Step | Logit correlation | Argmax, NPU / CPU | Harness / runtime logits |
|---|---:|---|---|
| 0 | 0.9999973866 | 8678 / 8678 | Byte-identical |
| 1 | 0.9999920389 | 198 / 198 | Byte-identical |
| 2 | 0.9999972735 | 2 / 2 | Byte-identical |

Worst residual normalized maximum error: **0.004046247**, layer 63 at step 1,
below the unchanged **0.005** bound. Every correlation exceeds **0.9999**;
reference hashes, finite values, dimensions and argmax checks pass. The current
runtime reproduced the generated IDs on a second request, whose three logit
dumps pass `--runtime-prefix`. Three reset replays of inputs 248045, 8678, 198
reported **0/3 differing runs** for observed buffers and recurrent-state hashes.
This confirms repeatability against an NPU run, not independent state accuracy.

Full-precision metrics, reference metadata and dated logs are retained under
ignored `Models/qwen38-27b/production-multitoken/{results.json,logs/}`.
Stage 5b is complete for these three positions. Independent state/head/FFN-partial
checks remain open, as do block-prefill, serving and final resource acceptance.
The power-mode request still failed; no NPU throughput claim is made.

## Stage 5c1: strict FFN down-partial diagnostics

The current merge includes prefill `fast`/`lean` route selection and manifest
v3 support for BF16 GEMM packing. The Qwen3.8 dense export remains manifest v1;
its complete manifest is still compatible apart from source build identity,
and all 82 recorded binary hashes pass. The sequential layer designs and
Qwen3.5 recipe have no changes in this update. The current C++ CLI builds and
CTest passes 3/3. No new NPU run or kernel rebuild was performed in this stage.

Added `model/ffn_partials.py`, which loads one down matrix at a time and checks
both K pieces [0,8192) and [8192,17408), their sum and the closing residual.
The independent FP64 dot products use captured NPU `h` and pre-FFN residual;
this isolates down arithmetic/packing and closure, not up/gate/norm accuracy.
The comparator requires exact capture sizes, finite consumed values, valid
layer coverage and fixture/export identity. It records capture hashes and
fails if any boundary exceeds its strict normalized max-error bound of 0.005.
This adds a partial-level diagnostic gate; the existing full-layer residual
and logit bounds are unchanged.

TDD: tests first failed to import the absent checker, then 29 cases passed.
A further failing regression established that a rounded-input diagnostic cannot
turn the original comparison into PASS; **30 checker tests now pass**. Coverage
includes cancelling partial errors, zero references, exact threshold rejection,
non-finite/incorrect-size captures and missing later tokens.

Analyzed retained stage-5b captures for **64 layers × 3 positions** (192 cases)
without loading the full model on the NPU. The weight SHA256 still matches the
stage-5b model. These are historical NPU captures analyzed with the new checker,
not hardware measurements of the rebuilt runtime.

| Boundary | Worst normalized maximum error | Layer / position | Result |
|---|---:|---|---|
| Partial 0 | 0.004358931 | 49 / 2 | PASS, 192/192 |
| Partial 1 | 0.007325846 | 60 / 2 | **FAIL, 191/192 pass** |
| Sum | 0.004957893 | 60 / 2 | PASS, 192/192 |
| Closing residual | 0.002240655 | 59 / 2 | PASS, 192/192 |

The failing second partial at layer 60 / position 2 is largely explained by
input rounding: `gemv_q4_prep_f32_rt` converts `h` to BF16 before table
preparation. The optional `--diagnose-bf16` reduces that case's error to
**0.0000918061** against the rounded input. It never changes acceptance:
the original FP32-input comparison still fails and the CLI returns nonzero.
The remaining diagnostic error includes table/GEMV arithmetic; this experiment
does not establish a corrected kernel.

Evidence is retained in `production-multitoken/ffn-partials.json` and
`ffn-layer60-diagnostic.json` under the ignored model directory. Checker
implementation and diagnosis are complete; **FFN numerical acceptance remains
open**, alongside independent up/gate/norm, state and per-head comparisons.
This finding is reproduced and corrected in the isolated experiments below.
Full-model propagation and the other independent stage checks remain open.

## Stage 5c2: direct FP32 table preparation and isolated replay

The verified production kernel reproduced the layer-60 failure in a standalone
three-position replay: maxrel **0.0073258458** for the second partial, with the
captured FFN input and pre-FFN residual unchanged. Only that layer's pool,
constants and state were resident. The repeated positions carried state forward
from zero and received the retained previous layer's output at each position.

Added `gemv_q4_prep_f32_direct_blocks` for split dense FFNs. It quantizes the
original FP32 input directly to INT16 using per-32-value power-of-two scales,
ties-to-even rounding and endpoint saturation. Offset-correction sums also use
the original FP32 values. The existing 2.25*K table ABI, partition, packing and
buffer sizes are unchanged. Single-piece dense, BF16 and non-dense preparation
retain their existing paths; the generator regression verifies this scope.

With the corrected `lx`, the failing partial's error is **0.0000988872**.
All four boundaries pass at all three positions of layer 60, and the FFN input
and pre-FFN residual are byte-identical to the original capture. This is a
kernel correction against the unchanged FP32 reference, not the rounded-input
diagnostic used in stage 5c1.

A single-core hardware probe verifies 32 blocks spanning zero, both signs,
endpoint saturation, cancellation and scales from 2^-100 to 2^100. Shifts and
INT16 values match CPU rounding exactly. For block sums, the BF16 hi/lo ABI
requires an absolute floor at the minimum normal value when the low part is
subnormal; the 0.005 FFN gate is unchanged.

Experimental `lx` and `ax` were built from the same model-derived spec. Every
generated translation unit was checked against that recipe before the accepted
build. An earlier experimental build used a stale PER_CALL=2 source instead of
1 and produced non-finite pre-FFN values; that build is excluded from acceptance.
The accepted builds use the regenerated sources. Maximum core program sizes:
**lx 15584 bytes**, **ax 14400 bytes**, below 16384. No extra activation-table
storage was introduced.

`utilities/qwen35_layer_replay.py` prepares and scores linear or full-attention
replays, carries recurrent state/KV across positions, pins inputs and binary
hashes, and rejects changed configuration before scoring. The sweep utility
runs one layer at a time. The kernel fix was preceded by the failing hardware
replay; source-generator, route-scope and replay-integrity regressions were
also tested red before implementation.

The corrected kernels then passed **64 isolated layers × 3 positions**, with
**192 NPU dispatches**. All captured FFN inputs and pre-FFN residuals remain
byte-identical. Every boundary passes the unchanged 0.005 gate:

| Boundary | Worst normalized maximum error | Layer / position |
|---|---:|---|
| Partial 0 | 0.0003117401 | 54 / 1 |
| Partial 1 | 0.0001995445 | 48 / 1 |
| Sum | 0.0003320237 | 54 / 1 |
| Closing residual | 0.0000744078 | 62 / 1 |

The full open-engine suite passes **914 tests, 47 skipped**, including ten
replay/generator/sweep regressions. Empty/interrupted sweeps cannot retain a
successful summary. This completes the isolated split-down correction, not
end-to-end propagation through all corrected layers.

Reproduction is documented in the new
[FP32 down replay skill](../../../.agents/skills/open-qwen35-fp32-down-replay/SKILL.md).
Artifacts and logs are retained under ignored `Models/qwen38-27b/ffn-replay/`.
The production export remains intact; corrected full-model logits, independent
up/gate/norm, state/head accuracy, prefill and serving acceptance remain open.

## Stage 5c3: complete corrected export and propagation regression

A clean build from `c718cbe` produced all **41 kernel sets / 82 verified binary
hashes** in `Models/qwen38-27b/production-fp32-kernels`. The source build key is
`sha256:e9cf679f2870b23c689ab09ab3820713a6345d6dfe62defadabaf63a5d069e39`;
the model-derived spec and weight packing are unchanged. Every retained core
fits program memory: maxima lx 15584, ax 14400, ln 3424, LM head 2432,
attention products 1984 and GEMM 4784 bytes (<16384). Old build folders with
unwritable container-owned project files were preserved before rebuilding.
The prior verified export was preserved. No model reconversion was needed.

The CPU fixture was regenerated against the new export. All 195 reference
capture hashes match the existing CPU reference; its build identity is new.
The unchanged model hash and 65 reused pool hashes are recorded. Native harness
and runtime were rebuilt, CTest passed 3/3, the config/manifest parser passed,
and the open-engine suite passed **914 tests, 47 skipped**.

On the NPU, the complete 64-layer / three-position chain executed 198 dispatches.
These are propagated corrected outputs, not isolated historical layer inputs.
The strict comparator correctly reports **FAIL** without changing its bounds:

| Position | Logit correlation | Argmax NPU / CPU | Residual failures |
|---|---:|---|---|
| 0 | 0.9999977820 | 8678 / 8678 | None |
| 1 | 0.9999932894 | 198 / 198 | Layers 55, 56, 57, 58, 59, 61, 63 |
| 2 | 0.9999972692 | 2 / 2 | None |

All 192 residuals were checked; 185 pass. Worst maxrel is **0.0083300389**
at layer 63 / position 1 against the unchanged **0.005** bound. Passing logits
and equal generated tokens do not override those residual failures.

The independent FP64 down checker passes all **192** propagated layer/position
pairs at all four boundaries. Worst maxrel:

| Boundary | Error | Layer / position |
|---|---:|---|
| Partial 0 | 0.0002570118 | 45 / 1 |
| Partial 1 | 0.0002904925 | 43 / 1 |
| Sum | 0.0002560792 | 41 / 2 |
| Closing residual | 0.0001103267 | 63 / 1 |

This establishes local down-projection accuracy from captured FP32 `h`, including
where the full residual fails. It does not establish up/gate/norm or
attention/state accuracy, nor identify which of those causes the accumulated
error. Further independent checks must localize it before another correction.
Runtime/harness logits are byte-identical at all three positions. A second
request reproduced IDs, and full-buffer reset replay reported 0/3 differing runs.
These repeatability checks do not prove independent CPU state accuracy.

The new export is marked `VALIDATION_FAILED.txt` and is **not promoted** as
numerically accepted. Evidence lives in
`Models/qwen38-27b/production-fp32-multitoken/`: `results.json`,
`build-audit.json`, `input-sha256.json`, `ffn-partials.json` and `logs/`.
The new [corrected production decode skill](../../../.agents/skills/open-qwen38-fp32-production/SKILL.md)
documents reproduction. The power-mode request failed; no throughput claim.
Next: localize the full-chain error with independent up/gate/norm and
attention/state checks, then rerun the unchanged acceptance gates. Prefill,
serving and final resource acceptance remain open.

## Stage 5c4: independent FFN front validation

Added `open_kernels/model/ffn_front.py`. It reads the captured FP32 pre-FFN
residual, BF16 normalized input and FP32 combined activation `h`, and computes
RMSNorm plus `SiLU(Wgate @ norm) * (Wup @ norm)` in FP64 from the same stored
weights. The primary reference does not round norm to BF16. Every boundary
keeps the strict maxrel <0.005 bound. Export/fixture identity, complete requested
coverage, exact buffer sizes and finite inputs are enforced; JSON pins capture
hashes and a failed numerical check exits nonzero.

A separate comparison uses the captured normalized input. BF16-rounded norm
and activation references, plus the predicted activation change from rounding,
are diagnostics: none can turn a failed primary check into PASS. Raw up/gate
projections are not retained in these kernels, so this validates their combined
activation only, not each projection independently or attention/state accuracy.

All 64 layers × 3 positions were checked from the corrected full-chain captures;
no new hardware run or kernel build was performed:

| Boundary | Passing cases | Worst maxrel | Layer / position |
|---|---:|---:|---|
| Post-attention norm | 192/192 | 0.0036529193 | 63 / 1 |
| Norm → combined up/gate activation | **191/192** | **0.0071636206** | 11 / 0 |
| Combined activation from captured BF16 norm | 192/192 | 0.0016461914 | 36 / 0 |

At the failing layer 11 / position 0, the captured-input comparison has error
0.0003658505. Using an ideally BF16-rounded norm gives error 0.0003203473;
the predicted activation error due to that rounding alone is 0.0074496288.
The captured norm differs numerically from the ideal BF16 norm in three values.
This isolates a local precision loss dominated by norm-input quantization; it
does not establish that this is the cause of the seven position-1 full-chain
residual failures. All seven of those layer/position pairs pass the new local
front gates, so accumulated error and attention/state still need investigation.

TDD: missing checker → failing tests → implementation; an additional failing
diagnostic test enforces that the rounded reference cannot mask the failure.
All **40 new tests pass**; full open-engine suite: **954 passed, 47 skipped**.
Coverage includes both layer layouts, mixed BF16/FP32 offsets, non-finite/shape
errors, zero references, the strict threshold, missing positions, stale export
identity and CLI failure status. No production arithmetic or threshold changed.

Evidence: `production-fp32-multitoken/ffn-front.json`,
`ffn-front-layer11.json` and `logs/ffn-front-*` under ignored model artifacts.
Next: reproduce layer 11 with an isolated kernel and evaluate preserving FP32
norm input through up/gate preparation, then rerun local and full-chain gates.
The complete corrected export remains unaccepted. Separate up/gate captures,
independent state/head checks, prefill and serving are still open.

## Stage 5c5a: isolated FP32 norm-to-up/gate correction

Added an experimental two-core design in `utilities/ffn-front-replay/`. It uses
the existing FP32 RMSNorm helper (`ln_nr32`), the real production Q4 pools and
the existing GEMV/SiLU arithmetic. A control build uses legacy BF16-rounded
input preparation; the candidate uses direct FP32-to-INT16 preparation. It
captures **up, gate and h separately**, allowing independent projection gates.
The production layer layout, kernel export and arithmetic were not changed.

Both builds ran all three retained pre-FFN residual inputs for layer 11 on the
NPU. The input and FP32 norm buffers match bytewise between the builds. The
control h also matches the original full-layer capture bytewise at all three
positions, reproducing the primary T0 FAIL **0.0071636206**. The corrected T0
error is **0.0001517240**, with the same unrounded FP64 reference and 0.005 bound.

| Corrected boundary | Worst maxrel across 3 positions |
|---|---:|
| Norm | 0.0000063849 |
| Up projection | 0.0000937135 |
| Gate projection | 0.0000796810 |
| Combined activation h | 0.0001517240 |

All independent gates pass in the corrected probe. Program .text is 6304 bytes
on the GEMV core and 2944 on the norm helper, both below 16384. Those small
probe cores do not establish that the full lx/ax images will fit after integration.
The experiment streams one layer's weights and needs no resident full NPU model.

The preparation/scoring tool pins the kernel, run configuration, model, pool,
source capture and input hashes; it rejects missing hash coverage or changed
inputs. Output sizes and finite values are checked. It invalidates a stale score
before validation and reports nonzero for a failed numerical gate. Model hashing
streams the weight file. Output bands contain up[64], gate[64], h[64], not three
contiguous whole vectors.

TDD: **9 new tests**, including projection errors hidden by the product, mixed
band decoding, non-finite outputs, missing/changed input hashes and stale-score
invalidation. The hardware control fails and corrected probe passes. Full suite:
**963 passed, 47 skipped**. No new full-model hardware result is claimed.

Reproduction is in the new
[FP32 front replay skill](../../../.agents/skills/open-qwen35-fp32-front-replay/SKILL.md).
Evidence: ignored `Models/qwen38-27b/ffn-front-replay/` with `results.json`,
build/hardware logs, hashes, core sizes and separate FAIL/PASS score files.
Next is production integration: widen the norm-to-up/gate stream, handle the
buffer/manifest ABI explicitly, verify complete lx/ax program memory, then run
local and full-chain regression. The seven full-chain residual failures and
independent state/head, prefill and serving gates remain open.

## Stage 5c5b: integrated FP32 norm-to-up/gate

The split dense lx/ax path now streams the post-attention norm as two FP32
halves through `ln_nr32` and uses direct FP32 table preparation for up/gate.
Layer-entry attention norm and unsplit dense recipes retain BF16. The existing
FP32 down preparation is reused; model weights and packing are unchanged.

Norm scratch grows from three to five 4-KiB elements at hidden 5120. Both act
buffers grow by 8192 bytes; the manifest declares `ffn_norm_dtype: float32`.
The capture reader follows this field and compatibility checks reject old
layouts. The native manifest/config checker accepts the new complete export.
Old exports and captures remain intact and must not be relabelled.

TDD: six new tests failed before implementation and pass after it, covering
scratch boundaries, unchanged unsplit recipes, explicit ABI compatibility and
FP32 capture decoding. Full suite: **969 passed, 47 skipped**; CTest **3/3**.

A complete export contains **41 sets / 82 verified hashes**. Build key:
`sha256:b707550f6c1d82a71a12998672e41e248ae70d4171757bfb5d1714432be3b3ae`.
Every retained core fits the 16384-byte program budget: lx maximum **15584**,
ax **14400**. All 195 freshly generated CPU reference hashes match the previous
reference; model and all 65 reused pool hashes were verified. The fixture was
bound to the completed export before hardware execution.

The NPU completed **198 dispatches, 64 layers × 3 positions**. All **192
residuals pass** the unchanged 0.005 bound; worst **0.0044951978** at L61/T1.
All three logits pass, with correlations **0.9999986446 / 0.9999957289 /
0.9999973938** and matching argmax **8678 / 198 / 2**. The seven residual
failures recorded before norm integration no longer occur on this fixture.
Runtime logits match harness captures byte for byte at every position, and a
second request reproduces the first. Full-buffer reset replay reports **0/3**
differing runs, including recurrent-state hashes; this measures repeatability
against an NPU reference.

The new propagated captures pass all **192 independent down checks**. Worst
normalized errors are partial0 **0.0003103112**, partial1 **0.0003736917**,
sum **0.0003362342**, closing residual **0.0000686012**. These local gates
use captured FP32 h and do not replace the full-chain comparison.

The independent FP64 front checker also passes **192/192**: worst norm
maxrel **0.0000172679**, combined activation **0.0009100331**. The diagnostic
activation conditioned on captured norm is at most **0.0009182833**. L11/T0's
primary combined activation error is now **0.0001456326**, below the same
0.005 bound. This uses newly propagated inputs; individual up/gate captures
remain limited to the earlier isolated probe.

Reproduction and ABI details are in the new
[FP32 norm integration skill](../../../.agents/skills/open-qwen38-fp32-norm-integration/SKILL.md).
Artifacts are under ignored `Models/qwen38-27b/production-fp32-front-kernels/`
and `production-fp32-front-multitoken/`, including build/input audits, source
diff, reference hashes, numerical results and logs. This three-position result
does not establish independent recurrent-state/head accuracy, block-prefill
parity, serving isolation or final resource acceptance.

## Stage 5c6: independent full-attention head comparison

Added `open_kernels/model/attention_heads.py`. It reads retained Q/K/V/gate
projections and gated BF16 head outputs, and independently computes Q/K
RMSNorm, partial RoPE, causal GQA and sigmoid in FP64. Each layer starts its
reference KV history at position zero. The default covers all full-attention
layers and fixture positions; restricted layer selection is explicit.

The enforced per-head gates are **absolute max error <0.02** and **cosine
>0.9999**. Normalized error is diagnostic, not a substitute for the absolute
bound. Ideal BF16 output-rounding error is also diagnostic and cannot turn a
failed unrounded-reference comparison into success. The checker rejects bad
coverage, stale export identities, malformed sizes and non-finite data, records
capture/effective-norm hashes, and returns nonzero on a failed numerical gate.

On the retained FP32-integrated run, **1109/1152 heads pass**, across 16 layers,
24 heads and 3 positions. **43 fail** the absolute bound: layers 51 (5), 55 (1),
59 (7), 63 (30); positions 0/1/2 account for 11/16/16 failures. All cosine gates
pass (minimum **0.9999399829**). Worst absolute error **0.2226126094** is at
L59/T2/H7. Its normalized error is only **0.00368121**, illustrating why the
absolute gate must be enforced separately. Maximum normalized error over all
heads is **0.0183480320**.

Even ideally rounding the FP64 output to BF16 exceeds 0.02 in **24 cases**,
with a maximum rounding error of **0.0696828407**. The remaining **19 failures**
cannot be explained by output rounding alone. At the worst observed head the
NPU also differs from the ideally rounded output by **0.25**. These diagnostics
do not yet isolate QK normalization, KV quantization or softmax arithmetic.

TDD: **41 new tests** cover scalar-reference agreement, head grouping/order,
partial RoPE, causal history, independent absolute/cosine gates, output-rounding
masking, zero/non-finite data, capture layout, coverage and CLI failure. Full
suite: **1010 passed, 47 skipped**. No kernels were rebuilt or new NPU run made.
Artifacts: `production-fp32-front-multitoken/attention-heads{,-summary}.json`
and test logs. The export now carries a scoped `VALIDATION_FAILED.txt` marker.
Previously passed full-chain logits/residual and FFN/runtime gates remain valid;
this stricter local check identifies an additional, unresolved acceptance gap.

Next: isolated attention replay with separate QK/KV/softmax/output boundaries
and an FP32 output experiment. Retained fixtures contain no recurrent-state
snapshots; independent DeltaNet state validation still needs capture support.
Individual projections, final LM head, prefill and serving remain separate.

## Stage 5c7a: isolated attention output and KV precision

Added `utilities/attention-replay/`, reusing the production split attention
worker/functions with six cores and four heads per core. One dispatch runs all
three positions with hardware KV history; each layer starts from zero KV.
The control calls the production BF16 finish and promotes its result. The
experimental finish preserves the same arithmetic through its FP32 output.
Both retain scaled Q hi/lo pairs, BF16 K/V, O accumulators and padded softmax
max/sum state. No projection pools or resident full model are loaded.

The control reproduces all retained production outputs **byte for byte** across
16 layers × 3 positions, including the same **43/1152** failing heads and worst
absolute error **0.2226126094**. FP32 output reduces failures to **21/1152**:
L51 (5), L59 (3), L63 (13), worst **0.1974321142** at L59/T2/H9. Every cosine
passes, minimum **0.9999418362**. Inputs, KV and all debug buffers match bytewise
between modes. Primary acceptance remains FAIL at the original absolute-error
and cosine bounds, against the unrounded FP64 reference.

| Conditional diagnostic | BF16 output | FP32 output |
|---|---:|---:|
| Softmax/PV from captured Q/K/V, worst absolute error | 0.0002344581 | 0.0002344581 |
| Final output from captured Q/K/V | 23 FAIL, worst 0.1096219261 | 0 FAIL, worst 0.0003921240 |
| Final output from captured KV and independently recomputed ideal Q | 23 FAIL, worst 0.1096586925 | 0 FAIL, worst 0.0003919703 |

These diagnostics localize the remaining discrepancy primarily to KV
representation at the observed boundaries. They do not replace the primary
reference or validate a higher-precision KV implementation. An output-only
change is insufficient and has not been integrated into production.

The build wrapper pins mode, token count, geometry, source key and binaries
only after successful compilation. Preparation/scoring verify exact input hash
coverage, captured constants against model norms, dimensions and finite values;
stale scores are removed before validation. The control must also reproduce
production output bytes. Both numerical score commands return exit 1; both
hardware runs return exit 0 with 16 dispatches each.

Core .text maxima: control **15968**, FP32 **15680**, below 16384. The initial
three-output-FIFO design exceeded the core's two output DMA channels; core 0
now emits KV and debug on one shared FIFO. The new
[attention replay skill](../../../.agents/skills/open-qwen38-attention-replay/SKILL.md)
documents the build and this constraint. Artifacts are under ignored
`Models/qwen38-27b/attention-replay/`, including mode-specific build/audit/score
JSON, captures, logs, `results.json` and `build-audit.json`.

Added 11 host regressions for packing, debug decoding, causal KV history,
mode/source identity, input hashes, stale scores and conditional diagnostics
that cannot hide primary errors. Full suite: **1021 passed, 47 skipped**.
Production kernel sources, ABI, weights and packing are unchanged. Next is an
isolated higher-precision KV experiment before considering production fit and
full-depth regression; independent DeltaNet state, LM head, prefill and serving
remain separate acceptance work.

## Stage 5c7b: paired-BF16 K/V with FP32 output

Added `utilities/attention-kv-replay/` with two modes sharing the same widened
cache transport. The control zeroes low planes and uses production row
arithmetic; the experiment retains K/V hi+lo through QK and PV. Both preserve
FP32 gated outputs. An unchanged FP64 oracle recomputes QK norm/RoPE, causal
GQA and sigmoid from retained projections at the original absolute-error and
cosine bounds. No resident model or projection pools are loaded.

| Check, 16 layers × 3 positions × 24 heads | BF16 KV control | Paired KV |
|---|---:|---:|
| Primary heads passing | 1131 /1152 | **1152 /1152** |
| Worst absolute error (L59/T2/H9) | 0.1974321142 | **0.0009283289** |
| Minimum cosine | 0.9999418362 | **0.9999999991** |
| Worst K boundary error | 0.0613304117 | 0.0002521316 |
| Worst V boundary error | 0.1993179321 | 0.0001831055 |
| Maximum linked core program bytes | 15184 | 14992 |

Control outputs/debug and reconstructed KV exactly reproduce the earlier FP32
replay. Inputs, Q and high K/V planes are identical across modes. Conditional
softmax/PV and output comparisons also pass, but primary acceptance is against
the independent unrounded reference. Both hardware runs complete 16 dispatches;
control scoring returns exit 1, paired scoring and the controlled comparison
return exit 0. Scope remains three-position attention on captured projections.

The cache row is `[K_hi,K_lo,V_hi,V_lo]`, **8192 bytes** rather than 4096.
Four planes are acquired through the unchanged 2048-byte input FIFO element.
With output FIFO depth 2, the initial design exceeded local memory:
**68112 >65536 bytes**. Depth 1 saves 4096 bytes; the compiler's actual
allocation ends at **64016**, including a 6144-byte stack. Both instrumented
builds fit program memory. The four host data buffers total **816 KiB**.
Production ax fit and performance have not been established.

The current merged sources have build key `db351c64dc4e…`; the retained
fixture/export remains `b707550f6c1d…`. The replay validates its current source,
geometry and binary hashes separately from the fixture's original export.
Both identities are recorded, and mismatches on either side fail. The old
export is not relabeled. Attention source files are unchanged by the latest
merge; the exact control replay additionally confirms the observed behavior.

TDD: 10 layout/precision cases were introduced before the implementation;
five additional provenance regressions failed before the identity handling
was implemented. All 15 pass. Full open-engine suite: **1038 passed, 47 skipped**.
Artifacts, hashes and failure/success logs are retained under ignored
`Models/qwen38-27b/attention-kv-replay/`. The
[paired-KV replay skill](../../../.agents/skills/open-qwen38-paired-kv-replay/SKILL.md)
documents reproduction, DMA/L1 constraints and resource inspection.

Production attention/cache/output-projection ABIs, model weights and packing
remain unchanged. Next assess the full ax and FP32 output-to-projection
contract, integrate the correction, rebuild the complete export and rerun
full-depth decode. Independent DeltaNet state/LM-head, prefill and serving
remain separate acceptance work.

## Validation

| Check | Result |
|---|---|
| `specs/open-engine/tests` | 1038 passed, 47 skipped |
| `utilities/q4nx-build/tests` | 113 passed, 42 subtests passed |
| Standalone C++ CLI build | Passed |
| CTest | 3/3 passed |
| Manifest generation from the checked-in spec and local container | Both derive `qwen35`, 64 layers, 41 kernel build sets, 64 AB lanes and down split 8192 + 9216 |

The open-engine suite was rerun for stage 5c7b on the merged tree. The converter
and C++ build/CTest entries retain their earlier stage coverage; they were not
rerun for this isolated experiment and do not validate the merged runtime.
The latest full-model evidence belongs to the explicitly identified retained
export. Stage 5c7b runs only isolated attention with current sources. Weights
were not reconverted, and skipped tests extend neither scope.

Stage 1 reproduction:

```bash
OPENBLAS_NUM_THREADS=1 ironvenv/bin/python -m pytest specs/open-engine/tests/test_compare_decode.py -q
OPENBLAS_NUM_THREADS=1 ironvenv/bin/python -m pytest specs/open-engine/tests -q
```

The full suite passes with the new comparator tests included. Existing numerical
references and kernel arithmetic were not changed.

## Findings

### Decode comparison regression fixed

The comparator previously truncated logits to the shorter array, printed
residual errors without enforcing them, and stopped checking layers when
either capture was missing.

The following FP32 captures reproduced a false positive:

```text
ref_logits.bin = [1, 3, 2, 4, 0]
y_logits.bin   = [1, 3, 2, 4]
ref_res0.bin   = [1, 3, 2, 4, 0]
y_res0.bin     = [-1, -3, -2, -4, 0]
```

The previous comparator printed residual correlation `-1` and normalized
maximum error `2`, then reported **PASS** with exit `0`. Stage 1 rejects both
the truncated logits and, independently, the incorrect residual. This fixes
acceptance of invalid captures; it does not demonstrate a numerical change
in real model outputs.

### Local container metadata normalized

The initial nested `text_config` failed C++ `Manifest::check_model` with
`config.json lacks 'head_dim'`. Stage 2 provides a converter command to fix
this metadata layout and verifies the normalized production directory against
the same C++ check. The source directory and weight contents are preserved.

### Block prefill includes host computation

`Core::block_layer_linear` invokes host RMSNorm and `host::deltanet_block`
between NPU GEMMs. Block prefill therefore uses a hybrid CPU/NPU execution
route; it must be distinguished from sequential NPU execution when reporting
coverage and performance. No performance measurements were made in this stage.

## Stage 5c7c: integrated paired attention

The wide Qwen3.5-family geometry now uses paired-BF16 K/V and FP32 gated
attention output in complete `ax` layers. Cache rows are 8192 bytes in
`[K_hi,K_lo,V_hi,V_lo]` order. Each output element carries two FP32 heads;
`o_proj` consumes six 4-KiB elements directly through FP32 table preparation.
This retains the existing FIFO/DMA topology and 6144-byte stack. The manifest
explicitly declares the cache/output dtypes and validates global/per-layer KV
row sizes. Other geometries retain their existing ABI. Weights and packing
are unchanged.

Host block attention preserves both cache planes across blocks. Paired-cache
NPU block attention is not implemented: the runtime reports host attention
for this prefill route. Public BF16 cache inspection keeps its original logical
head count by returning the high plane; snapshots/reset retain all bytes.
Independent block-prefill accuracy is not established by this change.

Added `--build-root` for separate writable compiler projects, and an export
audit checking recipe/source identity, all binary hashes and linked core
program sizes. Earlier standalone replay generators reject the new ABI and
require their matching source revision. Native manifest tests now reparse
independent fixtures instead of copying the non-copyable GGUF ownership field.

TDD covers geometry, FP32 capture precision, legacy shape compatibility,
cross-block low-plane retention and isolated build workspaces. The completed
host suite reports **1049 passed, 47 skipped**; native CTest reports **4/4 PASS**.
The new complete export has **41 sets / 82 verified binary hashes**. Every core
fits 16384 bytes: fullest `lx` is **15584**, fullest `ax` is **13760**.
Build key: `sha256:fa34220099d565bfeb514765e3c72665f768a42ee17e9ba37dd7fa3402788405`.
Artifacts: `Models/qwen38-27b/production-paired-attention/`, including
`build-audit.json`, `input-sha256.json`, build logs and the matching decode fixture.

Reproduction: [integration skill](../../../.agents/skills/open-qwen38-paired-attention-integration/SKILL.md).

All **195 CPU reference hashes are unchanged**, and model/65 weight-pool hashes
are verified. Available memory fell to approximately 28 GiB before resident
execution. A TDD-tested cfg transformer therefore reuses one layer-weight BO,
retaining all layer states and capture/dispatch ordering. Logical buffers fall
from **16.618 to 2.960 GiB**; source and transformed cfg hashes are recorded.

The streamed run completes **198 dispatches / 64 layers × three positions**.
Independent production attention passes **1152/1152 heads**: worst absolute
error **0.0003238841** (layer 51, position 2, head 6), minimum cosine
**0.9999999988341115**. Logit correlations are **0.9999986691 / 0.9999964904 /
0.9999982652**, with matching argmax **8678 / 198 / 2**.

Full-chain acceptance remains **FAIL**, with **189/192 residual checks passing**.
At position 1, layers 59/61/63 have normalized max errors **0.0055346809 /
0.0051228118 / 0.0058082229**, exceeding the unchanged 0.005 bound. The export
carries `VALIDATION_FAILED.txt`. Local attention success does not override the
full-chain failures; this build must not be promoted as fully validated support.

Resident-runtime identity, repeated requests and reset are not yet tested for
this export. The streamed run is not evidence for those gates or for throughput.
Next localize the remaining full-chain error, then rerun its acceptance gate;
independent state/head/projection, block-prefill and serving remain separate.

All **192 FFN front** and **192 split-down** checks pass on propagated captures.
Worst normalized norm/front errors are **0.0000164919 / 0.0005658096**.
Down partial0/partial1/sum/closing-residual maxima are **0.0002353918 /
0.0001898039 / 0.0001780994 / 0.0000365645**. These conditional local gates do
not establish independent projection or recurrent-state correctness.

## Stage 5c8: attention entry, projection and sublayer gates

Added `open_kernels/model/attention_projections.py`. The checker covers all
16 full-attention layers × three positions using retained incoming residuals.
It recomputes FP64 entry RMSNorm and Q/gate/K/V projections without substituting
captured BF16 norm. Captured-norm and ideal-rounded-norm comparisons remain
separate diagnostics. Output projection and residual addition are checked
against captured gated attention. A separate sublayer reference recomputes
unrounded projections, Q/K norm, RoPE, causal KV history, softmax, gating and
output projection across every position.

All primary normalized maximum errors must be <0.005; zero references require
exact zero. The checker rejects non-finite/malformed captures, stale export
identity, incomplete requested coverage, and the unsplit norm path whose closing
operation overwrites the entry capture. It records 96 capture/input hashes and
112 effective-weight hashes. These are conditional checks on incoming residuals;
they do not validate independent model state or source-tensor conversion.

Results on the unchanged paired-attention export:

| Boundary | Passing cases | Worst normalized max error |
|---|---:|---:|
| Entry norm/Q/gate/K/V (all required per case) | 48/48 | Q 0.0045171202 |
| Output projection from captured gated attention | 48/48 | 0.0002811151 |
| Residual addition from captured output | 48/48 | 0.0000000726 |
| Full attention branch output from unrounded entry reference | **38/48** | **0.0236300737** |
| Post-attention residual from that sublayer reference | 48/48 | 0.0023039741 |

The ten branch-output failures are L11/T1, L15/T1, L23/T1, L23/T2, L27/T1,
L35/T1, L39/T1, L47/T1, L55/T0 and L59/T0. Worst is L23/T1: absolute error
0.0191318717 on reference output magnitude 0.8096407976. Scoring only the
larger residual would hide these failures. The checker returns **exit 1**.

Entry norm/Q/gate/K/V maxima are 0.0030703652 / 0.0045171202 / 0.0026948500 /
0.0036089342 / 0.0029681746. With captured norm, Q/gate/K/V errors fall to
0.0002698095 / 0.0003015196 / 0.0003423122 / 0.0004635156. Together with the
passing head and output-projection gates, this identifies entry precision and
its amplification through attention as the next controlled experiment. It does
**not** prove that BF16 norm alone causes the three full-chain failures.

TDD: initial import failure; the composed gate then added two failing regressions
before implementation. **33 new tests** cover independent references,
cancellation hidden by rounded inputs, causal input history, separate output/
addition gates, capture integrity, provenance, coverage and CLI failure status.
Full suite: **1082 passed, 47 skipped**. No kernel, weight, reference or threshold
was changed; no new hardware dispatch/full-model residency was needed. The
41-set/82-hash export audit still passes. Full-chain comparison still returns
exit 1 with the same three residual failures and matching logit argmax.

Evidence: `Models/qwen38-27b/attention-projections/` contains results, test logs,
export audit and the repeated full-chain comparison. The export's failure marker
remains in place. Next: an isolated FP32-entry-norm versus BF16-control replay
on matching inputs/history, then integration only if the local gates improve.
Resident runtime/reset, DeltaNet state, LM head, prefill and serving remain open.

## Stage 5c9: controlled FP32 attention-entry replay

Added `utilities/attention-entry-replay/` to test the entry-precision hypothesis
on all 16 full-attention layers × three positions. The BF16 control pins copies
of the retained production ax binaries. The candidate builds a private full ax
with FP32 entry RMSNorm (`ln_nr32`) and direct FP32 Q/gate/K/V table preparation.
It appends 20480 bytes of norm scratch after the original 266240-byte act buffer;
other offsets, weights, paired attention and FFN stages are unchanged. Exact
source anchors and independent experiment/source/binary hashes prevent accidental
reuse after source drift. Production recipes and manifests are not modified.

Both variants receive the same retained incoming residuals, constants and weight
pools. State/act are reset between layers and K/V history is retained across
positions. Each completes **48 NPU dispatches**. Control reproduces all **96
activation/residual captures byte for byte**, including its ten branch failures.
The shared `attention_projections.score_layer` oracle preserves all independent
FP64 math and the strict normalized max-error threshold **<0.005**.

| Primary gate | BF16 control worst | FP32 candidate worst |
|---|---:|---:|
| Entry norm | 0.0030703652 | 0.0000103846 |
| Q projection | 0.0045171202 | 0.0001849842 |
| Gate projection | 0.0026948500 | 0.0002755196 |
| K projection | 0.0036089342 | 0.0002652475 |
| V projection | 0.0029681746 | 0.0002784229 |
| Output projection from captured attention | 0.0002811151 | 0.0001703476 |
| Composed attention branch output | **0.0236300737** | **0.0005290482** |
| Composed post-attention residual | 0.0023039741 | 0.0000867198 |

The FP32 candidate passes **48/48 cases**, removing all **10 local failures**.
The control score returns exit 1 as expected; candidate scoring returns exit 0.
Final comparison passes with **185 identical common input hashes**, matching
effective weights and exact reproduction of the retained control oracle results.
This establishes the local correction on forced retained inputs. It does not
establish full-model propagation or prove that all three full-chain residual
failures disappear when layers receive corrected upstream outputs.

Candidate program memory: all 15 compute cores fit 16384 bytes, maximum
**13760**; GEMV cores **9824**, norm helper **3600**. Compiler-assigned L1 buffer
end reaches **59920/65536 bytes**, including reserved stack space; the **6144-byte
stack is unchanged**. Data BO sizes total **254085120 bytes** (~242.31 MiB),
using one reused layer pool. Native Python 3.14 / mlir-aie 1.4.2 /
Peano 21.0.0.2026080301+c9c5ecb7; the retained native harness runs on XDNA2.
No full-model residency was required.

TDD adds eight replay/config/provenance/comparison tests, with recorded red then
green results. The existing 33 projection tests cover the extracted shared oracle.
Full suite: **1090 passed, 47 skipped**. Artifacts and logs are in
`Models/qwen38-27b/attention-entry-replay/`; build/resource metadata pins generated
designs, binaries and core ELF evidence. Candidate source hash:
`fcbd7273f296c66ed434ee19da637291286dc6a36408dadef7983d76901b35e5`.

Reproduction: [FP32 entry replay skill](../../../.agents/skills/open-qwen38-fp32-attention-entry/SKILL.md).
Next integrate the entry dtype/scratch/stream ABI into production, build a fresh
complete export and repeat full-depth gates. The retained production export's
three residual failures and failure marker remain unchanged; resident runtime/
reset, independent state/LM-head, prefill and serving acceptance remain open.

## Stage 5c10: production FP32 attention entry and full-depth validation

Integrated FP32 entry RMSNorm and direct FP32 Q/gate/K/V preparation into the
paired, split dense attention recipe. The manifest declares `attn_entry_dtype:
float32`; entry scratch grows from three to five 4096-byte stream elements.
Full-attention act size is **274432 bytes**, an increase of **8192 bytes**.
Later scratch offsets move consistently in the recipe and generated sequences.
Linear-attention entry remains BF16. Model tensors, packing, FFN precision and
paired attention semantics are unchanged.

The shared norm helper emits two FP32 halves for attention entry. Its existing
FIFO depths and **6144-byte stack** are preserved. The production checker reads
the declared dtype after full manifest/fixture validation; old layouts and
unknown dtypes fail. The native parser also rejects unknown entry dtypes.
The historical private replay now refuses integrated recipes and documents
its required checkout, preventing accidental reinterpretation of old buffers.

TDD: five new contract tests and the native parser test failed before the fix;
a further regression first exposed the historical replay's missing ABI guard.
The manifest-to-capture dtype path has its own regression. Full suite:
**1097 passed, 47 skipped**; native CTest **4/4 PASS**. The model config passes
the rebuilt C++ production parser.

Built a fresh complete export: **41 kernel sets, 82 verified binary hashes**.
Maximum .text is **15584 bytes** for lx and **13760** for ax, below 16384.
Compiler-assigned L1 buffer ends are **64384** and **59920** bytes respectively,
below 65536 with the existing stack. Native toolchain: Python 3.14, mlir-aie
1.4.2, Peano 21.0.0.2026080301+c9c5ecb7. Build key:
`sha256:63d7cd0879422f0f31f0b9d1e018799b8db868a3f110e3ee5f908a3faa5fc8da`.

Model and all 65 reused weight-pool hashes match retained provenance. All
**195 regenerated independent CPU reference hashes are unchanged**. The NPU
run completes **198 dispatches**, **64 layers × three positions**, using one
reloaded weight BO while retaining every layer's state. Logical data buffers
are **2.960 GiB**; this is a correctness run, not a resident-runtime benchmark.

| Gate on the new propagated captures | Result | Worst error |
|---|---:|---:|
| Full-depth residual | **192/192 PASS** | **0.0040853661** (L63/T1) |
| Attention heads | **1152/1152 PASS** | absolute **0.0004209284** |
| Entry norm/Q/gate/K/V | **48/48 PASS** | V **0.0003181145** |
| Conditioned output projection | **48/48 PASS** | **0.0001953298** |
| Composed attention branch | **48/48 PASS** | **0.0005515331** |
| Composed post-attention residual | **48/48 PASS** | **0.0000732911** |
| FFN norm/front | **192/192 PASS** | **0.0000160763 / 0.0013213623** |
| Split down partial0/partial1 | **192/192 PASS** | **0.0003836580 / 0.0002338498** |
| Split down sum/closing residual | **192/192 PASS** | **0.0002655461 / 0.0000957904** |

Except the explicitly absolute per-head metric, these are normalized maximum
errors under the unchanged **0.005** bound. Attention cosine minimum is
**0.999999998443142**, above 0.9999; absolute bound remains 0.02. Logit
correlations are **0.9999988740 / 0.9999971796 / 0.9999982962**; argmax
**8678 / 198 / 2** matches CPU at every position. The preceding export's three
full-depth residual failures and ten attention-branch failures do not reproduce
on the new propagated captures. No reference or threshold was relaxed.

Artifacts: `Models/qwen38-27b/production-fp32-entry/`, including export/resource,
input/reference audits, numerical JSON, captures and logs. Old exports and their
failure markers remain untouched. Reproduction is documented in the
[FP32 entry integration skill](../../../.agents/skills/open-qwen38-fp32-entry-integration/SKILL.md).

Resident runtime/harness byte identity, repeated requests and reset for this
export remain untested: available memory was about 30 GiB versus the estimated
35–40 GiB full-load requirement. Independent recurrent-state/LM-head checks,
block-prefill parity, serving isolation and final resource/performance gates
remain open. Next add independent state/LM-head validation; run resident gates
separately when memory is available. Passing this fixture is not complete model
support or proof for arbitrary prompts/context lengths.

## Stage 5c11a: final norm/head checks, state captures and resident replay

Added opt-in `make_decode.py --capture-boundaries`. It dumps the complete linear
recurrent state after each layer/position and BF16 final RMSNorm output before
LM-head dispatch. It verifies the supported tail contract and leaves all existing
dispatches/default captures unchanged. Separate fixture directories preserve
previous evidence. No kernels, native binaries, weight packing or references
were changed.

Added `lm_head.py`, an independent FP64 tail checker conditioned on captured
last-layer residuals. It streams native Q8 weights in 256-row slabs and computes
unrounded RMSNorm → logits without materializing the full head matrix. Captured
norm → logits is a separate required projection gate; ideally rounded BF16 norm
is diagnostic only. Every position and all 248320 padded vocabulary rows are
checked. Norm and projection errors must remain <0.005; projection correlation
must exceed 0.9999 and argmax must match. Identical constant outputs are handled
without undefined correlation. Malformed/non-finite captures, incompatible
identity/dimensions, unsupported head format and incomplete row coverage fail.
Results pin nine captures, fixture metadata and both effective weight hashes.

TDD: missing-checker collection failure and two failing capture-contract tests
preceded implementation. **33 new tests** cover BF16 cancellation hidden by
conditioned references, near-tie argmax, zero references, exact row coverage,
Q8 row/K ordering, provenance, bad inputs and CLI failure. Full host suite:
**1130 passed, 47 skipped**.

The unchanged 41-set/82-hash export audit passes. A new streamed 64×3 NPU run
completes 198 dispatches; all **387 existing captures reproduce byte for byte**.
It adds **144 state captures (468910080 bytes)** and **three norm captures**.
All 195 independent CPU reference hashes are preserved. The full-chain
residual/logit comparator remains PASS.

| Tail gate (three positions) | Result | Worst normalized error |
|---|---:|---:|
| Final RMSNorm against unrounded FP64 | 3/3 PASS | 0.0021158452 |
| Composed RMSNorm → logits | 3/3 PASS | 0.0008821369 |
| Projection from captured norm | 3/3 PASS | 0.0000110584 |

Composed correlations are **0.9999993625 / 0.9999987783 / 0.9999989898**;
argmax **8678 / 198 / 2** matches the local independent reference. The primary
reference was not replaced with rounded or device-derived norm values.

Available memory rose to about **60 GiB**, allowing the previously deferred
resident checks on this same export. The original resident harness cfg completes
198 dispatches and reproduces all **387 streamed captures bytewise**. The hash-
verified native CLI runs all 64 layers and three tokens twice; request 2 reproduces
the first. All **three runtime logit captures are byte-identical** to resident
harness output and pass the independent full-depth comparator. Forced IDs
248045,8678,198 with `--det-step 3 --det-full` produce **0 of 3 differing replays**.
This validates repeatability against an NPU reference, not independent recurrent
state accuracy. The power-mode request failed; no throughput claim is made.

Evidence is under `Models/qwen38-27b/head-state-validation/`: `decode/` retains
new boundaries, `resident/` retains resident captures, and the audits, numerical
JSON and logs pin both runs. The integrated export's build key is unchanged:
`63d7cd0879422f0f31f0b9d1e018799b8db868a3f110e3ee5f908a3faa5fc8da`.
Next independently validate the captured multi-token DeltaNet state. Remaining
all-layer primitive/projection, prefill, serving and final acceptance gates are
not implied by the passing tail or reset checks.

### Isolated FP32 linear-entry validation (2026-10-14)

The private full-lx candidate preserves FP32 norm through QKV/Z and alpha/beta,
appending 20480 bytes of act scratch. Conv history stays BF16 and the production
export is unchanged. Its builder generates recipe-specific translation units in
a private directory and verifies their hashes, removing dependence on stale
GEMV entries. Rolled AB loops fit all 11 cores: maximum .text 15248 and L1 buffer
end 64384 bytes; stack remains 6144 bytes.

Both variants complete 144 NPU dispatches. The BF16 control reproduces 432
captures bytewise and every retained oracle row. FP32 passes 143/144 cases
versus 121/144 for BF16. All output and Q/K/V/decay/beta checks pass; worst
composed output error falls from 0.0317001700 to 0.0044733739. One state fails
at L57/T1: 0.0054339342 against 0.005. Conditioned glue/recurrence pass 144/144;
95/6912 head-output diagnostics remain failing. Acceptance bounds are unchanged.

The CPU history-rounding diagnostic predicts a 0.0054568534 state effect at
L57/T1 and matches the candidate there within 0.0002856729. This supports the
next isolated higher-precision conv-history experiment, not a production PASS.
Tests: 12 new TDD cases; full host suite 1210 passed, 47 skipped. Reproduction is
in the linear-entry skill, with evidence under Models/qwen38-27b/linear-entry-replay.
