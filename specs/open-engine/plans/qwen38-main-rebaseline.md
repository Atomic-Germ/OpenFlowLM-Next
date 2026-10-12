# Qwen3.8-27B validation report

Updated 2026-10-12. This branch adds strict decode/source validation, FP32
split-FFN input preparation, and integrated paired-BF16 KV with FP32 gated
attention/output-projection preparation. The new complete export passes all
**1152 independent attention-head checks** at 64 layers × three positions;
worst absolute error is **0.0003238841**, below 0.02.

Full-chain acceptance remains **FAIL**: **189/192 residuals pass**; layers
59/61/63 at position 1 exceed 0.005 (worst **0.00580822**). Logit correlation and
argmax pass at all three positions. The hardware run streams layer weights
through one shared BO while retaining every layer state. Resident-runtime
identity/reset for this export remains untested because available memory is
below the full-load requirement. This is not complete model support.

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
[the production preflight skill](../../../.opencode/skill/open-qwen38-production-preflight/SKILL.md).
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
[FP32 down replay skill](../../../.opencode/skill/open-qwen35-fp32-down-replay/SKILL.md).
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
The new [corrected production decode skill](../../../.opencode/skill/open-qwen38-fp32-production/SKILL.md)
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
[FP32 front replay skill](../../../.opencode/skill/open-qwen35-fp32-front-replay/SKILL.md).
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
[FP32 norm integration skill](../../../.opencode/skill/open-qwen38-fp32-norm-integration/SKILL.md).
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
[attention replay skill](../../../.opencode/skill/open-qwen38-attention-replay/SKILL.md)
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
[paired-KV replay skill](../../../.opencode/skill/open-qwen38-paired-kv-replay/SKILL.md)
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

Reproduction: [integration skill](../../../.opencode/skill/open-qwen38-paired-attention-integration/SKILL.md).

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
