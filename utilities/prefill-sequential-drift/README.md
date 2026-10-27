# Sequential long-prefix drift diagnosis

## Current standard FP32 export

Use `post_targets.py` with the audited `post_boundaries.py` result. It validates
all three prompt lengths and all ordinary/traced routes again, including source,
build, model, reference, raw capture and prefix identities. A completed numerical
FAIL is a valid diagnostic input; its acceptance result must remain unchanged.
The previous tools and their historical artifacts are retained without edits.

```bash
OPENBLAS_NUM_THREADS=1 ironvenv/bin/python utilities/prefill-sequential-drift/post_targets.py run \
  --baseline Models/qwen38-27b/post-prefill-production/boundaries-native-v1 \
  --target 0:125,205,239 --target 40:239 --target 41:239 \
  --target 53:125 --target 54:125 --target 57:205 --target 58:205 \
  --out Models/qwen38-27b/post-prefill-drift/targets-v1
OPENBLAS_NUM_THREADS=1 ironvenv/bin/python utilities/prefill-sequential-drift/post_targets.py audit \
  --out Models/qwen38-27b/post-prefill-drift/targets-v1
```

This selection covers the first temporal crossing L54/P125, shallowest failing
layer L41/P239, worst residual L58/P205, their preceding layers and a shared-input
L0 control. Only one layer's decoded weights is cached at a time. No NPU dispatch,
kernel build, model repacking or full-model residency is required. Use a fresh
output directory, and do not overlap weight replay with resident NPU validation.

The replay reuses the independent oracle and causal state/counterfactual helpers.
Every selected reference must first reproduce the retained FP32 reference bytes.
Both sequential and hybrid native histories are evaluated independently. Signed
error components must close; scalar maxima cannot be added. The gap between
native and ideal output driven by native inputs includes local arithmetic AND
accumulated native state error. Ideal replay state is not a native state capture.
The separate audit replays the oracle and requires byte-exact NPZ arrays and
identical measurements. Exit zero means diagnosis completed, not model acceptance.

### Validated FP32 checkpoint — 2026-10-27

`targets-v1` and its separate audit complete with 6686 bound input files,
9 exact reference reconstructions and 18 byte-exact NPZ replays. Peak decoded
weight cache: 1.42578125 GiB. The three selected failure targets are dominated
by current-input effects; the largest sequential implementation gap over the
selection is 1.43036e-5. Sequential total gates pass 5/9 and hybrid 9/9.
The original sequential numerical FAIL remains unchanged. Earlier layers must
be examined to locate the initial source; this result does not establish a fix.
Eight new TDD tests; full host suite: 1933 passed, 47 skipped.

### Sweep every layer at selected positions

The same evaluator can attribute P125/P205/P239 through all 64 layers. It still
replays every causal input row up to P239; these are selected output positions,
not a scan of every possible output. Use a new directory and a separate audit:

```bash
replay_targets=()
for layer in {0..63}; do replay_targets+=(--target "$layer:125,205,239"); done
OPENBLAS_NUM_THREADS=1 ironvenv/bin/python utilities/prefill-sequential-drift/post_targets.py run \
  --baseline Models/qwen38-27b/post-prefill-production/boundaries-native-v1 \
  "${replay_targets[@]}" \
  --out Models/qwen38-27b/post-prefill-drift/all-layers-v1
OPENBLAS_NUM_THREADS=1 ironvenv/bin/python utilities/prefill-sequential-drift/post_targets.py audit \
  --out Models/qwen38-27b/post-prefill-drift/all-layers-v1
```

Require exactly 192 reference reconstructions and 384 replayed array sets.
Rank implementation-gap and input-history contributions separately from total
error. Each layer has its own reference scale: ratios between layers' `maxrel`
values do not measure causal amplification. Confirm any proposed native replay
target from its signed vectors and causal history before changing arithmetic.

### Validated all-layer checkpoint — 2026-10-27

`all-layers-v1` and its separate audit complete: 192 exact reference outputs,
384 byte-exact array-set replays, 6686 bound inputs, and 18/18 exact overlaps
with `targets-v1`. Peak decoded-weight cache remains 1.42578125 GiB. All 35
focused tests pass again; no evaluator or production code changed.

Sequential selected total-output gates pass 168/192; hybrid passes 192/192.
First failures at P125/P205/P239 are L54/L50/L41. The maximum sequential
implementation gap is 0.0000631635 (L63/P239); the maximum history effect is
0.0002090308 at the same point. Current-input effects dominate each of the 24
failing selected cases. The native baseline's overall numerical FAIL remains.

Before L41, the three largest selected implementation gaps are at L3, making
that full-attention layer an early capture candidate. Replay its real sequential
L2 inputs with the current standard ax binary and causal KV before inspecting
projection, attention and FFN boundaries. Establish exact native residual
identity first. This is a diagnostic target, not a confirmed root cause.

## Current full-ax L3 native replay

`native_l3.py` reuses the current standard ax binary and manifest ABI. It packs
only L3 and reads all sequential L2 inputs from the audited 257-token trace.
KV starts at zero and advances through P239 without resets. Each position saves
its residual, complete activation scratch and the complete active KV prefix;
unused capacity is not dumped. The immutable CPU reference retains its original
build key; current export identity is verified separately against current source.

```bash
OPENBLAS_NUM_THREADS=1 ironvenv/bin/python utilities/prefill-sequential-drift/native_l3.py prepare \
  --baseline Models/qwen38-27b/post-prefill-production/boundaries-native-v1 \
  --out Models/qwen38-27b/post-prefill-drift/native-l3-v1
XILINX_XRT=/opt/xilinx/xrt LD_LIBRARY_PATH=/opt/xilinx/xrt/lib OPENBLAS_NUM_THREADS=1 \
  ironvenv/bin/python utilities/prefill-sequential-drift/native_l3.py run \
  --out Models/qwen38-27b/post-prefill-drift/native-l3-v1 \
  --harness Models/qwen38-27b/prefill-sequential-drift/harness-l0-v1/run_kernel
OPENBLAS_NUM_THREADS=1 ironvenv/bin/python utilities/prefill-sequential-drift/native_l3.py audit \
  --out Models/qwen38-27b/post-prefill-drift/native-l3-v1
```

Use a fresh output directory. The runner checks the harness executable and all
hashes in its adjacent `build-evidence.json`, including source/header identities.
The existing harness can be reused only while that check passes. Use the shared
resident-validation lock; never overlap another NPU job. No kernel rebuild or
full-model residency is needed. Preserve failed attempts and never replace an
exact control requirement with a numerical tolerance.

After all 240 residuals match the native trace byte-for-byte and the separate
audit verifies all 720 files, score independent boundaries:

```bash
OPENBLAS_NUM_THREADS=1 ironvenv/bin/python utilities/prefill-sequential-drift/l3_boundaries.py run \
  --capture Models/qwen38-27b/post-prefill-drift/native-l3-v1 \
  --out Models/qwen38-27b/post-prefill-drift/l3-boundaries-v1
OPENBLAS_NUM_THREADS=1 ironvenv/bin/python utilities/prefill-sequential-drift/l3_boundaries.py audit \
  --out Models/qwen38-27b/post-prefill-drift/l3-boundaries-v1
```

The scorer checks FP64 entry and causal attention/output from native layer
inputs across every position, 5760 attention heads conditioned on captured
Q/K/V/gate, and FFN front/down at P125/P205/P239. It does not expose separate
raw up/gate projections or softmax accumulators. Captured KV is retained for
subsequent diagnosis; ideal recomputed KV is not a native state measurement.
A separate audit repeats CPU calculations and compares the entire report.
Passing conditioned gates cannot override the full-model sequential FAIL or
identify a unique root cause. Signed attribution on a common residual scale
is required before choosing a precision change.

### Validated L3 checkpoint — 2026-10-28

`native-l3-v1`: all 240 residuals are byte-exact to the full-model trace; separate
audit verifies all 720 captures (307,691,520 bytes). Logical NPU buffers total
254,072,832 bytes. Existing harness/build dependencies and kernels were reused.
`l3-boundaries-v1` and its separate recalculation audit pass all 240 projection/
attention cases, 5760 conditioned heads and three selected FFN front/down cases.
Worst entry projection maxrel is 0.0002980986 (V/P197); worst conditioned head
absolute error is 0.0000566654 (P76/H20). All original thresholds are unchanged.

25 new tests pass; full host suite: 1958 passed, 47 skipped. These local checks
do not close sequential full-model accuracy. Next attribute signed boundary
changes on a common residual scale, with exact retained mixed-oracle controls
and actual native KV, before choosing a precision experiment.

## Historical expanded export

CPU-only causal attribution at L60/P37–39 and L33/P240–242 using the verified
expanded 257-token native run. These include the first temporal residual failure
and the shallowest failing layer, plus neighboring positions. Compare sequential
and the passing hybrid route; the retained audit already requires both hybrid
schedules to match exactly. No model acceptance or NPU dispatch occurs here.

```bash
OPENBLAS_NUM_THREADS=1 ironvenv/bin/python utilities/prefill-sequential-drift/check.py run \
  --baseline Models/qwen38-27b/prefill-expanded-cli/boundaries-native-v1 \
  --out Models/qwen38-27b/prefill-sequential-drift/run-v1
OPENBLAS_NUM_THREADS=1 ironvenv/bin/python utilities/prefill-sequential-drift/check.py audit \
  --out Models/qwen38-27b/prefill-sequential-drift/run-v1
```

Use a fresh output directory. Validate the complete retained audit, preparation,
source/build/model identity and all native output hashes at all three lengths.
Remeasure the selected length from raw captures before interpreting it. Keep
its completed overall numerical FAIL; the diagnostic does not upgrade it.

Decode and cache effective Q4NX weights for one layer at a time. Replay every
causal row from zero conv/recurrent/KV state separately for independent-reference
inputs and each route's native inputs. Compute FFN only at selected positions:
other FFN outputs do not update the same layer's state, and preceding-layer
residuals already supply its future inputs. First reproduce every selected
independent reference bytewise after its original FP32 output rounding.

At each selected position also evaluate its native input on a copy of the
reference trajectory's state immediately before that position. State copies
are essential because the independent oracle mutates recurrent state. This
counterfactual must never alter the reference or another selected row.

Each NPZ contains native output, ideal output driven by native inputs/history,
reference-history output with the current native input, unrounded reference
output, stored reference, input vectors and these signed components:

- `reference_rounding`: unrounded CPU reference minus stored FP32 reference.
- `current_input`: reference-history output with native input minus unrounded reference.
- `input_history`: ideal native-history output minus the reference-history counterfactual.
- `implementation_gap`: native output minus ideal native-history output.
- `propagated`: current-input plus input-history effects.
- `total`: native output minus stored independent reference.

The first four components must sum to total within FP64 rounding error. All
output errors use the same reference scale; their scalar maxima cannot be
added because coordinates differ and signed errors may cancel. Input error
uses its own input scale and is reported separately. Zero-reference/nonzero
errors remain explicit JSON-safe FAIL values.

The implementation gap includes local arithmetic and the native state built
up over the prefix. Actual native sublayer/state captures are not available
in these files: do not attribute the gap to a particular kernel or treat the
ideal-history state as native state. This is not a corrected model execution.

The CPU audit rereads provenance and native captures, replays the oracle again,
requires byte-exact agreement for every saved array and recomputes all metrics.
Exit zero means the diagnostic completed, not that selected numerical gates
or the model passed. Preserve incomplete reports and never overwrite an
existing run. `model_acceptance` remains false in both run and audit.

Tests cover conv/recurrent/KV carry, omitted output positions, independent
history copies, reset between routes, signed cancellation/reference rounding,
invalid/nonfinite inputs, complete case inventory and persisted failures.

## Validated checkpoint — 2026-10-24

`run-v1` and its independent causal CPU re-audit complete: 6222 bound input
files, 6/6 exact reference reconstructions and 12 byte-exact NPZ replays.
Peak decoded-weight cache is 1.42578125 GiB. Twelve RED-to-GREEN tests; host
suite **1756 passed / 47 skipped**. Original native captures are unchanged.

| Sequential target | Current input | Input history | Implementation gap | Total maxrel |
|---|---:|---:|---:|---:|
| L33/P241 | 0.0050773486 | 0.0000522907 | 0.0000162030 | 0.0050229750 |
| L60/P38 | 0.0076863502 | 0.0000208493 | 0.0013556628 | 0.0090391466 |

Reference-rounding components are zero in these fixtures. Propagated maxrel is
0.0050250579 / 0.0076834838: the independent layer on native causal inputs
still fails at both crossings. Current input is the largest component, while
L60 also has a measurable implementation gap. Scalars must not be summed.
Sequential passes 4/6 selected output gates; hybrid passes 6/6 with worst
maxrel 1.8360597e-5 and implementation gap <=4.384963e-7.

Follow incoming drift at L32/P241 and L59/P38, and examine L0/P241, where
retained sequential residual error is already about 0.0012928 on the common
embedding input. Use that evidence to choose actual native sublayer/state
captures. Overall long-prefix sequential accuracy remains FAIL; these results
do not prove a particular arithmetic or state operation is responsible.

## Selecting preceding layers

`targets.py` reuses the causal replay and provenance checks without changing
`check.py` or its retained evidence. Specify each layer once and list unique
prompt positions; layer indices must be 0–63 and positions 0–256. Targets are
normalized and stored in the report. Audit uses exactly that recorded selection.

```bash
OPENBLAS_NUM_THREADS=1 ironvenv/bin/python utilities/prefill-sequential-drift/targets.py run \
  --baseline Models/qwen38-27b/prefill-expanded-cli/boundaries-native-v1 \
  --target 0:240,241,242 --target 32:240,241,242 --target 59:37,38,39 \
  --out Models/qwen38-27b/prefill-sequential-drift/predecessors-v1
OPENBLAS_NUM_THREADS=1 ironvenv/bin/python utilities/prefill-sequential-drift/targets.py audit \
  --out Models/qwen38-27b/prefill-sequential-drift/predecessors-v1
```

Layer zero uses FP32 model embeddings for every route and never reads a
nonexistent layer -1 residual. Its current-input, input-history and propagated
vectors must be exactly zero: otherwise the shared-input control is invalid.
A nonzero layer-zero implementation gap belongs to that layer's arithmetic or
native history, though residual-only captures cannot separate its operations.
Other layers read the preceding layer's captured causal prefix, excluding the
continuation row. Full-attention layers carry independent K/V at every position.

The same independent reference reconstruction, signed closure and fresh causal
CPU audit apply to every selected case. Refuse duplicate layers/positions,
out-of-range targets, incomplete artifacts and overwritten runs. Model acceptance
remains false. No new native build or model residency is required.

### Predecessor checkpoint — 2026-10-24

`predecessors-v1` and a separate causal CPU re-audit complete: **9/9 reference
reconstructions and 18/18 NPZ replays** are byte-exact. Both routes pass
9/9 selected output gates. Peak decoded-weight cache remains 1.42578125 GiB.
Fifteen RED-to-GREEN tests; full host suite **1771 passed / 47 skipped**.

| Sequential target | Current input | Input history | Implementation gap | Total maxrel |
|---|---:|---:|---:|---:|
| L0/P241 | 0 | 0 | 0.0012928460 | 0.0012928460 |
| L32/P241 | 0.0032113675 | 0.0001437686 | 0.0000152968 | 0.0030828957 |
| L59/P38 | 0.0041249896 | 0.0000384913 | 0.0000115911 | 0.0040749071 |

Reference rounding is zero for these fixtures. L32/L59 primarily transmit input
error; L0 isolates a first-layer arithmetic/native-history discrepancy on exact
shared embeddings. Hybrid L0/P241 error is 1.6313961e-6. The same token (1235)
occurs at P37/88/139/190/241; retained sequential L0/P37 error is already
0.0011891532, compared with 0.0012928460 at P241. This does not distinguish the
current arithmetic from recurrent history or identify a faulty kernel.

Next use a single production L0 weight pool with the complete embedding prefix
through P242. Require exact native residual identity at every position before
interpreting available entry/conv/recurrence/output/FFN boundaries at P37/P241
and neighboring rows. Keep production arithmetic and primary references intact.
Overall sequential long-prefix FAIL and model_acceptance=false remain unchanged.

## Production L0 causal capture

`native_l0.py` prepares a single production L0 pool with original model
embeddings at P0–242. It validates the retained long-prefix audit and current
recipe/bundle compatibility, repacks through the shared packer, and binds all
inputs and sources. The standard `open_kernels/harness` executes unchanged
production lx kernels. State and scratch are initialized once, then carried
through the entire prefix; only the incoming embedding is replaced per step.

Build instructions and XRT runtime selection are in
[the L0 replay skill](../../.agents/skills/open-qwen38-sequential-l0-replay/SKILL.md).
The build-tree XRT library is suitable for linking; runtime must use the
installed XDNA driver under `/opt/xilinx/xrt`. An attempt using build-tree
runtime libraries is retained as `native-l0-v1` and failed before dispatch.

```bash
OPENBLAS_NUM_THREADS=1 ironvenv/bin/python utilities/prefill-sequential-drift/native_l0.py prepare \
  --baseline Models/qwen38-27b/prefill-expanded-cli/boundaries-native-v1 \
  --out Models/qwen38-27b/prefill-sequential-drift/native-l0-v2
XILINX_XRT=/opt/xilinx/xrt LD_LIBRARY_PATH=/opt/xilinx/xrt/lib OPENBLAS_NUM_THREADS=1 \
  ironvenv/bin/python utilities/prefill-sequential-drift/native_l0.py run \
  --out Models/qwen38-27b/prefill-sequential-drift/native-l0-v2 \
  --harness Models/qwen38-27b/prefill-sequential-drift/harness-l0-v1/run_kernel
OPENBLAS_NUM_THREADS=1 ironvenv/bin/python utilities/prefill-sequential-drift/native_l0.py audit \
  --out Models/qwen38-27b/prefill-sequential-drift/native-l0-v2
OPENBLAS_NUM_THREADS=1 ironvenv/bin/python utilities/prefill-sequential-drift/l0_boundaries.py \
  --capture Models/qwen38-27b/prefill-sequential-drift/native-l0-v2 \
  --out Models/qwen38-27b/prefill-sequential-drift/l0-boundaries-v1
```

Use fresh directories; preparation and native captures cannot be overwritten.
The runner holds the common resident-validation lock and checks that the NPU is
idle. It stores all 729 complete residual/activation/state buffers. Every FP32
residual must match the full-model native trace bytewise, including signed zero.
Tolerance-based agreement cannot authorize boundary interpretation. A separate
audit rechecks all input/capture hashes and identity measurements; failures stay
explicit and do not become model acceptance.

`l0_boundaries.py` requires that exact audited control, then uses existing
independent entry/conv/recurrence references through every causal position from
zero. It retains strict per-head failures separately from whole-tensor gates.
At P36–38 and P240–242 it also checks Z, post-recurrence gating, output projection,
residual addition, FFN norm/activation and both down partials. These latter
checks are conditioned on captured inputs, not independent full-model evidence.
Raw FFN up/gate projections are not retained in production scratch; the checker
can observe their combined activation, not infer separate projection accuracy.

This replay uses approximately 0.240 GiB of logical NPU buffers and stores
906,743,808 capture bytes. CPU scoring decodes one layer only. The original
production binaries, baseline captures and overall sequential long-prefix FAIL
remain unchanged; passing local gates is not a correction or full-model PASS.

### Native L0 checkpoint — 2026-10-24

The isolated production replay completes 243 dispatches and reproduces every
retained full-model L0 residual exactly. Independent audit verifies 729 captures.
All 243 entry/conv/state gates and 11,664 strict recurrence head cases pass;
all six selected projection/front/down cases pass. Host suite: 1792 passed,
47 skipped. Production arithmetic and the overall long-prefix FAIL are unchanged.

At P37/P241, the post-recurrence norm/gate BF16 boundary has maxrel
0.0029327378 / 0.0029708429 against its unrounded conditioned reference.
The full-prefix composed recurrence output error stays below 2.121e-5.
These differently scaled errors do not quantify final residual contribution.
Next perform signed vector attribution of post-output rounding through output
projection and FFN before selecting an FP32 experiment. Keep the exact native
control and independent references intact.

## Signed post-output attribution

`post_attribution.py` analyzes the exact audited native L0 replay without an NPU
run. It carries two independent histories from zero: the existing mixed-precision
layer oracle (required to reproduce stored residual references bytewise) and the
FP64 entry/conv/recurrence reference. Native state never seeds either history.
One layer's effective weights are cached; the mixed/FP64 copies materialize
approximately 4.282 GiB of weights, not a resident full model.

```bash
OPENBLAS_NUM_THREADS=1 ironvenv/bin/python utilities/prefill-sequential-drift/post_attribution.py run \
  --capture Models/qwen38-27b/prefill-sequential-drift/native-l0-v2 \
  --out Models/qwen38-27b/prefill-sequential-drift/post-attribution-v1
OPENBLAS_NUM_THREADS=1 ironvenv/bin/python utilities/prefill-sequential-drift/post_attribution.py audit \
  --out Models/qwen38-27b/prefill-sequential-drift/post-attribution-v1
```

The six targets are P36–38/P240–242. Each NPZ retains the intermediate final
outputs, raw post/recurrence/Z vectors and a telescoping signed decomposition:
reference storage rounding, mixed-oracle versus FP64 math, recurrence, Z,
post-output BF16 rounding, remaining post arithmetic, output projection,
residual addition and conditioned FFN arithmetic. The post vectors are propagated
through the nonlinear FP64 FFN before taking differences. All output effects use
one stored-reference scale; maxima must not be added. The decomposition depends
on the stated replacement order and does not claim a unique nonlinear attribution.

`without_post_rounding` subtracts the rounding contribution while freezing the
other downstream error vectors. It is a CPU counterfactual, not a corrected NPU
execution; removing rounding can expose cancellation with another operation.
The audit reconstructs both causal trajectories and requires byte-identical NPZ
arrays and metrics. Missing references, changed controls or incomplete inventories
invalidate the result. Diagnostic completion never changes model acceptance.

### Post-output attribution checkpoint — 2026-10-24

All six references reconstruct bytewise; signed closure is exact. A separate
causal audit reproduces every NPZ array and metric. At P37/P241, native residual
maxrel is 0.0011891532 / 0.0012928460; the post-rounding contributions are
0.0012200074 / 0.0013051784. The frozen-downstream CPU prediction without that
rounding is 0.0000308542 / 0.0000123325, roughly 39x/105x lower. At P242 it
worsens from 0.0000202869 to 0.0000891240 because rounding and projection errors
partially cancel. These are local diagnostic effects, not model acceptance.

Nine RED-to-GREEN tests; full host suite 1801 passed / 47 skipped. No build or
NPU run was needed. Next validate a private FP32 post-output and direct-FP32
output-preparation path against exact production controls, retaining all 243
positions and the cancellation case at P242.
