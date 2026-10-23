# Running a stock Qwen3.6-MoE decode step on the open kernels

> The engine that keeps the weights resident and serves the app is
> `src/open_qwen36/` (built on the same packing and the same kernels). This
> directory is the batch harness path and the fp64 reference it is scored by.

This directory is the model-side half of the open path: it takes OFLM's own
`.q4nx` weight file, packs each layer's weights into the byte order the open
kernels stream, writes a driver program for `../harness/run_kernel.exe`, and
computes an fp64 CPU reference to score the result against.

Nothing here is closed: the container is parsed by [q4nx.py](q4nx.py), the pool
layouts by the recipe's packing plan (`../recipes/qwen36moe.py`, applied by
`../recipes/pack.py` -- the same plan `src/open_qwen36/pools.cpp` interprets),
the math by [replica.py](replica.py). OFLM's engine is never loaded. (The weights are still OFLM's file — the GGUF path that
replaces it is a separate piece of work; see the repo plan.)

## Use

```
python open_kernels/model/make_decode.py --layers 8 --tokens 1
open_kernels\harness\out\run_kernel.exe open_kernels\model\out\run_decode.cfg
python open_kernels/model/compare_decode.py --tokens 1
```

The program names the kernels at `../designs/layer_x/build_lx0` … `build_ax1`,
`../designs/ln/build` and `../designs/lm_head_q8/build_full`, which is where
`../export_qwen36_kernels.py` builds them (it also copies them to
`src/xclbins/<model>/open_kernels/` for the engine). The model directory comes
from `OFLM_MODEL_DIR` (default `~/.oflm/models/Qwen3.6-35B-A3B-NPU2`).

`make_decode.py` writes into `out/` (small per-layer blobs, the reference, the
`.cfg`) and `out/pools/` (the 512 MB-per-layer weight pools and the 542 MB
lm_head pool — budget 512 MB × layers of disk AND of RAM, since every pool is a
resident device buffer for the whole run). `--reuse-pools` keeps them across
runs; `--cfg-only` rewrites just the program.

### Decode acceptance

CPU Q4 reference preparation gathers values and coefficients into contiguous
chunk-major arrays. The indices, FP32 multiplication/addition and output values
are unchanged; this avoids tensor-wide strides during dequantization. The
scalar-layout tests check bit identity, including strided input chunks. To
compare the retained legacy implementation against a real model tensor:

```bash
OPENBLAS_NUM_THREADS=1 python utilities/benchmark-q4-decode.py --model-dir MODEL_DIR
```

The utility fails on any differing output byte and reports dequantization time
only. Its timings do not measure NPU inference or end-to-end model throughput.

Pass `make_decode.py --kernel-dir DIR` to target a retained runtime export
instead of mutable design build directories. The export must have the same
manifest as the current model/recipe except for its source build key; all
required binaries must match `toolchain.json`. The generated harness config
uses that export's paths, and reference metadata retains its original build
key. This validates an existing compatible export, not a fresh source build.
Use the same option for `--cfg-only`; existing reference provenance is checked
and cannot be relabelled as a different build.

After completing the CPU reference, `make_decode.py` also writes
`decode_reference.json`. It records the manifest-derived hidden/logit sizes,
layer and token counts, context capacity, spec/build identifiers, seed token, routing mode and
SHA256 hashes of every reference residual and logit file. This is fixture
provenance; it does not hash all source weights or compiled kernels.

`compare_decode.py` checks **all** declared tokens and layers by default.
If supplied, `--tokens` must match the recorded count. Missing files, unequal
sizes (including partial float bytes), non-finite values or changed reference
hashes fail the comparison. Logits must have correlation > 0.9999 and equal
argmax; each layer residual must have
`max(abs(actual - reference)) / max(abs(reference)) < 0.005`.
A zero reference residual requires an exactly zero result. Top-5 and residual
correlation remain explicitly diagnostic, not additional acceptance gates.

To also check a standalone CLI run made with `--dump-logits PREFIX`, pass
`--runtime-prefix PREFIX` to the comparator. It requires `PREFIX_t0.bin`,
`PREFIX_t1.bin`, etc. for every token declared in the fixture, validates their
size and finite values, and requires byte identity with the harness logits
(including the sign of zero). CPU-reference gates still apply. Extra runtime
steps beyond the fixture's declared count are not covered by this comparison.
The prefix is a path relative to the working directory, or an absolute path:

```bash
python open_kernels/model/compare_decode.py --out path/to/fixture \
  --runtime-prefix path/to/fixture/engine
```

Fixtures without `decode_reference.json` must be regenerated with
`make_decode.py` **without `--cfg-only`**. A cfg-only rewrite checks that its
dimensions, context capacity, seed, routing mode, spec/build identifiers and pinned references
match the existing fixture; it cannot create or update reference hashes.
Interrupted reference generation removes the old metadata so it cannot be
mistaken for a completed run. These checks validate captured residuals/logits,
not recurrent state, source tensor conversion or serving behavior.

Re-running `make_decode.py` after a run adopts the NPU router's expert choice
wherever it differed from the reference's own, and says so. That is deliberate:
the 8th of eight routed experts is often a near-tie, and a legitimate tie-break
difference would otherwise show up as a large numerical error. `--strict-routing`
turns the adoption off.
The fixture records this mode as `device-assisted`; other families and
`--strict-routing` use `independent`. The comparator prints the mode so a
conditional MoE comparison is not presented as an independent routing check.

## What runs where

Per layer, two dispatches on one xclbin context:

| | linear-attention layer (`lx`) | full-attention layer (`ax`) |
|---|---|---|
| dispatch 0 | norm → qkv/z → glue → DeltaNet → post → out → norm(+residual) → router | norm → q/gate/k/v → attention → o → norm(+residual) → router |
| driver | `moeroute2` — rewrite the expert fills' DDR offsets to the router's top-8 | same |
| dispatch 1 | the MoE block (8 routed experts + shared expert + combine) | same |

Then, once per token, the final norm and the q8 lm_head. The attention layers
share one instruction stream; `attnpos` rewrites its three position-dependent
words per token (KV window length, new-row offset, RoPE record offset).

A layer must be ONE xclbin context: a context switch restarts the AIE cores, so
the array state a fused layer carries between its two dispatches would be lost.

## Provenance

`replica.py` is vendored from phlegm's `tools/kernel-interp/` (`decode_step.py`,
`full_forward.py`), where every math element was verified against buffers
captured from OFLM's own engine. The pool laws came the same way
(`build_pools.py`, checked byte-for-byte for both layer types of this model);
they now live as ops of the recipe's packing plan, and the original
hand-written packer is frozen as the oracle in
`specs/open-engine/tests/legacy_pools.py`, which `test_pack_plan.py` holds the
plan interpreter to.

The kernels themselves are `../designs/`, and their provenance is
`../PROVENANCE.md`.

## Results (2026-09-04, Strix, Windows + XRT)

Stock `Qwen3.6-35B-A3B-NPU2`, decode from position 0 with zeroed state, scored
against `replica.py` in fp64 on the same weights.

| run | logits corr | argmax | top-5 | layer residuals |
|---|---|---|---|---|
| 3 linear layers, 1 token | 0.999998 | same | identical | 1.000000 |
| 8 layers (attention at 3, 7), 1 token | 0.999998 | same | identical | 1.000000 |
| 8 layers, 3 tokens | 0.999998 / 0.999996 / 0.999991 | same at each position | identical | ≥ 0.999995 |
| **40 layers — the whole model** | **0.999997** | **same (846)** | **identical** | **≥ 0.99992** |

Expert routing happens on device. All 24 of 24 (token, layer) selections in the
3-token run matched the reference; 38 of 40 in the whole-model run, the two
exceptions being 8th-slot ties with probability gaps of 7.5e-5 and 8.4e-5.

Whole model, 82 dispatches, 21 GB of resident pools: **323 ms/token (3.1 tok/s)**
— `lx0` 5.2 ms × 30, `lx1` 2.0 × 30, `ax0` 3.8 × 10, `ax1` 1.7 × 10, lm_head
31 ms, and 4 ms total for the 41 instruction patches. The box was holding 21 GB
of 47.6; phlegm's 30-layer runs on the same hardware imply ~220–275 ms at this
layer count, so treat this as an upper bound rather than the kernels' number.

### Two-piece Qwen3.5 FFN down validation

Use completed decode captures and the same verified kernel export:

```bash
OPENBLAS_NUM_THREADS=1 python open_kernels/model/ffn_partials.py \
  --model-dir MODEL_DIR --kernel-dir KERNEL_DIR --out FIXTURE_DIR --layers 0,3
```

The checker loads one layer's down matrix at a time and covers every fixture
position for each requested layer. It computes both K partitions in FP64 from
the captured NPU `h`, then checks each partial, their sum and the closing
residual with normalized maximum error strictly below 0.005. Zero references
require exact zeros. Missing, truncated or non-finite captures fail; the fixture
must identify the selected hash-verified export. JSON output records scope,
metrics and hashes of consumed captures; any failed comparison returns nonzero.

This isolates the down projection and residual closure. Because the input `h`
and pre-FFN residual come from the NPU, passing does not validate up/gate/norm,
attention heads or recurrent state. Captures are not regenerated by this tool;
rechecking old captures after a source update is not a new hardware run.

`--diagnose-bf16` additionally projects a BF16-rounded copy of `h` to isolate
input-rounding error. Those diagnostic metrics never replace the original
FP32-input comparison or turn its failure into success.

The split dense kernel's FP32 down input now has a direct INT16 table-preparation
entry. To reproduce the failing case without loading the whole model, use
`utilities/qwen35_layer_replay.py`; `utilities/gemv-fp32-prep/sweep.py` runs
layers serially with experimental lx/ax builds. Both retain the independent
FP64 comparison from captured FP32 `h`. Build and replay commands, including
mandatory regeneration of kernel translation units, are documented in
[the FP32 down replay skill](../../.agents/skills/open-qwen35-fp32-down-replay/SKILL.md).

For full-model propagation after this correction, build a separate complete
export, regenerate the CPU fixture with `--kernel-dir`, and score the new
propagated captures at all layers. The
[corrected production decode skill](../../.agents/skills/open-qwen38-fp32-production/SKILL.md)
documents export provenance, memory requirements, runtime identity and all-layer
partial checks. Isolated replay alone does not establish full-model propagation.

### Qwen3.5 norm and combined up/gate validation

```bash
OPENBLAS_NUM_THREADS=1 python open_kernels/model/ffn_front.py \
  --model-dir MODEL_DIR --kernel-dir KERNEL_DIR --out FIXTURE_DIR --layers 0,3
```

This reads the captured FP32 pre-FFN residual and `h`, and decodes `xm` using
the manifest's norm dtype (BF16 by default, FP32 for the split dense route). Its
primary FP64 reference recomputes RMSNorm from the residual and stored effective
norm weights, then computes `SiLU(Wgate @ norm) * (Wup @ norm)` without a BF16
rounding step. Norm and combined activation each require maxrel <0.005.
A separate comparison conditions the combined activation on captured `xm` to
help distinguish norm/input rounding from subsequent projection/SiLU error;
its success cannot override the primary gate. BF16-rounded norm/activation
comparisons and the predicted rounding effect are diagnostic only. Each matrix pair is released before loading the next layer.

The kernels do not retain separate up/gate outputs, so this does not independently
validate those two projections: compensating errors can hide in their product.
It also does not establish attention/state accuracy or full-model acceptance.
The checker requires compatible export/fixture identities, exact capture sizes,
finite inputs, and every position for the selected distinct covered layers.
JSON includes scope, metrics and capture hashes; failures return nonzero.

An isolated hardware experiment at `utilities/ffn-front-replay/` runs FP32
RMSNorm and captures up, gate and h separately. Its BF16-preparation control
reproduces the layer-11 error; direct FP32 preparation passes the local gates.
See the [FP32 front replay skill](../../.agents/skills/open-qwen35-fp32-front-replay/SKILL.md)
for build, pinned-input replay and scoring commands. This experiment does not
modify production layer layouts or establish corrected full-model propagation.

For the integrated split dense route, `manifest.layout.ffn_norm_dtype` is
`float32`: norm scratch grows from three to five 4-KiB elements at hidden 5120.
The front checker selects the capture dtype from this field; older compatible
recipes without it use BF16. The new recipe deliberately refuses old BF16
exports/act buffers. Build a complete new export and regenerate the fixture;
model weights and their packing are unchanged. Reproduction and ABI details are
in the [FP32 norm integration skill](../../.agents/skills/open-qwen38-fp32-norm-integration/SKILL.md).


### Per-head gated full-attention validation

```bash
OPENBLAS_NUM_THREADS=1 python open_kernels/model/attention_heads.py \
  --model-dir MODEL_DIR --kernel-dir KERNEL_DIR --out FIXTURE_DIR
```

The default covers every full-attention layer and every fixture position from
zero. `--layers 3,7` restricts coverage explicitly. The checker reads retained
FP32 Q/K/V/gate and manifest-declared BF16 or FP32 gated outputs, loads effective Q/K norm weights, then
recomputes RMSNorm, partial RoPE, causal grouped-query attention and sigmoid
in FP64. Reference KV history is rebuilt from the captured projections without
BF16 rounding. No full-model residency or new NPU dispatch is needed.

Every head must satisfy **absolute max error <0.02 and cosine >0.9999**.
Normalized max error, ideal BF16 output-rounding error and error against the
rounded reference are diagnostics; none can override a primary failure.
Exact zero pairs pass; a zero/nonzero mismatch fails. Missing, malformed or
non-finite captures, incompatible exports and invalid layer coverage fail.
JSON records every head, capture hashes and hashes of the effective norm
weights (little-endian float64). Numerical failure returns exit code 1.

This checks the attention core conditioned on captured projections, including
its accumulated KV effects on the gated output. It does not independently
validate Q/K/V projection weights, exact KV-cache bytes, output projection,
DeltaNet recurrent state or final LM head. Those require separate evidence.


For low-memory hardware localization use `utilities/attention-replay/`. It
reuses the production split attention functions, carries hardware KV across
positions and captures scaled Q, K/V, softmax accumulators and sums. The BF16
control reproduces the retained production outputs bytewise. Its FP32-output
variant isolates the final conversion; it is not a production precision fix.
Build, replay, pinned-input scoring and the remaining KV error are documented
in the [attention replay skill](../../.agents/skills/open-qwen38-attention-replay/SKILL.md).

`utilities/attention-kv-replay/` adds an isolated `[K_hi,K_lo,V_hi,V_lo]`
BF16-pair cache with FP32 output. Its widened BF16 control reproduces the earlier
FP32 replay exactly; the paired variant passes all 1152 retained head checks
(worst absolute error 0.00092833). This result covers captured projections at
three positions, not full-model propagation. The experimental cache row doubles
to 8192 bytes. The standalone probes require their matching earlier BF16
production ABI and reject the integrated paired ABI.
Build, provenance checks and L1/DMA constraints are documented in the
[paired-KV replay skill](../../.agents/skills/open-qwen38-paired-kv-replay/SKILL.md).

The Qwen3.8 production geometry now integrates this correction: paired KV rows,
FP32 gated output, and direct FP32 preparation for `o_proj`. Its manifest declares
`attn_kv_dtype: bfloat16_pair` and `attn_output_dtype: float32`. Rebuild the complete
export and regenerate fixtures before checking this wider layout. Other shapes
retain their existing ABI; model weights do not need repacking.

Use `export_qwen36_kernels.py --build-root DIR` to keep compiler work separate
from retained builds. `utilities/audit-kernel-export.py` checks current source
identity, build/export binary hashes and linked program budgets. The
[integration skill](../../.agents/skills/open-qwen38-paired-attention-integration/SKILL.md)
contains build and full-depth validation commands. Block-prefill uses host
attention for paired KV, with an explicit runtime diagnostic; its independent
accuracy gate remains separate from sequential decode.

For a limited-memory full-depth hardware check, run
`utilities/stream-decode-weights.py --cfg FIXTURE_DIR/run_decode.cfg --out FIXTURE_DIR/run_streamed.cfg`
and pass the new cfg to the harness. It reloads a shared layer-weight BO before
each dispatch and retains all states/capture ordering. The original fixture and
CPU reference are unchanged. This checks sequential propagation with streamed
weights; resident runtime identity/reset and performance require separate runs.

### Full-attention projections and sublayer

```bash
OPENBLAS_NUM_THREADS=1 python open_kernels/model/attention_projections.py \
  --model-dir MODEL_DIR --kernel-dir KERNEL_DIR --out FIXTURE_DIR
```

This checker covers every full-attention layer and fixture position by default;
`--layers 3,7` explicitly restricts coverage. It requires the split norm route,
which retains the entry `AA_XN` capture. The unsplit closing norm overwrites that
region and is rejected. The fixture must match the hash-verified export.

FP64 RMSNorm and Q/gate/K/V projections use the captured incoming residual and
effective model weights, without substituting rounded norm values. Captured-norm
and ideal-BF16-norm projection comparisons are separate diagnostics. Output
projection and residual addition are checked against captured gated attention.
A separate sublayer gate recomputes unrounded projections, causal KV history,
Q/K norm, RoPE, attention, gating and output projection from all incoming
residual captures. This catches errors hidden by conditioning each operation
on its device-produced input.

Every primary normalized maximum error must remain below 0.005; zero references
require exact zero. Missing/non-finite captures and invalid coverage fail. JSON
records every position, input/capture hashes and effective-weight hashes;
numerical failures return exit 1. This does not validate independent whole-model
state, source-tensor conversion, linear attention, or LM-head accuracy. Passing
local gates cannot override a failed full-chain residual gate. No new NPU run
or full-model residency is required.

The historical `utilities/attention-entry-replay/` (checkout `dfefe4d`) tests FP32 entry norm and direct FP32
Q/gate/K/V preparation in a private full-ax design. Its appended scratch region
is explicitly experimental; no production manifest is rewritten. It reuses one
weight pool and retains K/V history across forced input positions. The BF16
control must reproduce production activation/residual bytes and oracle results.
Use the [FP32 entry replay skill](../../.agents/skills/open-qwen38-fp32-attention-entry/SKILL.md)
for build, resource audit, prepare/run/score and controlled comparison commands.
The experiment shares the production checker's unrounded `score_layer` oracle.
It refuses the current integrated ABI; reproduce it with its matching checkout.
Passing it requires subsequent ABI integration and a new full-depth run before
making claims about whole-model residual or runtime accuracy.

Production full attention now declares `attn_entry_dtype: float32` for the
validated paired/split geometry. Entry RMSNorm feeds Q/gate/K/V without BF16
rounding; full-attention scratch grows by 8192 bytes. `attention_projections.py`
reads that dtype only after manifest/fixture validation. Rebuild the complete
export and regenerate fixtures; weights need no repacking. Reproduction and
acceptance scope are in the [FP32 entry integration skill](../../.agents/skills/open-qwen38-fp32-entry-integration/SKILL.md).

### Final norm and Q8 LM-head validation

`make_decode.py --capture-boundaries` additionally dumps `y_norm[_tN].bin`
(BF16 final RMSNorm output) and `y_stateL[_tN].bin` (each complete linear state)
after every position. It supports the declared BF16 norm/Q8-head tail and
leaves the dispatch order and default capture set unchanged. Use a separate
fixture directory to retain earlier evidence. Compatible copied reference/input
files can be bound with `--cfg-only`; reference hashes are still checked.
Streaming weights preserves these captures and every layer's state.

```bash
OPENBLAS_NUM_THREADS=1 python open_kernels/model/lm_head.py \
  --model-dir Models/qwen38-27b/Qwen3.8-27B-NPU2 \
  --kernel-dir Models/qwen38-27b/production-fp32-entry/kernels \
  --out Models/qwen38-27b/head-state-validation/decode
```

The checker computes final RMSNorm in FP64 from the captured last-layer residual
and streams native Q8 weights in 256-row blocks. It compares both composed
norm-to-logits and projection from captured norm, across the entire padded
vocabulary. Norm and both projection gates require normalized maximum error
<0.005. Projection gates also require correlation >0.9999 and equal argmax;
identical constant outputs are accepted without an undefined correlation.
Ideal-BF16-norm results are diagnostic only and cannot override primary failure.

All fixture positions, exact capture sizes, finite values, manifest identity
and complete Q8 row coverage are required. JSON pins nine captures for a
three-position run and the effective norm/head weight hashes. The default
projection slab is 256 × hidden FP64 values; no full head matrix or resident
model is needed. This is an independent tail check conditioned on captured
residuals. It does not validate upstream recurrent state. The new state dumps
supply data for that separate multi-token check; capturing them is not a PASS.

### Multi-token DeltaNet recurrence validation

```bash
OPENBLAS_NUM_THREADS=1 python open_kernels/model/deltanet_state.py \
  --model-dir Models/qwen38-27b/Qwen3.8-27B-NPU2 \
  --kernel-dir Models/qwen38-27b/production-fp32-entry/kernels \
  --out Models/qwen38-27b/head-state-validation/decode
```

This checker uses retained `--capture-boundaries` snapshots. It carries its own
FP64 `[head, key, value]` state from zero across every recorded position, taking
Q/K/V, decay and beta from the captured DeltaNet records. Captured previous
states never feed the reference. It checks both updated state and recurrent
output, all padded state rows, and the bit-exact manifest-declared BF16/FP32 conv-history shift from
captured QKV projections. Every declared linear layer and position is required,
with at least two positions, exact buffer sizes, finite values, compatible
manifest/build identity and a complete zero-state input. JSON records hashes
of all inputs and per-head metrics. No weights or NPU execution are needed.

Whole-state and whole-output gates retain normalized maximum error <1e-4 and
cosine >0.9999999. Per-head scores use the same bounds but remain separately
reported diagnostics, consistent with the standalone DeltaNet checker. A
whole-tensor PASS does not imply every head passes; both statuses must be
reported. This isolates recurrence and state storage conditioned on captured
records, not the norm/projection/conv/alpha/beta computation producing them.
Residual-to-state accuracy and complete model acceptance remain separate gates.

### DeltaNet input preparation and composed state

```bash
OPENBLAS_NUM_THREADS=1 python open_kernels/model/deltanet_inputs.py \
  --model-dir Models/qwen38-27b/Qwen3.8-27B-NPU2 \
  --kernel-dir Models/qwen38-27b/production-fp32-entry/kernels \
  --out Models/qwen38-27b/head-state-validation/decode
```

This checker reads one linear layer's effective QKV, norm, conv, alpha/beta,
decay and bias weights at a time. From captured incoming residuals it computes
FP64 RMSNorm, projections, causal depthwise convolution, SiLU, Q/K L2 norm,
decay/beta and recurrent state/output. Primary conv history and state start at
zero and carry independently without rounding or captured-state feedback.
Each record field has its own comparison, so large V values cannot hide an
incorrect decay or beta. Primary norm/projection/composed gates use the existing
0.005 normalized maximum error bound.

Separately, captured-norm QKV projection and captured-input glue checks isolate
preparation arithmetic. Glue retains normalized maximum error <1e-4 and cosine
>0.99999; the captured-record recurrence checker and its per-head diagnostics
are included unchanged. All declared linear layers/positions are required,
with finite, exactly sized captures, a zero-state input and verified export
identity. JSON pins incoming residuals, activation/state buffers, and effective
FP64 weights. These checkers do not modify runtime arithmetic or kernels.

Three CPU rounding experiments round entry norm, conv history, or both to BF16.
Their predicted states/outputs are compared both to captures and to the primary
reference. These are diagnostic estimates; they never replace primary gates
or demonstrate a hardware fix. A numerical failure produces complete JSON and
exit status 1. Passing full-depth residuals or conditioned recurrence does not
override a failed composed-state comparison.

### Isolated FP32 linear-entry replay

The historical experiment at checkout `46c845a` uses
`utilities/linear-entry-replay/` to compare BF16 entry with a
private full-lx FP32 norm → QKV/Z/alpha/beta candidate. It reuses the independent
DeltaNet preparation/composed-state checker above, forcing retained incoming
residuals while carrying conv/recurrent history across positions. The BF16
control must reproduce all activation, residual and state captures bytewise
and match the retained oracle results. Input, weight, binary and generated
source hashes pin both variants. Generated kernel entries come directly from
the recipe in a private directory, so stale global translation units cannot
silently change the experiment.

The FP32 candidate appends 20 KiB of activation scratch; model weights, the
production export and state layout are unchanged. Conv history remains BF16.
Preparation, serial hardware execution, resource auditing and comparison
commands are documented in
[the linear-entry skill](../../.agents/skills/open-qwen38-fp32-linear-entry/SKILL.md).
Only one layer's weight pool is needed at a time. Numerical failures retain
complete JSON and return exit status 1; diagnostic rounding never replaces the
unrounded primary reference. This experiment does not establish corrected
full-depth propagation or promote the candidate into production.

### Isolated FP32 conv-history replay

At checkout `46c845a`, `utilities/conv-history-replay/` keeps FP32 entry in both variants and compares
BF16 versus FP32 conv history. The candidate processes each conv tile in two
512-channel halves, retaining the existing FIFO depths and weights. Its state
buffer grows by 61440 bytes; its recurrent-state offset grows by the same amount.
The production export is unchanged. See the
[conv-history skill](../../.agents/skills/open-qwen38-fp32-conv-history/SKILL.md)
for build, resource audit, causal replay and comparison commands.

Shared `read_boundaries`, `compare_sequence` and `score_sequence` APIs accept
an explicit `conv_dtype='fp32'` for this private ABI. Their default remains BF16;
misdeclared dtype, malformed buffers and non-finite history fail. Both storage
formats require bit-exact history shifts, including signed zero. The unrounded
primary reference and all numerical bounds are unchanged. Replaying the control
must reproduce the previous FP32-entry captures and oracle rows before candidate
results can be accepted; full-depth propagation and head diagnostics remain
separate validation requirements.


### Production FP32 DeltaNet entry and history

The validated wide dense geometry (hidden 5120, 16 key heads, 48 value heads,
head dimension 128, four conv taps, split norm/down) now preserves FP32 entry
norm through QKV/Z/alpha/beta preparation and stores conv history in FP32.
Other geometries retain their existing storage. The manifest explicitly declares
`linear_entry_dtype` and `linear_conv_dtype`; the shared checkers select storage
from these fields while keeping the unrounded FP64 primary reference.

Linear activation scratch is 393216 bytes. State is 3317760 bytes, with
recurrent state beginning at byte 122880. Native host prefill follows the same
history dtype, including short-block carry, and rejects inconsistent manifest
offsets and sizes. Rebuild the full kernel export and regenerate decode fixtures
for this ABI; model weights and pool packing do not change. The historical
isolated replay builders reject the integrated recipe.

See the [production integration skill](../../.agents/skills/open-qwen38-fp32-linear-integration/SKILL.md)
for build, resource audits and full-depth checks. Host scalar/vector arithmetic
uses explicit multiply-then-add compiler semantics so legacy and FP32 tests
remain comparable. Host unit tests do not establish full model block-prefill
parity; resident runtime/reset and serving acceptance require separate checks.


### DeltaNet head readout diagnostics

```bash
OPENBLAS_NUM_THREADS=1 python utilities/deltanet_readout.py \
  --model-dir Models/qwen38-27b/Qwen3.8-27B-NPU2 \
  --kernel-dir Models/qwen38-27b/production-fp32-linear/kernels \
  --out Models/qwen38-27b/production-fp32-linear/decode
```

This CPU-only tool carries the unchanged independent FP64 recurrence from zero,
then separately evaluates `S^T q / sqrt(128)` using the captured updated state.
It reports primary head state/output scores, observed versus exact captured-state
readout, and the effect of state error on that readout. The cancellation ratio
is `max(sum(abs(S*q))) / max(abs(sum(S*q)))` per head; an exactly zero readout
has a null ratio. It is a sensitivity diagnostic, not an acceptance threshold.

Three-term and four-term paired-BF16 estimates isolate operand representation
and the omitted low×low product. They use FP64 accumulation and do not emulate
AIE accumulation or establish that a replacement kernel passes. No estimate
replaces the primary reference or its bounds. All layers/positions, sizes,
finite values and export identity are required; hashes pin the captures and
script. Exit status 1 preserves the complete JSON when any primary whole-tensor
or per-head gate fails, even when a diagnostic estimate fits the observations.


### Isolated DeltaNet recurrence precision replay

`utilities/deltanet-replay/` extracts all production DeltaNet records and runs
one head at a time with the original padded state/slice layout. The control uses
the unchanged production header and entry points; both state and output must
match bytewise across every position. The candidate uses three BF16 components
and all nine products, with its own causal state from zero. Neither path loads
model weights or substitutes a captured state into the primary FP64 reference.

Build, resource audit, preparation and serial hardware commands are in the
[recurrence replay skill](../../.agents/skills/open-qwen38-recurrence-replay/SKILL.md).
Copied sources, binaries, records, complete outputs and run identity are hashed;
configure removes stale dumps. Missing, incorrectly sized or non-finite captures
fail. The control's identity gate is reported separately from numerical head
precision. Candidate acceptance retains maxrel <1e-4 and cosine >0.9999999 for
every head and zero padding; numerical failure returns 1 with complete JSON.

The measured control reproduces all 6912 head steps bytewise. The isolated
triple-component candidate reduces head-output failures from 92 to 4, with all
state checks passing. The `factored` mode closes those remaining failures using
compensated sums and `decay*S_old^T*q + (k^T*q)*delta`, avoiding readout of the
rounded updated state. All 6912 state/output/padding checks pass at unchanged
bounds; worst output maxrel is 2.6966e-6. Stored states match the triple candidate
bytewise. The independent primary oracle and causal carry are unchanged.

The [factored readout skill](../../.agents/skills/open-qwen38-factored-readout/SKILL.md)
records commands and evidence. Its standalone core uses 8720 bytes of text,
L1 buffer end 22016 and stack 6144.

The private full-lx builder, `utilities/deltanet-replay/build_full_layer.py`,
uses `--shared` to deduplicate identical arithmetic helpers at link time.
The direct build exceeds program memory (18592 bytes); sharing reduces it to
15840 bytes, leaving 544 bytes on each main core. All 11 cores fit, with L1
buffer end 62336 and unchanged 6144-byte stacks. Compiler dependency files
must prove that the private recurrence headers were selected: IRON copies TUs,
so their original source directory is insufficient for quoted includes.

`full_layer_replay.py` runs all linear layers with forced incoming residuals
and each variant's own causal state. The control must reproduce complete
production buffers bytewise. The candidate must retain identical recurrence
inputs and match the isolated factored state/output captures for every head;
independent residual and recurrence oracles retain their existing thresholds.
The [full-layer skill](../../.agents/skills/open-qwen38-factored-full-layer/SKILL.md)
records build and validation commands. The measured candidate passes all 144
full-layer residual/recurrence cases and 6912 head-state/output checks, with
worst residual error 0.0013026962 <0.005. All 13824 head captures match the
isolated candidate bytewise. These forced-input measurements are separate from
the integrated full-depth validation below.

### Production compensated DeltaNet recurrence

The validated wide dense Q4_1 recipe selects `dnx_factored.h` and shares its
arithmetic helpers across full-layer entry points. It computes the output from
the old state and rank-one update with compensated accumulation, while using
triple-BF16 products for the stored state. Other geometries and Q8/mixed-role
configurations retain the legacy recurrence. Weight packing, activation/state
buffers and scheduling are unchanged.

The manifest's `linear_recurrence: factored_compensated` distinguishes the new
arithmetic from an older export with the same buffer ABI. Rebuild the export
through the normal recipe and create fresh decode fixtures; existing model
weights need no repacking. The
[integration skill](../../.agents/skills/open-qwen38-factored-integration/SKILL.md)
documents build, dependency/resource audits and full-depth validation commands.
Private experiments retain their own recurrence-header injection; reproduce
historical control results with their recorded checkout and export.

The retained recurrence-only 41-set export passes 64-layer, three-position residual/logit checks:
192 residuals pass, with worst maxrel 0.0033520097 <0.005. All 6912 recurrence
head-state/output checks pass at unchanged maxrel <1e-4 and cosine >0.9999999;
worst state/output errors are 9.3486e-7 / 2.0365e-6. Attention, FFN and LM-head
boundary checks pass. This does **not** close numerical acceptance: the full
DeltaNet input checker passes 143/144 cases, with normalized K at L34/T0
measuring 0.0058359393 >0.005. Its failure status remains visible in that
historical export. The integrated consistent-sum correction below addresses it.

To separate norm and QKV projection effects on retained layer captures:

```bash
OPENBLAS_NUM_THREADS=1 python utilities/deltanet_projection.py \
  --model-dir MODEL_DIR --kernel-dir KERNEL_DIR --out FIXTURE_DIR --layer 34
```

This CPU diagnostic loads one layer's Q4_1 weights. Each variant carries its
own unrounded conv history from zero; matching a diagnostic never replaces the
primary record gate. Invalid/incomplete captures fail, and primary failures
return exit 1 with complete JSON and source/weight/capture hashes. Q8 projection
diagnostics are not implemented. At L34/T0 the norm-only K effect is 1.8758e-5
and QKV-only effect 0.0058319225. A controlled hardware replay gives identical
entry records with legacy and factored recurrence on the same forced inputs.
The isolated QKV experiment below tests the activation-table arithmetic.

### Isolated QKV affine-sum precision

`utilities/qkv-precision/` builds a private control and a consistent-sum Q4
projection. The original affine epilogue combines INT16-quantized nibble products
with sums of the original FP32 inputs. The candidate computes both terms from
the same quantized inputs, preserving INT16 values, scales, table ABI and weight
packing. A zero-effective-weight regression exposes the original cancellation
error. Production kernels and recipe selection are unchanged by this experiment.

The probe dumps full QKV and activation tables. Control projections must match
production bytewise; candidate projections and causal CPU record preparation
are scored against an independent unrounded FP64 reference at the unchanged
0.005 bound. At L34/T0, normalized K error decreases from 0.0058319225 to
0.0003864200 when both projections use the same CPU conv oracle. The retained
production NPU record has error 0.0058359393. This isolated result does not yet
close the production input gate or validate NPU conv/recurrence after the change.

The complete 48-layer × 3-position isolated sweep passes: all 144 control
projections match production bytewise, and all 144 candidate projections/CPU
records pass. Worst candidate K error is 0.0004477865; worst QKV error is
0.00004684383. INT16 values/scales remain byte-identical across the variants.

The [QKV precision skill](../../.agents/skills/open-qwen38-qkv-precision/SKILL.md)
documents private builds, resource audits, single-layer replay and the causal
sweep. Full-layer validation is described below; production integration needs
a fresh export and full-depth validation.

### Consistent Q4 sums in a complete linear layer

`utilities/qkv-full-layer/` applies the same private table correction to a
complete `lx`, retaining the production compensated recurrence and unchanged
streams, ABI, packing and stack. Actual compiler dependency files must resolve
the private table/GEMV headers and production factored recurrence headers.
All 11 cores fit: maximum text **15936 bytes**, **448 bytes** program headroom,
maximum L1 buffer end **62336**, stack **6144**.

Each layer receives retained production residuals, starts from zero state and
carries its own conv/recurrent history over three positions. The control must
reproduce complete buffers bytewise; candidate QKV must match the isolated
consistent-sum kernel bytewise. Independent checks cover records and composed
state/output, strict per-head recurrence, Z/post/output projections, FFN front,
both down partials and the whole-layer residual. The residual reference uses
the existing mixed FP64/FP32 replica; all thresholds remain unchanged.

Commands and measured results are in the
[full-layer runbook](../../.agents/skills/open-qwen38-consistent-full-layer/SKILL.md).
These forced-input layer checks use one layer's weights at a time. Production
selection and new full-depth propagation checks remain separate requirements.

The complete private replay passes **144/144 cases** and **6912 strict head
cases**. All **432 control buffers** are byte-exact to production; candidate
entry norms and QKV match their respective controls in all 144 cases. L34/T0
K error is **0.0003889343** (control **0.0058359393**); worst candidate K and
residual errors are **0.0004466512 / 0.0006191985**. FFN front and both down
partials pass all cases. The control's numerical input gate remains 143/144.


### Production consistent Q4 activation sums

The validated wide dense Q4_1 recipe now uses the same INT16-reconstructed
activation for both affine Q4 terms in direct FP32 preparation. This avoids
cancellation error when quantized dot products are combined with original-input
sums. Selection is restricted to the validated factored-recurrence and FP32
attention-entry geometry. Both full `lx` and `ax` use it, including their FFN;
other geometries and Q8/mixed-role recipes retain their existing arithmetic.

The manifest records `q4_activation_sum: quantized_fp32`. Buffer ABI, weight
packing, compensated recurrence and scheduling are unchanged. Rebuild through
the normal recipe; existing weights require no repacking. The compiled-source
and dependency audit is `utilities/audit-consistent-sum.py`. Build and validation
commands are in the [production runbook](../../.agents/skills/open-qwen38-consistent-integration/SKILL.md).

The fresh export contains 41 sets / 82 verified binary hashes. Maximum lx/ax
text is 15936 / 13760 bytes, L1 buffer ends 62336 / 59920, and stacks remain
6144. All 66 model/pool hashes and 195 independently regenerated CPU reference
files match retained provenance; CPU oracle arithmetic and thresholds are unchanged.

The fresh 64-layer × three-position NPU run passes all 192 residual checks
(worst 0.0030399006 <0.005). Logit correlations are
0.9999989597 / 0.9999982780 / 0.9999986060; argmax 8678 / 198 / 2 matches CPU.
DeltaNet input and composed state/output pass all 144 cases. Worst normalized
K error is 0.0009785607 at L34/T0; this is a new propagated-input measurement,
separate from the forced-input private replay above. All 6912 strict head
cases pass, with worst state/output errors 1.2624e-6 / 2.3722e-6. All seven
boundary checkers pass, including 1152 attention heads, 48 attention projection/
sublayer cases, three LM-head cases and 192 FFN front/down cases each. Worst
FFN front/partial0/partial1 errors are 0.0002516594 / 0.0000958907 / 0.0000693562.

The complete model also passes resident runtime validation: all three logits
are byte-identical to the harness after a repeated request, and full-buffer
reset reports zero differing replays out of three. This validates the tested
three-position decode and lifecycle. The separate all-layer FFN projection
checks below supplement it. Remaining source/packing checks, block-prefill
parity, serving isolation and
resource/performance acceptance remain open. Paired-KV block attention still
uses host attention; these sequential NPU checks do not establish fully NPU prefill.


### Separate FFN up/gate across all layers

`utilities/ffn-front-replay/front_build.py` builds the isolated probe with
production consistent Q4 preparation; `sweep.py` replays all captured pre-FFN
residuals using one pool BO. It requires both byte-exact production norm/H and
independent FP64 up/gate/activation gates, so a matching product cannot hide
an incorrect individual projection. Original model/pool/capture provenance,
compiled sources, header dependencies, configuration and all outputs are pinned.
The input ABI is FP32 residual followed by BF16 norm weights; reference arrays
must be serialized explicitly because the checker promotes them to FP64.

All 64 layers × three positions pass: **192/192** independent cases, with
**192/192** norm and **192/192** activation arrays byte-exact to production.
Worst up/gate errors are **0.0000919118 / 0.0001016790**, norm/activation
**0.0000189398 / 0.0002516594**, all below the unchanged **0.005** bound.
The two probe cores use **6400 / 2944 bytes** of text, L1 ends **33024 / 53248**,
and **2048-byte** stacks. Logical NPU buffers total **0.217 GiB**; full-model
residency is unnecessary. The production export and its 82 binary hashes are
unchanged. This diagnostic xclbin is not an application kernel export.

The [FFN projection runbook](../../.agents/skills/open-qwen38-ffn-projection-sweep/SKILL.md)
records complete commands and evidence. These isolated measurements supplement
the full-depth decode checks; source conversion, prefill, serving and performance
acceptance remain separate requirements.

### FP32-history and paired-KV host prefill boundaries

`replica_block.py` supports FP32 convolution history (`conv_f32`) and four-plane
paired-BF16 KV (`kv_pair`), preserving the legacy BF16 defaults.
[The host boundary checker](../../utilities/prefill-boundaries/README.md) compiles
the production CPU stages and compares independent FP64 results with both
one-token and padded 256-token calls. Prompts of **1, 2, 255, 256 and 257 tokens**,
each followed by one continuation token, pass at the actual 27B dimensions.
Outputs, state and cache match sequential host calls bytewise; history, cache
prefix/tail and padding gates also pass. Worst normalized error is **6.1067e-6**
against a **1e-3** limit. This conditions the stages on synthetic projections;
NPU GEMMs, complete model prefill and serving require separate validation.

### Real-weight prefill GEMM diagnostics

[The prefill projection checker](../../utilities/prefill-projections/README.md)
exercises all five production GEMM shapes with layer 0/3 weights and three
captured decode inputs in two 256-column layouts. It checks linear Q/K/V/Z,
attention Q/K/V/gate, output, FFN up/gate and down separately, against FP64
products with both original and BF16-rounded inputs. Complete output sizes,
finite data, zero inactive columns, slot identity and original input provenance
are required. This is a conditioned projection diagnostic; complete causal
prefill and hybrid/sequential route parity remain separate acceptance gates.

### Causal prompt slice and prefill-to-decode handoff

[The four-layer prefill checker](../../utilities/prefill-slice/README.md) tokenizes
a real Qwen chat prompt, carries independent CPU conv/recurrent/KV state through
all prompt positions, and predicts one continuation step. It compares all
prompt logits and the continuation with sequential NPU and both hybrid
schedules; hybrid layer-major/block-major logits must be byte-identical.
The continuation's final-layer residual is checked from both split-down
partials. A bounded-memory Q8 head reference shares weight unpacking across
prompt positions. This slice does not establish full-depth or long-context
prefill acceptance.

The 20-token prompt plus continuation passes on all three routes: **63/63**
logit comparisons, worst correlation **0.9999989974**, and exact hybrid schedule
identity. Continuation final-layer residual maxrel is **0.0005230136** or better
against **0.005**.

The same checker now prepares shared causal prefixes of **255/256/257 tokens**,
with independent state snapshots and a separate CPU-selected continuation for
each length. All **9 native runs** pass: **2313/2313** logit comparisons,
minimum correlation **0.9999975248**, maximum continuation final-layer residual
error **0.0002930006 <0.005**. All **774** hybrid logit files match bytewise
between schedules. This covers the 256-token block boundary on four layers;
full-depth prefill, all-layer residual coverage and long-context acceptance
remain open.

The checker also supports `prepare --layers 64`, with one layer's dequantized
weights cached at a time and independent causal state retained for continuation.
The runner derives its depth and final residual capture (layer 63) from the
reference manifest and requires references for every selected layer. On Linux,
full resident runs require at least 35 GiB available RAM; insufficient memory
leaves a failed report without starting a native process. Preparing a full-depth
CPU fixture is not evidence of NPU prefill acceptance. See the checker's README
for preparation and resume commands.

Full-depth testing now distinguishes the routes: the 20-token prompt plus
continuation passes sequential NPU prefill (**21/21** logits, minimum correlation
**0.9999943778**). Both hybrid schedules fail position **17** at correlation
**0.9998908204**, below **0.9999**, despite matching argmax, continuation and
byte-identical schedules. A retained-reference depth scan passes at
4/8/16/32/48 layers and fails at 60 (positions 16 and 18). Hybrid precision
remains open; see the checker's README for the failed gates and reproduction
commands. The scan does not identify a specific faulty layer or operation.
