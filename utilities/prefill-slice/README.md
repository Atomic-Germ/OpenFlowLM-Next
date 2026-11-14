# Causal prefill validation

Run a real, locally tokenized chat prompt through layers 0–3 (three DeltaNet
layers and one full-attention layer), then perform one NPU decode step after
prefill. This checks the complete four-layer computation and its state handoff;
it is not full-depth model acceptance or a useful-text evaluation of the slice.
`prepare --layers 64` selects the complete model; the default remains four layers.
The native run takes its depth from the reference manifest, including the final
residual capture at layer 3 or 63. Full-depth acceptance requires actual native
results in addition to preparing a CPU reference.

The independent CPU reference uses `replica_qwen35.layer_decode` with its
existing FP64/FP32 layer arithmetic and effective packed-weight quantization.
The Q8 LM-head reference unpacks 256 vocabulary rows at a time and multiplies
all normalized prompt rows in FP64, then stores FP32 logits. A small real-format
Q8 fixture checks this batched path against the existing single-vector reader. It
carries its own conv, recurrent and KV state from position zero. No native
capture or generated token is fed into the reference. Its own last-prompt
argmax selects the continuation input. Native output tokens must match both
CPU predictions before the continuation comparison is accepted.

```bash
# NumPy and tokenizers must be installed in this environment.
OPENBLAS_NUM_THREADS=1 ironvenv/bin/python utilities/prefill-slice/check.py prepare \
  --out Models/qwen38-27b/prefill-slice/short-batched \
  --model Models/qwen38-27b/Qwen3.8-27B-NPU2 \
  --kernels Models/qwen38-27b/production-consistent/kernels \
  --original Models/qwen38-27b/production-consistent/input-sha256.json \
  --message 'What is 2 + 2?'
cmake --build /tmp/qwen38-main-merge-next/build --target open_qwen36_cli -j 2
OPENBLAS_NUM_THREADS=1 LD_LIBRARY_PATH=/opt/xilinx/xrt/lib \
  ironvenv/bin/python utilities/prefill-slice/check.py run \
  --out Models/qwen38-27b/prefill-slice/short-batched \
  --cli /tmp/qwen38-main-merge-next/build/open_qwen36_cli
```

`prepare` uses the model's tokenizer and the existing Qwen closed-thinking chat
format from `replica_prompt.prompt_ids`. It pins the exact message/IDs, original
model hash, tokenizer, config, compatible export identity, reference sources
and every reference file. Weights are dequantized one layer at a time and the
same layer cache serves all prompt positions. The continuation reuses the CPU
state and reloads one layer's weights at a time.

`run` executes three native processes serially: sequential NPU prefill, hybrid
layer-major prefill, and hybrid block-major prefill. All continue with explicit
`--decode-route npu`. The hybrid route uses NPU GEMMs and CPU DeltaNet/attention;
paired-KV attention has no block NPU attention implementation. Ambient
`OFLM_OPEN_*` experiment flags are removed; layer-major is explicitly enabled,
stage-major disabled, and block attention remains on the host. No kernel is
rebuilt or changed by this checker.

Every prompt position and the continuation requires full, finite logits,
correlation **>0.9999** and the same argmax within the real vocabulary. The
`_t0` and last `_pN` aliases must be byte-identical. After the continuation,
the final layer's captured pre-FFN residual plus both down partials must match the
independent final residual with normalized maximum error **<0.005**. Other
layer residuals are retained by the CPU but not captured by this CLI check.
The two hybrid schedules must produce byte-identical logits for all positions.
These are the existing decode thresholds, without relaxation.

Each route retains its command, log, every logit dump and final layer act
buffer. The report records hashes, per-position scores and token identity.
Missing or stale input/reference data is refused; expected output dumps and
prior success are invalidated before a rerun. A run failure leaves a failed
report. Run the CPU preparation and resident NPU tests serially to avoid
competing for memory. Source conversion, all-layer residual coverage,
full-depth prefill and serving are separate gates.

## Full-depth preparation and resident run

```bash
OPENBLAS_NUM_THREADS=1 ironvenv/bin/python utilities/prefill-slice/check.py prepare \
  --out Models/qwen38-27b/prefill-full/short \
  --model Models/qwen38-27b/Qwen3.8-27B-NPU2 \
  --kernels Models/qwen38-27b/production-consistent/kernels \
  --original Models/qwen38-27b/production-consistent/input-sha256.json \
  --layers 64 --message 'What is 2 + 2?'
OPENBLAS_NUM_THREADS=1 LD_LIBRARY_PATH=/opt/xilinx/xrt/lib \
  ironvenv/bin/python utilities/prefill-slice/check.py run \
  --out Models/qwen38-27b/prefill-full/short \
  --cli /tmp/qwen38-main-merge-next/build/open_qwen36_cli
```

CPU preparation keeps one layer's dequantized weights at a time, retaining causal
state for the subsequent CPU-selected continuation. It saves both prompt and
continuation residual references for every selected layer. A 20-token, 64-layer
fixture contains **149 reference files**: 64 prompt residual arrays, 64
continuation residual vectors and 21 logit vectors. The runner requires the
complete reference-file inventory; relabelling a four-layer fixture as 64 layers
is rejected. The native CLI captures only the **last** continuation residual;
retaining all CPU references does not establish all-layer native residual coverage.

Full resident NPU runs need approximately **35–40 GiB available RAM** on the
validated 2048-context export. The Linux checker reads `/proc/meminfo` before
verification and again before each route, and refuses a 64-layer run below
35 GiB. A failed preflight invalidates previous success, leaves a failed report
and starts no native process. This is a minimum headroom check, not a memory
reservation. CPU preparation does not require full model residency. Run it
separately from resident native tests.

The full-depth CPU fixture for the message above has been prepared and audited:
**64 layers × 20 prompt positions + one continuation**, all **149 files** have
the expected sizes, finite values and recorded hashes. The prompt residuals for
layers 0–3 are byte-identical to the earlier four-layer fixture. CPU-selected
token IDs are `[17, 478]`; these two tokens are a continuation prefix, not a
complete answer. Host regression: **1419 passed, 47 skipped**, native CTest
**4/4**. Preparing this fixture alone is not NPU acceptance.

### Full-depth measurement and remaining precision failure

On 2026-10-19, all three native routes loaded **64 weight pools plus LM-head**
and completed the 20-token prompt followed by NPU decode. Sequential prefill
passes **21/21** logit comparisons, minimum correlation **0.9999943778**, and
final continuation residual maxrel **0.0003932455**.

Both hybrid schedules fail **one of 21 positions**: at prompt position **17**,
correlation is **0.9998908204**, below the unchanged **>0.9999** gate. Their
final continuation residual passes at **0.0018923571 <0.005**; all argmax values
and generated IDs `[17, 478]` match CPU. All **22** hybrid logit files are
byte-identical across schedules. This is a numerical acceptance failure even
though token IDs and the continuation agree. Full-depth hybrid prefill remains
unaccepted. Evidence: `Models/qwen38-27b/prefill-full/short/report.json`.

### Hybrid depth diagnostic

`depth.py` applies the same production tail reference to saved CPU prompt
residuals at selected depths. It then runs the same complete prompt on native
hybrid layer-major prefill with those depths. It never feeds native residuals
into the oracle. Every position retains the original correlation/argmax gates;
this diagnostic omits continuation and intermediate native residual captures.
The source fixture, residual files, model/export, checker sources, CLI, logs
and outputs are pinned by hashes. CPU preparation and native runs are serial.

```bash
OPENBLAS_NUM_THREADS=1 ironvenv/bin/python utilities/prefill-slice/depth.py prepare \
  --fixture Models/qwen38-27b/prefill-full/short \
  --out Models/qwen38-27b/prefill-full/depth-scan
OPENBLAS_NUM_THREADS=1 LD_LIBRARY_PATH=/opt/xilinx/xrt/lib \
  ironvenv/bin/python utilities/prefill-slice/depth.py run \
  --out Models/qwen38-27b/prefill-full/depth-scan \
  --cli /tmp/qwen38-main-merge-next/build/open_qwen36_cli
```

The 2026-10-19 scan produced:

| Depth | Minimum correlation | Failing positions |
|---:|---:|---|
| 4 | 0.9999989974 | none |
| 8 | 0.9999962133 | none |
| 16 | 0.9999970907 | none |
| 32 | 0.9999800818 | none |
| 48 | 0.9999196879 | none |
| 60 | 0.9998648409 | 16, 18 |

All argmax values match CPU. The scan passes through depth 48 and fails at 60;
it does not identify the first faulty operation or prove an error originates
in layers 48–59. `Core::gemm_run` converts FP32 inputs to BF16 via `host::tile_x`;
accumulated input rounding is a candidate for a controlled precision experiment,
not an established cause. No production kernel or arithmetic changed in this
diagnostic. Regression: **1425 passed, 47 skipped**; native CTest **4/4** from
the unchanged native sources. Both failed reports are retained; thresholds
remain unchanged.

## Crossing a 256-token block

`prepare --boundaries` prepares directories `255`, `256` and `257` beneath
`--out`. Supply a message that tokenizes to at least 257 tokens. These cases
are prefixes of the same tokenized chat, potentially ending inside the user
message; they are numerical state-carry checks, not complete chat requests.
The message and exact prefix IDs are retained in each reference manifest.

The CPU processes the longest prefix once per layer and deep-copies conv,
recurrent and KV state at each boundary. It calculates all prompt logits in
one batch, then selects each prefix's own greedy continuation and advances its
independent saved state. Layer weights are shared across the three continuation
cases before being released. Tests compare checkpointed state with separate
sequential runs and exercise in-place mutation and distinct continuation IDs.

For the retained boundary fixture, prepare with the same model, kernel and
original-provenance arguments as above, adding `--boundaries` and using this
message repeated **12 times** (including its trailing space):

```text
A language model reads a sequence of tokens and preserves information from earlier positions. Each layer updates recurrent state or an attention cache before passing its output to the next layer. Splitting the same sequence into blocks must preserve the causal order and the next token prediction.
```

Then run the cases serially:

```bash
for length in 255 256 257; do
  OPENBLAS_NUM_THREADS=1 LD_LIBRARY_PATH=/opt/xilinx/xrt/lib \
    ironvenv/bin/python utilities/prefill-slice/check.py run \
    --out "Models/qwen38-27b/prefill-slice/boundaries/$length" \
    --cli /tmp/qwen38-main-merge-next/build/open_qwen36_cli || break
done
```

Every case runs all three schedules and applies the same logit, token,
continuation residual and hybrid identity gates. Passing the 257-token case
requires a partial second prefill block followed by a decode step; four-layer
results still do not establish full-depth prefill or long-context acceptance.

On 2026-10-18 all **9 runs** passed: **2313/2313** prompt/continuation logit
comparisons, matching real-vocabulary argmax throughout. Minimum correlation
was **0.9999994801** for sequential NPU and **0.9999975248** for both hybrid
schedules. Continuation final-layer residual errors were:

| Prompt tokens | Sequential maxrel | Hybrid maxrel (both schedules) |
|---|---:|---:|
| 255 | 0.0002930006 | 0.0000525120 |
| 256 | 0.0000940882 | 0.0000306139 |
| 257 | 0.0002093509 | 0.0002889360 |

All **774** hybrid logit files, including last-prompt aliases, were byte-exact
between schedules. All cases predicted `[220, 220]` on this truncated slice;
the full distributions, not text quality, are the numerical acceptance target.
References, output hashes and logs are retained in
`Models/qwen38-27b/prefill-slice/boundaries/`. Regression: **1405 passed,
47 skipped**, native CTest **4/4**. Production kernels and native arithmetic
were unchanged; no new kernel or native library build was required.

## Short-prompt measurement

On 2026-10-18 the 20-token prompt above passed all three routes: **63/63**
position comparisons (20 prompt positions plus one continuation per route).
Minimum correlation was **0.9999995111** for sequential NPU and
**0.9999989974** for both hybrid schedules, with matching argmax throughout.
Final continuation residual maxrel was **0.0000364223** / **0.0005230136**
(sequential / hybrid), below **0.005**. All **22** hybrid logit files, including
the last-prompt alias, were byte-identical across schedules. The four-layer
slice's two predicted token IDs were `[220, 220]`; this is numerical validation,
not evidence that a truncated model gives useful answers.

The batched head was additionally compared against two retained real-model
single-vector references: worst maxrel **5.6245e-7**, same argmax, and unchanged
layer residuals. The reference uses FP64 head accumulation; it is numerically
close to, not byte-identical with, the older FP32 accumulation. The run used
`tokenizers 0.23.3`. Regression: **1397 passed, 47 skipped**, native CTest **4/4**.

## Paired-input follow-up

The private [paired BF16 input experiment](../prefill-paired-input/README.md)
reproduces all 66 original logits bytewise, then repeats each projection with
the BF16-rounded residual of its FP32 input. On the same 64-layer short prompt,
both hybrid schedules now pass all 21 position gates: minimum correlation
**0.9999488108**, final continuation residual **0.0011956505**. All 22 sequential
logit files remain exact and the hybrid schedules agree in 22/22 files.
Production engine sources and kernels are unchanged; this is a controlled
precision result, not a production default or full-context acceptance.

## Auditing full-depth boundary references

Use `prepare --layers 64 --boundaries` with the same repeated message as the
four-layer boundary fixture. Weights are streamed by layer; the three causal
state snapshots and CPU-selected continuation IDs remain independent.

```bash
OPENBLAS_NUM_THREADS=1 ironvenv/bin/python utilities/prefill-slice/boundaries.py \
  --out Models/qwen38-27b/prefill-full/boundaries \
  --slice Models/qwen38-27b/prefill-slice/boundaries
```

The audit requires all **1155** references (384/385/386 files), exact float32
sizes, finite data and matching hashes. It verifies model/source/export identity,
exact ID files and that each continuation was selected by that case's CPU logits.
It compares all **128** adjacent-length prompt residual prefixes and **511**
logit prefixes bytewise, then compares the **12** layer-0..3 prompt arrays with
the prior four-layer fixture. Continuation residuals are deliberately independent:
different prefix lengths and model depths can select different next tokens.

The older four-layer checker hash may differ because it predates full-depth
support; every other oracle/packing/recipe source hash must agree. A missing,
changed or inconsistent reference leaves `reference-audit.json` failed. A passing
reference audit explicitly records `native_acceptance: false`; it is not a
hardware result. Full-depth native runs use the ordinary `check.py run` command
for each length, serially, with the private candidate CLI when measuring paired
input precision.

After the three native runs, add `--cli` to audit their actual captures:

```bash
OPENBLAS_NUM_THREADS=1 ironvenv/bin/python utilities/prefill-slice/boundaries.py \
  --out Models/qwen38-27b/prefill-full/boundaries \
  --slice Models/qwen38-27b/prefill-slice/boundaries \
  --cli Models/qwen38-27b/prefill-paired-input/paired/build/open_qwen36_cli
```

This does not launch NPU work. It recomputes the per-case numerical gates using
the paired-input capture auditor, requires the same CLI hash for all cases,
and checks **1533** native common-prefix logit identities across lengths on all
three routes. Each case already checks both hybrid schedules against each
other. The result goes to `native-audit.json`, separate from reference-only
acceptance. Missing files, incomplete routes, stale hashes, a failed numerical
gate or any changed common-prefix logit cannot produce a passing native audit.
A passing audit covers final L63 continuation residuals; intermediate native
prefill residuals require separate captures.

On 2026-10-19 full-depth CPU preparation and the reference audit passed: all
**1155 files**, **128 residual prefixes**, **511 logit prefixes**, and **12**
first-four-layer arrays matched their required sizes, hashes and identities.
Each case independently selected continuation IDs **[248046, 198]**. These
are truncated chat prefixes used to exercise numerical state carry, not complete
chat requests or an answer-quality test. The same 41-set production-consistent
export is used. TDD: **11 new tests**, host suite **1447 passed, 47 skipped**.
Evidence is in `Models/qwen38-27b/prefill-full/boundaries/`; the reference audit
alone does not establish native acceptance.

### Full-depth boundary measurement — 2026-10-19: precision FAIL

All nine resident runs completed with the private paired-input CLI and explicit
NPU continuation. Acceptance remains **FAIL**; the short-prompt correction is
insufficient on these longer prefixes. No threshold was relaxed.

| Tokens | Sequential logits passing | Each hybrid route passing | Sequential continuation maxrel | Each hybrid continuation maxrel |
|---|---:|---:|---:|---:|
| 255 | 183/256 | 186/256 | 0.0026296296 | 0.0018321448 |
| 256 | 183/257 | 186/257 | 0.0015172531 | **0.0096012460 (FAIL)** |
| 257 | 184/258 | 187/258 | 0.0042892308 | 0.0019920430 |

Overall **1668/2313** position gates pass. On each length, the sequential route
first fails at position **80**, with minimum correlation **0.9351669042** at
position **220**. Both hybrid schedules first fail at **82**, with minimum
**0.9422035563** at **205**. Their argmax also differs from CPU at **181, 183,
234**. Sequential argmax matches at every position, and all three routes emit
the CPU-selected continuation IDs `[248046,198]`. These token matches do not
override the failed numerical gates.

The two hybrid schedules agree bytewise in **774/774** logit files, including
last-prompt aliases and continuation. All **1533/1533** common-prefix identities
across lengths pass on all three routes. `native-audit.json` returns failure
because numerical gates fail, not because captures are missing or inconsistent.
Failures before position 255 show this is
not solely a transition across the 256-token boundary. The 256-token hybrid
continuation additionally fails the final residual bound. This does not yet
identify the responsible layer or operation.

The retained CPU reference includes every layer's prompt residual, but this
native CLI captures only L63 after continuation. Next add native intermediate
residual captures and localize the first layer/position divergence, prioritizing
positions 80/82, 205/220 and the hybrid argmax mismatches. Verify any instrumented
control against these captured logits before using its internal boundaries.
All reports and **2331 native output files** are retained beneath
`Models/qwen38-27b/prefill-full/boundaries/`. No additional kernel/native build
was needed; the measured CLI hash remains
`f2b8882051782078bc3d5717ada5762c1ea220a79a9bb3b8a09aa2ff917be054`.
