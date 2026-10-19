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
**4/4**. The full-depth NPU comparisons remain a separate acceptance step.

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
