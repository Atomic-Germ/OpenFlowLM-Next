# Causal prefill slice

Run a real, locally tokenized chat prompt through layers 0–3 (three DeltaNet
layers and one full-attention layer), then perform one NPU decode step after
prefill. This checks the complete four-layer computation and its state handoff;
it is not full-depth model acceptance or a useful-text evaluation of the slice.

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
layer 3's captured pre-FFN residual plus both down partials must match the
independent final residual with normalized maximum error **<0.005**. Other
layer residuals are retained by the CPU but not captured by this CLI check.
The two hybrid schedules must produce byte-identical logits for all positions.
These are the existing decode thresholds, without relaxation.

Each route retains its command, log, every logit dump and final layer act
buffer. The report records hashes, per-position scores and token identity.
Missing or stale input/reference data is refused; expected output dumps and
prior success are invalidated before a rerun. A run failure leaves a failed
report. Run the CPU preparation and resident NPU tests serially to avoid
competing for memory. Source conversion, long prompts across 256-token blocks,
all-layer residual coverage, full-depth prefill and serving are separate gates.

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
