# Real Qwen3.8-27B full-model bring-up

Status (2026-09-28): real weights are converted, all64 layers execute over
three autoregressive tokens and reset replay, but the numerical gate is **FAIL**:
three inherited bounds are exceeded. This is PR4/B7; full-model acceptance,
runtime support and catalogue promotion remain pending.

Follow-up: [full-depth precision experiments](qwen38-model-precision.md) add
compensated projection/FFN and opt-in attention/LN modes. Output correction alone
reduces the failures to two; the gate remains open. A reproducible first BF16
divergence is localized to layer1/channel2931, propagated from layer0 GEMV error.

Follow-up (2026-10-01): [segmented down precision](qwen38-down-segment-precision.md)
closes that channel's rounding regression and makes both norms in the first
three layers exact. Full acceptance still fails four numerical checks; the
next isolated target is layer1 residual addition at channel3390. PR4 remains
incomplete.

Follow-up: [exact residual addition](qwen38-residual-precision.md) fixes
all384 captured residual sums, including channel3390. Its experimental full
replay still fails11 numerical checks (versus4 previously); it is not promoted
as the accepted default. The next isolated target is layer3 attention.

Follow-up: [real attention rounding](qwen38-attention-boundary.md)
makes layer3 new K/V and gated attention output exact using block-carry Q/K/V/gate
projection and precise attention. The full replay fails6 checks versus11, but
final residual/norm errors increase, so it remains experimental. The first
changed norm boundary is now layer4 xn, one local RMSNorm error at channel786.
PR4 remains incomplete; runtime and catalogue defaults are unchanged.

Follow-up (2026-10-02): [RMSNorm rounding](qwen38-norm-precision.md)
closes layer4/channel786 and reduces local norm differences from26 to9.
The first ten norm boundaries are exact. Full acceptance still fails6 checks;
final residual/norm errors decrease but exceed their bounds. The next isolated
target is layer4 FFN activation14949, whose BF16 rounding differs despite exact
xm. The new norm mode is slower and remains opt-in; PR4 is not complete.

Latest follow-up: [FFN activation rounding](qwen38-activation-precision.md)
closes layer4 h14949 with compensated products and exact FP32 additions.
The full replay fails2 checks versus6, but final residual/norm maxrel increase;
it remains experimental. The next first norm error is local to layer5/channel3295
and is reproduced separately. PR4 still requires numerical closure and runtime
integration; defaults and catalogue support are unchanged.

Latest follow-up: [RMSNorm scale compensation](qwen38-norm-scale-precision.md)
closes layer5/channel3295 by retaining product residuals. The first eighteen
norm boundaries are now exact. Full replay still fails2 checks, but final
residual/norm maxrel fall to0.0229053 and0.0128205. The first changed norm is
layer9 xn, with14 propagated differences; a separate layer8 FFN trace isolates
one activation rounding error at749 despite exact up/gate. The mode
adds runtime cost and remains experimental. PR4 is not complete.

Main integration (2026-10-03): [merge regression review](qwen38-main-merge.md)
checks main through06b98bc. Its general head-order fix matches our existing27B
converter; the new block-prefill route is separate from this NPU-only harness.
The rebuilt FFN and full replay produce identical captures to the preceding
stage. Both numerical failures remain; there is no measured decode precision
improvement or regression from this merge.

Follow-up (2026-10-03): [small-gate sigmoid precision](qwen38-sigmoid-precision.md)
closes layer8 activation749 and makes the first 21 norm boundaries exact.
The complete replay nevertheless worsens to six numerical failures from two:
four layer47 attention heads plus the final residual/norm. Tokens, reset and
state checks pass. The mode remains experimental; PR4 is incomplete. The next
target is layer10 DeltaNet before its post norm, where seven differences are
propagated from earlier operations in that layer.

Main integration follow-up (2026-10-03): [shared WideDeltaNet merge](qwen38-main-oct3.md)
includes upstream/main b16e6ab via origin/main 0ceb46d. All8190 full-model
captures match the preceding stage bytewise after rebuilding FFN. The six
numerical failures remain; upstream's rolled-band fix targets H2560, not H5120.

## Sources and format

Weights live in the existing ignored `Models/qwen38-27b/` directory.
`utilities/download-qwen38-27b.py` pins and verifies these sources:

- Official [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B/tree/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0),
  revision `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`.
- [ggml-org Q8 GGUF](https://huggingface.co/ggml-org/Qwen3.8-27B-GGUF/tree/71bc7b627595dc8a91039addd9c791ae548d6747),
  revision `71bc7b627595dc8a91039addd9c791ae548d6747`.
  `Qwen3.8-27B-Q8_0.gguf`:28595763648 bytes, SHA256
  `aab65c67ef0dad127960efef9247f1832bca105faa1c7a052cc039b223cf86a1`.

The GGUF `.src_sha` PRIMARY entry matches the official revision. Its config
SHA256 is the existing fixture's
`191e0af232104ed8b65258cf3fb2b842e288008baca7633c11b82a1ac7203aab`.
Source metadata, tokenizer, chat template and license accompany the weights;
per-file hashes are in `Models/qwen38-27b/source/sources.json`.

## Converter fixes

The previous size detector selected9B for hidden5120. `QWEN35_27B` now has an
exact5120 match, aliases for Qwen3.5/3.8-27B, a registered converter and a text
packing configuration. Layer projections use Q4_1 and the LM head Q8.
No new weight format is introduced. Vision and direct HF-tensor conversion for
this variant explicitly report not implemented.

The old converter assumed a two-to-one V/K ratio and split QKV into equal halves.
The [upstream GGUF conversion](https://github.com/ggml-org/llama.cpp/blob/master/conversion/qwen.py)
reorders value heads from grouped to tiled order for its broadcast implementation.
The wide path restores the actual16/48 geometry: Q/K rows4096 remain intact,
V rows6144 and their associated gate/AB/A/dt/conv/output axes are reordered with
ratio3. Attention Q/gate is deinterleaved separately.

Independent comparison against bounded ranges of the official first HF shard:

| Tensor | GGUF raw vs HF cosine | Restored vs HF cosine |
|---|---:|---:|
| Layer0 alpha | 0.1559182572 | 0.9999855490 |
| Layer0 conv | 0.7268157311 | 1.0 (exact) |

The alpha difference after restoration is its Q8 quantization. The verifier is
`utilities/verify-qwen38-head-order.py`; source ranges and results are retained
in `Models/qwen38-27b/source`. No entire BF16 shard was downloaded for this check.

A streaming safetensors writer keeps only one converted tensor in memory.
Embedding and head use512-row batches, avoiding whole-vocabulary FP32 arrays.
The final file is published only after successful serialization. The open wide
recipe reads native48-row BF16 alpha/beta companions; unused padded quantized
AB tensors are omitted. Canonical data:851 tensors,19150495744 bytes.
The completed container uses native `[row_blocks,column_blocks,chunk_bytes]`
shapes. Preflight accepts that grid and the flattened chunk-list representation,
while rejecting an incorrectly transposed grid even when its byte count matches.
Container SHA256:
`5f5c282ae81b63df34e5f9ce42c616d73b454691c4b8d253027b2e08a1fe3d84`.

## Full-model execution preparation

The existing standalone schedule now supports an explicit64-layer composition,
with48 linear and16 full attention layers, separate state and shared streamed
weight buffers. The eight-layer default is preserved. Each full pass performs
592 layer dispatches plus final norm and LM head; three tokens and full reset
replay require3564 dispatches.

Header preflight checks every required tensor, including layer63, both Q/gate
halves, dtype/shape, bounds and overlapping ranges. It does not validate tensor
values or hardware. The full-model utility uses production packing and the
existing independent FP64 oracle. Only the seed token is loaded at sequence
start/reset; subsequent device inputs come from device-selected logits.

## TDD and reproduction

Five schedule/preflight tests failed before implementation, followed by two
variant-routing tests and three ordering/streamed-output tests. Implemented
changes pass789 open-engine tests (47 skipped) and86 converter tests (38 subtests).
The native3D grid contract added one further failing test before its fix.

Use [open-qwen38-model](../../../.opencode/skill/open-qwen38-model/SKILL.md).
Logs: `/tmp/wide-model-{download,convert,head-order,unit,converter-unit}.log`.

## Hardware result: execution passes, numerical acceptance fails

The existing open harness completed **3564 dispatches**, using11 sequential
contexts and1937929356 bytes of declared BO storage. No kernel or library was
rebuilt for this stage. Host control selects the token and loads its embedding;
all neural operators execute on the NPU. The model source, packed fixtures and
accepted kernel artifacts are hash checked. The captured model is language only.

| Position | Input | Device / reference next token | Full-logit correlation |
|---|---:|---:|---:|
| 0 | 248045 | 8678 / 8678 | 0.9999975546293374 |
| 1 | 8678 | 198 / 198 | 0.9999865019445474 |
| 2 | 198 | 2 / 2 | 0.9999965601912532 |

All full-vocabulary comparisons exceed the inherited0.9999 correlation bound
and select the same argmax. All4095 decode reset comparisons are byte-identical,
including per-layer inputs, outputs, scratch and state. State isolation, padding,
cache-row preservation, activation chaining and canaries pass.

Nevertheless, **18894/18896 slice checks and4121/4122 decode checks pass**;
`compare` correctly exits1 and both result files retain `passed: false`:

| First failing capture / field | Normalized max error | Bound |
|---|---:|---:|
| `cold-1-layer39`, attention head11 output | 0.02287581699 | <0.02 |
| `cold-1-layer63`, final layer residual | 0.02633355729 | <0.005 |
| `cold-1`, final normalized vector | 0.01282051282 | <0.008 |

The supplementary conditional Q8 head passes all three tokens: worst maxrel
9.18510e-6 and minimum cosine0.9999999999166488. Whole DeltaNet states pass
their inherited0.02 bound (worst0.00538648); stricter head-local1e-4 diagnostics
pass only3/144 layer-token tensors and remain reported separately.

## Boundary diagnosis and next gate

`utilities/diagnose-wide-model-layer.py` reads captures without changing them
and replays attention layers from device inputs using the independent CPU oracle.
It validates fixture/kernel hashes and capture canaries. Conditional results are
diagnostics only and never replace independent end-to-end acceptance.

At `cold-1-layer63`, conditional input/post norms are BF16-exact, Q/K/V/gate
projection maxrel is at most1.37e-7, attention0.00033784, output projection
2.28e-5 and FFN4.63e-5. Replaying the entire layer from device input/cache agrees
with the device within0.00056890 but still differs from the independent path
by0.02575001. Thus the large final residual failure is primarily propagated
input/state error amplified by the last layer, not a local FFN or LM-head fault.

At layer39, conditional head11 maxrel is0.00041118; replaying the layer from
device input/cache agrees with the device within1.26e-5. Its first inherited
head failure likewise cannot be attributed to a large local attention error.
These measurements localize propagation but do not identify a single faulty
primitive or prove which arithmetic change will close the full-depth gate.

Next: trace accumulated rounding across earlier layers/tokens and validate a
precision correction against these unchanged real weights, seed and bounds.
Do not advance to runtime/catalogue promotion or weaken acceptance based only
on matching tokens. This run does not establish prompt quality, prefill,
long-context behavior, performance or the B8 definition of done.

```bash
OPENBLAS_NUM_THREADS=1 ironvenv/bin/python utilities/diagnose-wide-model-layer.py --tag cold-1-layer39
OPENBLAS_NUM_THREADS=1 ironvenv/bin/python utilities/diagnose-wide-model-layer.py --tag cold-1-layer63
```

Artifacts: `open_kernels/designs/wide_deltanet/build_full_model/`, including
`model-provenance.json`, `slice-results.json`, `decode-results.json` and both
`cold-1-layer*-diagnosis.json` files. Logs:
`/tmp/wide-model-{prepare,hardware,compare,diagnose39,diagnose63}.log`.
