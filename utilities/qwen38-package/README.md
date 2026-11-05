# Qwen3.8-27B package gate (Phases 16 and 23)

Two plan phases ask for the same guarantee from two directions: the converter
must find every tensor `qwen35.pack_plan()` names exactly once and silently
ignore none (Phase 16), and the release must be a reproducible package linked to
a validated kernel set through the normal model-add mechanism with
`details.family = qwen3.5` (Phase 23).

The converter deliberately only *warns* when a name-map entry is absent, because
a family whose `lm_head` is synthesized from the embedding still converts. That
is correct for a converter and insufficient for a release, so this tool turns
that warning into a gate.

```bash
source utilities/env.sh
OPENBLAS_NUM_THREADS=1 ironvenv/bin/python utilities/qwen38-package/verify.py \
  --model-dir Models/qwen38-27b/Qwen3.8-27B-NPU2 \
  --kernels Models/qwen38-27b/production-v2/kernels-model \
  --report Models/qwen38-27b/production-v2/package-verify.json
```

## What it proves

**Coverage.** It flattens the plan into its ops and resolves each `op["tensor"]`
against the packed container. A miss, a stray tensor no op reads, or a
non-27B geometry fails the run. `{l}` ops are expanded **per layer type**: a
`linear_attention` op is never required on an attention layer. Expanding every
`{l}` op over all 64 layers is the obvious implementation and it is wrong — it
reports 48 phantom misses (16 full-attention layers x 6 ops) for a package that
is complete. A test pins exactly that.

**Reuse is not duplication.** The fused `q_proj` is read twice, once for the Q
half and once for the gate half, so a tensor may legitimately be read by two
ops. The inventory counts reads rather than forbidding them.

**Consistency.** The bundle's `manifest_version` must be 4, its family
`qwen35`, its `real_vocab` must equal the model's, its toolchain build key must
agree with its manifest, every declared artifact must exist, and the linear
post-output region must be the standard FP32 one
(`float32` at 393216 for 24576 bytes). The model directory must carry
`config.json`, `model.q4nx`, `tokenizer.json`, `tokenizer_config.json` and
`chat_template.jinja`.

`model_acceptance` stays false: a complete package is not a numerically accepted
model, and Phases 20-22 own that verdict.

## Validated checkpoint — 2026-10-31

`production-v2/package-verify.json`: **851/851 required tensors resolved, zero
missing, zero unused** out of 851 packed tensors; 46 builds and 92 declared
binaries present; manifest v4, family `qwen35`, `real_vocab` 248077 agreeing with
the model; build key `643d3b47…` consistent with the toolchain record.

The first run of this tool reported 432 "missing" tensors. It was the tool that
was wrong, not the package: it expanded every `{l}` op over all 64 layers
regardless of layer type. The fix is type-gated expansion and the test above is
its regression.

## Registry link

`src/model_list.json` already carries the shipped entry with
`details.family = qwen3.5`, `vlm: false`, a Hugging Face URL and a ModelScope
mirror, so Phase 23's registry requirement is satisfied by the existing entry
rather than a new one. Vision is explicitly out of scope for this task and the
entry does not advertise it.
