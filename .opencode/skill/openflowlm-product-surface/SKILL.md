---
name: openflowlm-product-surface
description: What OpenFlowLM-Next ships and how a model reaches the NPU - oflm pack, oflm add, oflm pull, the q4nx-build packer, and the open-kernel catalogue. Use when packaging a model, diagnosing "no open kernels found" or a kernel-set mismatch, onboarding someone to the project, or explaining where any of it came from.
---

# The product surface

A load-bearing accident with a load-bearing result. The fork happened at
FastFlowLM 1.0.1 — immediately before they changed their packer and broke
every model not packed by that internal version, which mobilised the
community. The first symptom was personal: a finetune that would not load on
hardware that had been deliberately undersold. Their docs said kernels were
per-model and sold a service to add new ones. So: write a packer.

That packer then packed every NPU2 model on HuggingFace not on the FastFlowLM
account, and when that team was absorbed by AMD, the field landed. A great deal
came out of the simple desire to load a finetune.

**There is no upstream.** AMD does not support this project and is adversarial
toward it. Every open file here is community-written. `oflm add` has no closed
parallel because their answer to a new model was a service engagement, and
this one's answer is a converter. Do not plan around an upstream merge or a
fix "coming in a later version" — if a capability does not exist, it gets
built here.

## What this replaced, and the shape of it

The pre-fork state, all of it now gone:

- **A build that did not easily produce an installable package.** Now
  `cmake --preset linux-default` → `cpack`, and an RPM carrying 18 open-kernel
  manifests, 5 BERT sets and Whisper, with no partial files in it.
- **Developer instructions shown to end users**, because nothing could work
  out how to link a kernel set that did in fact exist. Now family resolution
  is automatic and a miss still installs.
- **`flm pack`, poor.** Now a dispatcher to a real packer with per-family
  converters, per-role dtype maps, `--quant`, `--prune-ffn`, and a refusal to
  write a container known not to load.
- **Manual kernel builds.** Now the catalogue derives the specs, the exporter
  builds the sets, and `toolchain.json` records versions, commit and sha256s
  so a rebuild can be diffed against a previous one.

These were not unrelated conveniences. They were all one problem: a container
format treated as a fixed artifact instead of as an **output with a derivable
input**. Once `q4nx-build/configs/` is the input and the manifest the output,
the kernel build can read the same quant map, `spec_hash` can include it, and
"which kernels belong to this container" stops being a question a human
answers.

## Complete open-route inventory

**This is the answer to "what can we build?", and it is read from the code, not
inferred.** Four independent axes, because a route needs all four:

1. **Packer** — `utilities/q4nx-build`: 18 `ModelArch` values, 18 converters
   registered. No gap: every arch the packer knows can emit a container.
2. **Recipe** — `open_kernels/recipes/families.py`: 10 dispatchable families
   (`gemma3 granite hunyuan lfm2 llama3 phi3 qwen2 qwen3 qwen35 qwen36moe`),
   six of them sharing `recipes/dense.py`. Plus exactly one family explicitly
   marked not implemented (`gptoss`).
3. **Spec derivation** — `HF_FAMILIES` in `recipes/spec.py` maps a model's
   `model_type` to a builder. This is the authoritative model_type → recipe
   bridge; 22 model_types are routed.
4. **Built sets** — `src/xclbins/*/open_kernels`, which is **gitignored**
   (kernels are built, never checked in, and never shipped inside a model).

### Routes that are complete (pack → derive → build → run)

| recipe | geometry built | engine | notes |
|---|---|---|---|
| `qwen35` | h1024/2048/2560/4096 dense | `open_qwen36` | lx/ax dense line |
| `qwen36moe` | h2048-L40 | `open_qwen36` | Qwen3.6-MoE 35B-A3B, verified running |
| `qwen3` | h1024/2048/2560/4096 | `open_npue` | |
| `llama3` | h2048/2560/3072/4096 | `open_npue` | |
| `gemma3` | h768/h1152/h2560 | `open_npue` + `open_gemma3` | |
| `phi3` | h3072-L32 | `open_npue` | Phi-4-mini |
| `qwen2` | h2048-L36 | `open_npue` | also serves Qwen2.5-VL's text half |
| `lfm2` | h2048-L16/L30 | `open_npue` | short_conv |
| *(whisper)* | v3 turbo | `open_whisper` | own xclbins, not a recipe |
| *(embeddings)* | all-minilm, bge ×3, gte, nomic, embed-gemma | `open_embedding` | no kernels |

### Routes that need only a BUILD (the code is all here)

**These are the actionable list.** Nothing below needs a kernel written.

> **These three are DONE and one is PROVEN ON HARDWARE (2026-10-01).**
> The durable fix for all of them was a `src/model_list.json` **entry**, not a
> hand-written spec: `recipes/specs/` is gitignored, and
> `gen_catalogue_specs.py` *deletes every spec* and regenerates from the
> catalogue on each build. A hand-written spec is wiped, silently.

- **`granite`** — Granite 4.2-3B. **Built, linked, and PROVEN on hardware**:
  `oflm run granite:3b` answers correctly ("Paris"), and there are **no closed
  granite kernels installed**, so it ran entirely on kernels built in this
  checkout. `spec_hash sha256:fea8ba46bd39`; catalogue URL is
  `OpenFlowLM/Granite-4.2-3B-NPU2` (that repo's README carries
  `oflm-family: granite`; the `vegahyo/` original does **not**, so `oflm add`
  cannot infer the family from it).
- **`nanbeige`** — Nanbeige4.1-3B, the `phi3` recipe. **Needs nothing.** It is
  `model_type: llama` at h2560-L32, so it resolved to the pre-existing
  `llama3-h2560-L32` set with an **identical `spec_hash` (3213be526345)** and
  byte-identical contents. Built again for the proof; it deduplicated.
- **`hunyuan`** — **Kernels DONE for both dense sizes, engine MISSING.**
  - `hy-mt2:1.8b` → `hunyuan-h2048-L32`, `sha256:d6ef13a70b6e`
  - `hy-mt2:7b` → `hunyuan-h4096-L32`, `sha256:96485aee63e2`

  Both build (`dx`, `dx_attn`, `ln`, `lm_head_q4`, 4 GEMM prefill sets) and both
  link. The 1.8B needed one `catalogue.py` attn tuple added —
  `(128, 16, 4, 128, qk_norm, attn_gate=False, post_rope=True, bias=False)`,
  which is the 7B's post-rope qk-norm at a GQA group of 4, so the compiled path
  already existed. **Packing: the dense pair only converts from a Q8_0 source**;
  from Q4_1 the `lm_head` comes back a float passthrough the q4_1 head cannot
  read, and the converter refuses with a clear message.

  **The remaining gap is an ENGINE, not a kernel:** `SupportedModelFamily` in
  `src/include/AutoModel/model_families.hpp` has no `hunyuan` member, so the
  runtime refuses with *"family 'hunyuan', which this build has no engine for"*.
  Kernels + converter + catalogue can all be complete and the model still cannot
  run. Treat family presence in that enum as a separate checklist item.

### Routes that need CODE

- **`gptoss`** — the one family with a written, reasoned refusal
  (`recipes/families.py` `NOT_IMPLEMENTED`, plan in
  `.claude/plans/gptoss-bringup.md`). The arithmetic is done and tested
  (`model/replica_gptoss.py`); what is missing is that no kernel computes it —
  experts want a clamped SwiGLU, `o_proj`/router/all three expert projections
  carry a bias no design has room for, the MoE FFN must compose with
  sliding-window layers, and the expert intermediate equals hidden
  (`OPEN-MOE-WIDE-FF`). Packer side is complete; this is purely kernel work.
- **DeepSeek-R1 / R1-0528** — in the catalogue, but **no `ModelArch` and no
  recipe at all**. It needs a converter *and* a recipe. The largest genuine hole.

### A model_type worth remembering

`gemma3_text_only` exists in `HF_FAMILIES` because Gemma's own 1B-class towers
publish that model_type rather than `gemma3_text` — and its absence is why
`oflm add` on Gemma3-1B derived no spec and found no kernels while every other
Gemma3 size worked. When a container is right but no kernels link, check the
model_type is in `HF_FAMILIES` before anything else.


## Why the closed stack cannot do this

Not a claim about their intentions — only about what the structure implies.
Their registry entries carry hand-written `details` per model, their containers
are pinned to whatever an internal packer produced, and "kernels are
per-model" is stated in their own docs with support sold for new ones. There is
no way to derive a spec from a catalogue entry, so a new finetune is a support
engagement by construction. Whether they run a reproducible build internally is
not something we can observe; the absence of a way to *express* one is visible
in the artifact either way.

## The four commands

| | does |
|---|---|
| `oflm pack` | GGUF/safetensors → q4nx container, via the bundled `q4nx-build` |
| `oflm add` | install a container, resolve its family, link a kernel set |
| `oflm pull` | fetch from the registry; live listings, not a frozen snapshot |
| `oflm list` | installed + available, network-free in the fast path (~0.15 s) |

`oflm pack` is a **dispatcher** to the bundled Python packer, not a C++ port.

## Family resolution, in order

1. exact `oflm-family` in the repo README frontmatter
2. `config.json` `model_type`
3. the model's name

Normalised exactly, so `qwen3_5` and `qwen3.5` are the same family. A
`oflm-family` value is a declaration by whoever made the conversion, so it
outranks name inference. When no kernel set matches, `oflm add` **still
installs** and prints a prefilled GitHub issue link naming the model — a model
that loads on the closed kernels is a working install, and a request for open
kernels is not an error.

## The packer

`utilities/q4nx-build/`, a family-per-model registry. This is the piece with
no closed parallel, and the reason is structural: the container format is an
*output* of the packer and its configs are the *input*, so the kernel build can
read the same quant map the packer used and the two cannot disagree. That
agreement is why `spec_hash` includes the quant map, and why the kernel side
**refuses rather than assumes** when it cannot resolve one — a kernel set that
disagrees with a container's quant is worse than no kernels.

Per-family converter classes (`q4nx/models/qwen35.py` and friends) plus a
JSON config per family carrying the block geometry and a per-role dtype map.
`--quant` overrides the family's default; roles that pin their own type in the
config keep it.

Two things to know before changing it:

- **The embedding is always bf16**, written unconditionally, no dtype lookup.
  AMD's equivalent carries a `# this should be bf16` comment beside a
  `tied_embedding` branch that skips writing the embed at all — and the
  container that started the 27B investigation had an **I8** embed, which the
  closed loader refused with a size mismatch. Ours cannot produce that.
- **`.gitignore` has `!utilities/q4nx-build/q4nx/models/**`**, which re-admits
  that directory's `__pycache__` along with the converters. There is a
  re-ignore line after it. Keep it.

## The kernel catalogue

`open_kernels/gen_catalogue_specs.py` derives one buildable spec per
`(family, size)` from the registry — 44 catalogue models became 21 sets
covering 30 models. Specs under `open_kernels/recipes/specs/` are **generated
and gitignored**; the hand-written ones are gone, and a spec you create there
will not be committed.

Kernel identity is `(family, geometry)`, not the tokenizer: `real_vocab` is no
longer part of it, the engine reads the model's tokenizer bounds and clamps to
padded vocab. A new finetune of a known family therefore reuses the family's
kernel set with no work.

Dense Qwen3.5 currently covers hidden 1024 / 2048 / 2560 / 4096, and 5120 once
`of_lni` clears (see `open-qwen36-kernels`).

## Rules

- **`oflm add` overwrites a differing file and says so.** Installing a model
  twice is an update, not a no-op; the common reason is a newer conversion.
  It compares content (size plus head/tail hash — hashing 16 GB per install
  costs more than the copy it saves) and logs the swap, so a 20 GB → 16 GB
  change is never invisible.
- **Never hand-edit a container's config.** The engine walks
  `num_hidden_layers`, the kernel recipe builds for `intermediate_size`, and
  both must match the tensors actually in the file. Repack instead. If a pack
  produced an inconsistent config, that is a bug in the packer.
- `/tmp` is an 18 GB tmpfs and `/` sits near capacity. Before packing a large
  model, check `df`, and delete the source GGUF afterwards if you are short —
  the container is the artifact.
- The closed kernels are a **fallback**, and are versioned independently of
  this project. A model that needs a newer FastFlowLM than you have will
  install and then fail deep in a closed loader; that is a version gate, not a
  bug in anything here.
- **To see NPU memory, use `xrt-smi examine --batch`.** Bare `xrt-smi` prints
  only its help text, so a grep for `%` finds nothing and the device looks
  idle when it is fully loaded. Concluding "the NPU is not being used" from
  that is a measurement error — confirm with a BO submission in the run log
  (`Submitted BO …`) before believing it.
- **`[ a = b* ]` is string equality, not a glob.** A `[ $m = Granite* ]` guard
  is false for `Granite-4.2-3B-NPU2`, which silently passed the wrong
  `--family` and registered Granite as `hunyuan` — so the engine refused it
  with a plausible-looking family error. Use `case "$m" in Granite*) …`.

## Open bugs found 2026-10-01

- **Server dies on cached prompt + system-prompt change.** Reproduced on
  `granite:3b` via `oflm-test --llm` (3 PASS, then 2 ERROR): the log reads
  `Use cached prompt!` → `Matched 1 out of 3 messages` →
  `System prompt changed! Clearing context...` → 1686-token prefill →
  `Start generating...` → death immediately after `Submitted BO 389`, with no
  signal or assertion text. **Not** a kernel or length problem: a 700-token
  single-turn prompt generating a long answer runs clean. The suspect is the
  context-reuse/KV-cache path after a system-prompt change. Unfixed.
