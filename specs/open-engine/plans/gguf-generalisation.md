# Plan: GGUF generalisation -- closing the false gates

**Status:** proposed. The hard part is done and shipped: container, loader, kernel linking,
packaging, and diffusion end to end from a GGUF. What remains is *generalisation within
families* -- and most of it is the tool telling the truth about why something failed.

**Evidence base:** twelve `oflm add` runs against the installed package (2026-10-31), sorted
into five buckets. Bucket 1 is first because it hides or fakes the others.

## 1. The kernel hash mixes packing with geometry  (bucket 1 -- sweeps the field)

Measured, gemma-3-4b-it against the shipped set:

    Gemma3-4B-NPU2 set spec:   vocab 262208   real_vocab 262145
    gemma-3-4b-it GGUF:        token_embd [2560, 262144]

Everything else is equal -- hidden 2560, 34 layers, 8/4 heads, head_dim 256, rope 1e6,
sliding window 1024. Only `vocab` differs, by 64: the set was built from a container whose
vocabulary was padded to 262208, the GGUF reports the model's own 262144. Because `vocab` is
in `ModelSpec.kernel_hash`, a *packing* choice rejects every GGUF of the family, and the
message says "different geometry", which sends the user to build kernels that are already
installed.

Change:
* `kernel_hash` covers what the kernels tile: geometry, head layout, rope, norm, activation,
  quant law. **Vocabulary padding is not geometry** -- hash the chunk-rounded value (the
  tiling-relevant one) and validate the real vocabulary at load, where it belongs.
* The same applies to any other field that is a container's packing choice rather than a fact
  about the model. Audit the spec for them while doing this (`real_vocab` already exists and
  is the right shape).
* **One matching path.** Today two messages disagree for the same family:
  `gemma-3-4b-it` says "no set matches this geometry, the nearest is different", while
  `translategemma-4b-it` says "no GGUF-direct set found for Translategemma-4B-Instruct-NPU2"
  -- i.e. it built a set name from the model instead of matching by geometry. Geometry
  decides; the name is a hint for the tag only.
* `build-new-spec.py` writes hashes into `recipes/catalogue.py` and the frozen test lists, so
  this change ripples there: regenerate, and expect the tests to name any catalogue entry
  whose hash moved.

Verify with, in this order: `gemma-3-4b-it` (fernandoruiz, Q4_0), `translategemma-4b-it`
(mradermacher, i1), `gemma-4-E4B-it` (unsloth), and **Qwen3-VL-4B must still install** (it
works today; it is the regression check).

## 2. Two parser gaps  (bucket 4 -- cheap, and both are "read what llama.cpp wrote")

* **LFM2.5-1.2B**: `gguf: metadata lfm2.attention.head_count_kv is not a number ()`. The
  GGUF carries that key as a per-layer *array* (llama.cpp does this for hybrid models). The
  reader must accept scalar or array, and the spec must take the per-layer value. Same for
  any sibling key (`head_count`, `key_length`) that can be arrayed.
* **Qwen3.8-9B-Distill**: `config.json layer_types differ from the kernel set's (32 layers)`.
  The model is 33 layers per its set; the derived `layer_types` says 32. Fix the derivation
  (`_sliding_window_pattern` → the same pattern the set was built with), and make the message
  say which side disagrees rather than a count that reads like a model fault.

Both are one field each, and both currently refuse a model that would run.

## 3. Qwen3.5 numerics  (bucket 3 -- the one that "works" and lies)

Every Qwen3.5 tune tested (4B/2B/0.8B, prithivMLmods and mradermacher) installs, loads, and
emits gibberish. That is worse than a refusal.

* Start from the skill's q4_1-vs-f32 twin notes; the CPU reference is the gate.
* **Until it passes, refuse qwen3.5 GGUF-direct by name.** A stated limitation is honest; a
  fluent wrong answer is not. That is a one-line change in the family table, and it can be
  removed the moment the numerics pass.

## 4. Missing sizes  (bucket 2 -- the scripted path)

`Qwen3-VL-8B`, `gemma3-1b`. Exactly the Qwen sizes precedent: `build-new-spec.py` (SPEC →
BUILD → GATE → CATALOGUE → REBUILD → REGISTER), then the size is a record. Nothing here
should need a new code path -- if it does, that is the finding.

Same pass: **derive the packaging list from imports** rather than remembering it (the
converter's modules, `open_kernels/*.py`), so "our module isn't installed" stops recurring.

## 5. ASR, then embedding  (bucket 5 -- not started)

* **Whisper**: the engine exists and runs on xclbins; models are q4nx. The work is the GGUF
  name map plus geometry from the metadata, then a per-geometry kernel set. Both whisper
  repos tested (Xviers, tory-labs) fail identically, so one mapping fixes both.
* **Embedding**: six families, five architectures, pooling from each `1_Pooling/config.json`.
  The `EmbedFamily` design and the six-family table were written earlier and are recoverable
  from `backup/pre-embedding-nuke`; take the design, not the code, and land it behind a
  cosine gate. `embeddinggemma-2` is multimodal and belongs with the encoder work, not here.

## Rules this plan assumes (they are the product)

* **User:** install → `oflm pull` (suggestion list) or `oflm add <gguf>` → `serve`/`run`/
  `image`. Never `pack`, never `source`, never a build.
* **Identity comes from the file** (shape, metadata, tensors). The name is a hint.
* **Kernels ship with the app**, all of them, shared by geometry. Never assume a container
  ships.
* **`model_list.json` is a suggestion list**, not the boundary of what can run.
* **If we have enough information to tell the user what to do, we have enough to script it.**
  A gate that refuses what a script could do is a bug.

## Failure modes to avoid (all of these happened)

* Reframing documents instead of fixing the path.
* Optimising one model, then generalising from it.
* Treating the dev/app seam as a design decision.
* In-tree tests as proof: verify with `oflm add` + `run`/`image` **from the installed package**.
* Adding a guard in front of a mechanism that already works (`ensure_venv()` was right there).
