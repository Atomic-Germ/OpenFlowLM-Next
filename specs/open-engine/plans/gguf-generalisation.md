# Plan: GGUF generalisation -- closing the false gates

**Status:** in progress. The hard part is done and shipped: container, loader, kernel linking,
packaging, and diffusion end to end from a GGUF. What remains is *generalisation within
families* -- and most of it is the tool telling the truth about why something failed.

**Evidence base:** twelve `oflm add` runs against the installed package (2026-10-31), sorted
into five buckets. Bucket 1 is closed; bucket 2's first case (Qwen3-VL-8B) is built and
running; 3 (qwen3.5 numerics), 4 (two parser gaps) and 5 (ASR, embedding) are open.

## 1. Bucket 1 -- closed, and it was TWO false gates, not one

Measured, gemma-3-4b-it against the shipped set: every field equal (hidden 2560, 34
layers, 8/4 heads, head_dim 256, rope 1e6, sliding 1024) except `vocab` -- 262208 in
the set (a container's padding), 262144 in the GGUF. That was real, but it was not the
whole bucket.

**1a. The spec came from the wrong place.** `oflm add` peeked `config.json` before the
GGUF. `mradermacher/translategemma-4b-it-i1` ships a config with no `rope_scaling`,
while the GGUF's metadata says `factor 8.0` -- the law llama.cpp actually applied. So a
model whose geometry matched `Gemma3-4B-NPU2` exactly was refused, `geom_set` came back
empty, and the gate fell back to a *name* from the registry: "no GGUF-direct open kernel
set found for Translategemma-4B-Instruct-NPU2". Two messages, one family, both wrong.
Now, for a GGUF install, the file is the authority; config.json is a fallback for what
the file cannot carry (tokenizer, chat template, vision tower). `rope_ignore` also gets
the source now (it was being passed `None`).

**1b. `vocab` in `kernel_hash` -- but not as this plan guessed.** The head kernel is
compiled per row count (bands and their split), so vocab is not simply "not geometry":
the relation is ASYMMETRIC. A set built for a padded vocabulary serves a narrower head,
because the pack zero-fills the rows past the model's own (`pools.cpp`: "padded rows:
zeroed scales/codes"); a wider head cannot be served. A hash cannot express one-sided
containment, so:

* `vocab` left `ModelSpec.kernel_hash` (`_KERNEL_INDEPENDENT`), with the reason written
  where the field list is;
* `ModelSpec.serves(model)` carries the one-sided requirement -- same geometry, and at
  least as many head rows -- and is what every matching site now asks;
* `Manifest::check_model` accepts `config vocab_size <= set rows` and refuses wider, so
  the install cannot succeed and the load fail (it would have: `vocab_size` was an
  equality check).

**1c. One matching path.** `scan_kernel_sets()` replaces the name lookup plus the
separate geometry check. It returns the set, its name (a label only -- the tag, the
xclbin link) and *why each other set was refused*, and the refusal message is built from
those notes. A set of the same family at a different geometry is named as such; an
unrelated alphabetical neighbour is not.

**1d. A missing size is BUILT, not handed over as a command.** When nothing installed
serves the geometry, `oflm add` writes the spec it already derived from the file and runs
`utilities/build-new-spec.py --spec <name>` itself (spec, build, gates, catalogue,
rebuild, register). It needs a checkout; an installed package has no recipes and no
toolchain, and there the message says the size is ours to ship, with the shape -- not
"run this python script", which the user cannot do. Verified: `unsloth/Qwen3-VL-8B-Instruct-GGUF`
built `Qwen3-VL-8B-Instruct-NPU2`, installed, and answered a prompt on the NPU.

**Also found by 1a:** `minja` (vendored, upstream 3e4c61c) rejects a TRAILING COMMA in a
dict/array literal, which Jinja allows. Gemma 3's translation template ends a 600-entry
language map with one: the model loaded on the NPU and died in the template parser.
Patched locally in `src/include/minja/minja.hpp` (both parsers).


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

## 4. Missing sizes  (bucket 2 -- the scripted path, now driven by `oflm add`)

`oflm add` builds the set itself when nothing installed serves the geometry (1d above), so
this is no longer a bucket of "run this command".

* **Qwen3-VL-8B: done.** Built, installed, runs (`20/36 layers resident`, coherent reply).
  The spec, its frozen `spec_hash` and its expected name were written by the scripts --
  nothing hand-edited.
* **gemma3-1b: a DESIGN step, not a catalogue point.** hidden 1152 is not a multiple of
  the 256-wide q4 k-tile, so the recipe refuses the shape by name ("hidden and
  intermediate must be multiples of 256") and `build-new-spec.py` moves the spec to
  `recipes/candidates/`. The fix is a 128-wide k-tile (pool layout, gemv, lm_head) or a
  padded hidden -- a recipe change, and it is the same wall `candidates/gemma-3-1b-it.json`
  already recorded. Do not expect a size inside a family to be a subset of the process:
  the *widths* decide, and a width can need a design.

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
