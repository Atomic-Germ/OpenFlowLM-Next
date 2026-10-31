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

**1e. Identity: the family is the model's, the size is the file's.** Two more false
statements, both from a name doing work a number should do:

* A kernel set names the KERNELS, not the model. `details.family` is the engine/adapter
  (`qwen3`, `qwen3-it`, `qwen3-tk`, `qwen3vl` are four entries for four Qwen3 4B
  containers) while one set serves every model of a geometry -- so a text Qwen3-4B
  fine-tune matching `Qwen3-VL-4B-Instruct-NPU2`'s kernels was logged (and would have
  been installed) as `qwen3vl`. The set now only supplies a family as the LAST resort,
  when the model itself names none.
* The family choice between the arch and the base chain is explicit
  (`family_precedence`): the base wins when it REFINES the arch (`qwen3` -> `qwen3-it`,
  which is what an Instruct-2507 fine-tune is), and the arch wins otherwise (`qwen3vl`
  keeps its VL family against a text base). The log names the source it used.
* The tag size comes from the model's NUMBERS, in order of exactness: a GGUF's tensors
  (the real parameter count), a safetensors index over the dtype, the config's geometry
  (a VLM's `text_config` included -- the top level has no `hidden_size`, which is why
  `GRaPE-1.5-TEST-DONTDL` had no size at all), the bytes/quant guess, and last the set
  that serves it. `mradermacher/GRaPE-1.5-i1-GGUF` (arch `qwen35`, name says 1.5) now
  installs as `qwen3.5`, `4.2b`, on `Qwen3.5-4B-NPU2` -- every fact from the file.

**Bucket 3 is now a refusal, not a silent wrong answer.** That GRaPE install loads and
answers, and the answer is gibberish -- the same wall as every Qwen3.5 GGUF-direct model.
`oflm add` refuses `qwen3.5` GGUF-direct by name (`GGUF_DIRECT_UNREADY`), says why, and
points at the q4nx container; `--force` still installs it for the numerics work.



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

## 3. Qwen3.5 numerics  (bucket 3 -- FIXED: three pack bugs)

**Status: qwen3.5 GGUF-direct runs on the open kernels.** Qwen3.5-0.8B answers
("Hello! How can I assist you today?", "42", "Paris"). The record settles the "was it ever
good" question too: on the open kernels Qwen3.5-4B used to *segfault in prefill at layer 0*
and Qwen3.8-4B gave garbage, so this was new work, not a regression -- what worked was the
closed container path, which is AMD's and has to go.

**The instrument** (`open_kernels/model/gguf_ref.py`): a GGUF *tensor* reader (the vendored
one is metadata-only) with F32/F16/Q4_0/Q4_1/Q8_0/Q4_K/Q5_K/Q6_K dequantizers and the
HF-name map llama.cpp's conversion implies, so `replica_prompt.py --model-dir <gguf>` runs
the fp64 reference on the file. `--ids` gives it the engine's token list. It agreed with a
container built from the same GGUF (q4nx-build) to cos 0.9946 at layer 0, which is how the
mapping was cleared before any engine change was blamed.

**Three bugs, all in the PACK -- ours, not the kernels'.** Found by diffing the two
manifests' pack plans and the `OFLM_OPEN_DUMP_POOLS` dumps per `dst` offset, one tensor at a
time (every other consts region and every projection was already bit-identical):

1. **`conv_transpose`** read a GGUF's `ssm_conv1d` as `[taps][channels]`; llama.cpp stores
   `[channels][taps]`. Every channel got another channel's filter (maxabs diff 1.78 on a
   tensor whose own maxabs is 1.76). Fix: de-interleave when the tensor's dims say so.
2. **`lmhead_q8`'s GGUF branch** wrote the head pool in the *file's* raster; the kernel
   reads the *supertile* raster (the q4nx branch permutes into it). Logits came out at cos
   0.33 with every hidden state at 0.998 -- which is what "plausible but wrong" looks like.
   Fix: the same permutation the container branch uses.
3. **The fused q|gate projection** is per head in a GGUF (`[q_h|gate_h|...]`) and
   `[all q|all gate]` in the pool. cos 0.14 for the whole tensor; de-interleaved, 0.99999.
   Fix: a `qgate_rows`/`qgate_gate` pair on the op, emitted by `proj_op` only for a GGUF
   source (`spec.reads_gguf`), because a container is already in the pool's order.

After the fixes: every layer's hidden state is 0.965-0.995 against the fp64 reference, the
last-position logits are within the pool law, and the model answers coherently. The manifest
changed, not the kernels -- the shipped set needs a rebuild (or its manifest regenerated) to
carry it.

**The family is open.** The five sets' manifests were regenerated from the recipe
(`qwen35-0.8b/2b/4b/9b/27b` -> `src/xclbins/<set>/open_kernels/manifest.json`; the kernels
are untouched, and a manifest is the only thing the pack fix changes). `GGUF_DIRECT_UNREADY`
is empty again, so `oflm add` accepts qwen3.5 GGUF-direct. Measured:

| size | evidence |
|---|---|
| 0.8B | answers ("Hello! How can I assist you today?", 17+25 -> 42, Paris); every layer 0.965-0.995 vs the reference |
| 4B | last-position logits **cos 0.993**, same argmax (mradermacher's i1 is a quant of the BASE checkpoint, so it completes text rather than answering -- the numerics are right, the model is not instruct) |
| 2B / 9B / 27B | sets carry the fixed pack; unmeasured here |

**`oflm add` picks the NEWEST manifest among the sets that serve a geometry.** It used to
take the first root (and I had made the installed one win), so a fresh `export-kernels.py`
in the tree was silently shadowed by the installed copy -- the exact workflow of testing a
pack fix. A kernel fix lands in a rebuild, so the rebuild wins.

**Still open, and unrelated to the pack:** `OFLM_GGUF_POOL_F32=1` does not load --
"GGUF-direct block prefill is not implemented for the 'linear' layer kind (only the dense
one)". The f32 twin's *route* has no hybrid layer kind; the message used to say "MoE models"
and sent us looking at the wrong thing (the user was right: dense models are not broken,
qwen3.5 is hybrid). That blocks the q4_1-vs-f32 A/B, not the model.

**A full build is the cmake workflow, and that is the only entry point that matters.** Clone
the repo, run `cmake --workflow --preset linux-package`, install the RPM: every family and
every size is built from `recipes/specs/*.json` plus the diffusion families, and the package
carries them. Verified after the qwen35 fix: 448 MB installed, 2886 xclbins + 8 ELFs, all
five qwen35 sizes, and Qwen3.5-4B's manifest inside the RPM carries the `qgate` ops. A spec
is added by `oflm add <repo>` from the repo root (SPEC -> BUILD -> GATE -> CATALOGUE ->
REBUILD -> REGISTER); from anywhere else it says so instead. No hand-run export, no
hand-edited manifest.

**The staging dir is now unique per run and removed on exit** (`tempfile.mkdtemp` + `atexit`).
It used to be named after the spec's hash, so a leftover made every later export of that spec
die on `shutil.rmtree` -- hundreds of lines above the diffusion message the log ends on, which
is what made that failure read as a klein-8B problem. `cmake --build build --target
clean-kernels` sweeps the scratch (and `distclean-kernels` the exported sets too).




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
