# Plan: load GGUF directly at runtime, make `.q4nx` a legacy format

**Status:** proposed, not implemented.
**Issue:** [#14](https://github.com/Atomic-Germ/OpenFlowLM-Next/issues/14) — "we should support gguf and treat q4nx as a legacy format".
**Spec impact:** one new requirement, `OPEN-GGUF-RUNTIME`. No existing requirement
changes; the `.q4nx` reader, the packing plan, the manifest and the kernels stay
byte-for-byte as they are.

## Why

Today a model travels `GGUF → q4nx-build (offline convert) → model.q4nx →
engine`. The `.q4nx` container is OFLM's own invention; every model must be
converted before the open engine can run it, and the converter (`utilities/q4nx-build`)
is the one place that still carries AMD/OFLM's weight-format decisions. GGUF is
the community's container — every HuggingFace model has a GGUF already, llama.cpp
reads it, and it is a stable, versioned, self-describing format.

Making the engine read a `.gguf` directly removes the conversion step and, with
it, the last format dependency on AMD's pipeline. `.q4nx` stays readable (as a
legacy format) so existing installs keep working, but it stops being the format
new models are produced in.

Three decisions were made up front and this plan is written against them:

1. **True runtime support.** The engine mmaps the `.gguf` and packs weights
   straight out of it. No `.q4nx` is ever written, no offline conversion runs.
   "q4nx is legacy" means the engine can still *read* it, not that it must
   produce it.
2. **Vendor llama.cpp's reader in C++.** A single self-contained GGUF parser
   lives in-tree and runs on Windows, Linux and the NPU host. (See "License"
   below — it is MIT, with the one real constraint being the `ggml.h` type-size
   helpers `gguf.cpp` includes.)
3. **Quant coverage.** First cut reads `q4_0`, `q4_1`, `q8_0` and `q4_K`
   natively (the four formats OFLM's own pipeline already knows), and treats
   `q5_K` / `q6_K` / `F16` / `F32` / `BF16` as "dequantize-then-requantize to
   q4_1" fallbacks.

## What already exists (do not rebuild)

The GGUF work is mostly a *weight reader* plus a small provisioning step. The
metadata side is already done:

- `open_kernels/recipes/spec.py` — `ModelSpec.from_gguf_metadata` reads llama.cpp's
  key names and derives the full hyperparameter tuple for every family the open
  engine runs: `qwen35moe` / `qwen3next`, `qwen35`, `qwen3`, `llama`, `gemma3`,
  `hunyuan-dense`, `granite` (OPEN-SPEC-DERIVE). Kernel-set generation already
  works from a GGUF's metadata.
- `utilities/q4nx-build/configs/*.json` — the `name_map` tables (GGUF tensor name
  → HF name) for every family, and `utilities/q4nx-build/q4nx/gguf_tensor.py` —
  the ggml block layouts (`unpack_q4_0` / `unpack_q4_1` / `unpack_q8_0` /
  `unpack_q4_k`) and the `q4_k` transcode. These are the ground-truth for how
  ggml quant blocks map onto the engine's chunk layouts.
- `src/open_qwen36/pools.cpp` — the transcode seam the runtime reader plugs into:
  `requant_q4_1_chunks` (q8→q4_1), `q4k_to_q4_1_chunks` (Q4_K→q4_1),
  `q8_half_tile` (q8→16-row half-tile). A GGUF source is a fourth source form
  through the same three chunk ops.
- `utilities/oflm-add` — already derives a model's spec and links its kernel set
  by `spec_hash` (OPEN-ADD-KERNEL-LINK). The GGUF path is "derive the spec from
  GGUF metadata, which `spec.py` already does".

So the kernel set (manifest.json + xclbins) and its build path are **unchanged**:
a GGUF model still gets a family kernel set built from its `ModelSpec`; what
changes is only where the weights come from.

## What we take from `../llama.cpp`

`llama.cpp` sits at `../llama.cpp` (MIT, "Copyright (c) 2023-2026 The ggml
authors"). The reusable pieces, in decreasing order of how much we take:

| piece | location | what we take |
|---|---|---|
| GGUF parser | `ggml/include/gguf.h`, `ggml/src/gguf.cpp` | The reader half: `gguf_init_from_file`, `gguf_get_n_kv`, `gguf_find_key`, `gguf_get_val_*`, `gguf_get_arr_*`, `gguf_get_n_tensors`, `gguf_get_tensor_{name,ne,type,offset,size}`, `gguf_free`. The writer half (`gguf_set_*`) is not needed and can be dropped. |
| Quant reference | `ggml/src/ggml-quants.c/h`, `ggml/src/ggml-common.h` | Only the block structs (`block_q4_0/1`, `block_q8_0`, `block_q4_K/5_K/6_K`) and the `dequantize_row_q*` functions — the slice needed to turn q5/q6/float tensors into q4_1 on the fallback path. The quantizers and the SIMD kernels are not needed. |
| Format + conventions | `gguf-py/gguf/` (reader/constants/quants/vocab), `convert_hf_to_gguf.py`, `src/llama-model.cpp` | Reference for tensor-name and metadata semantics, and for `gguf-py/gguf/vocab.py` (GGUF tokenizer → `tokenizer.json`) used by the provisioning step. Not vendored; `q4nx-build` already depends on `gguf` (the PyPI package IS gguf-py). |

### License

- `gguf.h` / `gguf.cpp` and `ggml-quants.c` are MIT, with **no third-party
  copyright notices** in those files (only "reference implementation for
  deterministic creation of model files" comments; the k-quants/IQ code is
  ggml-authored MIT).
- `gguf.cpp` is not strictly self-contained: it includes `ggml.h` (for the
  `ggml_type` enum), `ggml-backend.h` and `ggml-impl.h`. The parser path uses
  nothing from `ggml-backend` — only the `ggml_type` enum and four size helpers
  (`ggml_type_size`, `ggml_blck_size`, `ggml_row_size`, `ggml_nbytes`). We
  therefore vendor a **trimmed** copy plus a small shim for those helpers,
  rather than all of ggml.
- Requirement on us: retain the MIT copyright + permission notice at the top of
  every vendored file, keep a `third_party/ggml-gguf/README.md` that names the
  upstream source and commit, and note the attribution in the repo's license
  section. No code written here is a reimplementation that avoids the license —
  we *want* llama.cpp's reader, and MIT permits it so long as the notice rides
  along.

## The seam

The engine already reads a weight container through one class —
`src/open_qwen36/q4nx_file.hpp` — and `pools.cpp`'s ops call it to fetch
tensors by name and by `chunk_bytes`. The GGUF reader slots in beside it:

- `Q4nxFile` grows a sibling `GgufFile` (or a common `WeightFile` interface) that
  answers the same questions: `has(name)`, `raw(name)`, `bf16(name)`,
  `f32(name)`, `bf16_row(name)` (embedding lookup), `chunk_bytes(name)`.
- The chunk ops in `pools.cpp` gain a fourth source form: when the source is a
  GGUF tensor, its quant type selects one of the transcodes below, then the
  existing permutation (`std_perm`, `expert_stripes`, …) runs on the resulting
  q4_1/q8 chunks exactly as it does today.

### The transcode (ggml block → pool chunk)

A ggml tensor's blocks are laid out row-major over 32-value blocks along the
last (K) dim, which is the same raster as OFLM's 32×256 chunks (chunk f = rows
`32*(f//ncol)` × cols `256*(f%ncol)`). So the mapping is a per-tile transcode,
no chunk-index law changes — the same guarantee the q8 and Q4_K sources already
give (OPEN-PACK-PLAN / OPEN-QUANT-Q4K).

| GGUF type | → pool form | transcode |
|---|---|---|
| `Q4_1` (20 B / 32 vals: fp16 d, fp16 m, 16 B nibbles) | q4_1 chunk (5120 B) | d/m fp16→bf16; re-pack 32 nibbles/block into OFLM's 16-lane interleave |
| `Q4_0` (18 B: fp16 d, 16 B nibbles) | q4_1 chunk | m = bf16(−8·d), d fp16→bf16; same nibble re-pack (the q4_0→q4_1 substitution OPEN-QUANT-FORMAT proposes) |
| `Q8_0` (34 B: fp16 d, 32 int8) | q8 chunk (8704 B) | d fp16→bf16, codes verbatim |
| `Q4_K` (144 B / 256 vals: fp16 d + dmin, 12 B 6-bit scales/mins, 128 B nibbles) | q4_1 chunk | fold S·scales → bf16 d and M·mins → bf16 m (the OPEN-QUANT-Q4K fold), nibble re-pack |
| `Q5_K` / `Q6_K` / `F16` / `F32` / `BF16` | q4_1 chunk | `dequantize_row_q*` → float32 tile → `requant_q4_1` (existing arithmetic) |

The nibble re-pack is the one genuinely new bit of bit-shuffling, and it is the
same shape as `q4k_to_q4_1_chunks`'s de-interleave, only from ggml's 2-nibbles-per-byte
(low nibble = even value) layout instead of Q4_K's 16-contiguous-bytes layout.

The `lm_head` (and any `q8_perm` role) is read through the same table: a `Q8_0`
`output.weight` becomes a q8 chunk; a `Q4_K` or quantized head is re-quantized to
q4_1 for the q4 head kernels, or refused where the kernel set demands q8 — the
existing "container must agree with the kernel set" check, now stated over GGUF
types instead of chunk byte counts.

## Provisioning: where `config.json` and the kernel set come from

The engine still needs two things it does not get from a bare `.gguf`:

1. **`config.json`** — `hf_config_check` in the manifest is a `config.json`-keyed
   check, and the app's model classes read `config.json` for preprocessing. A
   GGUF model has no `config.json`; its metadata is in the file. So the first
   load / `oflm-add` step writes one: `ModelSpec.from_gguf_metadata` (already in
   `spec.py`) derives the spec, and a small writer emits the equivalent
   `config.json` beside the `.gguf`. This is Python, not C++, and lives in
   `oflm-add` / the provisioning path, not in the engine.
2. **The kernel set** — built from the `ModelSpec`, exactly as today, and linked
   into `<model dir>/open_kernels` by `spec_hash` (OPEN-ADD-KERNEL-LINK). No
   change.

The engine's own GGUF read is therefore only about weights: it reads the
`manifest.json` beside the kernels (unchanged), the `config.json` beside the
`.gguf` (unchanged, just now generated from the GGUF), and the weights from
`model.gguf` instead of `model.q4nx`.

## The engine-side selection

`Core::load_weights` opens `model.q4nx` today (`core.cpp:82`). It becomes:
look for `model.gguf` first, then `model.q4nx` (legacy), refusing a model dir
that has neither. The `Engine`/`Core` otherwise touch nothing: the packer
(`pools.cpp`) and the kernels are identical either way, because the transcodes
above produce the same pool bytes.

## New requirement

### OPEN-GGUF-RUNTIME: the engine reads a GGUF container

**Applies to:** openflowlm-next (`src/open_qwen36/gguf_file.{hpp,cpp}`,
`pools.cpp`, `core.cpp`, `third_party/ggml-gguf/`, `utilities/oflm-add`)

**Test category:** unit (reader, transcodes, both sides of the NumPy/C++ line)
+ manual (the hardware run: NPU + a GGUF whose `.q4nx` twin already runs)

The engine shall load weights from a `.gguf` beside the model's `config.json`,
deriving each tensor's role and source quant type from the GGUF itself, and pack
the pool through the same manifest plan as the `.q4nx` path. A GGUF whose
metadata disagrees with the manifest's `hf_config_check` is refused at engine
construction, naming the key, exactly as a disagreeing `config.json` is today.

**Acceptance criteria:**

- **Byte equality with the `.q4nx` path is the anchor.** For each family, a GGUF
  converted to `.q4nx` by `q4nx-build` (the ground-truth weights) packs to the
  *same pool bytes* through the GGUF reader as through the `.q4nx` reader —
  FNV-1a agreement per pool in NumPy and C++ (`tests/test_gguf_runtime.py`,
  `src/open_qwen36/pools_test.cpp`), just as `q8`/`Q4_K` sources are held to
  today.
- A synthetic ggml `Q4_1` tensor read through the reader equals the same
  `(d, m, nibble)` triples packed from the equivalent `.q4nx` chunk, element for
  element; the fp16→bf16 conversion of d/m differs only where the two formats'
  scale precision differs (≤ a bf16 half-ulp, as `q4k_to_q4_1` documents).
- `Q4_0` maps to `m = −8·d` (no rounding: ×8 is an exponent shift), matching the
  OPEN-QUANT-FORMAT substitution.
- `Q8_0` reads as q8 with verbatim codes; a `q8_perm` role over a non-Q8_0 GGUF
  tensor is refused naming the tensor and its GGUF type.
- `Q4_K` folds to q4_1 via the same S·scales / M·mins products
  `q4k_to_q4_1_chunks` implements, from ggml's 6-bit scale packing
  (`get_scale_min_k4`).
- A q5/q6/float tensor falls back to `dequantize_row_*` → `requant_q4_1` and
  lands in the same chunk positions a q4_1 tensor of the same shape would.
- An unknown GGUF quant type, or a tensor whose name maps to no role, is refused
  naming it — never silently skipped.
- `gguf_init_from_file` on a truncated / wrong-magic / v1 file fails loudly with
  the llama.cpp error, not a crash.
- The reader classifies without touching weight bytes beyond what packing
  touches (header + metadata are read; tensors are read lazily like the `.q4nx`
  mmap), and `drop_pages()`-equivalent behaviour keeps a 22 GB GGUF from
  pinning the whole working set after packing.

**Procedure (manual):** for each family the open engine runs, take the GGUF the
shipped `.q4nx` was converted *from*, build its kernel set from
`ModelSpec.from_gguf_metadata` (or reuse the shipped set — same spec hash), and
run the existing slice + `chat.py` + `oflm-test --llm` procedure with
`model.gguf` in place of `model.q4nx`. Logits corr ≥ 0.99999, same argmax and
top-5, engine bit-identical to the `.q4nx` run. Details to a `.claude/plans/`
log as the other families do.

## Changes

| file | change |
|---|---|
| `third_party/ggml-gguf/` (new) | vendored `gguf.h` + `gguf.cpp` (reader half), `ggml-quants.h` + the `dequantize_row_q*` slice of `ggml-quants.c`, a `ggml_types.h` shim for the `ggml_type` enum + 4 size helpers, and a `README.md` naming the upstream source/commit and the MIT notice. |
| `src/open_qwen36/gguf_file.hpp/.cpp` (new) | `GgufFile`: the `Q4nxFile`-shaped interface over `gguf_init_from_file` — `has/raw/bf16/f32/bf16_row/chunk_bytes`, plus a `quant(name)` returning the GGUF type, and the ggml-block→chunk transcodes (`ggml_to_q4_1_chunks`, `ggml_to_q8_chunks`). |
| `src/open_qwen36/pools.cpp` | the three q4 chunk ops + `q8_perm`/`lmhead_q8` accept a GGUF source: when the file is a `GgufFile`, transcode by quant type, then apply the existing permutation. `chunk_bytes(name)` for a GGUF reports the *pool* chunk size (5120 / 8704) after the transcode. |
| `src/open_qwen36/core.cpp` | `load_weights` prefers `model.gguf`, falls back to `model.q4nx`. |
| `open_kernels/model/gguf.py` (new) | the NumPy mirror of the transcodes + a `Gguf` reader for the fp64 replica / slice comparison, so the oracle scores the GGUF path (it must agree with the device path). |
| `utilities/oflm-add` | write `config.json` from `ModelSpec.from_gguf_metadata` when a model dir has only a `.gguf`; extract `tokenizer.json` via `gguf-py/gguf/vocab.py` when absent; link the kernel set by `spec_hash` (unchanged). |
| `specs/open-engine/tests/test_gguf_runtime.py` (new) | the unit criteria above; FNV-1a agreement with `pools_test.cpp`. |
| `src/open_qwen36/pools_test.cpp`, `CMakeLists.txt` (both levels) | the C++ half of the agreement; add the vendored sources to the build. |
| `specs/open-engine/spec.md` | `OPEN-GGUF-RUNTIME`; a GGUF sentence in `OPEN-PACK-PLAN`'s "source forms" paragraph. |

Roughly: ~200 lines of vendored-but-trimmed reader, ~250 lines of transcode +
`GgufFile`, the existing `requant_q4_1`/`q4k_to_q4_1` functions shared, plus the
provisioning writer.

## Order (TDD)

1. `third_party/ggml-gguf/` vendor + shim; a tiny `gguf_file` unit that opens the
   checked-in GGUF fixture and asserts metadata/tensor counts. Confirm the
   parser builds with no ggml beyond the shim.
2. `tests/test_gguf_runtime.py`: write the transcode criteria (q4_1/q4_0/q8_0/
   q4_K + fallback) against synthetic ggml bytes, confirm they fail for the
   right reason (no `GgufFile`, no transcode).
3. `open_kernels/model/gguf.py`: the NumPy reader + transcodes; the Python
   criteria go green, and `test_pack_plan.py`'s frozen pools stay untouched.
4. `gguf_file.cpp` + `pools.cpp`: the C++ transcode; `pools_test.cpp` FNV-1a
   agreement goes green. `test_pack_plan.py` must still pass unmodified.
5. `core.cpp` selection (`model.gguf` first). `oflm-add` provisioning writer.
6. Manual hardware run per family; record in a `.claude/plans/` log, add
   `OPEN-GGUF-RUNTIME` to `spec.md`, archive this plan.

## What this does NOT do

- **Write `.q4nx`.** The runtime reader produces pool bytes, never a container.
- **Native q4_0 kernels.** q4_0 reads through the q4_1 substitution
  (`m = −8·d`), the OPEN-QUANT-FORMAT change; no new kernel.
- **Native q5/q6 GEMVs.** Those stay on the dequantize-then-requantize fallback
  forever unless a real evaluation says otherwise.
- **Vision GGUF (mmproj) at runtime.** The vision tower still reads
  `vision_weight.q4nx`; the VLM families keep using the converter for the tower
  until this path is proven for text. Images are out of scope for the first cut.
- **Replacing `ModelSpec.from_gguf_metadata`** — it stays in `spec.py` (Python),
  used by kernel-set generation and the `config.json` writer. The engine does not
  re-derive the spec from GGUF metadata at runtime; it reads the manifest and
  the generated `config.json`, as it does today.
- **Model-family reorders that differ between GGUF and the current `.q4nx`.** If a
  family's `.q4nx` carries a reorder the GGUF does not (or vice versa), that is a
  per-family finding recorded under the byte-equality acceptance criterion — the
  transcode seam assumes "same raster, different quant block", which the
  byte-equality test will falsify if it is not true for a family. The reference
  for what a GGUF needs is llama.cpp's `src/llama-model.cpp` loaders.

## Alternatives considered

- **Transparent conversion (GGUF in, `.q4nx` written on first load).** Much
  smaller, reuses `q4nx-build` whole. Rejected: it keeps the AMD format in the
  hot path and does not deliver "q4nx is legacy".
- **Reimplement the GGUF parser from the format spec.** No license question at
  all (~200 lines, the layout is documented in `gguf.h`). Rejected for now:
  we want llama.cpp's battle-tested parser; reimplementing is the fallback if
  trimming `gguf.cpp`'s ggml dependency proves messy.
- **Parse GGUF in Python and emit a container descriptor.** Keeps C++ minimal but
  reintroduces a build-time/per-model step and two sources of truth for weights.
  Rejected in favour of a single C++ reader.
- **Support every ggml quant type natively.** Not needed: q5/q6/float cover the
  tail through a fallback the packer already has, and the families OFLM ships
  are q4_1/q8/Q4_K in practice.
