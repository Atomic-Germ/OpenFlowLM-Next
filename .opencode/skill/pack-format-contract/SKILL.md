---
name: pack-format-contract
description: The contract between `q4nx-build` / `oflm pack` and the engines + open kernels that read what it writes. Use when adding a model family, a weights format or an open pipeline, when a packed model refuses to load or `oflm add` says "no installed open kernel set has spec_hash ...", or when a pack produces a directory a reader rejects.
---

# The pack ↔ reader contract

`oflm pack` is a thin handoff to the bundled `q4nx-build`
(`src/pull/pack_command.cpp` execs it, argv verbatim). So **every format question
about `oflm pack` is a question about `utilities/q4nx-build`**, and every answer
is on one of these two sides:

| writer | reader |
|---|---|
| `utilities/q4nx-build/q4nx/` (`cli.py`, `model_assets.py`, `model_converter.py`, `models/*.py`, `open_*.py`, `deploy.py`, `arch_detect.py`) | `src/open_*`, `src/common/AutoModel`, `src/include/lm_config.hpp` |
| `open_kernels/recipes/load.py` (`spec_from_model_dir`, `tokenizer_vocab`, `container_chunk_bytes`) | `open_kernels/recipes/spec.py`, `families.py`, `catalogue.py` |
| `utilities/dit-chain/export_bundle.py`, `open_kernels/export_dit_kernels.py` | `src/open_diffusion/engine.cpp` |
| `open_kernels/export_whisper_kernels.py` | `src/open_whisper/*` |

The two halves have no shared test, which is why they drift: `utilities/q4nx-build/tests/`
and `specs/open-engine/tests/` each pass while the packed artifact does not load.

## The five places to check when adding anything

1. **`config.json`'s `model_type`** must be in `open_kernels/recipes/spec.py`'s
   `HF_FAMILIES` (for `spec_from_model_dir`) *and* in
   `src/include/AutoModel/model_families.hpp`'s `modelFamilyMap` (for the closed
   engine) *and* in `model_assets._OFLM_FAMILY_OF_MODEL_TYPE` (for the README tag).
   Three lists, one value. `qwen3_6_moe` was missing from the first,
   `k2_horizon` from the third.
2. **The container's chunk size** must be in `spec.py`'s `CHUNK_FORMAT` or
   `AMBIGUOUS_CHUNK`. A byte count the deriver does not know is a `SpecError`,
   which means `oflm add` will never find that model's kernel set.
3. **The keys the recipes `_need`** (`rope_theta`, `head_dim`, ...) must be in the
   config the packer writes — including `generate_config_from_gguf`, the no-source
   fallback, which used to emit llama.cpp's `general.architecture` string as
   `model_type`, drop `rope_theta` and pop `head_dim`.
4. **The files `oflm add`'s `REQUIRED_FILES` needs** must exist in the output
   directory, for every entry point (`assemble_model_assets` *and*
   `assemble_model_assets_hf`, `open_embedding`, `open_causal`,
   `open_diffusion`, `open_whisper`).
5. **The default list** of anything both sides emit (FLUX.2's sizes, edits) must
   have ONE definition, read by both.

## Does a kernel change force a container rebuild? (No, with one exception)

A kernel set is DERIVED from a container, never the other way round. `oflm add`
computes `spec_hash` from the container alone -- `config.json`, `tokenizer.json`,
and the `model.q4nx` safetensors HEADER only, no weight bytes -- and matches it
against the `spec_hash` in an installed `manifest.json`. So:

- **A kernel rebuild is a drop-in replacement.** Same derived hash, same link, the
  engine finds it. `model.q4nx` is the input and its bytes are not consulted by
  the kernel build, so a re-pack produces the same file anyway.
- **Re-packing does not strand a kernel set**, as long as the shape and the
  per-role weight FORMAT stay put. That is the one thing to watch: change a
  family's `default_tensor_type` (or the `*_Q8_0_NAMES` pins in
  `models/qwen35.py`) and the container derives a different spec, so the old set
  no longer matches -- and `ofml add` goes back to closed kernels.
- **A kernel set cannot serve a container of a different spec.**
  `pools.cpp`'s `q8_perm` REFUSES a non-q8 source, and `test_pack_plan` proves the
  reverse, so the check is load-time and loud in both directions. The set must
  match the container.
- **FLUX.2 klein is the exception.** `bundle.json` carries a layout hash that ties
  the *packed* GEMM/VAE weights to a kernel set and the engine refuses any other,
  so a kernel-side layout change DOES require re-running
  `q4nx-build --open-diffusion`. That is why `open_diffusion.klein_sizes()` and
  `export_dit_kernels.py` both read `klein_pipeline.SIZES/EDIT_SIZES`.

The export cache is per model directory (`src/xclbins/<extra.model>/open_kernels`),
so two shape-identical models build independently even when their build directory
NAMES coincide.

### Finetune twins share one set; they do not duplicate it

`oflm add` links `<model dir>/open_kernels` -> the chosen kernel set with a
symlink (`_make_dir_link`), so the install is never a copy. A finetune whose
derived `spec_hash` EQUALS its twin's links to the twin's set and there is no
second build at all -- one spec, one set, N symlinks. Measured on the installed
containers:

| spec | models sharing it |
|---|---|
| `gemma3-4b` | Gemma3-4B, Medgemma-4B, Medgemma-1.5-4B, Translategemma-4B |
| `qwen3-4b` | Qwen3-4B, Qwen3-4B-Thinking-2507 |
| `qwen25-3b` | Qwen2.5-3B, Qwen2.5-VL-3B |
| `llama31-8b` | Llama-3.1-8B, Deepseek-R1-Distill-Llama-8B |
| `lfm2-1.2b` | LFM2-1.2B, LFM2.5-1.2B |

`find_open_kernels` matches on the manifest's `spec_hash` EXACTLY, and a set under
the model's own directory name wins over another root's match. No match prints
the export command rather than staying silent
(`test_open_kernels_link.py::test_no_match_prints_the_export_command`), and
`--open-kernels <dir>` takes a set as given without any hash check
(`test_override_takes_the_directory_as_given`).

**The trap for user-packed finetunes: `real_vocab` is in the hash.** A fine-tune
that adds vocabulary tokens links to NOTHING, because its tokenizer's id count
moved -- LFM2.5-1.2B-Thinking is exactly this (64402 ids against the 1.2B's 64400),
which is the only reason it has its own spec file and therefore a second build of
byte-identical kernels. `oflm add --dry-run` prints the `spec hash` to compare
against the twin's set; `--open-kernels` links it by hand when they differ.

## Verify (run this before and after any change)

```bash
source /opt/xilinx/xrt/setup.sh
source ironvenv/bin/activate

cd utilities/q4nx-build && python -m pytest tests -q          # the packer
cd specs/open-engine  && python -m pytest tests -q             # the recipes
cd specs/open-diffusion && python -m pytest tests -q
cd utilities/oflm-add && python -m pytest tests -q
cd utilities/oflm-test && python -m pytest tests -q
cd build && cmake --build . --target oflm -j8                  # the engines
```

`q4nx-build`'s tests need the *system* python (gguf / torch / safetensors);
`specs/*`'s need `ironvenv`. Two failures are pre-existing on a machine without
`libubsan.so` or on Linux:

- `specs/open-engine/tests/test_qwen35_wide_glue.py::test_small_bank_pointer_arithmetic_in_compiled_cpp`
- `utilities/oflm-add/tests/test_system_registry.py::test_the_xclbin_link_falls_back_to_a_junction`

### Derive from the real things

The strongest check is against installed models, not fixtures — `~/.config/oflm/models`
holds containers and `/opt/openflowlm/share/oflm/xclbins/*/open_kernels/manifest.json`
holds the kernel sets:

```python
import sys; sys.path.insert(0, "open_kernels")
from recipes.load import spec_from_model_dir
from pathlib import Path
for d in sorted(Path.home().joinpath(".config/oflm/models").iterdir()):
    if not (d / "model.q4nx").is_file(): continue
    try:
        s = spec_from_model_dir(d)
        print(d.name, s.family, s.spec_hash()[:22], s.quant)
    except Exception as e:
        print(d.name, type(e).__name__, e)
```

Every printed `spec_hash` must equal the `spec_hash` in the matching installed
`manifest.json`. A mismatch is a container no `oflm add` can serve with open
kernels, and it comes from the pack ⇄ recipe drift above.

## Fixed 2026-10-10 (see `test_packed_format_contract.py`, `test_build_spec_wiring.py`, `test_pack_recipe_contract.py`)

| drift | fix |
|---|---|
| Granite's folded multipliers were written only on `assemble_model_assets_hf`, a path Granite (GGUF-only) never reaches → every Granite config shipped UNFOLDED, `spec.py`/`dense.py` refused it | fold in `assemble_model_assets`; the HF branch's call (which named a non-existent `reader`) is gone |
| `--quant Q4_K`'s 4736-byte chunks were not in `CHUNK_FORMAT` → those containers never derived a spec | `4736` maps to `q4_1`, because `pack.q4k_to_q4_1` transcodes into the pool; naming it a distinct format would split `spec_hash` for byte-identical kernels while `quant_hash()` still built into the same directory |
| `generate_config_from_gguf` emitted llama.cpp arch strings as `model_type`, no `rope_theta`, no `head_dim` | `GGUF_ARCH_TO_MODEL_TYPE` + both keys |
| GGUF-with-no-source `tokenizer.json`'s Unigram `model.vocab` is a list; `.values()` raised `AttributeError` out of `oflm pack --build-spec` and `oflm add` | `tokenizer_vocab` handles both shapes |
| `assemble_model_assets_hf` wrote no `tokenizer_config.json` when the source shipped none → `oflm add` rejected the directory | `synthesize_hf_tokenizer_config` |
| `--build-spec` / `-t` were missing from the recorded pack command; `spec.json` was not deployed or recognised | recorded in `_packed_command`, added to `deploy.MODEL_FILES` and `oflm_add.ALL_FILES` |
| `qwen3_6_moe(_text)` accepted by the packer, no recipe | aliased into `HF_FAMILIES` |
| `ARCH_TO_FAMILY` had no granite/k2/hunyuan; `FAMILY_ALIASES` no k2/hunyuan | both filled; `derive_family` no longer exits before the open-kernel link |
| open embedding recorded no dtype and read every tensor as f32 | manifest carries it; the engine widens BF16 and refuses a dtype it cannot widen |
| FLUX.2: packer always emitted edit schedules, exporter defaulted to no edits | one `klein_pipeline.SIZES/EDIT_SIZES`; the engine refuses at load a configuration the model advertises and the set has no ELF for |
| `inject_oflm_keys` wrote `eos_token_id = 248044` into every family | only the Qwen3.5/3.6 model types it is correct for |
| `_build_spec` raised out of `oflm pack` on a container the recipes refuse | prints the reason, writes no spec.json, leaves the pack standing |

## Open, needs a decision

### RESOLVED 2026-10-10: the specs were the wrong side, and they moved

The installed containers are the ground truth `src/model_list.json` is written
against, so the four stale specs were corrected against them. The kernel sets now
need a rebuild from the corrected specs, which this machine can do.

| spec | was | now |
|---|---|---|
| `qwen35-9b.json` | `quant: "q4_1"` | `{"linear_out": "q8"}` |
| `qwen36-35b-a3b.json` | `quant: "q4_1"` | `{"attn","linear","linear_out": "q8"}` |
| `qwen35-27b.json` | `real_vocab: 248320` | `248077` |
| `phi4-mini-4b.json` | `real_vocab: 200064` | `200029` |

Plus `gemma3_text_only` aliased into `HF_FAMILIES`: Gemma3-1B-NPU2 declares it and
`_OFLM_FAMILY_OF_MODEL_TYPE` already answered for it, so the recipe refusing it was
one-sided drift.

**Why it mattered beyond the hash.** `pools.cpp`'s `q8_perm` refuses a container
whose tensor is not 8704-byte q8, and `test_pack_plan` proves the reverse. The
shipped sets were built from q4_1 specs, so their `std_perm` read the containers'
q8 chunks and re-quantized them to q4_1 on the way into the pool: **correct but
silently degraded**, paying the 4-bit cost on 251 of the 35B's 733 quantized
tensors the container had already stored at q8. With the specs corrected those
projections stream at q8, and the 35B's pool region grows past the 512 MB default,
which `qwen36moe.py` documents as designed ("a q8 variant of the same shape needs
more and rounds up to the next MB").

**All four corrected specs pass `recipe()` with no `OPEN_KERNELS_UNVALIDATED`
opt-in** -- they are inside the validated catalogue, so the export is a normal one.

**The tests that encoded the stale invariant.** `test_quant_q8`'s "every checked-in
spec reads as all-q4_1", `test_quant_mixed`'s "every spec is all-q4_1",
`test_one_context`'s default, `test_qwen35` / `test_spec_derive`'s comparisons
against a bare config derivation, `test_recipe_layout`'s `LAYOUT_27B` and the
hardcoded `build_s256` / `build_lx0` / `build_qwen35_lx_h4096` directory names all
assumed the q4_1 default. They now build the all-q4_1 reference explicitly
(`dataclasses.replace(default_spec(), quant="q4_1")`), which is more honest: the
default IS the q8 one, and a q8 role is supposed to add `quant_hash`'s suffix to
the build directory. `test_lfm2.test_no_shipped_models_hash_moved` still freezes
fourteen hashes, four of which moved on purpose -- its docstring now says which and
why. `test_quant_q8.Q8_CHECKED_IN` and the `real_vocab` test beside it are the
regression guards that replace the old blanket all-q4_1 assertion.

## Which container has which spec, and who is still missing one

Derived from the 36 installed containers against the 15 checked-in specs
(`open_kernels/recipes/specs/*.json`), matched by the derived `spec_hash`:

| links to | container(s) |
|---|---|
| `qwen36-35b-a3b` | Qwen3.6-35B-A3B |
| `qwen35-9b` | Qwen3.5-9B |
| `qwen35-27b` | Qwen3.8-27B |
| `qwen3-4b` | Qwen3-4B, Qwen3-4B-Thinking-2507 |
| `qwen25-3b` | Qwen2.5-3B, Qwen2.5-VL-3B |
| `gemma3-4b` | Gemma3-4B, Medgemma-4B, Medgemma-1.5-4B, Translategemma-4B |
| `phi4-mini-4b` | Phi4-mini-Instruct |
| `llama31-8b` | Llama-3.1-8B, Deepseek-R1-Distill-Llama-8B |
| `lfm2-1.2b` | LFM2-1.2B, LFM2.5-1.2B |
| `lfm2.5-1.2b-thinking` | LFM2.5-1.2B-Thinking |

**Derivable but still missing a spec file** -- these families have recipes, so a
`_build_spec`-style derivation and a `*.json` is all they need (and each then needs
its own kernel-set export): DeepSeek-R1-0528 (qwen3), LFM2-2.6B,
LFM2-2.6B-Transcript (lfm2), Llama-3.2-1B, Llama-3.2-3B, Nanbeige4.1 (all llama3),
Qwen3-0.6B / 1.7B / 8B / 4B-Instruct-2507 / VL-4B (qwen3), Qwen3.5-0.8B / 2B / 4B
(qwen35 -- the 2B and 0.8B derive `linear_out: q8`, and both are in
`catalogue.MIXED_CORE_FITS`, so they are valid, not just valid-by-luck).

**Refused, on purpose.** GPT-OSS and Whisper are not open-recipe families;
Gemma4-{E2B,E4B,12B} have none either. Gemma3-1B stores 1280-byte chunks, which
`specs/open-engine/plans/container-formats.md`'s OPEN-QUANT-FORMAT requires
refusing by name -- the error already says so, so refusing it is correct rather
than drift. LFM2.5-1.2B-Thinking's tokenizer had 64402 ids against the 1.2B spec's
64400, so it needed its own spec file before it could link.
open-recipe families. Gemma3-1B stores 1280-byte chunks, which
`specs/open-engine/plans/container-formats.md`'s OPEN-QUANT-FORMAT requires
refusing by name -- the error already says so, so refusing it is correct rather
than drift. LFM2.5-1.2B-Thinking's tokenizer has 64402 ids against the 1.2B spec's
64400, so it needs its own spec file before it can link.
