---
name: open-gguf-direct
description: Direct GGUF loading for the open engines (supersedes q4nx for compatible quants). Use when installing a GGUF model, debugging a GGUF refusal, rebuilding the GGUF manifest view, extending GGUF support to a family, or checking quant/pool numerics.
---

# GGUF-direct

A GGUF is a first-class model source. `oflm add <repo-or-file>` installs one
compatible `.gguf` as `model.gguf` beside the tokenizer files and links a
kernel set whose manifest carries a `gguf` view. There is no q4nx conversion,
no manual pack, and no separate model entry for a quantized upload.

The front door is the same one as every other model: read
`open-kernel-pipeline` first. This file is the GGUF-specific half: which file
is chosen, how it is packed, what the engine swaps, and where the traps are.

## The install, in one command

```bash
oflm add Qwen/Qwen3.5-4B-GGUF
oflm add mradermacher/Some-Model-i1-GGUF:Q4_K_M
oflm add ./local-model.gguf
oflm add <repo> --dry-run
```

What the installer does:

1. **The file is the authority.** The GGUF header names the architecture,
   widths, rope law, vocabulary and tensor types. `config.json` is a hint for
   what the file cannot carry (tokenizer, chat template, vision tower). Peeking
   the config first is what made `translategemma-4b-it` look like a different
   geometry: its config omits `rope_scaling`, while the GGUF says factor 8.0.
2. **Picks the file `oflm add` would install.** `choose_gguf_file()` uses the
   quant preference below, refuses multi-part files and unsupported quants
   with a reason, and verifies the chosen file's tensor types against the
   header. A local file is verified from disk (no network); a cached file is
   verified from the HF cache.
3. **Derives the family from the GGUF architecture**, then the base chain, then
   the kernel set. `--family` is a last-resort override.
4. **Finds the set by geometry**, requiring the manifest's `gguf` section. If
   no set serves the size and a checkout is present, it builds it with
   `utilities/build-new-spec.py` — the same one-command path as a q4nx model.
5. **Links the set** at `<model>/open_kernels` and writes the model entry to
   `settings.json`.
6. **Packs on first use.** Chat GGUF streams its pools directly; an embedding
   GGUF is staged into a checkpoint and packed to `.npue` on first load (see
   below).

A repo that ships both `model.q4nx` and a GGUF installs the q4nx: it is the
curated container, and the GGUF is a lossy re-encoding of the same weights.

## Which quants are accepted

`GGUF_QUANT_PREFERENCE = ["Q4_1", "Q4_0", "Q6_K", "Q8_0", "Q5_K", "Q4_K", "IQ4"]`

The pool the kernels stream is q4_1, so the source format decides how many
quantization steps the weights take on the way in:

- **Q4_1** — packed near-exactly (the file's fp16 `d`/`m` widen to f32/bf16).
- **Q4_0** — re-encoded exactly (the implicit −8 becomes q4_1's min).
- **Q6_K / Q8_0 / Q5_K / Q4_K / IQ4** — dequantized and re-quantized to q4_1,
  so the source's error is added to q4_1's. The same Qwen3.5-4B as Q4_K_M
  measured cos 0.75 against 0.993 for Q4_1: still coherent, but two lossy
  steps where one would do.

Anything else (F16, BF16, Q2_K, Q3_K, Q5_0, `*_4_4` layouts, …) is refused at
install with the quant named. `:QUANT` on the repo argument pins one file; the
quant must run to the end of the stem, because `Q4_0_4_4` is a different
tensor layout, not Q4_0.

For **NpuEmbeddings** GGUF (`bert` / `nomic-bert`) the preference is
`NPUE_GGUF_PREFERENCE = ["Q8_0", "F16", "BF16", "F32", "Q6_K", "Q5_K", "Q4_K",
"Q4_1", "Q4_0", "IQ4"]`. Those packers dequantize every tensor to F32, so
source precision is all that matters; Q8_0 first (half the bytes of F16 for
the same result). Measured against the author's safetensors: Q8_0 0.9996,
Q4_0 0.966 (MiniLM) / 0.91 (nomic).

## The pool law

The manifest's `gguf` section is the q4_1 program's own view, with every
projection op in its GGUF-reading form (`std_perm_gguf`). The engine reads
`model.gguf` directly and packs the pools from it.

**The default is the q4_1 twin, not the f32 twin.** Both sections name the same
chunk law and kernels; the difference is the pack op. The engine merges the
GGUF section's pack ops onto the base section's geometry (same tensors, same
chunk counts, only the byte offsets move) and swaps the layer kernels, the
head and the globals. That is what every shipped q4nx container runs, so a GGUF
on it is numerically the same model those containers serve.

`OFLM_GGUF_POOL_F32=1` opts back into the f32-scale twin, the probe that
isolates f32 chunk handling. The twin packs the same 32×256 tiles but widens
the GGUF block scales/mins **exactly** fp16 → f32 (mantissa <<13, exponent
+112; lossless), with q4 chunk 6144 B and q8 head chunk 9216 B. It is not the
default because it is not reliable on every family: with weights that pack
byte-identically to a container's chunks, the f32 twin turns layer 0 into
all-NaN for gemma3-4b and Nanbeige, while the q4_1 twin reproduces the
container's per-layer numbers to ~4 significant figures. Qwen3 and Llama 3.2
work on either.

Do **not** skip the `gguf` section even when the chunk law matches the base.
Its pack ops are the GGUF-reading ones; without it the packer runs `std_perm`
and dies with "lm_head.weight is not in a q4nx container".

`OFLM_GGUF_POOL_F32=1` is also not a route for hybrid families' block prefill:
Qwen3.5's linear-attention layers hit "block prefill is not implemented for the
'linear' layer kind". The error names the layer kind; it is not an MoE problem.

## The manifest and the engine

- A set is GGUF-capable when its `manifest.json` has a `gguf` section
  (`manifest_supports_gguf()`). The gate is the manifest, never a family-name
  list: the registry's family (`qwen3.5`) and the recipe's (`qwen35`) are
  different strings for the same model, and a list in one namespace silently
  refuses the other.
- The engine swaps the whole manifest view when the model directory has
  `model.gguf`; a `model.q4nx` wins if both exist.
- `Manifest::files()` dedupes names across the main and GGUF views, because the
  two views describe the same build and the same files.
- `config.json` is optional for a GGUF: the engine derives the checked fields
  from GGUF KV (`derive_config`). Fields the GGUF cannot carry (Granite's
  folded multipliers, for example) still need a config.json, and are refused
  by name when absent.
- The parser is `src/gguf/gguf_file.hpp` / `.cpp` (shared). There is no
  `src/open_qwen36/gguf_file.hpp` in this tree; that path belongs to older
  branches. `GgufFile` also provides `f32_all()` for whole-tensor dequant and
  `kv_str_array()` for metadata arrays.

## Tensor names and pack ops

`GgufFile::gguf_name()` maps HF-style manifest names
(`model.layers.N.self_attn.q_proj.weight`) to llama.cpp names
(`blk.N.attn_q.weight`). `pools.cpp`'s `std_perm_gguf` packs Q4_0/Q4_1
byte-exactly and requantizes Q8_0 / Q4_K / Q6_K to the pool's q4 layout.
Everything else is refused at pack time, pointing at `q4nx-build`.
`put` / `pack_norm` convert f32/f16 small weights to the bf16 consts blob.

## Embedding GGUF (two different paths)

- **`embed-gemma`** (`EMBEDDING_FAMILIES`) loads through the existing bf16
  matmul kernels; the engine reads `model.gguf` and dequantizes. It needs no
  `gguf` manifest section. Its design set is
  `Embedding-Gemma-300M-OpenNPU2`.
- **NpuEmbeddings** (`bge`, `nomic`, `minilm`, `gte`) is not a GGUF-direct q4_1
  model. `oflm add` installs the GGUF, and on first load
  `src/open_npue_adapter/npue_embedding.cpp` stages a checkpoint from it in
  `<dir>/.npue_gguf`, calls `npue::prepare_model_auto()`, and writes
  `<dir>/<dir-name>.npue`. The staging directory is removed afterwards. The
  packers still read a HuggingFace checkpoint; the adapter is the fork-owned
  bridge, and nothing under `src/open_npue/` is modified.
- `gte-multilingual` has no llama.cpp GGUF; it installs as safetensors.
- A repo that ships **both** safetensors and a GGUF keeps the safetensors:
  that is the packers' native input, and the GGUF is a lossy re-encoding.
- Nomic declares named task prompts, so the server requires `prompt_name`
  (e.g. `search_document`) and returns HTTP 400 without one.

Verify an embedding install with `utilities/gguf-embed-check.py` and
`oflm-test --embedding`. Measured cosines against the author's safetensors:
bge-base Q8_0 0.99962, nomic Q8_0 0.99813, MiniLM Q8_0 0.99946.

## Vision and diffusion

- A VLM GGUF installs its text weights as `model.gguf` and fetches the vision
  projector (`*mmproj*.gguf`) when the repo or its base carries one.
  `--text-only` refuses images instead, which is the same path a text-only
  checkpoint takes.
- A diffusion GGUF (FLUX.2 klein) takes its own install path and its own
  kernel set; see `open-diffusion` / `open-diffusion-edits`.

## Numerics and verification

The GGUF numeric gate is `open_kernels/model/gguf_ref.py` plus
`replica_prompt.py --model-dir <gguf install> [--ids ...]`: a GGUF *tensor*
reader (F32/F16/Q4_0/Q4_1/Q8_0/Q4_K/Q5_K/Q6_K) with the HF-name map, so the
fp64 reference runs on the file itself. Compare per-layer outputs with
`OFLM_OPEN_DUMP_POOLS` dumps keyed by `dst` offset when a layer disagrees.

For a whole install: `oflm check <tag>`, then `oflm-test --llm --model <tag>`
(or `--embedding`). `oflm-test` currently checks `finish_reason` and length;
read the generated text too.

## Fixed GGUF pack bugs (do not reintroduce)

All three were qwen35-only, which is why only qwen35 was broken:

1. `conv_transpose` read `ssm_conv1d` as `[taps][channels]`; a GGUF stores
   `[channels][taps]`.
2. `lmhead_q8`'s GGUF branch wrote the file's raster instead of the pool's
   supertile raster (logits cos 0.33 with hidden states at 0.998).
3. The fused q|gate is per head in a GGUF (`[q_h|gate_h|...]`) versus
   `[all q|all gate]` in the pool. `qgate_rows` / `qgate_gate` on the op carry
   the third, emitted only for a GGUF source. It is a manifest change, not a
   kernel change.

## Merge and build state

- The shared parser is `src/gguf/gguf_file.hpp`; `src/open_qwen36/gguf_file.hpp`
  is from older branches and does not exist here.
- The `rope_il` (llama3 interleaved RoPE) and `kv_pair` (paired-BF16 KV)
  features are both merged; a GGUF twin carries whichever the family needs.
- MoE GGUF-direct is ready: `GGUF_EXPERT_OPS_READY = True`, and
  `expert_stripes_gguf` / `expert_down_gguf` read the expert tensors from the
  GGUF source.
- `GGUF_DIRECT_UNREADY` is empty. A family goes back on that list only when its
  text is measured wrong; an install that loads and emits fluent gibberish is
  worse than a refusal.
- The canonical build is `cmake --workflow --preset linux-package`; it exports
  every spec's GGUF twin and packages the sets. `python
  utilities/export-kernels.py` is the kernel-only path.
