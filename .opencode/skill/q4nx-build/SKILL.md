---
name: q4nx-build
description: How utilities/q4nx-build converts GGUF/HF-safetensors models into Q4NX containers - the converter lifecycle, the configs/*.json contract, the GGUF innermost-first shape rule, tensor storage sinks, and what adding a new family requires. Use when converting a model, debugging a container that loads but decodes garbage, adding a model family to the converter, or touching name_map / _store_q / _pack / arch detection.
---

# q4nx-build

The converter that produces `model.q4nx` containers. Derived from the code,
not from anyone's memory: every claim below carries a `file:line` you can check,
and where the code does *not* pin something down it says so.

**First, for any container that decodes garbage:** the kernels and the container
read the same bytes, so a converter bug passes every slice compare, the engine's
bit-identity check, and a prompt test, and the model still answers in fragments.
Check the container against HF before you suspect a kernel:
`open_kernels/model/container_vs_hf.py`. Then come back here.

Sibling skills: `q4nx-streaming-pack` (packing larger than RAM, MoE pruning, the
catalogue wall), `open-qwen38-27b-kernels` (the value-head scramble and the
container-vs-HF workflow), `npu-offload-pipeline` (the kernel side).

## What the tool is for

It began as `FLM_Q4NX_Convert`, which only built containers for models already
hand-coded into a supported list; there was no way to run a custom model or
fine-tune. `q4nx-build` generalized that, and most of its odd shape follows from
that history: a config-driven name map, an architecture registry, and a
speculative-packing path that refuses to crash on an architecture it was not
written for. Upstream shipped breaking format changes with no migration path
(ROCm/FastFlowLM#690), which is why this exists separately.

## The converter lifecycle

Base class `__Q4NX_Converter`, `q4nx/model_converter.py:114`. Only **one** method
is abstract: `convert()` (`:161`). Everything else is concrete and inherited.

Order of operations for one pack:

1. **Factory** — `create_converter()` (`:1375`) opens a `GGUFReader`, resolves the
   arch via `get_model_arch_from_gguf` (`:1255`), looks the class up in
   `_MODEL_REGISTRY`, instantiates. HF equivalent: `create_hf_converter` (`:1412`).
2. **Subclass `__init__`** — sets `self.gguf_reader`, `self.gguf_tensors`, then
   calls `self.initialize()`.
3. **`initialize()`** (`:153`) — reads GGUF tensors/metadata, or the HF index,
   then `_load_config()`.
4. **`_load_config()`** (`:192`) — loads `configs/<arch>.json`, maps
   `default_tensor_type` to a `GGMLQuantizationType` (**rejects anything but
   Q4_0/Q4_1/Q8_0/Q4_K**, `:202-211`), reads optional `vision_config`/`audio_config`,
   then calls `_create_name_maps()`.
5. **`_create_name_maps()`** (`:263`) — the `{bid}` expansion (below).
6. **CLI sets attributes** — `pad_to_fit`, `prune_ffn`, `prune_moe_ffn`,
   `prune_experts`, `imatrix_path_hint` (`cli.py:901-906`) are plain attributes,
   not constructor arguments. `--quant` calls `set_default_tensor_type`, which
   **re-runs `_create_name_maps`** (`:247`); a same-value `--quant` returns
   early (`:243-244`) so the maps are not built twice.
7. **`convert()`** — the subclass method. For `-t vision` the CLI calls it twice,
   language then vision (`cli.py:925-927`).
8. **`_export_weights()`** (`:439`) — plain `safetensors.torch.save_file` to
   `model.q4nx` / `vision_weight.q4nx` / `audio_weight.q4nx`.

**Ordering caveat:** the base `initialize()` reads tensors *then* loads config
(`:154-158`), but `Qwen35.initialize` (`models/qwen35.py:127`) and
`Qwen35Moe.initialize` (`models/qwen35moe.py:72`) call `_load_config` **first**.
Both orders are live in the tree; the subclasses pre-populate `self.gguf_tensors`
in `__init__` so either works. Do not assume one is canonical.

## Registration is a side effect of import

There is no `@register` decorator. `__init_subclass__` (`:147-151`) takes a
**class keyword argument**:

```python
class Qwen3(__Q4NX_Converter, model_arch=ModelArch.QWEN3):
```

and inserts it into `_MODEL_REGISTRY`. Importing the module is what registers it,
which means **`q4nx/models/__init__.py` is the only thing that populates the
registry** — a converter whose module is never imported there does not exist.

## Adding a family: the exact checklist

Follow `models/k2.py` (23 lines, the minimal example; `tests/test_k2_arch.py`
pins it).

1. `constants.py` — add the member to `ModelArch` (`:7-27`), its spellings to
   `ModelArchNames` (`:57-78`, used by `-f` and GGUF `general.architecture`
   matching), and its config file to `ModelArchConfigs` (`:80-100`).
2. `configs/<name>.json` — see the schema below.
3. `q4nx/models/<name>.py` — subclass with `model_arch=`, implement `__init__`
   and `convert`. Nothing in the base calls `super().__init__` for you.
4. `q4nx/models/__init__.py` — add the import and the `__all__` entry.
5. `arch_detect.py` — **only** if you want auto-detection without `-f`.

Then add a test that reaches the family end-to-end (registry entry, arch
resolution without `-f`, `-f` still short-circuits) the way `test_k2_arch.py`
does. That test is what catches step 4 being forgotten.

## The configs/*.json contract

19 config files for 20 arch members. Top-level keys: `q4nx_config`,
`default_tensor_type`, `name_map`, plus optional `vision_config`, `audio_config`.

- `q4nx_config` — only `row_block_size`, `col_block_size`, `parallel_size`,
  `keep_block_in_2D` are ever read (`:198-201`). Note `configs/lfm2.json` also
  carries `"optimize_ddr": true`, which **no code in the tree reads** — a dead key.
- `name_map` — `{ <role>: { gguf_name, q4nx_name, [default_tensor_type] } }`. The
  role key is for humans and iteration order; it is **never looked up by name**.
- `default_tensor_type` — `"Q4_0" | "Q4_1" | "Q8_0" | "Q4_K"`. Per-entry overrides
  win over the file default (`:307-310`); only the Qwen3.5 and Gemma4 configs use
  them. `get_ggml_type` additionally accepts `"BF16"` (`:258`).

Example, verbatim from `configs/llama.json`:

```json
"name_map": {
  "embedding":    { "gguf_name": "token_embd.weight",
                    "q4nx_name": "model.embed_tokens.weight" },
  "q_proj":       { "gguf_name": "blk.{bid}.attn_q.weight",
                    "q4nx_name": "model.layers.{bid}.self_attn.q_proj.weight" },
  "lm_head":      { "gguf_name": "output.weight",
                    "q4nx_name": "lm_head.weight" }
}
```

### `{bid}` expansion: detected, not declared

Both names are `str.format(bid=...)`'d (`:303-304`), but the layer range comes
from **detecting indices against the real tensors** (`:276-293`): the template
becomes a regex (`re.escape` + `\{bid\}` → `(\d+)`, anchored), matching indices
are collected, and `num_layers = max(found) + 1` — computed **per pattern**, so
two patterns can disagree. A pattern matching nothing gets `range(0)` and is
skipped (`:300-301`). Non-`{bid}` templates map unconditionally even when absent
from this GGUF; the print is suppressed (`:320-321`) and the absence surfaces via
`_warn_missing_config_entries` (`:100-112`).

This same regex expansion is duplicated in four places: `config_coverage_report`
(`:92`), `qwen35.py:692`, `cli._report_speculative` (`cli.py:251`), and the
detection loop itself. Change one and you probably need to change all four.

### Configs are heavily duplicated

The five `qwen3.5_{0.8b,2b,4b,9b,27b}.json` share one **identical** `name_map`
and differ only in `vision_config`; the code de-duplicates them on that basis
(`:82-86`). `llama.json`, `granite.json` and `k2.json` have byte-identical
`name_maps`, as do `qwen3.json` and `hunyuan.json`. A new size of an existing
family means a new config file that mostly copies the old one.

## GGUF shapes are innermost-first — this is the rule that bites

**The columns (matmul K axis) are `shape[0]`, not `shape[-1]`** —
`gguf_tensor.py:471`:

```python
if wants_quantized_target and self.shape and self.shape[0] % 32 != 0:
```

Every native unpacker takes `self.shape[0]` as its column count (`:438, 442,
448, 487, 489, 495`, and `:525-529` in `_requantize_to`), and each reshapes the
flat buffer as `(-1, columns)`. GGUF writes dimensions innermost-first, so the
logical `[rows, cols]` weight has `cols` at index 0.

Using `shape[-1]` sends a Q6_K `[hidden, vocab]` embedding (vocab is almost
never a multiple of 32) down the float-passthrough escape hatch, which then
crashes a head that needed a quantized triple — hunyuan's `lm_head` padding path
(`gguf_tensor.py:464-470`). **That bug shipped.**

A second guard: `len(self.shape) < 2` forces the float path regardless of target
(`:398-399`) — 1-D tensors are never block-packed.

Note the asymmetry that makes this easy to get wrong: on the **numpy/HF** side
the same logical axis is at index **1** (`qwen35.py:195, 199` pass
`np_w.shape[1]`). Only the GGUF side is reversed.

## Storage sinks: which one to use

There is no `_store` in the base. On the **GGUF** path, `GGUFTensor.unpack()`
(`:376`) returns **either a 1-element list** (float passthrough) **or a 3-tuple
`(d, m, qw)`** — the arity *is* the format discriminator, and callers branch on
`len()` (`granite.py:95`, `hunyuan.py:53`). Then `_pack` (`:707`) dispatches on
tensor type to `pack_q4k` / `_pack_q8nx` / `_pack_q4nx`.

On the **HF** path the sinks are per-family, and two families disagree:

- `Qwen35._store_q` (`:609`) — **Q8_0 for a specific four names**
  (`lm_head`, `ssm_out/alpha/beta_proj`, `:607`), Q4_1 for everything else.
  It quantizes to Q8_0 first, dequantizes, then re-quantizes "to match the
  GGUF-derived reference" so both paths produce identical bytes.
- `Qwen35Moe._store_q` (`:851`) — **inverted**: Q4_1 only for the three big
  expert mats (`:845-849`), Q8_0 for everything else quantized.
- `_bf16` (`:841`, and inline in the dense families) — embeddings, norms,
  `ssm_a`/`ssm_dt` as float32, alpha/beta twins, MoE routers/gates.

Copy the neighbouring family's policy; do not invent one. Getting this wrong
produces a container that loads and decodes garbage.

`_pack_q4nx` has a second float escape: if `m is None` it returns `d` as bf16
verbatim (`:922-923`).

## The two paths, and what the HF path cannot do

The branch is `self.gguf_reader is not None`, inside every `convert()` (e.g.
`models/qwen3.py:29`) and in `initialize()` (`:154`). CLI branch at `cli.py:869`.

Enforced differences — each is a deliberate refusal, not an oversight:

- **No imatrix on HF** → every prune flag is refused: `qwen35.py:235`,
  `qwen35moe.py:382`, `cli.py:870-872`.
- `--pad-to-fit` is GGUF-only (`qwen35.py:152`).
- `--quant Q4_K` is refused on HF (`model_converter.py:233-238`).
- The generic `_convert_hf` (`:399`) stores tensors **unquantized** and raises for
  any non-language `weights_type`. Granite (`granite.py:139`) and Hunyuan
  (`hunyuan.py:45`) refuse the HF path outright with that reason spelled out —
  the open kernels' q4_1 GEMV cannot read unquantized bytes.
- Reverse-coverage warnings are GGUF-only (`:334`), because the HF path builds its
  own iterators and would warn on every entry.

## Speculative packing: never crash on an unknown architecture

- `_WarnDict` (`:34-57`) — a name map that passes unknown names through unchanged
  and records them in `.unknown`, instead of raising. The point is that an
  unsupported architecture yields a best-effort container plus one summary, not
  a traceback.
- `config_coverage_report` (`:60-97`) — ranks configs by how much of this GGUF
  their own templates would cover, so the log can name the least-wrong `-f`.
- `_report_speculative` (`cli.py:232`) — the end-of-pack roll call. It drops
  config entries that a converter fallback actually supplied (tied `lm_head`,
  synthesized vision weights) before reporting "missing", so it only names
  tensors the runtime will really lack.

A speculative pack is **not** support for an architecture. The log says so; do
not publish one.

## Tests: all pure unit, no model files

Every test in `tests/` is synthetic — hand-built byte blobs, `SimpleNamespace`
fake readers, `torch.randn`, one tiny tmpdir Whisper checkpoint. No network, no
GGUF, no on-disk safetensors. That is why the suite runs in 1.4 s and why it can
run in CI.

Four exist specifically to catch bugs that already shipped:

- `test_qwen35_vheads.py` — llama.cpp stores Qwen3.5 value heads tiled; the old
  hard-coded `grp = 2` and "v is the second half" scrambled the 27B in **all 48**
  linear-attention layers while every slice compare passed, because the kernels
  and the reference read the same scrambled bytes.
- `test_qwen35_mtp.py` — the MTP block leaking into packs, and `_prune_meta`
  returning `{}` so `config.json` claimed 65 layers for a 64-layer container.
- `test_q4_source_repack.py` — cross-format repack must **preserve codes, not
  re-quantize**. Pins the min-sign conventions: Q4_K→Q4_1 must *not* sign-flip,
  Q4_1→Q4_K must, Q4_0→Q4_1 folds the `-8d` offset into the min.
- `test_granite_fold.py` — Granite's multipliers fold into the already-quantized
  tensor so codes do not move, and the deployed `config.json` states the
  post-fold value.

### Known coverage gaps

Verified absent — worth adding before you rely on these areas:

- **No test asserts the `shape[0]` vs `shape[-1]` rule**, despite it being a
  documented shipped bug. This is the highest-value missing test.
- **No test exercises `_create_name_maps`** — not the `_WarnDict` passthrough,
  not `{bid}` detection, not `_missing_config_entries`. All four duplicated regex
  sites are therefore unpinned.
- No test for `set_default_tensor_type`'s same-value early return, or the HF
  Q4_K refusal.
- No byte-layout test for `_pack_q4nx`/`_pack_q8nx`/`pack_q4k` against a real
  kernel reader; `pack_q4k`'s docstring points at an external suite
  (`specs/open-engine/tests/test_quant_q4k.py`).

## Rules

- **`pyproject.toml` is the only dependency declaration.** `torch` is floored
  (`>=2.4`), not pinned, so ROCm/CPU/CUDA wheels all satisfy it. Do not add an
  exact torch pin, and check `torch.__version__` still reports the ROCm build
  after touching `ironvenv`.
- Before adding a GGUF unpack, read the innermost-first rule above. Getting the
  axis wrong is silent.
- Copy the neighbouring family's `_store_q` quant policy rather than inventing
  one; the engine's GEMV expects a specific choice per tensor class.
- When a pack passes every kernel test but the text is wrong, suspect the
  container first and check it against HF before touching kernels.
- If closed behavior cannot be reproduced, return an explicit `not implemented`
  error rather than silently depending on the closed component.
