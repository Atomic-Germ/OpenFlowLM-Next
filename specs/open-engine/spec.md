# open-engine: the open engine (Qwen3.6-MoE, Qwen3 dense, Llama 3, Gemma 3, HunYuan dense) and its model recipes

Prefix `OPEN`. Home repo: openflowlm-next. Covers `src/open_qwen36/` (the
resident engine behind the app's `causal_lm` seam) and `open_kernels/recipes/`
(the ModelSpec → kernel-set generator whose `manifest.json` the engine reads).
Plan: `.claude/plans/open-kernels-phase3-model-recipes.md`.

Tests: `python -m pytest specs/open-engine/tests` (recipe, spec derivation,
build key, op range, packing plan) and `src/open_qwen36/manifest_test`
(the C++ manifest reader; built and run by `src/open_qwen36/build.cmd`, or
`ctest` after a CMake build of that directory). Hardware requirements are
documented procedures.

## Requirements

### OPEN-MANIFEST: the engine reads its kernel set from manifest.json
**Applies to:** openflowlm-next (`src/open_qwen36/manifest.cpp`, `core.cpp`, `engine.cpp`)
**Test category:** unit
**Tests:** `src/open_qwen36/manifest_test.cpp`, `tests/test_recipe_layout.py`, `tests/test_manifest_fixture.py`

The engine shall derive every layout constant, xclbin context, kernel,
per-layer-type verb sequence, buffer size and packing law from the
`manifest.json` beside the kernels. No model dimension, pool offset or kernel
name is a compile-time constant of the engine. A kernel directory without a
readable manifest, or whose manifest names a missing file, is not a kernel
set (`Engine::find_kernels` skips it). A model whose `config.json` disagrees
with the manifest's `hf_config_check` is refused at engine construction with
the offending key named.

**Acceptance criteria:**
- `Manifest::load` on the checked-in fixture (`tests/fixtures/manifest_qwen36.json`) yields 40 layers, two layer types with the 27B's buffer sizes and three-step programs, four contexts, six kernels with their patch kinds (`ax0` attnpos, `lx1`/`ax1` moeroute2), the tail `ln` → `lm`, and the MoE pool geometry `stripe 163840, up 655360, down_core 81920, pool_down 335544320, share 503316480 / 503971840 / 504627200`.
- A config with `hidden_size: 2560` → error naming `hidden_size`; `model_type: llama` → error naming `model_type`; a missing `num_experts` → error `lacks 'num_experts'`; a 24-layer config → error naming `num_hidden_layers`; `full_attention_interval: 5` → error naming `layer_types`; `full_attention_interval: 4` without `layer_types` → accepted.
- `manifest_version: 2` → refused by the parser.
- A manifest the packer or the engine could not execute is refused by the parser, naming the field: a pack op without a size `pools::apply` needs (a `std_perm` without `nch`, an `lmhead_q8` without `chunk_bytes`), or a `moeroute2` step on a kernel not built with the routed-expert patch table.
- The fixture equals the recipe's current output (`make_fixtures.py`) apart from the build key.

### OPEN-LAYOUT-FREEZE: the recipe reproduces the shipped 27B kernels
**Applies to:** openflowlm-next (`open_kernels/recipes/qwen36moe.py`, `designs/layer_x/`)
**Test category:** unit (constants) + manual (the rebuild)
**Tests:** `tests/test_recipe_layout.py`

The qwen36moe recipe shall derive, from the ModelSpec alone, every constant
that `designs/layer_x/layout.py`, `xcommon.py`, `lx.py` and `ax.py` carried
by hand on 2026-09-05, and the designs built from the recipe shall be the
kernels that shipped.

**Acceptance criteria:**
- `recipe(default_spec()).layout.constants()` equals the frozen `LAYOUT_27B` dict (every consts / act / state / pool / KV offset, `LMHEAD_POOL_BYTES 542113792`); `Common`, `Linear`, `Attn` equal their frozen dicts.
- Manual: `python open_kernels/export_qwen36_kernels.py --out <new> --check <previous export>` reports every `insts.bin` byte-identical and every `final.xclbin` identical apart from build stamps. Done 2026-09-05 against the kernels built from the hand-written sources: 6/6 streams identical, xclbins 75–82 stamp bytes each.

### OPEN-SPEC-DERIVE: ModelSpec from a model's own metadata
**Applies to:** openflowlm-next (`open_kernels/recipes/spec.py`)
**Test category:** unit
**Tests:** `tests/test_spec_derive.py`

`ModelSpec.from_hf_config` (the HF-style `config.json` FLM ships) and
`ModelSpec.from_gguf_metadata` (llama.cpp's key names) shall produce the
hyperparameter tuple for every supported family; an unknown family or a
missing key is an error naming it.

**Acceptance criteria:**
- The 27B's `config.json` fields (+ the tokenizer's 248070 ids) → a spec equal to `recipes/specs/qwen36-35b-a3b.json`; `layer_types` from the list when present, else from `full_attention_interval`.
- GGUF metadata for arch `qwen35moe` (or `qwen3next`) → the same hyperparameters (`real_vocab` = `vocab_size`, GGUF has no tokenizer-side count).
- HunYuan's `rope_scaling` (`type: dynamic`, an alpha) folds into ONE static base, `rope_theta * alpha^(d/(d-2))`, which is what a `hunyuan-dense` GGUF already carries in `rope.freq_base`; a `yarn` type, a factor or an mscale other than 1, `use_cla`, a bias or a MoE variant is refused by name.
- `model_type: llama` → `SpecError` naming `model_type 'llama'`; `general.architecture: gemma3` likewise; a missing `linear_num_value_heads` / `qwen35moe.expert_count` → `SpecError` naming the key.
- JSON round trip preserves the spec and its hash; an unknown field is refused.

### OPEN-OP-RANGE: a recipe fails at generation outside a template's validated set
**Applies to:** openflowlm-next (`open_kernels/recipes/catalogue.py`, `qwen36moe.py`)
**Test category:** unit
**Tests:** `tests/test_op_range.py`

Each kernel template declares the parameter points it has been validated at.
A recipe requesting another point shall raise `OpRangeError` naming the
template, the parameter and the validated set, before any build; more than 8
buffer arguments on a dispatch is likewise refused.

**Acceptance criteria:**
- The 27B spec passes every check.
- `head_dim=32` → `attn: head_dim=32 is outside the validated set {64, 128, 256}`; `hidden=3072` → `ln: width=3072 is outside the validated set {2048, 2560, 4096}`; `gemv_q4 K=3072` → refused by name; `quant='q4_k'` → refused. (The 128 / 2560 / 9728 points entered the sets with OPEN-FAMILY-QWEN3 on 2026-09-05; head_dim 64, num_heads 40 and K 8192 with OPEN-FAMILY-GRANITE on 2026-09-06. A test that spells out a set's membership fails on the one event that is never a regression, so the tests assert the refusal and not the contents.)
- The `attn` set grows only after a compare: `qk_norm_post_rope=True` entered it with OPEN-FAMILY-HUNYUAN and `head_dim=64` with OPEN-FAMILY-GRANITE, both on 2026-09-06; `head_dim=32` is still refused.
- Nine buffer arguments → `9 buffer arguments`.

### OPEN-ATTN-CONTEXT: decode cost grows linearly with position, on every family
**Applies to:** openflowlm-next (`open_kernels/designs/attn/attn.h`)
**Test category:** manual (`open_kernels/model/sweep_positions.ps1`, needs the NPU and two model containers)

**This is an observation, not yet a requirement** — it is written down because
it was measured, it is large, and nothing else in this spec records it.

A decode step's cost is dominated by a term linear in context position.
Measured 2026-09-06 on one box, one step per point, `--at-position`:

| position | Qwen3-4B `part0` (36 layers) | Granite-3B `part0` (40 layers) |
|---:|---:|---:|
| 0 | 56.7 ms | 60.0 ms |
| 256 | 442.6 ms | 578.6 ms |
| 1024 | 1585.3 ms | 2103.2 ms |
| 2048 | 3120.6 ms | 4121.6 ms |

Linear in both: **1.496 ms/position** (Qwen3-4B) and **1.983 ms/position**
(Granite), i.e. **41.6 µs and 49.6 µs per layer per position**. `lm_head` stays
flat at 4–6 ms throughout, so it is attention, not the GEMVs. At position 2048
a single token costs 3–4 seconds.

**The cost is not bandwidth and not arithmetic.** Granite reads *half* the KV
bytes per position per layer (KV_ROW 2048 against 4096) and does fewer MACs
(40×64 = 2560 against 32×128 = 4096), yet its slope is 19% steeper. Qwen3-4B
additionally carries `qk_norm`, which Granite does not — more work for the
faster one.

What does track is **head count**: 40/32 = 1.25 against a measured 1.19. The
hypothesis that fits is a fixed per-head cost in the position loop that does
not shrink when `head_dim` halves — i.e. `attn.h` not saturating the vector
unit at hd 64. Suggestive, not proven: two families is two points, and a third
(Llama 3.1 8B is 32 heads at hd 128, Gemma 3 4B is 8 at 256) would separate
head count from head width properly.

Consequence for anything quoting a tok/s number: **say the position.** Granite
measured 5.92 tok/s over 63 tokens and 1500 µs/layer at position 0; both are
true and they are not the same measurement.

### OPEN-BUILD-CACHE: the build key covers every build input
**Applies to:** openflowlm-next (`open_kernels/recipes/cache.py`, `export_qwen36_kernels.py`)
**Test category:** unit
**Tests:** `tests/test_build_cache.py`

The build key shall hash the recipe package's sources, every kernel source
the recipe's designs include, the ModelSpec (without its informational
`extra`) and the quant format. `export_qwen36_kernels.py` skips the build
when the destination's manifest already carries the key (`--force`
overrides). The KV / ptab capacity is a runtime buffer size in this tree,
not a build input, and is not in the key.

**Acceptance criteria:**
- The key is stable across calls and covers `recipes/qwen36moe.py`, `designs/layer_x/lx.py`, `designs/attn/attn.h`, `designs/gemv_q4/gemv_q4.h`, `designs/lm_head_q8/lm_head_q8.py`, `include/vecmath.h` (among others).
- Appending a comment to `attn.h` or to `qwen36moe.py` changes the key; changing `rope_theta` or `quant` changes it; changing `extra` does not.

### OPEN-PACK-PLAN: the packing plan reproduces the verified pool laws
**Applies to:** openflowlm-next (`open_kernels/recipes/pack.py`, `src/open_qwen36/pools.cpp`)
**Test category:** unit (Python interpreter) + integration (C++, through OPEN-FAMILY-QWEN36MOE)
**Tests:** `tests/test_pack_plan.py`, `tests/legacy_pools.py` (the frozen originals)

The recipe's plan (`expert_stripes`, `expert_down`, `std_perm`, `put`,
`conv_transpose`, the lm_head supertile order, the position table) applied
by `recipes/pack.py` shall produce, for a container with the 27B's tensor
shapes, exactly the bytes the hand-written packers produced (the ones
verified against pools captured from FLM's engine). `pools.cpp` interprets
the same plan and is verified by the hardware run.

**Acceptance criteria:**
- Layer pool, consts blob (linear and attention), lm_head pool and ptab are byte-equal to `legacy_pools.py` on random-byte tensors of the right sizes.
- A small weight larger than its slot is refused (`does not fit its 4096 B slot`).

### OPEN-FAMILY-QWEN36MOE: greedy agreement with the fp64 reference on the 27B
**Applies to:** openflowlm-next (`src/open_qwen36/`)
**Test category:** manual (needs the NPU and the model)

Through the manifest path, the 8-layer slice of `Qwen3.6-35B-A3B-NPU2`
decoding `[248045]` greedily for 3 tokens shall match `open_kernels/model/out8t3`'s
fp64 logits at every position, and the full model shall answer a chat prompt
coherently.

**Procedure:**
1. `python -m recipes.manifest --model-dir ~/.flm/models/Qwen3.6-35B-A3B-NPU2 --out src/xclbins/Qwen3.6-35B-A3B-NPU2/open_kernels/manifest.json` (or a full `export_qwen36_kernels.py` run).
2. `src\open_qwen36\out\open_qwen36_cli.exe --model <model dir> --kernels src/xclbins/Qwen3.6-35B-A3B-NPU2/open_kernels --ids 248045 --max-tokens 3 --layers 8 --dump-logits <dir>/y --twice`
3. Correlate `y_t{0,1,2}.bin` with `open_kernels/model/out8t3/ref_logits{,_t1,_t2}.bin` over the first 248070 ids: corr ≥ 0.9999, same argmax; the second request reproduces the first.
4. `python src/open_qwen36/chat.py "Explain what an NPU is in two sentences."` → a coherent two-sentence answer ending in `<|im_end|>`.

**Result 2026-09-05:** corr 0.999998 / 0.999996 / 0.999991, argmax and top-5 identical at every position, request 2 reproduced request 1 (8 layers, 35 ms/step). Full model: see the plan's "Phase A result".

### OPEN-FAMILY-QWEN3: Qwen3 dense on the open kernels
**Applies to:** openflowlm-next (`open_kernels/recipes/qwen3.py`, `designs/dense/dx.py`, `designs/lm_head_q4`, `src/open_qwen36/`)
**Test category:** manual (needs the NPU and `FastFlowLM/Qwen3-4B-NPU2`); the recipe's arithmetic is unit-tested in `tests/test_qwen3_dense.py`

A Qwen3 dense model (GQA with q/k RMSNorm, full RoPE, no attention gate,
silu-gated FFN, a q4_1 lm_head) shall run on the open kernels from its
`config.json` alone: the `qwen3` recipe derives the layouts, the packing plan
(`model.layers.N` names, the general pool-order law), the one-run program and
the kernel builds (`dx`, `ln` at the model's width, `lm_head_q4`); the engine
is unchanged. The kernel points the family needs (K = 2560 / 9728 GEMVs, HD 128
attention with 32/8 heads and full RoPE, the 2560-wide norm, the q4 head) are
in the catalogue's validated sets only once this procedure has passed.

**Procedure:**
1. `python open_kernels/export_qwen36_kernels.py --model-dir ~/.flm/models/Qwen3-4B-NPU2` (WSL) → `src/xclbins/Qwen3-4B-NPU2/open_kernels/{dx,ln,lm_head_q4}` + `manifest.json`.
2. `python open_kernels/model/make_decode.py --model-dir ~/.flm/models/Qwen3-4B-NPU2 --layers 4 --tokens 2 --out open_kernels/model/out_q3`, then `open_kernels/harness/out/run_kernel.exe open_kernels/model/out_q3/run_decode.cfg` and `python open_kernels/model/compare_decode.py --tokens 2 --out open_kernels/model/out_q3`: every layer's residual corr > 0.9999, logits corr > 0.9999, same argmax at both positions.
3. `src\open_qwen36\out\open_qwen36_cli.exe --model <model dir> --kernels src/xclbins/Qwen3-4B-NPU2/open_kernels --ids 151644 --max-tokens 3 --layers 4 --dump-logits <dir>/y` matches step 2's reference logits (the engine's packer, manifest path and attnpos on the dense stream).
4. `python src/open_qwen36/chat.py "Explain what an NPU is in two sentences." --model <model dir> --kernels src/xclbins/Qwen3-4B-NPU2/open_kernels` → a coherent answer ending in `<|im_end|>`.

**Result 2026-09-05 (Qwen3-4B):** step 2 logits corr 0.999997 / 0.999994, same argmax and top-5, residual corr ≥ 0.999996 in every layer at both positions; step 3 identical through the engine, request 2 reproduced request 1; step 4 a coherent two-sentence answer ending in `<|im_end|>` at token 58 (272 ms/token). Details: `.claude/plans/open-kernels-phase-b-qwen3-dense.md`.

### OPEN-FAMILY-LLAMA3: Llama 3 on the dense recipe
**Applies to:** openflowlm-next (`open_kernels/recipes/dense.py`, `spec.py`, `designs/dense/dx.py`, `src/open_qwen36/`)
**Test category:** manual (needs the NPU and `FastFlowLM/Llama-3.1-8B-NPU2`); the derivation, the RoPE scaling and the 8B layout are unit-tested in `tests/test_llama3.py`

A Llama 3 model (GQA without q/k norms, full RoPE with the llama3 frequency
scaling, eps 1e-5, silu FFN, untied q4_1 head) shall run on the open kernels
from its `config.json` alone through the dense recipe: `qk_norm` / `norm_eps`
become the `ATTN_QKNORM` / `LN_EPS` knobs, the scaled inverse frequencies are
computed host side (`ModelSpec.rope_inv_freq`, in the manifest, used by both
position-table builders), and widths that overflow a core's memory are handled
by the recipe (one chunk per weight element; one norm output element per call).

**Acceptance criteria (unit):**
- HF and GGUF derivations agree; `rope_inv_freq()` equals transformers' `_compute_llama3_parameters` for the 8B's parameters; a non-llama3 `rope_scaling` and tied embeddings are refused.
- The 8B layout: 8 KB norm elements, one chunk per weight element (`TAB_BYTES 32256`), `PER_CALL 1`; Qwen3-4B keeps two.

**Procedure (manual):** as OPEN-FAMILY-QWEN3 with `Llama-3.1-8B-NPU2`, `out_l3`, prompt id 128000, and `chat.py` (which switches to the Llama 3 template when the tokenizer has `<|start_header_id|>`).

**Result 2026-09-05 (Llama-3.1-8B):** slice logits corr 1.000000 / 0.999993, same argmax and top-5, residual corr ≥ 0.999994 every layer; identical through the engine; a coherent two-sentence answer ending in `<|eot_id|>` at token 79 (203 ms/token). Details: `.claude/plans/open-kernels-phase-c-llama3.md`.

### OPEN-FAMILY-GEMMA3: Gemma 3 on the dense recipe
**Applies to:** openflowlm-next (`open_kernels/recipes/dense.py`, `spec.py`, `designs/dense/dx.py`, `designs/ln/ln_nr32.cc`, `harness/stream_patch.hpp`, `src/open_qwen36/`)
**Test category:** manual (needs the NPU and `FastFlowLM/Gemma3-4B-NPU2`); the derivation, the two RoPE tables, the window's row counts and the 4B layout are unit-tested in `tests/test_gemma3.py`

A Gemma 3 text model (GQA with q/k RMSNorm, GeGLU-tanh, sandwich norms, five
sliding-window layers per global one, a local and a linearly scaled global
RoPE, the tied head stored as q4) shall run on the open kernels from its
`config.json` through the dense recipe: the activation is a generated kernel
knob, the sandwich norms are the `ln_nr32` entry plus the design's sandwich
program, the sliding window is a per-token `attnpos` patch (the fill's offset
and length, the record's row counts) on a second kernel entry sharing the
global layers' stream, and each layer type has its own position table. The
container's folded `1 + w` norms and sqrt(hidden) embeddings are used as
stored.

**Acceptance criteria (unit):**
- HF and GGUF derivations agree; the global table is `1e6^(-2i/256) / 8`, the local `1e4^(-2i/256)`; `window_rows` gives `valid = min(p, 1023)`, `nf = max(1, valid)` for a 1024 window.
- The 4B layout: two layer types sharing one design, `dx` / `dx_local` with windows 0 / 1024 on the same stream, `ptab` / `ptab_local` globals, six consts per layer.
- A silu activation, softcapping, or a `query_pre_attn_scalar` unequal to the head dim is refused by name.

**Procedure (manual):** as OPEN-FAMILY-QWEN3 with `Gemma3-4B-NPU2`, `out_g3`, 6 layers (five local, one global), prompt id 2; then `open_qwen36_cli --at-position 1100 --layers 6` (finite logits through the window path); then `chat.py` (the Gemma template when the tokenizer has `<start_of_turn>`).

**Result 2026-09-05 (Gemma3-4B):** slice logits corr 0.999998 / 0.999998, same argmax and top-5, residual corr 1.000000 every layer; identical through the engine; a finite step at position 1103; a coherent two-sentence answer ending in `<end_of_turn>` at token 43 (96 ms/token). Details: `.claude/plans/open-kernels-phase-d-gemma3.md`.

### OPEN-FAMILY-HUNYUAN: HunYuan dense on the dense recipe
**Applies to:** openflowlm-next (`open_kernels/recipes/dense.py`, `spec.py`, `designs/attn/attn.h`, `designs/dense/dx.py`, `utilities/q4nx-build`)
**Test category:** manual (needs the NPU and a converted `Hy-MT2-7B-NPU2`); the derivation, the folded RoPE base, the post-RoPE norm order and the 7B layout are unit-tested in `tests/test_hunyuan.py`

A HunYuan V1 dense model (Hy-MT2-7B and the Hunyuan-{1.8,4,7}B dense line:
Llama 3.1 8B's GQA shape, eps 1e-5, silu FFN, a tied head) shall run on the
open kernels from its `config.json` alone through the dense recipe. Two
things are new, and neither is a ModelSpec field:

- **The q/k RMSNorm weight multiplies AFTER RoPE** (`query_layernorm(apply_rotary_pos_emb(q))`),
  where every other family norms first. RoPE is orthogonal and the rotary dim
  is the whole head, so the RMS is unchanged by the rotation; what moves is the
  per-dim weight, which does not commute with the pair rotation. It is
  `attn.h`'s `ATTN_QKNORM_POST`, set from `recipes.dense.QKNORM_POST_ROPE`.
- **The vocabulary is not a whole number of head bands** (128167). The lm_head
  is built, packed and read at the rounded count (`dense.lm_rows`, 128192) with
  the converter zero-padding the tensor, while `hf_config_check` still holds the
  model's own `vocab_size` and `real_vocab` bounds the argmax.

The NTK-alpha RoPE scaling is folded into one static base by the spec builder
(OPEN-SPEC-DERIVE), so the position tables need nothing new.

**Acceptance criteria (unit):**
- HF and GGUF derivations agree; `rope_theta` is `1e4 * 1000^(128/126)` from either source; the layout equals Llama 3.1 8B's (`PER_CALL 1`, `TAB_BYTES 32256`, 8 KB norm elements) with `LMHEAD_BANDS 2003`.
- `QKNORM_POST` is True for `hunyuan` and False for `qwen3` / `llama3`; `qk_norm_post_rope=True` is refused by the catalogue until this requirement's procedure has run.
- The manifest carries `vocab 128192` / `real_vocab 128166` while `hf_config_check.vocab_size` is 128167; a config.json carrying the padded count is refused by name (`manifest_test`, fixture 4). The tokenizer defines ids 0..128165 (127957 vocab entries plus 209 added tokens, no gaps), so 128166 is its id count; config.json's 128167 is the embedding table's row count, one row no token maps to, inherited from the base model.

**Procedure (manual):**
1. Convert: `q4nx-build -i tencent/Hy-MT2-7B-GGUF` (the Q8_0 file requantizes to q4_1 with the least loss), then copy the HF repo's `config.json` beside the resulting `model.q4nx` / `tokenizer.json`.
2. The ONE new kernel point (HD 128, 32/8 heads, full RoPE, qk-norm AFTER RoPE) has no standalone fixture -- `designs/attn` is gated through the whole layer -- so step 3's per-layer residual correlation against `replica_dense.py` IS its compare. Build it with `OPEN_KERNELS_UNVALIDATED=1` until that passes, then add `True` to the `attn` template's `qk_norm_post_rope` set in `recipes/catalogue.py` (done 2026-09-06).
3. Then as OPEN-FAMILY-QWEN3 with `Hy-MT2-7B-NPU2`, `out_hy`, prompt id 127958, and `chat.py` (which switches to the HunYuan turn format when the tokenizer has `<|extra_0|>`).
4. The chat check is a translation instruction, not a chat question -- Hy-MT2 is a translation model: `python src/open_qwen36/chat.py "Translate the following text into French. Note that you should only output the translated result without any additional explanation: The neural processing unit runs the model on the laptop."` -> the French sentence, ending in `<|eos|>`.

**Result 2026-09-06 (Hy-MT2-7B, Strix, Windows + XRT):** 4-layer slice, 2 greedy
tokens from id 127958 -- logits corr 1.000000 / 0.999996, same argmax (101773)
and top-5 at both positions, every layer's residual corr >= 0.999996 (maxrel
<= 3.9e-3); identical through the engine, request 2 reproduced request 1, and
the head's padded rows 128167..128191 came back exactly zero. All 32 layers,
a French translation instruction: a correct sentence ending in `<|eos|>` at
token 33 (231 ms/token, 4.3 tok/s). `qk_norm_post_rope=True` is now in the
catalogue. Details: `.claude/plans/open-kernels-phase-e-hunyuan.md`.

### OPEN-FAMILY-GRANITE: IBM Granite on the dense recipe
**Applies to:** openflowlm-next (`open_kernels/recipes/spec.py`, `dense.py`, `families.py`, `src/open_qwen36/`, `utilities/q4nx-build`)
**Test category:** manual (needs the NPU and `vegahyo/Granite-4.2-3B-NPU2`); the derivation, the multiplier fold and the 3B layout are unit-tested in `tests/test_granite.py`

An IBM Granite dense model (GQA without q/k norms, unscaled RoPE, eps 1e-5,
silu FFN, untied q4_1 head) shall run on the open kernels from its
`config.json` alone through the dense recipe. **Granite is the first
`head_dim = 64` point**, and the first at `num_heads = 40`; nothing in the
design changes for it, because `ATTN_HD` / `ATTN_NH` are compile-time macros
and `attn.h` already carries HD 64's `kScale = 0.125f`.

Granite is Llama plus four scalar multipliers — `attention_multiplier`
(replacing the implicit `hd**-0.5`), `embedding_multiplier`,
`residual_multiplier`, `logits_scaling`. `ModelSpec` expresses none of them and
`attn.h` hard-codes `1/sqrt(HD)`, so the recipe **requires a container whose
multipliers have been folded into the weights** by q4nx-build
(`q_proj *= attention_multiplier * sqrt(hd)`, `o_proj`/`down_proj *=
residual_multiplier`, `embed_tokens *= embedding_multiplier`,
`lm_head /= logits_scaling`). For 4.2-3B the only non-unit factor is
`attention_multiplier = 0.015625` at hd 64, so the fold is `q_proj *= 0.125`
and the folded config reads `attention_multiplier = 0.125 = 64**-0.5` exactly —
a power of two, so the fold is exact in bf16. The container records the
originals under `q4nx_folded_multipliers`.

**Acceptance criteria (unit):**
- HF and GGUF derivations agree; the RoPE table is the plain unscaled `1e7^(-2i/64)`; `rope_scaling` and tied embeddings are refused by name.
- An unfolded `attention_multiplier`, or any of the other three unequal to 1.0, is refused by name and names q4nx-build as the fix — from HF `config.json` and from GGUF metadata alike.
- `hf_config_check` carries `attention_multiplier`, so the **engine** refuses an unfolded container at load, not only the recipe at generation.
- The 3B layout: `PER_CALL 2`, `TAB_BYTES 18432` (the K = 8192 table), `ELN 5120`, `E_A 1024`, `KV_ROW 2048`, `PTAB_ROW 1024`, `LMHEAD_BANDS 1568`; one `dx` step per layer plus the `ln` + `lm` tail.

**Procedure (manual):** as OPEN-FAMILY-QWEN3 with `Granite-4.2-3B-NPU2`, `out_gr`, prompt id 100264 (`<|start_of_role|>` — *not* `config.json`'s `bos_token_id` 100283, which is `</documents>` and disagrees with `tokenizer_config.json`'s own bos). Three catalogue points entered with it: `attn.head_dim 64`, `attn.num_heads 40`, `gemv_q4.K 8192`.

**Prior evidence (2026-09-02, a different design):** these shapes have been run
and compared on this hardware before, by hand-written Granite kernels in
`vegah/FastFlowLM@feat/kernels` — all eight projection shapes cosine
1.00000000 under a one-hot activation, GQA attention 0.9993–0.9998, and a whole
layer in **four** dispatches at 1744.7 µs (13.6 tok/s device time). That is the
baseline the one-dispatch `dx` program should beat, and the reason head_dim 64
at hidden 2560 was expected to work at all. It is prior evidence for the
catalogue points, not a substitute for validating them on `dx`.

**Result 2026-09-06 (Granite-4.2-3B):** slice logits corr 0.999998 / 0.999990,
same argmax (38457) and an **identical top-5** at both positions, residual corr
0.999990–0.999999 in every layer; a coherent two-sentence answer through
`chat.py`, ending on `<|end_of_text|>` at token 52. Built on mlir-aie
1.4.2.dev16+g7e00b57 / Peano 21.0.0.2026080301, natively on Windows.

Through the app: the model loads 40/40 layers at context capacity 8192
(weights resident in 11 s), logs *"Granite on the open kernels"* and answers a
Norwegian prompt coherently, reasoning first. **That run used a catalogue entry
this PR no longer ships** -- per AGENTS.md the container belongs on
`Atomic-Germ/*-OpenNPU2` and installs through `flm-add`
(`flm-add <repo-or-directory> --family granite`), which is the supported path
until it is hosted there. The measurement above is what was run; the flm-add
install has not been re-verified end to end.

**Known rough edge:** the reasoning block is not parsed. Granite carries
`<think>` / `</think>` as real tokens (100274 / 100275) and its catalogue entry
sets `think: true`, but `Granite` implements no `parse_stream_content` /
`parse_nstream_content`, so the chain of thought is printed raw and only the
closing tag appears. Cosmetic, and separate from the kernel path.

**At zero context `dx` beats the hand-written kernels**: `part0` 60.0 ms over
40 layers is **1500 µs/layer**, against those kernels' 1744.7 µs at four
dispatches — 1.16×, the direction one dispatch per layer was expected to give.

Everything above that is the context term, and it is **not** Granite's: see
OPEN-ATTN-CONTEXT below. Decode measured 5.92 tok/s over 63 tokens because the
context grew underneath it, not because the family is slow.
