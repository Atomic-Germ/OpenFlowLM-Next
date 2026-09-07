# open-engine: the open engine (Qwen3.6-MoE, Qwen3.5 dense, Qwen3 dense, Llama 3, Gemma 3, HunYuan dense) and its model recipes

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
- `Engine::find_kernels` looks in this order and returns the first complete set, logging the directory that served: `FLM_OPEN_KERNELS_DIR`; `<model dir>/open_kernels`; then `<root>/xclbins/<model name>/open_kernels` over **every** root in `utils::xclbin_roots()` -- the user roots first (`$FLM_XCLBIN_PATH`, the directory holding `$FLM_CONFIG_PATH`, the user-level flm directory `flm-add` writes into), then the roots the closed path walks (the executable's directory, the CWD, `<exe>/../share/flm`, the configured prefix), then `config.exec_path` if a DEV_BUILD put it outside all of those. Not only the single root `utils::find_xclbin_path()` returns: a set `flm-add` linked under the user root and a set shipped in the install tree are both reachable, whichever of the two that function happens to pick. `find_xclbin_path` itself is unchanged -- it still walks the closed roots only, so which root serves a **closed** kernel does not move.

### OPEN-ADD-KERNEL-LINK: `flm-add` links a model to the kernel set matching its spec
**Applies to:** openflowlm-next (`utilities/flm-add/flm_add/__init__.py`)
**Test category:** unit
**Tests:** `utilities/flm-add/tests/test_open_kernels_link.py`

Open kernel sets belong to a `ModelSpec`, not to an official model name. When
installing a model, `flm-add` shall derive that model's spec the way the recipes
do (`recipes.load.spec_from_model_dir`: config.json, the tokenizer's real vocab,
and the per-role weight format read from the `model.q4nx` safetensors header) and
link the installed set whose `open_kernels/manifest.json` carries the same
`spec_hash`, at `<model dir>/open_kernels` — the candidate `Engine::find_kernels`
checks before any xclbins root. With no matching set installed, the model's
closed-kernel install is unchanged and the command that would build one is printed.

**Acceptance criteria:**
- Given two installed sets, one whose manifest `spec_hash` equals the model's and one whose does not, the matching one is chosen regardless of directory name ordering.
- When several sets match, the one filed under the model's own directory name wins.
- With no match, nothing is linked and the message names `export_qwen36_kernels.py --model-dir <model dir>`.
- `--open-kernels DIR` uses `DIR` without searching, and is refused when `DIR` has no `manifest.json`.
- `--xclbin-from` and `--no-xclbin` keep their existing behaviour.

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
- Re-run 2026-09-06 on the qwen35 working tree: 6/6 streams still identical, but `lx0` / `lx1` were 2688 B smaller because `xcommon.DN_FLAGS` passed `-DDNX_PAD=Common.DN_PAD` (140, the padded S row count) where `dnx.h`'s `kPad` is the hi/lo record stride (160). Fixed by removing the `DNX_PAD` knob (`.claude/plans/q-qwen35-handoff.md`, "the lx xclbin size").
- **Result 2026-09-06 (the confirming rebuild):** `--force` re-export of the 35B spec against the shipped set -- 6/6 `insts.bin` byte-identical and 6/6 `final.xclbin` stamps-only (76-83 bytes each), `lx0` / `lx1` back at 176 399 B. `manifest.json` differs only in `build_key` / `spec_hash`, the new `builds.lm_head_q8.env.LMHEAD_K`, the `qwen3_5_moe_text` alias in `hf_config_check.model_type` and four `spec` fields the dense families added -- no layout, kernel, program or packing-plan field moves. Log: `.claude/plans/q-hw-results.md`.
- **Result 2026-09-07 (after the native-q8 merge):** the same `--force` re-export, run as the gate for OPEN-QUANT-Q8 after eleven recipe files and every design changed -- 6/6 `insts.bin` byte-identical, 6/6 `final.xclbin` stamps-only (77-83 bytes), `lx0` / `lx1` at 176 399 B, `manifest.json` equal apart from `build_key`. A per-role quant map that is all q4_1 moves nothing. Log: `.claude/plans/q8-hw-results.md`.
- **Result 2026-09-07 (the gate for the Qwen3.5 sizes):** the same `--force` re-export, run before any of the 16-head DeltaNet work touched hardware -- `dn_glue.h` gained the `DNGLUE_NHEAD` knob and `lx.py` a per-half projection walk, both gated so a 32-head MoE spec passes the same flags and the same call sequence. 6/6 `insts.bin` byte-identical, 6/6 `final.xclbin` stamps-only (73-83 bytes), `manifest.json` equal apart from `build_key`. Log: `.claude/plans/q35-hw-results.md`.

### OPEN-SPEC-DERIVE: ModelSpec from a model's own metadata
**Applies to:** openflowlm-next (`open_kernels/recipes/spec.py`)
**Test category:** unit
**Tests:** `tests/test_spec_derive.py`

`ModelSpec.from_hf_config` (the HF-style `config.json` FLM ships) and
`ModelSpec.from_gguf_metadata` (llama.cpp's key names) shall produce the
hyperparameter tuple for every supported family; an unknown family or a
missing key is an error naming it.

**The weight format is per role, and only the model FILE says what it is.**
`config.json` does not record whether a projection is stored at q4_1 or q8, so
`ModelSpec.quant` is a map over the roles `attn`, `linear`, `linear_out`,
`shared`, `ffn`, `experts`, derived by `spec_from_model_dir` from the
container's safetensors header (8704-byte chunks = q8, 5120 = q4_1) or, on that
path, from a GGUF's tensor types -- and then narrowed to the roles the family's
designs can actually stream at q8 (`Q8_ROLES`), because the rest run on the
packer's re-quantizing fallback. `lm_head` is not a role: the family already
fixes the head's format. A role whose tensors disagree with each other is
refused naming the tensor that broke it. When every role is at the default the
map serialises, hashes and reads back as the bare string `"q4_1"`, so a model
with no q8 projection derives byte for byte what it derived before roles
existed. `OPEN_KERNELS_FORCE_Q4_1=1` forces the fallback for an A/B.

**Acceptance criteria:**
- The 27B's `config.json` fields (+ the tokenizer's 248070 ids) → a spec equal to `recipes/specs/qwen36-35b-a3b.json`; `layer_types` from the list when present, else from `full_attention_interval`.
- GGUF metadata for arch `qwen35moe` (or `qwen3next`) → the same hyperparameters (`real_vocab` = `vocab_size`, GGUF has no tokenizer-side count).
- HunYuan's `rope_scaling` (`type: dynamic`, an alpha) folds into ONE static base, `rope_theta * alpha^(d/(d-2))`, which is what a `hunyuan-dense` GGUF already carries in `rope.freq_base`; a `yarn` type, a factor or an mscale other than 1, `use_cla`, a bias or a MoE variant is refused by name.
- `model_type: llama` → `SpecError` naming `model_type 'llama'`; `general.architecture: gemma3` likewise; a missing `linear_num_value_heads` / `qwen35moe.expert_count` → `SpecError` naming the key.
- JSON round trip preserves the spec and its hash; an unknown field is refused.
- A container header with q8 attention / linear / out / shared projections and q4_1 routed experts gives `{"attn": "q8", "linear": "q8", "linear_out": "q8", "shared": "q8"}`; a stock container (only the head at q8) gives `{}`; a Qwen3.5 container gives `{"linear_out": "q8"}`. A role at two formats is refused naming the second tensor. An unknown quant role in a spec JSON is refused naming it.
- An all-`q4_1` map serialises as `"q4_1"`, hashes as `"q4_1"` and round-trips to it; a map with a q8 role changes `spec_hash` and round-trips unchanged. Every checked-in spec under `recipes/specs/` still reads `"quant": "q4_1"`, and the 27B's manifest fixture is byte-identical.

### OPEN-OP-RANGE: a recipe fails at generation outside a template's validated set
**Applies to:** openflowlm-next (`open_kernels/recipes/catalogue.py`, `qwen36moe.py`)
**Test category:** unit
**Tests:** `tests/test_op_range.py`

Each kernel template declares the parameter points it has been validated at.
A recipe requesting another point shall raise `OpRangeError` naming the
template, the parameter and the validated set, before any build; more than 8
buffer arguments on a dispatch is likewise refused.

The `attn` template's geometry is validated as a WHOLE TUPLE, not one
parameter at a time: `(head_dim, num_heads, num_kv_heads, rotary_dim, qk_norm,
attn_gate, qk_norm_post_rope)`, one entry per configuration a family procedure
has actually compared. Checking the parameters independently passed a
combination nobody had ever run -- (128, 16, 8), a GQA group of 2 at head dim
128 -- because each of 128, 16 and 8 had entered its own set from a different
model. A call site that does not pass `qk_norm_post_rope` is read as `False`.

**Acceptance criteria:**
- The 27B spec passes every check.
- `head_dim=64` on the 27B → `attn: ('head_dim', ..., 'qk_norm_post_rope') = (64, 16, 2, 64, True, True, False) is outside the validated combinations {...}` — head dim 64 is validated at Llama 3.2 1B's and Granite 4.2 3B's geometries only, so it is the COMBINATION that is refused here, not the value; `hidden=5120` → `ln: width=5120 is outside the validated set {1024, 2048, 2560, 3072, 4096}`; `gemv_q4 K=5120` → names `{1024, 2048, 2560, 3072, 3584, 4096, 6144, 8192, 9216, 9728, 10240, 12288, 14336}`; `quant='q4_k'` → refused.
- A combination of individually validated values that no procedure has run is refused: `(128, 16, 4, 128, True, False, False)` names itself, not one parameter, even though 128, 16 and 4 are each in a validated tuple.
- Points enter only after a compare. 128 / 2560 / 9728 entered with OPEN-FAMILY-QWEN3 on 2026-09-05; the post-RoPE tuple `(128, 32, 8, 128, True, False, True)` with OPEN-FAMILY-HUNYUAN on 2026-09-06; and on 2026-09-06 OPEN-FAMILY-QWEN3 added `gemv_q4` K 1024 / 3072 / 6144 / 12288, `ln` width 1024, `lm_head_q4` K 1024 / 2048 and the tuple `(128, 16, 8, 128, True, False, False)`, while OPEN-FAMILY-LLAMA3 added `gemv_q4` K 8192, `ln` width 3072, `lm_head_q4` K 3072 and the tuples `(128, 24, 8, 128, False, False, False)` and `(64, 32, 8, 64, False, False, False)`. On 2026-09-06 OPEN-FAMILY-QWEN35's 4B pass added `gemv_q4` K 9216, `lm_head_q8` K 2560 and the tuple `(256, 16, 4, 64, True, True, False)`; `deltanet heads=16` (Qwen3.5 2B / 0.8B) and `lm_head_q8 K=4096` (the 9B) stayed out because those runs did not pass. On 2026-09-07 OPEN-QUANT-Q8's pass added `gemv_q8` K 2048 and 4096 (Ornith-1.0-35B-A3B and five sibling containers); the Qwen3.5 4B's q8 variant added nothing, its `lx` build having overflowed program memory. OPEN-FAMILY-GRANITE added the tuple `(64, 40, 8, 64, False, False, False)` and `gemv_q4` K 8192 on 2026-09-06 -- the first entry at 40 heads. On 2026-09-07 OPEN-FAMILY-QWEN35's remaining three sizes passed and added `deltanet heads=16`, the tuple `(256, 8, 2, 64, True, True, False)` (the 2B / 0.8B), `lm_head_q8` K 1024 and 4096, and `gemv_q4` K 3584 -- the two points that had been held out since 2026-09-06 among them.
- Nine buffer arguments → `9 buffer arguments`.
- The `gemv_q8` template's validated `K` set holds 2048 and 4096, entered by OPEN-QUANT-Q8's hardware pass on 2026-09-07 (Ornith-1.0-35B-A3B); `gemv_q8 K=3072` names that set. Before that pass the set was empty and every q8 export needed `OPEN_KERNELS_UNVALIDATED=1`. The Qwen3.5 family's native-q8 pass on 2026-09-07 added nothing to it: its q8 `linear_out` GEMV reduces over `lin_value_width` (4096 on the 9B / 4B, 2048 on the 2B / 0.8B), not over `hidden`, so all four sizes compose at q8 with no override.
- `catalogue.MIXED_CORE_FITS` is the same idea one level up: not a template parameter but a PROGRAM MEMORY point, the `(family, hidden)` widths whose main core has been built carrying both weight formats' GEMV bodies. It holds `(qwen35, 4096)`, `(qwen35, 2048)` and `(qwen35, 1024)` from OPEN-QUANT-Q8's 2026-09-07 pass. Unlike a template point this one does not refuse: `recipes.load` narrows the container's q8 role away at an unlisted width and says so, because the alternative is an export that composes cleanly and then dies 60 s into `aiecc`.

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

**Mechanism found, and the guess above was wrong (2026-09-07).** It is not the
vector unit failing to saturate. It is **scalar float on the scalar unit**,
inside a loop that runs `heads x positions` times per layer: two `sexp()` per
head per position for the online softmax, one `* 1/sqrt(HD)` per head, and a
bf16 split and compare in the output accumulation. The ablation is the evidence,
one build, one session: dropping q's low bf16 half -- which HALVES the score
MACs -- moved a 185.3 ms step to 185.2, while dropping the single `* kScale`
beside it moved it to 160.1. The head-count correlation the paragraph above
found real (1.25 against 1.19) is explained by it: the cost is per head, and it
is not arithmetic.

Fixed for Granite in `attn.h` behind `ATTN_VEXP` / `ATTN_NHL` / `ATTN_RB`; the
slope goes **2.289 -> 0.0247 ms/position, 92x flatter**, measured end to end
through `flm bench` at 20.6x TTFT and 32.1x decode on a 1005-token prompt.
Every other family compiles byte-identically, so the observation above still
holds for them and this section stays an observation rather than a requirement
until a second family has been measured the same way.

**And prefill is the larger half.** Prefill costs about what a decode step costs
at that position, so it is **quadratic in the prompt length** -- 1005 tokens
took 1216 s before the fix. On a short prompt it reads as a flat per-token cost
and is invisible. Nothing here batches a prompt; that is untouched, and it is
now the dominant term for any document-shaped input.

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
- `designs/gemv_q4/gemv_q8.h` enters the key only for a spec with a q8 role (`KERNEL_SOURCES_Q8`): it is compiled by nothing else, so listing it unconditionally would move every shipped kernel set's key for a file none of them include.
- The key takes `quant` in its canonical form, so a role map hashes (a q8 role changes the key) and an all-`q4_1` map hashes the bare string, byte for byte what the key hashed before roles existed.

### OPEN-PACK-PLAN: the packing plan reproduces the verified pool laws
**Applies to:** openflowlm-next (`open_kernels/recipes/pack.py`, `src/open_qwen36/pools.cpp`)
**Test category:** unit (Python interpreter) + integration (C++, through OPEN-FAMILY-QWEN36MOE)
**Tests:** `tests/test_pack_plan.py`, `tests/legacy_pools.py` (the frozen originals)

The recipe's plan (`expert_stripes`, `expert_down`, `std_perm`, `q8_perm`,
`put`, `conv_transpose`, `transpose`, the lm_head supertile order, the position
table) applied by `recipes/pack.py` shall produce, for a container with the
27B's tensor shapes, exactly the bytes the hand-written packers produced (the
ones verified against pools captured from FLM's engine). `pools.cpp`
interprets the same plan and is verified by the hardware run.

**The q8 band law (`q8_perm`).** Where the recipe's quant map says a projection
runs at q8 (OPEN-QUANT-Q8), the plan carries `q8_perm` instead of `std_perm`
and the pool holds the container's own q8 values: each 8704-byte chunk is split
into two 16-row half-tiles of 5120 bytes -- `scales[128]` bf16 at `[0, 256)`
indexed `kb*16 + r`, `codes[4096]` int8 at `[256, 4352)` indexed `k*16 + r`,
zero pad -- and half-tile `c` of a 64-row band covers rows `16*(c%4)` and
k-tile `c/4`, so its source is file chunk `2*band + (c%4)/2` at half `(c%4)%2`.
A band is `K/64` half-tiles, twice the q4_1 bytes. The split is a byte
permutation, not arithmetic: the container's row-block stride is exactly 4096
codes, so a half-tile's codes are a verbatim slice. `nch` on a `q8_perm` op
counts POOL half-tiles; `chunk0` counts SOURCE file chunks, as it does for
`std_perm`. A `q8_perm` over a tensor the container does not store at q8 is
refused naming the tensor -- that is the check that a container agrees with the
kernel set it is being packed for.

**q8 sources.** The quantized chunk format is a property of the TENSOR, not of
the container: the stock 35B keeps only its lm_head at q8, its fine-tunes pack
attention, linear-attention and shared-expert projections at q8 and only the
routed experts at q4_1, and a Qwen3.5 container stores `ssm_out_proj` and
alpha / beta at q8. The three chunk ops (`std_perm`, `expert_stripes`,
`expert_down`) shall therefore read a q8 source (8704-byte chunks)
transparently and re-quantize it to q4_1 (5120-byte chunks) on the way into the
pool, and refuse any other chunk size naming the tensor. A q8 chunk and a q4_1
chunk hold the same 32-row x 256-column tile, so the chunk index laws, the
plan, the manifest and the kernels are unchanged; the packer is the only place
that knows. There is no separate re-quantizing op.

**Acceptance criteria:**
- Layer pool, consts blob (linear and attention), lm_head pool and ptab are byte-equal to `legacy_pools.py` on random-byte tensors of the right sizes.
- A small weight larger than its slot is refused (`does not fit its 4096 B slot`).
- On a synthetic container mixing one q8 and one q4_1 tensor: the q8 tensor's pool bytes equal `requant_q4_1` of its chunks put through the same `std_perm` order; the q4_1 tensor's are the verbatim chunk copy they were before q8 sources existed; and the whole pool has the same FNV-1a in NumPy and in C++ (`tests/test_pack_plan.py`, `src/open_qwen36/pools_test.cpp`, which writes the container as a real `.q4nx`).
- A container that cannot report a tensor's chunk size is read as q4_1, so the frozen pools above are unaffected.
- A tensor with 1280-byte chunks is refused by both packers, the message naming the tensor and `1280`; it says what the count probably is (8704 = q8, 4736 = Q4_K, 1280 / 2560 = a smaller chunk geometry) rather than guessing "FLM 1.0.3 / Q4_K?".
- `requant_q4_1` (q8 chunks -> q4_1 chunks) and `transpose` (`[rows, cols]` -> `[cols, rows]`) produce identical bytes in NumPy and C++: both sides build the same synthetic q8 chunks and assert the same FNV-1a of the output (`tests/test_qwen35.py`, `src/open_qwen36/pools_test.cpp`).
- Every value of a re-quantized block lands within `d/2` of its q4_1 reading, `d` being the block's stored scale: `m` is the minimum rounded toward -inf in bf16 and `d = (max - m)/15` rounded toward +inf, so `[m, m + 15d]` covers the block whatever bf16 did to either end.
**The q8 lm_head's supertile order is a function of K.** A band is 128 output rows = 4 row
quarters x `nk = K / 256` k-tiles, and the container holds chunk (rowblock32, ktile) at
`rowblock32 * nk + ktile`, so the pool order is
`pool k <- file (4 * (k // per_band) + k % 4) * nk + (k % per_band) // 4`, `per_band = 4 * nk`.
Both packers took `nk = 8` (K = 2048) as a constant until OPEN-FAMILY-QWEN35's 4B run.
The `lmhead_q8` op therefore carries `in_dim` (the hidden width) and an op without it is
refused at load rather than falling back to 2048.

**Acceptance criteria (continued):**
- The `lmhead_q8` order at K = 2048 / 2560 / 4096 is the law above, and at K = 2048 it is byte for byte the shipped 27B one (`tests/test_pack_plan.py`); an `lmhead_q8` op without `in_dim` is refused by both the NumPy packer and the manifest parser, naming the field.
- A `std_perm` without `nch` / `in_dim`, or a `transpose` without `rows` / `cols` / `elem`, is refused by the manifest parser naming the field.
- `transpose` takes an optional `dst_rows`: the destination row is widened to that many values and the tail zeroed (`[16, hid] -> [hid, 32]` with columns 16..31 zero, the 16-head DeltaNet's alpha / beta). It appears in a plan ONLY when it differs from `rows`, so a 32-head family's plan, manifest and build key do not move; `dst_rows` narrower than `rows` is refused by both packers. Both produce the same bytes (`tests/test_qwen35.py`, `src/open_qwen36/pools_test.cpp`).
- `model/q4nx.py` reads each q8 tensor the way the POOL holds it: as the container's own q8 when `native_q8(name)` (the projections the plan streams with `q8_perm`), else as the packer's q4_1. So a slice comparison measures the kernels whichever path a projection is on. `make_decode.py --requant` swings the whole run -- spec, plan, pools and reference -- onto the fallback for the A/B.
- A `q8_perm` half-tile round-trips exactly: dequantizing the two half-tiles of a chunk gives the same values as dequantizing the chunk, value for value. The band law matches a brute-force placement against the dequantized source matrix, and a q8 projection occupies exactly twice the q4_1 bytes.
- The NumPy and C++ packers produce the same `q8_perm` pool bytes (the same FNV-1a in `tests/test_quant_q8.py` and `src/open_qwen36/pools_test.cpp`), and a `q8_perm` without `nch` / `in_dim` is refused by the manifest parser naming the field.

### OPEN-QUANT-Q8: q8 projections run at q8
**Applies to:** openflowlm-next (`designs/gemv_q4/gemv_q8.h`, `designs/layer_x/`,
`designs/dense/`, `recipes/{spec,cache,catalogue,load,qwen36moe,qwen35,dense,pack}.py`,
`src/open_qwen36/{pools,manifest}.*`, `model/{q4nx,replica_qwen35,make_decode}.py`)
**Test category:** manual (needs the NPU and a q8-variant container); the half-tile
split, the band law, the quant-map derivation and the two packers' agreement are
unit-tested in `tests/test_quant_q8.py` and `src/open_qwen36/pools_test.cpp`

A weight projection the container stores at q8 shall be streamed to the main cores at
q8 -- 16-row half-tiles of the container's chunks in the `q8_perm` band law -- and
consumed by `gemv_q8_half_tile`, not re-quantized, whenever the design's core memory
allows. The recipe derives which projections those are from the container (or GGUF)
into `ModelSpec.quant` (OPEN-SPEC-DERIVE), the manifest carries it as `q8_perm` plan
entries, and the packer refuses a container whose tensor formats disagree with it,
naming the tensor. The re-quantizing fallback of OPEN-PACK-PLAN stays for every role a
family cannot stream at q8 -- the routed experts (their own stripe laws) and the MoE's
shared expert (it rides the routed experts' call sites, one nine-slot loop, for program
memory) -- and for the whole model under `OPEN_KERNELS_FORCE_Q4_1=1`.

Because a q8 variant bakes different pool offsets and fill sizes into its instruction
streams, it is a DIFFERENT kernel set: the quant map is in `spec_hash` and `build_key`,
and build directories carry its short hash -- but only when a role is q8, so every
shipped kernel set keeps the directory name it already builds into. "Kernels belong to
families": the family is shape plus quant map.

**Acceptance criteria (unit):** as OPEN-PACK-PLAN's q8 lines and OPEN-SPEC-DERIVE's
quant-map lines, plus:
- A q8 role doubles exactly its own pool region and nothing else; the shipped 27B's `POOL_BYTES` stays 536870912 byte for byte, and a q8 variant of the same shape rounds up to the next MB.
- The MoE's out-projection region is already twice the tensor, so a q8 `linear_out` moves no consts offset there; the qwen35 composition's is the tensor, so it doubles.
- `qwen36moe` refuses a q8 routed expert and a q8 shared expert by name; `dense` and `qwen35` refuse the roles they do not have; a quant the GEMVs cannot read (`q4_k`) is still refused naming it.
- Build directory names gain `_q<hash>` only when a role is q8; `ln` and the lm_head, which read no layer weights, keep theirs either way.
- With no q8 role the designs emit exactly what they emitted before: the same fifos, the same `ExternalFunction` set (no `gemv_q8_*` instantiated), the same core call sequence and the same host fills, for all six checked-in specs, and the generated `.cc` files are byte-identical.
- A spec whose GEMV roles MIX formats -- some q8, some still q4_1 -- generates the folded entry `gemv_q4_gyms` (one entry point, the destination chosen at runtime: below zero the band's y element, otherwise the act scratch at that offset) and NOT `gemv_q4_gy` / `gemv_q4_gms`; an all-q4_1 or an all-q8 spec generates the pair and no folded entry. Both `designs/layer_x` and `designs/dense` take the same switch.
- A mixed spec that also puts `ffn` at q8 is refused by name (`qwen35`, `dense`): the fold covers the q4_1 side only, and a second fold does not fit. Every role at q8 is not refused.
- **A derived quant map is narrowed to a mixed core that has been built.** Whether both formats' GEMV bodies fit in a core's 16 KB is not derivable -- only `aiecc` can measure it, and it does not print the shortfall -- so `catalogue.MIXED_CORE_FITS` is a validated set like any other, keyed by `(family, hidden)`: `(qwen35, 4096)`, `(qwen35, 2048)`, `(qwen35, 1024)`, the three widths whose mixed `lx` built and passed on 2026-09-07. At any other width `recipes.load.narrow_to_buildable` puts a q8 `linear_out` back to q4_1, so the packer re-quantizes it as it did before this requirement, and prints ONE line naming the width, program memory as the reason, `.claude/plans/q8m-hw-results.md`, and the fallback's measured cost (logits corr 0.999682). An all-q8 map is left alone -- one format on the core is not a mixed core -- and `OPEN_KERNELS_FORCE_Q4_1=1` still wins. Concretely: the Qwen3.5 4B (hidden 2560) derives `q4_1` and hashes as its shipped kernel set (`sha256:06e3163f...`); the 9B / 2B / 0.8B keep `{"linear_out": "q8"}`. `tests/test_quant_mixed.py`.

The fold exists because a mixed-format main core carries both formats' GEMV bodies and
16 KB of program memory does not hold three entry points; it is gated on the mix so that
no all-q4_1 and no all-q8 kernel set's object code moves (OPEN-LAYOUT-FREEZE).

**Procedure (manual):** Ornith-1.0-35B-A3B (q8 attention / linear / out projections):
export with `OPEN_KERNELS_UNVALIDATED=1`, then the 8-layer / 3-token slice against
`replica.py` fed the q8 values -- logits corr >= 0.99999, same argmax and top-5,
residual corr >= 0.9999 every layer (the 27B's bar, now against the shipped weights);
the engine CLI bit-identical to the harness; `chat.py`; ms/token beside the
re-quantized path's number. Then Qwen3.8-Distilled-9B with `linear_out` at q8 (its only
q8 projection): the same, and the logits correlation against the q8 reference must now
be >= 0.99999 where the re-quantized path scored 0.999682.

**Result 2026-09-07 (Ornith-1.0-35B-A3B and six sibling containers): PASS on the MoE family,
BLOCKED on Qwen3.5.** Log: `.claude/plans/q8-hw-results.md`.

The MoE half is done. Ornith-1.0 derives `{attn, linear, linear_out} = q8`, builds its own
kernel set (`build_*_q663fca7b`, `spec_hash sha256:640d27a72d49`) with `gemv_q8_gy` in
`gemv_q4_gy`'s place, and its 8-layer / 3-token slice against the replica fed the container's
own q8 values gives logits corr **0.999996 / 0.999998 / 0.999988**, argmax and top-5 identical
at every position, residual corr >= **0.999992** in every layer. The engine reproduced the
harness **bit for bit** (0.000e+00 over 248 320 logits) -- the C++ `q8_perm` packer against the
real container -- and `chat.py` answered coherently over all 40 layers at **155 ms/token
(6.46 tok/s)**. Six more containers (Ornith-1.5, Darwin-36B-Opus, Grug, BigBang 1.0,
Aquila-mini, and Atomic-Germ's own `Qwen3.6-35B-A3B-NPU2` mirror) derive the identical q8 spec
and pass on Ornith's kernels: corr **0.999988 to 0.999998**, argmax matching everywhere, worst
residual 0.999990, every engine run bit-identical.

**The A/B, at position 0 (the only like-for-like one -- the paths pick different second
tokens):** against the weights the author shipped, native q8 scores corr **0.999996** with an
identical top-5; the re-quantizing fallback scores **0.997631** and puts the wrong token 5th.
By the second token the fallback's greedy pick diverges. Speed: 155-170 ms/token at q8 against
140-162 re-quantized on the same six models, i.e. **~+9 % of wall time**, not the ~+50 % the
plan budgeted from weight bytes -- decode is not bound by the non-expert projections' DMA.

**Result 2026-09-07 (the mixed-format fold on hardware): PASS on three of the four Qwen3.5
sizes; the 4B alone is still over program memory.** Log: `.claude/plans/q8m-hw-results.md`.

Every Qwen3.5 container derives `linear_out: q8` while its other projections stay q4_1, so
the `lx` main core must hold BOTH GEMV bodies. The fold applies only to such a spec: the
core holds `gemv_q4_gyms` + `gemv_q8_gy` instead of `gemv_q4_gy` + `gemv_q4_gms` +
`gemv_q8_gy`, and its two GEMV translation units are compiled `-Oz`. `gemv_q4_gyms` went
through Peano for the first time here and compiled clean, and the 9B, 2B and 0.8B built,
ran and passed. The gate held on both formats before any of it: the shipped 35B re-exported
6/6 `insts.bin` byte-identical (xclbins stamps only, `lx0` / `lx1` at 176 399 B), and the
all-q4_1 Qwen3.5 4B 4/4 byte-identical -- so neither an all-q4_1 nor an all-q8 kernel set
moved.

**The second acceptance criterion is met.** Against the replica fed the container's own q8
`ssm_out_proj`, the 9B's 8-layer / 3-token slice gives logits corr **0.999999 / 0.999991 /
0.999993** with identical argmax and top-5 at every position and residual corr >= 0.999994
in every layer -- above the 0.99999 bar, where the re-quantizing fallback scored 0.999682.
The 2B gives 0.999998 / 0.999989 / 0.999986 (better than its own q4_1 run's 0.999979 /
0.999980) and the 0.8B 0.999993 / 0.999992 / 0.999990, both with argmax and top-5 matching
everywhere. Each engine run reproduced its harness **bit for bit** (0.000e+00 over 248 320
logits x 3) and answered the chat prompt coherently over all layers. Speed: 191 / 70 / 54
ms/token against the q4_1 path's 181 / 69 / 53, i.e. **+5.5 % on the 9B and about 1 ms on
the two small ones** -- the same "no measurable cost" the MoE found.

**The 4B still overflows**, in the same place and with the same message, and neither flag
lever recovers it: the fold plus `-Oz` on the two GEMV TUs is not enough, and `-Oz` on every
TU of the mixed core (the handoff's next lever, applied and then reverted to the byte)
changes nothing. The 4B is the only size whose hidden width is not a multiple of the 4 KB
element, so its glue side channel walks two unequal halves and its all-q4_1 `lx` is already
the largest of the four (178 239 B against the 9B's 156 623). What is left is a design
change -- splitting the FFN tail off that core -- or leaving the 4B on the re-quantizing
fallback its three siblings no longer need. `.claude/plans/q8m-hw-results.md` §2.

**No catalogue point was earned, and none was needed.** The q8 out projection's GEMV
reduces over `lin_value_width` -- 4096 on the 9B and 4B, 2048 on the 2B and 0.8B -- not over
`hidden`, so both K were already validated by the 35B pass and all four sizes compose with
no `OPEN_KERNELS_UNVALIDATED`.

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

**Result 2026-09-06 (the six q8 fine-tunes): all six PASS on the reference kernels, no rebuild.**
Ornith-1.0-35B-A3B, Darwin-36B-Opus, Grug-35B-A3B, BigBang1.0-35B-A3B, Aquila-mini-35B-A3B and
Ornith-1.5-35B-A3B each derive the reference 35B's ModelSpec exactly (`spec_hash
sha256:32e980528551`) and run on `src/xclbins/Qwen3.6-35B-A3B-NPU2/open_kernels`. Acceptance
(the replica fed the q4_1 the packer writes -- OPEN-PACK-PLAN's q8-source path): logits corr
**0.999987 to 0.999998** at both positions of an 8-layer / 2-token slice, **argmax and top-5
identical everywhere**, worst residual corr 0.999995, request 2 reproduced request 1 in every
fast proof, and all six answered the chat prompt coherently over 40 layers at 6.2-7.1 tok/s.

**Result 2026-09-07 (the same containers at native q8, OPEN-QUANT-Q8):** with the projections
streamed at q8 the seven q8 containers (the six above plus Atomic-Germ's 35B mirror) pass on a
q8 kernel set built from Ornith-1.0's spec -- corr 0.999988 to 0.999998 over an 8-layer /
3-token slice, argmax matching at every position, worst residual 0.999990, every engine run
bit-identical to the harness, chat coherent at 153-170 ms/token. `.claude/plans/q8-hw-results.md`.
Ornith-1.0's own export reports all six `insts.bin` byte-identical and all six `final.xclbin`
stamps-only.

Two earlier findings are corrected by this run: **Ornith-1.5 is a drop-in**, not a 41-layer
model needing its own kernel set -- its container carries a 41st layer's tensors but its
`config.json` says `num_hidden_layers: 40` with `mtp_num_hidden_layers: 1`, and the extra set is
the multi-token-prediction head this engine does not read; and **Ornith-1.0 is unblocked** by the
`qwen3_5_moe_text` alias, though a `manifest.json` generated before 2026-09-06 still refuses it
by name until it is regenerated.

**Quality, reported separately** (the same NPU run against the replica fed the container's own q8
weights, `make_decode --q8-weights`): logits corr **0.9966-0.9981** at position 0 and
**0.985-0.991** at position 1, and on Ornith-1.0 and BigBang the greedy pick at position 1 flips
between the top two candidates. Re-quantizing 251 q8 tensors to q4_1 costs about 1e-2 of logits
correlation -- three orders of magnitude more than the kernels' own 1e-5 -- and is enough to
change generated text. That is the evidence for a main-core q8 GEMV. Log:
`.claude/plans/q-hw-results.md`.

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

The same recipe composes the other three published Qwen3 dense shapes. All
three have now run this procedure (2026-09-06, results below) and their points
are in the catalogue:

| shape | points it added |
|---|---|
| 8B (4096 / 36 / 12288; also DynaGuard-8B, DeepSeek-R1-0528-Qwen3-8B) | `gemv_q4` K = 12288 only; `PER_CALL` drops to 1 |
| 1.7B (2048 / 28 / 6144, 16 heads) | `gemv_q4` K = 6144, `lm_head_q4` K = 2048, and the `attn` TUPLE (128, 16, 8) -- a GQA group of 2 that the catalogue's per-parameter check had been passing silently, which is why OPEN-OP-RANGE now validates the attention geometry as a whole tuple |
| 0.6B (1024 / 28 / 3072, 16 heads) | `ln` width 1024, `gemv_q4` K = 1024 and 3072, `lm_head_q4` K = 1024; the (128, 16, 8) `attn` tuple is the 1.7B's |

**Acceptance criteria (unit):**
- The 4B layout and manifest as `tests/test_qwen3_dense.py` asserts them.
- 8B: `PER_CALL 1`, `TAB_BYTES 27648`, `ELN 8192`, band split `(8, 2, 8, 24, 8)`, `LMHEAD_BAND_BYTES 163840`.
- 1.7B: `PER_CALL 2`, `TAB_BYTES 13824`, `ELN 4096`, band split `(4, 2, 4, 12, 4)`, head K 2048, `num_heads // num_kv_heads == 2`.
- 0.6B: `ELN 2048` (half an x-stream element), `TAB_BYTES 6912`, band split `(4, 2, 2, 6, 2)`, `attn_q_width == 2 * hidden`, `ln` built at 1024.

**Procedure:**
1. `python open_kernels/export_qwen36_kernels.py --model-dir ~/.flm/models/Qwen3-4B-NPU2` (WSL) → `src/xclbins/Qwen3-4B-NPU2/open_kernels/{dx,ln,lm_head_q4}` + `manifest.json`.
2. `python open_kernels/model/make_decode.py --model-dir ~/.flm/models/Qwen3-4B-NPU2 --layers 4 --tokens 2 --out open_kernels/model/out_q3`, then `open_kernels/harness/out/run_kernel.exe open_kernels/model/out_q3/run_decode.cfg` and `python open_kernels/model/compare_decode.py --tokens 2 --out open_kernels/model/out_q3`: every layer's residual corr > 0.9999, logits corr > 0.9999, same argmax at both positions.
3. `src\open_qwen36\out\open_qwen36_cli.exe --model <model dir> --kernels src/xclbins/Qwen3-4B-NPU2/open_kernels --ids 151644 --max-tokens 3 --layers 4 --dump-logits <dir>/y` matches step 2's reference logits (the engine's packer, manifest path and attnpos on the dense stream).
4. `python src/open_qwen36/chat.py "Explain what an NPU is in two sentences." --model <model dir> --kernels src/xclbins/Qwen3-4B-NPU2/open_kernels` → a coherent answer ending in `<|im_end|>`.

**Adapters (manual):** every Qwen3-dense adapter class selects the open engine when a
kernel set is installed for its model, and honours `FLM_QWEN3_ENGINE=open|closed`:
`Qwen3`, `Qwen3_IT`, `Qwen3_TK` and `DeepSeek_r1_0528_8b` (`model_list.json`
families `qwen3`, `qwen3-it`, `qwen3-tk`, `deepseek-r1-0528`). Verify with
`flm serve <tag>` on a model that has `open_kernels/` installed: the load logs
`<Family> on the open kernels (<dir>)` -- `Qwen3`, `Qwen3-IT`, `Qwen3-TK`,
`DeepSeek-R1-0528` respectively -- and `FLM_QWEN3_ENGINE=closed` restores the
`qwen3_npu` DLL for all four.

**Result 2026-09-05 (Qwen3-4B):** step 2 logits corr 0.999997 / 0.999994, same argmax and top-5, residual corr ≥ 0.999996 in every layer at both positions; step 3 identical through the engine, request 2 reproduced request 1; step 4 a coherent two-sentence answer ending in `<|im_end|>` at token 58 (272 ms/token). Details: `.claude/plans/open-kernels-phase-b-qwen3-dense.md`.

**Result 2026-09-06 (DynaGuard-4B-NPU2):** a fine-tune whose spec is identical to Qwen3-4B's in every field but `extra` -- step 2 logits corr 0.999999 / 0.999992, same argmax (11619) and top-5, residual corr >= 0.999996 every layer (maxrel <= 1.6e-3); step 3 identical through the engine, request 2 reproduced request 1; step 4 a coherent two-sentence answer ending in `<|im_end|>` at token 62. The SAME check ran first against `src/xclbins/Qwen3-4B-NPU2/open_kernels` with the same numbers, and its own export is byte-identical to that set (3/3 `insts.bin`, xclbins stamps only). Details: `.claude/plans/p0-p1-results.md`.

**Result 2026-09-06 (Qwen3-4B-Thinking-2507-NPU2):** same spec hash as Qwen3-4B (`sha256:602fa1836b21`) -- step 2 logits corr 0.999998 / 0.999994, same argmax (50179) and top-5, residual corr >= 0.999997 every layer; step 3 identical through the engine, request 2 reproduced request 1; step 4 (`--think`) a reasoning chain closed with `</think>` then a coherent two-sentence answer ending in `<|im_end|>` at token 285. Its `--no-build` export is byte-identical to `src/xclbins/Qwen3-4B-NPU2/open_kernels` (3/3 `insts.bin`), and the same slice passed against that directory directly. Details: `.claude/plans/p0-p1-results.md`.

**Result 2026-09-06 (Qwen3-8B-NPU2, and DynaGuard-8B / DeepSeek-R1-0528-Qwen3-8B on its kernels):** the 8B shape's first hardware run. Step 2 logits corr 0.999998 / 0.999997, argmax 104222 / 118063 matching the fp64 replica with identical top-5, residual corr 1.000000 (t0) / >= 0.999998 (t1) in all four layers, maxrel <= 1.1e-3; step 3 through the engine is BIT-IDENTICAL to step 2 (0.000e+00 over all 151936 logits) and request 2 reproduced request 1; step 4 a coherent two-sentence answer ending in `<|im_end|>` at token 63 (540 ms/token, 36 layers, a loaded box). This admits `gemv_q4` K = 12288 to the catalogue; the 4096 norm, the (128, 32, 8, 128, qk-norm, no gate) attention tuple and the K = 4096 q4 head were already in it. DynaGuard-8B (spec hash identical, `sha256:04374f23aede`) passed the same procedure on its own `--no-build` export, byte-identical to Qwen3-8B's in all six artefacts: corr 0.999997 / 0.999997, same argmax and top-5, `<|im_end|>` at token 59. DeepSeek-R1-0528-Qwen3-8B (the same shape, `real_vocab` 151671 instead of 151669) passed on correlation and residuals -- corr 0.999993 / 0.999996, residual corr >= 0.999997 every layer -- but its position-1 ARGMAX differs: the fp64 top two, 102188 and 108204, are 0.004 logits apart on a 14.5-logit scale and the NPU's ~0.015 per-logit deviation flips them, with slots 3-6 unchanged. Harness and engine agree bit for bit, so this is a tie inside q4_1 noise rather than a kernel disagreement, but it is the first time it has crossed `compare_decode`'s "same argmax" bar. `chat.py` handles this tokenizer now (the `<｜Assistant｜>` branch landed this session), answering coherently after a `</think>` chain. Details: `.claude/plans/k-new-points-results.md`.

**Result 2026-09-06 (Qwen3-1.7B-NPU2, and Qwen3-1.7B-NPU2-BASE on its kernels):** the 16/8-head shape's first hardware run, and the first run of the attention TUPLE (128, 16, 8, 128, qk-norm, no gate, pre-RoPE) -- a GQA group of 2 at head dim 128, which the catalogue's per-parameter check had been passing silently and now names. Step 2 logits corr 1.000000 / 0.999993, argmax 1121 / 17764 matching the fp64 replica, top-5 identical at t0 and slots 1-4 identical at t1 (slot 5 differs, 36976 against 78200), residual corr >= 0.999992 in every layer at both positions, maxrel <= 3.9e-3; step 3 through the engine BIT-IDENTICAL to step 2 and request 2 reproduced request 1 (4-layer slice 10-11 ms/token); step 4 a coherent answer ending in `<|im_end|>` at token 50 (139 ms/token, 28 layers). This admits `gemv_q4` K = 6144, `lm_head_q4` K = 2048 and the (128, 16, 8, 128, True, False, False) attention combination. Qwen3-1.7B-NPU2-BASE passed the same procedure identically on a `--no-build` export byte-identical in all six artefacts -- because it IS the same container: its `model.q4nx`, `config.json` and `tokenizer_config.json` have the same sha256 as `FastFlowLM/Qwen3-1.7B-NPU2`'s, chat template included, so it is the instruct model published under a `-BASE` name rather than a base checkpoint. Details: `.claude/plans/k-new-points-results.md`.

**Result 2026-09-06 (Qwen3-0.6B-NPU2):** the narrowest shape the recipe has produced. Step 2 logits corr 0.999999 / 0.999986, argmax 1121 / 460 matching the fp64 replica with top-5 identical at BOTH positions, residual corr >= 0.999982 in every layer at both positions (the loosest number in the batch, on the narrowest residual in the tree), maxrel <= 4.4e-3; step 3 through the engine bit-identical to step 2 and reproduced; step 4 a fluent answer ending in `<|im_end|>` at token 36 (57 ms/token, 28 layers -- the fastest model in this batch; the answer is factually wrong about NPUs, which is a 0.6B model being a 0.6B model, not a kernel result). This admits `ln` width 1024 -- the first width other than 2048 to take the fused single-core path -- plus `gemv_q4` K = 1024 and K = 3072 and `lm_head_q4` K = 1024. The three geometry firsts the handoff flagged all behaved: the half-used x-stream element (`ELN` 2048 against a 4096-byte element) is read correctly, with no position-independent offset on the first residual; the o-projection GEMV being wider than the layer's own residual (q width 2048, hidden 1024) changes nothing. Details: `.claude/plans/k-new-points-results.md`.

### OPEN-FAMILY-LLAMA3: Llama 3 on the dense recipe
**Applies to:** openflowlm-next (`open_kernels/recipes/dense.py`, `spec.py`, `designs/dense/dx.py`, `src/open_qwen36/`)
**Test category:** manual (needs the NPU and `FastFlowLM/Llama-3.1-8B-NPU2`); the derivation, the RoPE scaling and the 8B / 3.2-3B / 3.2-1B layouts are unit-tested in `tests/test_llama3.py`

A Llama 3 model (GQA without q/k norms, full RoPE with the llama3 frequency
scaling, eps 1e-5, silu FFN, a q4_1 head) shall run on the open kernels
from its `config.json` alone through the dense recipe: `qk_norm` / `norm_eps`
become the `ATTN_QKNORM` / `LN_EPS` knobs, the scaled inverse frequencies are
computed host side (`ModelSpec.rope_inv_freq`, in the manifest, used by both
position-table builders), and widths that overflow a core's memory are handled
by the recipe (one chunk per weight element; one norm output element per call).

`tie_word_embeddings` is not a refusal. Llama 3.2 (1B / 3B) ties the head to
the embedding table in `config.json`, but every container the recipe packs from
materialises `lm_head.weight` as its own q4 tensor -- FLM's `.q4nx` does it for
`Llama-3.2-{1,3}B-NPU2` (I8 `[32064, 5120]` / `[48096, 5120]`, the whole
128256-row head), and `utilities/q4nx-build` does it for a tied GGUF
(OPEN-FAMILY-HUNYUAN's converter). The derivation sees only `config.json`, so
the invariant is enforced where it is observable: `recipes/pack.py` refuses a
container that lacks the tensor, naming it.

**Acceptance criteria (unit):**
- HF and GGUF derivations agree; `rope_inv_freq()` equals transformers' `_compute_llama3_parameters` for the 8B's parameters; a non-llama3 `rope_scaling` is refused.
- `tie_word_embeddings: true` derives the SAME spec as `false` (the flag is not a spec field), the pack plan still names `lm_head.weight`, and a container without that tensor is refused by `pack.apply_op` naming `lm_head.weight`.
- The 8B layout: 8 KB norm elements, one chunk per weight element (`TAB_BYTES 32256`), `PER_CALL 1`; Qwen3-4B keeps two.
- Llama 3.2 3B (3072 / 28 / 8192, 24 heads): `PER_CALL 2`, `TAB_BYTES 18432`, `ELN 6144`, band split `(6, 2, 6, 16, 6)`, `OG_AOUT_ELEMS 3`, `LMHEAD_BANDS 2004`, `ln` built at 3072 and the head at K = 3072.
- Llama 3.2 1B (2048 / 16 / 8192, head_dim 64): `E_A 1024`, `KV_ROW 2048`, `PTAB_ROW 1024` (the RoPE record is 768 B at `rotary_dim` 64), `KV_PC 1`, 32 inverse frequencies.

**Procedure (manual):** as OPEN-FAMILY-QWEN3 with `Llama-3.1-8B-NPU2`, `out_l3`, prompt id 128000, and `chat.py` (which switches to the Llama 3 template when the tokenizer has `<|start_header_id|>`).

**Adapters (manual):** both Llama-3 adapter classes select the open engine when a
kernel set is installed for their model, and honour `FLM_LLAMA_ENGINE=open|closed`:
`Llama3` (`model_list.json` families `llama3.1`, `llama3.2`) and `DeepSeek_r1_8b`
(family `deepseek-r1`, a Llama-3.1-8B distill). Verify with `flm serve <tag>` on a
model that has `open_kernels/` installed: the load logs `Llama 3 on the open kernels (<dir>)`
or `DeepSeek-R1 on the open kernels (<dir>)`, and `FLM_LLAMA_ENGINE=closed` restores
the `llama_npu` DLL for both.

**Result 2026-09-05 (Llama-3.1-8B):** slice logits corr 1.000000 / 0.999993, same argmax and top-5, residual corr ≥ 0.999994 every layer; identical through the engine; a coherent two-sentence answer ending in `<|eot_id|>` at token 79 (203 ms/token). Details: `.claude/plans/open-kernels-phase-c-llama3.md`.

**Result 2026-09-06 (Deepseek-R1-Distill-Llama-8B-NPU2):** spec identical to Llama-3.1-8B's in every field but `extra` -- step 2 logits corr 0.999999 / 0.999987, same argmax (12451 / 37533), residual corr 1.000000 / >= 0.999987 every layer; top-5 identical at position 0 and slots 5-6 swapped at position 1 on a 0.025-logit tie; step 3 identical through the engine, request 2 reproduced request 1; step 4 a coherent answer ending in `<|end_of_sentence|>` at token 429. Its own export is byte-identical to `src/xclbins/Llama-3.1-8B-NPU2/open_kernels` (3/3 `insts.bin`), and the same slice passed against that directory directly. Caveat: `chat.py`'s template probe refuses this tokenizer (`tokenizer lacks ['<|end_of_text|>']`) because R1-Distill keeps Llama 3's `<|start_header_id|>` but renames the EOS tokens -- an engine-external gap, driven through `open_qwen36_cli` with DeepSeek's own template instead. Details: `.claude/plans/p0-p1-results.md`.

**Result 2026-09-06 (Llama-3.2-3B-NPU2 and Llama-3.2-1B-NPU2):** both 3.2 shapes' first hardware run, and both containers materialise `lm_head.weight` despite `tie_word_embeddings: true`, as the acceptance criteria above assume. **3B** (3072 / 28 / 8192, 24 query heads over 8 kv heads -- GQA group 3, the first odd group, `OG_AOUT_ELEMS` 3): step 2 logits corr 0.999998 / 0.999995, argmax 2 / 2 matching the fp64 replica with top-5 identical at both positions, residual corr 1.000000 in all four layers at t0 and >= 0.999996 at t1, maxrel <= 5.3e-3; step 3 through the engine bit-identical to step 2 and reproduced; step 4 a coherent answer ending in `<|eot_id|>` at token 72 (334 ms/token). **1B** (2048 / 16 / 8192, head_dim 64 -- `E_A` 1024, `KV_ROW` 2048, `PTAB_ROW` 1024, one KV band per core, a 768-byte RoPE record in a 1024-byte position row): step 2 logits corr 0.999999 / 0.999994, argmax 1757 / 1757 matching, top-5 identical at both positions, residual corr 1.000000 in all four layers at t0 and >= 0.999996 at t1, maxrel <= 2.4e-3; step 3 bit-identical and reproduced; step 4 `<|eot_id|>` at token 92 (262 ms/token). The 1B is the run that decides head dim 64, since a wrong q/k rotation there gives fluent nonsense rather than a crash: position 0 does not rotate and position 1 does, and both are clean in every layer, so the rotation is right. Together these admit `gemv_q4` K = 3072 and K = 8192, `ln` width 3072, `lm_head_q4` K = 3072 and the attention combinations (128, 24, 8, 128, False, False, False) and (64, 32, 8, 64, False, False, False). Note `chat.py`'s default `--max-tokens 64` truncates both models mid-sentence; 200 is enough. Details: `.claude/plans/k-new-points-results.md`.

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

**Adapters (manual):** both Gemma-3 adapter classes select the open engine when a
kernel set is installed for their model, and honour `FLM_GEMMA_ENGINE=open|closed`:
`Gemma3` (`model_list.json` family `gemma3`, e.g. `gemma3:4b`) and `Gemma3_Text_Only`
(family `gemma3-text`, e.g. `gemma3:1b`). Verify with `flm serve <tag>` on a model
that has `open_kernels/` installed: the load logs `Gemma 3 on the open kernels (<dir>)`
or `Gemma 3 (text) on the open kernels (<dir>)`, and `FLM_GEMMA_ENGINE=closed`
restores the `gemma_npu` / `gemma_text_npu` DLL. Images always need the closed
engine -- the open one has no vision path.

**Result 2026-09-05 (Gemma3-4B):** slice logits corr 0.999998 / 0.999998, same argmax and top-5, residual corr 1.000000 every layer; identical through the engine; a finite step at position 1103; a coherent two-sentence answer ending in `<end_of_turn>` at token 43 (96 ms/token). Details: `.claude/plans/open-kernels-phase-d-gemma3.md`.

**Result 2026-09-06 (Gemma3-4B-Text-NPU2, medgemma-1.5-4b-it-NPU2, Translategemma-4B-Instruct-NPU2):** three fine-tunes whose specs are identical to Gemma3-4B's in every field but `extra` -- step 2 logits corr 0.999998-0.999999 at both positions, same argmax, residual corr 1.000000 in all six layers (maxrel <= 6.1e-4); step 3 identical through the engine, request 2 reproduced request 1; a finite step at position 1101 through the window path for each; step 4 a coherent answer ending in `<end_of_turn>` at tokens 43 / 62 / 27. Each model's own export is byte-identical to `src/xclbins/Gemma3-4B-NPU2/open_kernels` (3/3 `insts.bin`, xclbins stamps only, manifests differing only in `spec.extra.model` and `build_key`), and the same slice passed against that directory directly first. medgemma's top-5 reorders in slots 2-4 at position 1 on a 0.02-logit tie, identically through harness and engine. Gemma3-4B-Text reproduces the reference model's answer token for token. Details: `.claude/plans/p0-p1-results.md`.

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

### OPEN-FAMILY-QWEN35: Qwen3.5 dense on the open kernels
**Applies to:** openflowlm-next (`open_kernels/recipes/qwen35.py`, `spec.py`, `qwen36moe.py`,
`designs/layer_x/lx.py`, `ax.py`, `xcommon.py`, `dnx.h`, `designs/dn_glue/glue_copy_e.cc`,
`designs/lm_head_q8`, `recipes/pack.py`, `src/open_qwen36/pools.cpp`, `manifest.cpp`,
`model/replica_qwen35.py`)
**Test category:** manual (needs the NPU and a Qwen3.5 container); the derivation, the
composed layout and the pack ops are unit-tested in `tests/test_qwen35.py`,
`tests/test_pack_plan.py` and `src/open_qwen36/{manifest_test,pools_test}.cpp`

A Qwen3.5 dense model (gated DeltaNet linear-attention layers with a gated
full-attention layer every fourth, a silu-gated dense FFN, q8 lm_head, `model_type`
`qwen3_5` or `qwen3_5_text`) shall run on the open kernels from its `config.json`
alone: the `qwen35` recipe composes the qwen36moe recipe's attention half (Layout /
Common / Linear / Attn with `ffn="dense"`, so the DeltaNet and attention constants are
the MoE's and not a re-derivation) with the dense recipe's FFN half, and `lx.py` /
`ax.py` build with the FFN tail and the plain norm helper selected by `R.kind`, ONE
instruction stream per layer type -- nothing is routed, so there is no part split. The
q8 `ssm_out_proj` is streamed at q8 where the derived quant map says so (`q8_perm`,
OPEN-QUANT-Q8) and re-quantized to q4_1 by the ordinary `std_perm` op otherwise -- the
plan the 35B uses for its q4_1 copy of that tensor; alpha and beta are read from
their bf16 `[heads, hidden]` copies through `transpose` into the `[hidden, heads]`
layout `glue_ab` reads. Images are refused as on the other VLM families.

**Acceptance criteria (unit):**
- `ModelSpec.from_hf_config` on the 9B / 4B / 2B / 0.8B `config.json` (fixtures under `tests/fixtures/`, the models' own files) gives family `qwen35`, `num_experts 0`, `intermediate` 12288 / 9216 / 6144 / 3584, the MoE's layer pattern, 16/4 (or 8/2) heads and `lin_value_heads` 32 (or 16); the nested `text_config` (`qwen3_5_text`) and FLM's flattened container config derive the same tower; `qwen3_5_moe` still derives to `qwen36moe`, and a config carrying `num_experts` is refused by name.
- Swapping only the FFN moves nothing in the attention half: the 27B spec and a dense twin of it (an FFN narrow enough to keep 10 KB weight elements) give identical DeltaNet / attention / state / KV / lm_head constants, and `qwen35.layout` is `qwen36moe.layout(..., ffn="dense")`, not a copy.
- The 9B layout: `PER_CALL 1` (a 12288-wide activation table leaves no room for two 10 KB weight elements beside the streams), `DN_ROWS 10 / DN_SLICES 13 / DN_PAD 130`, `S_ROWS 130`, `ELN 8192` (so the split `ln_y` / `ln_xn` norm entries), `E_A 2048` with 2 f32 heads per attention element and 4 og heads, `KV_ROW 4096`, `PTAB_ROW 2048`; the pool holds q4-sized `up | gate | down` first, at the same offsets for both layer types.
- No `moe` block, no `rout_idx_off`, no router or shared-expert tensor anywhere in the manifest; each layer type's program is one `run`, with `attnpos` on the full-attention kernel only.
- The manifest fixture parses in `manifest_test.cpp` (a linear-attention layer type with a one-step program and no `moe`); `ssm_out_proj` is a plain `std_perm` with no source-format field, and the two `transpose` ops carry the sizes `pools::apply` needs.
- **One record per value head.** The glue core emits `(NT - VALUE_TILE0) * HEADS_PER_TILE` records and the host drains one per value head; the two are equal only at 32 value heads (4 value conv tiles), so a 16-head model has 2 value tiles against its 4 key tiles. The value head's key head is `h / (lin_value_heads / lin_key_heads)` -- 2 value heads per key head at 32, one at 16.
- **The alpha / beta projection is padded to the accumulator's 32 lanes**, not narrowed: a W element stays 64 rows x 32 bf16 = 4 KB, `AB_ELEMS` is `hidden / 64` whatever the head count, and a 16-head model's `transpose` op carries `dst_rows` so columns 16..31 are zero. `dt_bias` sits at `lin_value_heads` floats inside `small`, not at a fixed 32.
- **The projection is walked in 4 KB halves.** The glue core holds ONE element of the layer-entry norm output, so the alpha and beta projections are re-streamed per half with the accumulator reset passed in (`glue_ab_e.cc`); a half carries `min(2048, hidden - h*2048) / 64` weight tiles, which is 32 and 8 at HID 2560. The side channel's fills are `2 + 4 * ceil(hidden*2 / 4096)` and the recipe refuses a hidden width whose count exceeds `LIMITS["shim_fills"]`, naming the number.

**Procedure (manual):** as OPEN-FAMILY-QWEN36MOE with `Qwen3.8-Distilled-9B-NPU2`,
`out_q35`, an 8-layer slice (six linear, two full), 3 greedy tokens from `[248045]`;
then the engine CLI, then `chat.py` (the Qwen template). The same procedure runs each
published size: 4B (passed 2026-09-06), 9B, 2B and 0.8B. The new kernel points (K 12288
GEMVs, `lm_head_q8` at K 4096, a 16/4-head gated attention at HD 256, `deltanet
heads=16`, an 8/2-head gated attention at HD 256) are built with
`OPEN_KERNELS_UNVALIDATED=1` until this passes, then added to `recipes/catalogue.py`.
The 4B is also the DENSE path's regression whenever the glue's projection walk changes:
its logits must reproduce byte for byte.
Thresholds as the 35B: logits corr >= 0.99999 against the replica **fed the same
re-quantized out_proj** (`replica_qwen35.py`'s default), same argmax and top-5, residual
corr >= 0.9999 every layer. Reported separately: the same slice against the replica fed
the q8 out_proj -- the quality cost of the re-quantization decision.

**Adapters (manual):** the Qwen3.5 adapter class selects the open engine when a kernel set
is installed for its model, and honours `FLM_QWEN35_ENGINE=open|closed`: `Qwen3_5VL`
(`model_list.json` family `qwen3.5`, e.g. `qwen3.5:4b`). Verify with `flm serve <tag>` on a
model that has `open_kernels/` installed: the load logs
`Qwen3.5 on the open kernels (<dir>)`, and `FLM_QWEN35_ENGINE=closed` restores the
`qwen3_5vl_npu` DLL. Images always need the closed engine -- the open one has no vision
path, and an image payload is refused with
`images need the closed Qwen3.5 engine (FLM_QWEN35_ENGINE=closed)`.

**Result 2026-09-06 (Qwen3.8-Distilled-4B-NPU2, HID 2560 / 32 layers / FFN 9216): PASS.**
Slice (8 layers, 3 tokens from `[248045]`, six linear and two full): logits corr **0.999999 /
0.999992 / 0.999989** against the replica fed the re-quantized weights, **argmax and top-5
identical at all three positions**, residual corr >= **0.999991** in every layer, maxrel <=
3.1e-03. The engine reproduced the harness **bit for bit** (max abs difference 0.000e+00 over
248 320 logits at every position) and request 2 reproduced request 1; every step trace shows
`route 0.00 / part1 0.0`, i.e. one instruction stream per layer type. All 32 layers answered the
chat prompt coherently, `[eos]` at token 85, 150 ms/token (6.66 tok/s). Points added to
`recipes/catalogue.py`: `gemv_q4 K=9216`, `lm_head_q8 K=2560`, `attn (256, 16, 4, 64, True, True,
False)`. Two bugs were fixed to get here, both outside this requirement's own code:
`lm_head_q8.py` never generated its `gemv_q4_prep_k{K}` TU, and the q8 head's pool order was
hardcoded to K = 2048 in both packers (OPEN-PACK-PLAN).

**Result 2026-09-07 (the 4B at native q8): the recipe narrows it.** The 4B's container stores
`ssm_out_proj` at q8 like its three siblings, so its `lx` main core would carry `gemv_q8_gy`
beside the q4_1 `gemv_q4_gy` and `gemv_q4_gms`; the build dies in `aiecc` with
`_XAie_LoadProgMemSection: Overflow of program memory`, and both flag levers (`-Oz` on the two
GEMV translation units, then on the whole core) were spent without recovering it. Hidden 2560
is therefore NOT in `catalogue.MIXED_CORE_FITS`, and the recipe derives `q4_1` for this size
with a one-line note rather than composing an export that cannot build (OPEN-QUANT-Q8). The
other three sizes keep their container's q8 role. The q4_1 path is unaffected and still passes: the shipped
kernel set (its manifest regenerated with `OPEN_KERNELS_FORCE_Q4_1=1`) answered the chat prompt
over all 32 layers at 124 ms/token (8.06 tok/s) after the native-q8 merge.
`.claude/plans/q8-hw-results.md` §2.

**Result 2026-09-06 (Qwen3.8-Distilled-9B-NPU2, HID 4096): NOT RUN -- the `lx` build does not
fit.** `ax`, `ln` (ELN 8192, the split norm entries) and `lm_head_q8` at K 4096 all build; `lx`
fails in aiecc with `'aie.tile' op allocated buffers exceeded available memory` on tile (2, 3),
the DeltaNet glue core, which sums to 68 096 B against 65 536. Everything on that core is
HID-independent except `xnb`, the private bf16[HID] copy of the layer-entry norm output that
`glue_ab_tile` walks tile by tile: 59 904 B of fixed allocations leave it **5 632 B, i.e.
HID <= 2816**. R5 in the handoff covered the norm helper's elements and the glue's *copy*
(`glue_copy_xn_e`) but never added up the glue core's L1. A fix means re-streaming the xn per
half (a `glue_ab_e.cc` with the accumulator reset passed in, a DENSE branch in `glue_body`, and
an interleaved `tg_s` fill sequence). Log: `.claude/plans/q-hw-results.md`.

**Result 2026-09-06 (Qwen3.8-Distilled-2B-NPU2 and Qwen3.5-0.8B-NPU2): NOT RUN -- both hang.**
All four kernel sets build for each, and both then time out on the first `lx` dispatch (ERT state
8) in the harness and the engine alike. Both have `linear_num_value_heads` 16 -- the
`deltanet: heads=16 is outside the validated set {32}` point -- and
`designs/dn_glue/dn_glue.h` carries `kNHead = 32` as a `static constexpr` that no recipe value
reaches. Making it a knob (as `DNX_ROWS` is) plus sizing the glue core's `acc_a` / `acc_b` /
`decay` / `beta` from it is a separate piece of work with its own compare.

**Result 2026-09-07 (all four sizes on the q4_1 path): PASS -- the family is complete.** The two
blockers above are fixed and every published size now runs the whole procedure. Every export used
`OPEN_KERNELS_FORCE_Q4_1=1`, because a Qwen3.5 container derives `linear_out: q8` and the
mixed-format `lx` core still overflows program memory (the q8 Result above); the native-q8 run
waits on that lever.

Before the models, the standalone `dn_glue` design was built at `DNGLUE_NHEAD=16` and compared
against the fp64 reference for a 16-head record set: **new conv state bit-exact (0 of 18 432 bf16
differ)** and cos 1.00000000 on k / q / v / decay / beta (maxrel <= 1.1e-05). Rebuilt at the
default 32 heads it gives what it always gave (0 of 24 576 differ, the same cosines), so the knob
costs the validated point nothing.

| size | HID / layers / FFN | slice logits corr (3 tokens) | argmax + top-5 | worst residual | engine vs harness | chat |
|---|---|---|---|---|---|---|
| 9B | 4096 / 32 / 12288 | 0.999999 / 0.999987 / 0.999992 | match | 0.999991 | 0.000e+00 | `[eos]` @54, 181 ms/tok (5.53 tok/s) |
| 4B | 2560 / 32 / 9216 | 0.999999 / 0.999992 / 0.999989 | match | 0.999991 | 0.000e+00 | `[eos]` @60, 138 ms/tok (7.25 tok/s) |
| 2B | 2048 / 24 / 6144 | 0.999998 / 0.999979 / 0.999980 | match | 0.999978 | 0.000e+00 | `[eos]` @63, 69 ms/tok (14.5 tok/s) |
| 0.8B | 1024 / 24 / 3584 | 0.999982 / 0.999993 / 0.999992 | match | 0.999984 | 0.000e+00 | `[eos]` @74, 53 ms/tok (18.7 tok/s) |

Each slice is 8 layers (six linear, two full), 3 greedy tokens from `[248045]`, against the
replica fed the re-quantized out_proj; maxrel <= 1.2e-02 in every layer of every run. `route 0.00
/ part1 0.0` in every step trace, i.e. one instruction stream per layer type. The 2B's positions 1
and 2 and the 0.8B's position 0 sit just under the 0.99999 logits bar (0.999979 / 0.999980 /
0.999982) while their argmax and whole top-5 match and every layer residual is >= 0.999978 --
the same fp32-vs-fp64 noise the 4B's 0.999989 is, wider because the residual is narrower.

**The 4B reproduced its 2026-09-06 numbers exactly** -- 0.999999 / 0.999992 / 0.999989, argmax
228793 / 695 / 3966, top-5 identical, residual corr >= 0.999991, and the same chat answer word for
word -- which is the DENSE-path regression for the per-half projection walk. Its `--check` against
the pre-fix export differs in `lx` alone (`insts.bin` 96 448 B against 95 136); `ax`, `ln` and
`lm_head_q8` are byte-identical and the manifest differs only in `build_key`.

**The 9B's glue core fits with 1 536 B to spare.** `xnb` is now one 4 KB element
(`memref<2048xbf16>` in the built design) whatever the hidden width, and the core allocates
64 000 B of its 65 536: stack 6 144, `qk` 16 384, `side` 3 x 4 096, `gact` 6 x 2 048, `gout`
4 x 2 048, `vt` 4 096, `xnb` 4 096, and the four f32[32] accumulators 512.

Points added to `recipes/catalogue.py`: `deltanet heads=16`, the `attn` tuple
`(256, 8, 2, 64, True, True, False)`, `lm_head_q8` K 1024 and 4096, `gemv_q4` K 3584. With those
in, all four sizes compose with no `OPEN_KERNELS_UNVALIDATED`. Log:
`.claude/plans/q35-hw-results.md`.

**Result 2026-09-07 (the same four sizes at native q8, OPEN-QUANT-Q8): 9B, 2B and 0.8B PASS;
the 4B has no kernels.** Each container stores `ssm_out_proj` at q8, so this is the family
running its own weights rather than a re-quantized copy of them. Against the replica fed
those q8 values, the same 8-layer / 3-token slice gives:

| size | slice logits corr | argmax + top-5 | worst residual | engine vs harness | chat |
|---|---|---|---|---|---|
| 9B | 0.999999 / 0.999991 / 0.999993 | match | 0.999994 | 0.000e+00 | `[eos]` @74, 191 ms/tok (5.25 tok/s) |
| 2B | 0.999998 / 0.999989 / 0.999986 | match | 0.999986 | 0.000e+00 | `[eos]` @61, 70 ms/tok (14.35 tok/s) |
| 0.8B | 0.999993 / 0.999992 / 0.999990 | match | 0.999991 | 0.000e+00 | `[eos]` @47, 54 ms/tok (18.36 tok/s) |

`routing None` and ERT state 4 on every dispatch -- the mixed core's earlier y acquire does
not disturb the fifos. The 2B is the one size whose q8 slice beats its own q4_1 slice
(0.999989 / 0.999986 against 0.999979 / 0.999980). The two paths diverge from the second
token: the 9B picks 220 / 248045 / 82 re-quantized and 220 / 3966 / 3966 at q8.

The **4B**'s `lx` still overflows program memory, so it stays on the re-quantizing fallback and
the recipe now narrows its derived map to q4_1 rather than offering an export that cannot
build; see OPEN-QUANT-Q8. The kernel sets went to
`src/xclbins/<model>/open_kernels_q8`, beside each size's untouched q4_1 baseline, and
`recipes/catalogue.py` did not move -- the q8 GEMV's K here is `lin_value_width`, 4096 or
2048, both already validated. Log: `.claude/plans/q8m-hw-results.md`.
