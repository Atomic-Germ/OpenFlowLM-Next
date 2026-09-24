# open-engine: the open engine (Qwen3.6-MoE, Qwen3.5 dense, Qwen3 dense, Llama 3, Gemma 3, HunYuan dense, Granite, Phi-3) and its model recipes

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
- An optional `hf_config_defaults` object names what an absent `config.json` key means: `check_model` compares the expected value against it instead of refusing for the missing key, and still refuses when the default disagrees (the phi3 fixture: a config without `head_dim` accepted, one without `partial_rotary_factor` refused against a 96-dim kernel set, one without `rope_scaling` refused against a longrope one). A key with no default stays a hard requirement.
- `gemm_block`, when present, is parsed per kind (`dense` | `linear` | `full`) with its weight map and, for the MoE kinds, its `moe_kernel`; the 35B fixture carries the linear and full routes, and a route naming a pack op past the plan, with a third step or whose MoE dispatch lacks the patch table is refused by name (OPEN-PREFILL-BATCH). For the `dense` kind, `attn_kernel` / `attn_args` name which attention kernel and buffer args drive the route's T single-token dispatches (default `dxB` / `pool, xres, consts, state, act, ptab`, so a manifest predating the fields still parses), letting a layer type with its own sliding window (Gemma 3's `dense_local`) name its own kernel and position table instead of sharing the whole model's one; `sandwich` and `act` (default `false` / `silu`) select the residual/norm chain and FFN activation the host stages compute. A manifest naming an `attn_kernel` that is not declared, or that is not built with the `attnpos` patch table, is refused by name.
- A manifest the packer or the engine could not execute is refused by the parser, naming the field: a pack op without a size `pools::apply` needs (a `std_perm` without `nch`, an `lmhead_q8` without `chunk_bytes`), or a `moeroute2` step on a kernel not built with the routed-expert patch table.
- The fixture equals the recipe's current output (`make_fixtures.py`) apart from the build key.
- `Engine::find_kernels` looks in this order and returns the first complete set, logging the directory that served: `OFLM_OPEN_KERNELS_DIR`; `<model dir>/open_kernels`; then `<root>/xclbins/<model name>/open_kernels` over **every** root in `utils::xclbin_roots()` -- the user roots first (`$OFLM_XCLBIN_PATH`, the directory holding `$OFLM_CONFIG_PATH`, the user-level oflm directory `oflm-add` writes into), then the roots the closed path walks (the executable's directory, the CWD, `<exe>/../share/oflm`, the configured prefix), then `config.exec_path` if a DEV_BUILD put it outside all of those. Not only the single root `utils::find_xclbin_path()` returns: a set `oflm-add` linked under the user root and a set shipped in the install tree are both reachable, whichever of the two that function happens to pick. `find_xclbin_path` itself is unchanged -- it still walks the closed roots only, so which root serves a **closed** kernel does not move.

### OPEN-ADD-KERNEL-LINK: `oflm-add` links a model to the kernel set matching its spec
**Applies to:** openflowlm-next (`utilities/oflm-add/oflm_add/__init__.py`)
**Test category:** unit
**Tests:** `utilities/oflm-add/tests/test_open_kernels_link.py`

Open kernel sets belong to a `ModelSpec`, not to an official model name. When
installing a model, `oflm-add` shall derive that model's spec the way the recipes
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

### OPEN-ADD-SYSTEM-REGISTRY: `oflm-add` finds the registry a real install ships
**Applies to:** openflowlm-next (`utilities/oflm-add/oflm_add/__init__.py`)
**Test category:** unit
**Tests:** `utilities/oflm-add/tests/test_system_registry.py`

`oflm-add` reads the official `model_list.json` for its defaults, and looks for it
beside the installed engine. The released engine installs as `flm`, not `oflm`, so
both names shall be searched -- `oflm` first, so a checkout build wins on a machine
that has both. `--system-list` shall be used when given, rather than searching, and
refused by path when it names something that is not a file. A refusal shall name the
paths actually tried, since the point of the message is to tell someone where to
look.

The same applies to the xclbins root: an engine directory found under either name.

**Acceptance criteria:**
- `--system-list PATH` returns `PATH` with nothing on `PATH` to find; a `PATH` that
  is not a file raises, naming it.
- With only `flm` on `PATH`, the list beside it is found.
- With both, the one beside `oflm` wins.
- `find_system_xclbin_root` finds `<dir>/xclbins` beside an installed `flm`.
- The refusal names every candidate it tried and `--system-list`.

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

`ModelSpec.from_hf_config` (the HF-style `config.json` OFLM ships) and
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
- `catalogue.MIXED_CORE_FITS` is the same idea one level up: not a template parameter but a PROGRAM MEMORY point, the `(family, hidden)` widths whose main core has been built carrying both weight formats' GEMV bodies. It holds `(qwen35, 4096)`, `(qwen35, 2048)` and `(qwen35, 1024)` from OPEN-QUANT-Q8's 2026-09-07 pass. Unlike a template point this one does not refuse: at an unlisted width `recipes.load` warns in one short line that q8 is NOT IMPLEMENTED YET there and narrows the container's q8 role away, because the alternative is an export that composes cleanly and then dies 60 s into `aiecc`. The line names the width, the reason (program memory) and the format it fell back to; what the fallback costs (0.999682) is recorded here rather than spent on a warning. The warning is worded as a gap in what has been built rather than a permanent limit -- the width wants a mixed core small enough to fit and nobody has built one yet (the maintainer's call on PR #26).

### OPEN-ATTN-CONTEXT: decode cost stays flat in the context position, on every family
**Applies to:** openflowlm-next (`open_kernels/designs/attn/attn.h`, `recipes/attnknobs.py`, `designs/dense/dx.py`, `designs/layer_x/ax.py`)
**Test category:** manual (the sweep below, needs the NPU and the model container); the geometry each family gets is unit-tested in `tests/test_attn_geometry.py`

A decode step's attention cost shall not grow with the context position beyond
a small per-position term: on the fast attention path -- the online softmax's
exponentials batched on the vector unit (`ATTN_VEXP`), the heads split over
`ACORES` cores (`ATTN_NHL`), cached rows blocked per kernel call (`ATTN_RB`) --
a step at position 2048 costs within 2x of a step at position 0 on the
families below. `1/sqrt(HD)` folds into q as an exponent shift at HD 64 / 256
and multiplies the scores on the vector unit at HD 128 (`ATTN_SCALE_IN_Q`).

A family enters the path by measurement, never by declaration:
`recipes/attnknobs.py: FAST_ATTENTION` lists the measured families; every
other family compiles the single-core attention it compiled before, byte for
byte. `ATTN_FAST=1` builds an unlisted family on the path for exactly that
measurement and is a probe variable (in the build key, OPEN-BUILD-CACHE).

**Acceptance criteria (unit, `test_attn_geometry.py`):**
- With `ATTN_FAST=1`, `dense.geometry` / `qwen36moe.attn` give: Qwen3-4B, Llama-3.1-8B, HunYuan 4 cores x 8 heads, RB 4; Gemma3-4B 4 x 2, RB 2; Gemma3-12B 4 x 4, RB 1; Phi4-mini 6 x 4, RB 4; Granite 5 x 8, RB 4; the 35B and Qwen3.5-9B 4 x 4, RB 1; Qwen3.5-0.8B 4 x 2, RB 1; LFM2-1.2B 4 x 8, RB 4. ACORES is the largest divisor of the HEAD COUNT that fits the columns, and a core's heads tile the og element they are written through (`kOGH = min(kNHL, kHPO)`, attn.h); RB x max(NHL, 8) is 8, 16 or 32.
- Without it, an unlisted family gets VEXP 0, one core, RB 1, ml packed (the shipped kernel); a listed one gets its fast geometry.
- `ATTN_FAST` is in `PROBE_VARS`; every family module exposes `probe_env`.

**Procedure (manual):** build the family with `ATTN_FAST=1` into a scratch
directory; one decode step at positions 0 / 256 / 1024 / 2048 through
`open_qwen36_cli --at-position` on the shipped set and the probe set; then
200-300 greedy tokens from the same prompt on both, logits dumped
(`--dump-logits`) and compared position by position until the first token
that differs. A near-tie flip (the two kernels' top-2 within ~0.05 logits,
corr > 0.9999 at that position) is not a defect. Passing: flat part0 across
the sweep, argmax agreement at every comparable position. Then list the family
in `FAST_ATTENTION`, export without the probe and install the set.

**Measured (2026-09-07/08, `.claude/plans/issue-16-hw-results.md`):**

| family | geometry | step @ 2048, shipped -> fast | greedy agreement |
|---|---|---|---|
| Granite-4.2-3B (hd 64) | 5 x 8, RB 4 | 1216 s TTFT -> 59 s on 1005 tokens | fp64 replica, coherent chat |
| Qwen3-4B (hd 128) | 4 x 8, RB 4 | 5050 -> 258 ms (19.6x) | 300/300, corr min 0.99993 |
| Llama-3.1-8B (hd 128) | 4 x 8, RB 4 | 4024 -> 427 ms (loaded box) | 54, then a 0.008-logit near-tie |
| Hy-MT2-7B (hd 128) | 4 x 8, RB 4 | 3395 -> 165 ms (20.6x) | 43, then a 0.05-logit near-tie |
| Gemma3-4B (hd 256) | 2 x 4, RB 2 | 512 -> 96 ms (5.3x; the local layers never grew) | 41, then a 0.06-logit near-tie |
| Qwen3.5-0.8B (hd 256, gated; `ax`) | 4 x 2, RB 1 | 176 -> 70 ms (6 attention layers of 24) | 200/200, corr min 0.99993 |
| Qwen2.5-3B (hd 128, q/k/v bias) | 4 x 4, RB 4 | 1605 -> 84 ms/token at 2048 (19.1x) | 143, then a 0.021-logit near-tie |
| Qwen3.6-35B (hd 256, gated; `ax`), 16-layer prefix | 4 x 4, RB 1 | 217 -> 43 ms part0 (four attention layers) | 85 (100 tokens; corr spread from expert flips) |
| LFM2-1.2B (hd 64, q/k normed; hybrid) | 4 x 8, RB 4 | 527 -> 35.6 ms/token at 2048 (14.8x) | 250/250, corr min 0.9999413 |

Qwen2.5 (2026-09-12) is the first family whose cores emit more than one og element:
16 heads over 2 kv heads means `HPO` is 2, so a core owning 4 heads writes them through two
2-head elements (`kOGH = min(NHL, HPO)`), on the narrowest attention element of any dense
family at 512 B. Per-token cost across positions 0 / 256 / 1024 / 2048 went 60 / 270 / 839 /
1605 ms on the shipped kernel and 64 / 65 / 69 / 84 ms on the probe -- 1.3x from end to end
of the sweep, against the 2x the requirement allows. Decode over 250 tokens went 5.03 ->
15.98 tok/s. Re-exporting with the probe unset reproduced the probe build: every instruction
stream byte-identical, the xclbins differing only in build stamps.

LFM2 (2026-09-13) is the first HYBRID measured, and it is what separates the
requirement's two halves cleanly. Ten of its sixteen layers are `short_conv`, whose
state is a fixed 16 KB window rather than a growing cache: the first two layers alone
cost 2.9 ms at position 0 and 3.1 ms at position 2048, flat as the block's arithmetic
says they must be. Adding the first attention layer takes the same slice from 4.7 to
86.6 ms, so one attention layer grows 81.7 ms across the sweep and the six of them
account for 490 ms of the whole model's 497 ms of growth. The conv layers contribute
nothing to measure. Per-token cost across positions 0 / 256 / 1024 / 2048 went 29.5 /
96.5 / 284.8 / 527.1 ms on the slow path and 28.8 / 29.5 / 35.1 / 35.6 ms on the probe
-- 1.24x from end to end of the sweep. Decode over 250 tokens went 14.29 -> 33.04
tok/s, 250/250 greedy tokens identical, and over 100 dumped positions the logits agree
at corr min 0.9999413 with no argmax disagreement anywhere. Re-exporting with the probe
unset reproduced it: all four instruction streams byte-identical, the xclbins differing
only in build stamps. Through `oflm serve`, `oflm-test --llm` went from 6.84 and 2.74
tok/s on the two rounds to 30.9 and 27.8 (`results/20260913_092125/windows/`); that the
second round no longer costs a quarter of the first is the flatness showing up end to
end, since it is the round that starts with the first one's answer in context.

The 35B's `ax` kernels rebuilt at the default knobs after the split was
plumbed into `ax.py` are byte-identical to the shipped set (`--check`).

**Measured (2026-09-12, the og split -- `attn_cores` on the head count):**

Until an og element was NHL wide, ACORES was the largest divisor of `NH / HPO`,
so NHL was always exactly HPO. Splitting the og fifo frees the two families
whose head count divides further than their og element count did:

| family | geometry | step @ 2048, before -> after | greedy agreement |
|---|---|---|---|
| Gemma3-4B (hd 256) | 2 x 4 -> 4 x 2, RB 2 | 94.2 -> 86.4 ms (1.090x) | `818,236743` both ways, identical |
| Phi4-mini (hd 128, 96-dim rotation) | 3 x 8 -> 6 x 4, RB 4 | 123 -> 117 ms (1.05x, n=4 each) | 250/250 identical |

Phi-4-mini was not in the branch that made the change -- it landed on main
first, and `attn_cores(NH)` reached it on the rebase. Position 0 is unchanged
within a jitter of +-25% on that path (no attention work there); position 2048
is stable to +-2% and is where the split shows. Every other dense family has
`NHL == HPO` and rebuilds byte-identical.

**Where the requirement came from (the observation, 2026-09-06):**

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
through `oflm bench` at 20.6x TTFT and 32.1x decode on a 1005-token prompt.
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
- `designs/gemm_q4_prefill/*` is in the key for the MoE family, since the GEMM xclbins of the block route are built from it (OPEN-PREFILL-BATCH).
- The key takes `quant` in its canonical form, so a role map hashes (a q8 role changes the key) and an all-`q4_1` map hashes the bare string, byte for byte what the key hashed before roles existed.

### OPEN-PACK-PLAN: the packing plan reproduces the verified pool laws
**Applies to:** openflowlm-next (`open_kernels/recipes/pack.py`, `src/open_qwen36/pools.cpp`)
**Test category:** unit (Python interpreter) + integration (C++, through OPEN-FAMILY-QWEN36MOE)
**Tests:** `tests/test_pack_plan.py`, `tests/legacy_pools.py` (the frozen originals)

The recipe's plan (`expert_stripes`, `expert_down`, `std_perm`, `q8_perm`,
`put`, `conv_transpose`, `transpose`, the lm_head supertile order, the position
table) applied by `recipes/pack.py` shall produce, for a container with the
27B's tensor shapes, exactly the bytes the hand-written packers produced (the
ones verified against pools captured from OFLM's engine). `pools.cpp`
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

**Source forms.** The quantized chunk format is a property of the TENSOR, not of
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

A Q4_K source (4736-byte chunks, what OFLM 1.0.3+ writes) is accepted the same
way and transcoded to q4_1 on the way in, for the same reason and through the
same seam -- but nearly free, because Q4_K's scale and min already carry the
pool's granularity and index (OPEN-QUANT-Q4K). So the three chunk ops read
three source forms and refuse everything else by name.

**Acceptance criteria:**
- Layer pool, consts blob (linear and attention), lm_head pool and ptab are byte-equal to `legacy_pools.py` on random-byte tensors of the right sizes.
- A small weight larger than its slot is refused (`does not fit its 4096 B slot`).
- On a synthetic container mixing one q8 and one q4_1 tensor: the q8 tensor's pool bytes equal `requant_q4_1` of its chunks put through the same `std_perm` order; the q4_1 tensor's are the verbatim chunk copy they were before q8 sources existed; and the whole pool has the same FNV-1a in NumPy and in C++ (`tests/test_pack_plan.py`, `src/open_qwen36/pools_test.cpp`, which writes the container as a real `.q4nx`).
- A container that cannot report a tensor's chunk size is read as q4_1, so the frozen pools above are unaffected.
- A tensor with 1280-byte chunks is refused by both packers, the message naming the tensor, `1280`, the three widths that ARE read (5120 q4_1, 8704 q8, 4736 Q4_K) and what 1280 probably is -- rather than guessing "OFLM 1.0.3 / Q4_K?", which is what it used to say about any unfamiliar width.
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
- `transpose_banked` is the dedicated wide-head AB operation: `[heads, hidden]`
  becomes `[ceil(heads/32), hidden, 32]`, with unused tail lanes zeroed. It
  requires `tensor`, `rows`, `cols`, `elem`, and uses `dst` as a byte offset.
  At 48 heads and hidden 5120, each bank is 327680 bytes (80 side tiles).
  Python and C++ test every element, heads 31/32/47, padding and destination
  bounds. Qwen3.5 plans select it only above 32 heads; existing plans retain
  `transpose`. This packing capability does not validate a wide-head NPU
  kernel. See `tests/test_qwen35_27b.py` and `plans/qwen35-27b-bringup.md`.
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

### OPEN-QUANT-Q4K: the packers read Q4_K containers
**Applies to:** openflowlm-next (`open_kernels/model/q4nx.py`,
`open_kernels/recipes/pack.py`, `src/open_qwen36/pools.cpp`,
`utilities/q4nx-build/q4nx/{gguf_tensor,model_converter,cli}.py`)
**Test category:** unit (the transcode, both packers, the reference, the converter's
writer) + manual (the hardware run: needs the NPU and a Q4_K container)
**Tests:** `tests/test_quant_q4k.py`, `src/open_qwen36/pools_test.cpp`

OFLM 1.0.3+ writes a third quantized chunk form, `q4k_block_t`, 4736 B per 32-row x
256-column tile. Packing the 35B MoE projections as q4_1 instead makes the closed runtime
decode infinite `////` or segfault, so this is not a preference and the open engine cannot
refuse it. Both packers shall accept it as a source for the three q4 chunk ops
(`std_perm`, `expert_stripes`, `expert_down`) and transcode it to the pool's q4_1 chunk on
the way in, exactly as a q8 source is re-quantized. `q8_perm` continues to demand q8: a
kernel set built to stream a projection at q8 is not satisfied by Q4_K.
`utilities/q4nx-build` shall also be able to WRITE the format (`--quant Q4_K`).

The chunk, everything column-major over the tile:

```
scales[8][32] uint8 @ [0, 256)      index g*32 + r   (32-column group g, row r)
mins  [8][32] uint8 @ [256, 512)    same index
qs    [256][16]     @ [512, 4608)   byte k*16 + r/2, even row in the low nibble
S     [32]    bf16  @ [4608, 4672)  index r          one super-block per chunk
M     [32]    bf16  @ [4672, 4736)  index r          stored NEGATED
value(r, k) = S[r] * scales[k/32][r] * nib + M[r] * mins[k/32][r]
```

**Nothing above the packer changes.** Q4_K's scale and min already have the pool's
granularity AND its index, so the transcode is `d = bf16(S*scales)`, `m = bf16(M*mins)`
in place, plus a byte de-interleave of the nibbles (Q4_K keeps a column's 32 rows in 16
contiguous bytes; the pool splits rows 0-15 and 16-31 into two 2048-byte planes, so q4_1
byte `h*2048 + k*8 + j` is Q4_K byte `k*16 + h*8 + j`, values and parity unchanged). No
chunk index law, plan, manifest, kernel or build key moves, and no kernel point is
needed.

**What it costs.** The exact product of a bf16 `S` and a uint8 `scales` needs 16
significand bits and the pool's `d` holds 8, so each group's scale and min take one bf16
half-ulp -- `2^-8` relative each. Where the two terms cancel the error can exceed `2^-8`
of the value, which is why the bound below is stated on `|scale term| + |min term|`.
Measured over all 108 Q4_K tensors of a real container: **0.48% relative weight RMS**,
against 8% for the q8 -> q4_1 re-quantization the packer already does.

**A source that cannot be read as Q4_K.** ggml has no Q4_K encoder, so a q5 / q6 / float
source under a Q4_K target is re-quantized onto q4_1's grid and then packed as Q4_K. The
two disagree on the sign of the min -- q4_1 stores an added `m` (<= 0), Q4_K a subtracted
magnitude (>= 0) -- so the converter's fallback negates it. Without that the weights come
out mirrored about each block's minimum and nothing downstream notices.

**Acceptance criteria:**
- A synthetic Q4_K chunk transcoded to q4_1 and read back as `nib*d + m` equals `S*scales*q + M*mins` computed in f64 from the same bytes, every value within `2^-8 * (|S*scales*q| + |M*mins|)` and no more. Measured worst case: 0.99 of that bound in C++, so the bound is tight rather than slack.
- The nibble de-interleave is exact: for every `(r, k)` the uint4 at q4_1 nibble `(r/16)*4096 + k*16 + (r%16)` equals the one at Q4_K byte `k*16 + r/2`, nibble `r%2`.
- The 256 metadata slots do not move: `d[i]` comes from `S[i%32] * scales[i]`, `m[i]` from `M[i%32] * mins[i]`, and every `m` is <= 0.
- NumPy and C++ produce the same transcoded chunks and the same `std_perm` pool (FNV-1a `0x685dc049ec1ca2d7` and `0xb02083912551d9d3` on the shared synthetic vector).
- A Q4_K tensor put through `std_perm` lands at the pool chunk positions a q4_1 tensor of the same shape does; a q4_1 tensor beside it is still a verbatim chunk copy, and `tests/test_pack_plan.py`'s frozen pools are unchanged.
- `q8_perm` over a Q4_K tensor is refused naming the tensor and both byte counts; a width that is none of 5120 / 8704 / 4736 is refused naming the width and what it probably is -- and the refusal no longer GUESSES Q4_K for an unfamiliar width, since 4736 is now read.
- `model/q4nx.py`'s `dq_tile` reads a Q4_K tensor as the transcoded q4_1 the NPU holds by default (the convention q8 already follows, so a slice comparison measures the kernels) and as the container's own Q4_K values under `requant=False`, which is the transcode's quality number.
- Writer and reader agree: chunks written by `q4nx-build`'s `pack_q4k` from known `(scale, min, quant)` triples read back through `dq_chunks_q4_k` with the quants bit-exact and the values within 1e-2 relative L2 (the super-block re-fit's own accuracy).
- The re-quantize fallback under a Q4_K target returns a non-negative min, and packing it round-trips to the q4_1 reading it came from.
- `--quant Q4_K` on the HF-safetensors path is refused: `_store_q` quantizes with its own fixed per-role targets, so it would write a q4_1 container while claiming Q4_K.

**Procedure (manual):** convert a GGUF for a validated shape twice, `--quant Q4_K` and
`--quant Q4_1`, and run both through `src/open_qwen36/` on the same kernel set. Then serve
each with `oflm serve` on the open engine and run `utilities/oflm-test --llm --tools` against
it. The two containers hold the same source weights, so the only variable is the format.

**Result 2026-09-08 (Qwen3.5-0.8B, condB fine-tune, 24 layers, Strix): PASS.** The Q4_K
container -- 108 tensors at 4736 B beside 79 at 8704 -- is the first one that exists
anywhere; no model on Atomic-Germ's Hugging Face or in OFLM's registry ships the format
yet. It loads, packs and decodes at **21.6 tok/s**, and its greedy output agrees with the
q4_1 twin's for **36 of 37 tokens**, diverging only where a sentence-final `.` and `,`
were a near-tie. Both continuations are coherent and say the same thing.

**Result 2026-09-09, through `oflm serve` (same model, same box): PASS.** `oflm-add` registers
the container and the engine loads it on the open kernels, 24 of 24 layers resident, at the
same spec hash the q4_1 twin derives -- the transcode is invisible above the packer, which is
the point. `oflm-test --llm` returns coherent, on-topic answers in stream mode. At temperature
0 the two containers agree for the first 148 of 215 characters on a fixed prompt and then
pick different phrasing for the same claim, at 16.67 tok/s against the twin's 16.86.
`oflm-test --tools` fails 5 of its 6 checks (no tool call issued), but the q4_1 twin fails the
same 5 identically, so that is the 0.8B model and not the format.

Getting there needed one converter fix, in this commit: `configs/qwen3.5_0.8b.json` pinned
`up_proj` to Q8_0 while the shipped container stores it at q4_1, and the other three qwen3.5
sizes never pinned it. A mixed `ffn` role is refused during spec derivation, so every
container this converter built for the 0.8B was unusable by the open kernels, at q4_1 as much
as at Q4_K. Unpinning it matches the shipped model and the sibling configs.

Still open: the fp64 slice comparison, whose kernels are unchanged by this requirement, so
their correlation is the one OPEN-FAMILY-QWEN35 already records.

### OPEN-QUANT-MXFP4: the readers decode GPT-OSS's MXFP4 expert chunks
**Applies to:** openflowlm-next (`open_kernels/model/mxfp4.py`)
**Test category:** unit
**Tests:** `tests/test_mxfp4_decode.py`

GPT-OSS ships its experts as MXFP4 - 4-bit FLOAT, sixteen unevenly spaced levels sharing one
E8M0 exponent byte per 32 values - where every other quantized form the engine reads is
4-bit or 8-bit integer. The reader shall decode a 2560-byte MXFP4 chunk to a defined
(32, 128) value array, and the branchless integer form the AIE tile computes shall agree
with the table on every code.

Layout, and the chunk is 32 rows by 128 columns: 128 E8M0 bytes at `[0, 128)` indexed
`g*32 + r`; 384 pad bytes at `[128, 512)` of which bytes 128..191 carry 32 bf16 output biases
in column block 0 only; 2048 nibble bytes at `[512, 2560)` indexed `(rg*64 + c)*16 + i`, with
the even/odd column split the converter pre-applies. Value is
`KVALUES[nibble] * 2^(e8m0 - 128)` with KVALUES twice the e2m1 levels - which makes them
integers, and is what lets a GEMV keep its multiply integer (`.claude/plans/gptoss-mxfp4-gemv.md`).

**Acceptance criteria:**
- Eight real chunks cut from the shipped container - spanning gate, up and down slabs at four
  column blocks, checked in as `fixtures/gptoss_mxfp4_chunks.npz` - decode element for
  element to their recorded values, under both the table and the tile's integer formula.
- `decode_int` equals `KVALUES` for all 16 codes, and four near-miss ladders (no doubling
  above 6, either knee moved, plain linear) do not.
- The sign bit is 0x08 and not 0x10.
- Every decoded scale is exactly a power of two, since E8M0 carries no mantissa.
- A column-block-0 chunk's padding holds a finite non-zero bias; a later column block's holds
  zeros.
- Raw codes and decoded values are not the same test: flipping every +0 code to -0 changes
  the codes and changes no value. A test asserting code equality against upstream fails on a
  decoder that is exact, because this container's GGUF source normalises the sign of zero -
  about 7% of codes differ where no value does.

**Verified by mutation 2026-09-14:** dropping the doubling above 6, swapping the row-split
halves, and reshaping the scale array transposed each break two tests. The suite is not
passing by construction.

### OPEN-PACK-CHUNK-FUSE: two half-width chunks make one pool chunk
**Applies to:** openflowlm-next (`open_kernels/recipes/pack.py`, `src/open_qwen36/pools.cpp`)
**Test category:** unit (`tests/test_chunk_fuse.py`, `src/open_qwen36/pools_test.cpp`)

`q4nx-build/configs/gpt-oss.json` is the only config with `col_block_size` 128, so a
GPT-OSS container holds 32 rows by 128 columns in each 2560-byte chunk where every other
family holds 32 by 256 in 5120. The packer shall place such a tensor with its own op,
`std_fuse`, which locates the k-tile's two 128-column halves in the container's supertile
raster and fuses them into one pool chunk, synthesising an all-zero chunk for a column
block past the container's own width.

**It is q4_1, in the ordinary layout.** `d[128]` as bf16 at byte 0, `m[128]` at 256, 2048
nibble bytes at 512 -- the same format the engine already reads, at half the columns. The
`I8` dtype these tensors carry is how every quantized tensor ships in a `.q4nx` container,
not a claim about the format; the experts at the same 2560 bytes are MXFP4 and ship as U8
(OPEN-QUANT-FORMAT).

**The fuse is eight byte-slice copies and no arithmetic.** The meta index is `b * 32 + r`,
which puts the low half's four blocks at metas 0..127 and the high half's at 128..255; the
nibble raster `(r // 16) * 512 * nb + b * 512 + i * 16 + (r % 16)` splits each source into
two 1024-byte planes by row half. So the copies are d from A then B, m from A then B, then
plane 0 from A, plane 0 from B, plane 1 from A, plane 1 from B -- interleaved, not
concatenated, which is the mistake the layout invites.

**The file raster is a supertile, and only this converter writes one.** Row block `rb` is
supertile `rb // rg` at position `rb % rg`, and its column block `q` lands at
`(rb // rg * ncol128 + q) * rg + rb % rg`; `rg` is 4 for the attention projections and the
experts and 2 for the `lm_head`. At `rg = 1` it reduces to the plain `rb * ncol + q` every
other family uses. Reading this container on the plain raster produces a full, plausible,
wrong pool, which is why the test asserts the two rasters disagree rather than only that
the supertile one round-trips.

**The pad is the same pass.** K = 2944 is 23 column blocks, an odd count, so a 3072-wide
pool's last k-tile has no high half in the container. It is synthesised as 2560 zero bytes,
and `d = m = 0` reads as exactly 0.0 -- so the fuse and the 2944-to-3072 pad happen
together rather than as a pack followed by a zero fill.

**Acceptance criteria:**
- A fused chunk read with the shipped `q4nx.dq_chunks_q4_1` has the low source's values at
  blocks 0..7 -- 0..3 from the low half, 4..7 from the high -- against a half-width reader
  written from the format definition rather than from the packer.
- Three near-miss interleavings (plain concatenation, the two nibble planes exchanged, the
  high half's `d` and `m` exchanged) each produce a different reading. The fuse is a byte
  permutation: the pool chunk's bytes are exactly the two sources', each once.
- `supertile_perm` is a permutation of `range(nrb * ncol128)`, is not the plain raster at
  `rg` 4 or 2, and is the plain raster at `rg` 1. A partial supertile is refused.
- `std_fuse` over a 2944-wide container into a 3072-wide pool places every pool chunk as
  the band law's (row block, k-tile) located in the supertile raster, and the synthesised
  24th column block reads back identically zero.
- Packing the same fixture on the plain raster gives a different pool for most chunks --
  the fixture discriminates.
- The NumPy and C++ interpreters produce byte-identical pools over
  `lcg_bytes(0x5EEDFACE, 8 * 23 * 2560)`: FNV-1a `0x21f7e3b732137cb2`, asserted on both
  sides.
- A container at 5120 is refused by `std_fuse`, naming both widths; a pool width that is
  not a whole number of 256-column k-tiles is refused (OPEN-WIDTH-PAD's guard, shared
  through `band_rowblock_ktile`); a source width that is not a whole number of 128-column
  chunks is refused; a pool narrower than the container is refused; and a tensor whose
  chunk count does not match the widths it claims is refused, naming the count it needs.
- `test_pack_plan.py`'s frozen q4_1 pools are byte-identical before and after: `std_perm`
  now defers to the shared band law rather than restating it, and the bytes do not move.

**Verified by mutation 2026-09-15:** writing the nibble planes A0 A1 B0 B1 instead of
A0 B0 A1 B1, swapping the high half's `d` and `m`, writing the supertile as the plain
raster, and pointing the synthesised column block at the last real chunk each break at
least one test. The suite is not passing by construction.

**Not covered here.** The expert tensor's own slab order is a separate law over the fused
`ffn_gate_up_down_exps.weight` (OPEN-PACK-EXPERT-ORDER). `std_fuse` is the attention
projections and the head.

### OPEN-PACK-EXPERT-ORDER: which slab of the fused expert tensor is gate, up and down
**Applies to:** openflowlm-next (`open_kernels/recipes/pack.py`)
**Test category:** unit (`tests/test_expert_order.py`)

`q4nx-build`'s GPT-OSS path fuses one layer's gate, up and down for every expert into a
single `ffn_gate_up_down_exps.weight`, shaped `[E, 3 * nslab, ncol128, rg, 2560]` -- the
shipped 20B's is `[32, 69, 23, 4, 2560]`. Nothing in the container names the three
projections. The packer shall read the slabs as **gate and up alternating every 128 rows
over the first `2 * nslab`, then down as one contiguous block**, and shall locate a
projection's row block within that using the same supertile raster as the attention
tensors (OPEN-PACK-CHUNK-FUSE), so a logical output row decomposes as
`(row // 128, (row % 128) // 32, row % 32)` -- slab, quarter, row in chunk.

**Why this needs measuring rather than reading off the shape.** The other plausible
reading -- three projections concatenated -- agrees with this one on down and swaps gate
and up. A packer that picks wrong feeds the clamped SwiGLU its two halves the wrong way
round, which is finite, plausible and silent: `(up + 1) * gate * sigmoid(1.702 * gate)`
evaluates perfectly well with the arguments exchanged, so nothing raises and only output
quality moves.

**The lever is the bias the converter writes twice.** `q4nx/models/gpt_oss.py` puts the
32 bf16 output biases at byte 128 of every column-block-0 chunk AND ships the named
`mlp.experts.{gate,up,down}_proj_bias` tensors. The duplicate distinguishes the three
projections by VALUE, so the order is pinned with no ambiguity and no appeal to the
converter's source.

**Acceptance criteria:**
- `expert_slabs(nslab)` is `[3, nslab]`, gate at `2s`, up at `2s + 1`, down at
  `2 * nslab + s`, and uses every slab of `range(3 * nslab)` exactly once.
- `expert_chunks(nslab, ncol128, rg)` is `[3, nslab * rg, ncol128]` and is a permutation of
  one expert's whole chunk range -- a map that aliases would pack one slab's bytes over
  another's with nothing raised.
- Against the shipped container's own bytes, for all 69 slabs of two experts: the 128
  biases carried in a projection's slab's four column-block-0 chunks equal the named
  `*_proj_bias` rows `128s .. 128s + 127` for the role and slab the map predicts.
- Four near-miss orders (three concatenated projections, gate and up exchanged, down
  first, all three alternating) each fail to reproduce those biases, and so does reading
  the tensor on the plain `rowblock * ncol + column` raster every other converter writes.
- The three roles' biases are non-zero and pairwise different, so the check above cannot
  pass on an order it did not measure.
- The rows past the projections' real 2880 -- 64 of the last slab's 128 -- are zero in the
  container, so a packer does not have to zero the tail itself.

**Measured 2026-09-15** on `GPT-OSS-20B-NPU2`'s 14.4 GB container at layers 0, 7 and 23:
every one of the 69 slabs matches exactly one (role, slab) pair, with no slab ambiguous at
any of the three layers, and the expert stride holds at expert 31 as well as 0 and 1. The
checked-in fixture (`make_gptoss_fixtures.py`, which also reproduces
`gptoss_mxfp4_chunks.npz` byte for byte) carries experts 0 and 31, so the test needs
neither the container nor the network.

**Not covered here.** This is the SOURCE law -- where a given expert weight lives in the
file. Where it then goes in the pool is the MoE block's layout, which is open
(OPEN-MOE-WIDE-FF), so there is no `apply_op` kind for the experts yet.

### OPEN-FAMILY-QWEN36MOE: greedy agreement with the fp64 reference on the 27B
**Applies to:** openflowlm-next (`src/open_qwen36/`)
**Test category:** manual (needs the NPU and the model)

Through the manifest path, the 8-layer slice of `Qwen3.6-35B-A3B-NPU2`
decoding `[248045]` greedily for 3 tokens shall match `open_kernels/model/out8t3`'s
fp64 logits at every position, and the full model shall answer a chat prompt
coherently.

**Procedure:**
1. `python -m recipes.manifest --model-dir ~/.oflm/models/Qwen3.6-35B-A3B-NPU2 --out src/xclbins/Qwen3.6-35B-A3B-NPU2/open_kernels/manifest.json` (or a full `export_qwen36_kernels.py` run).
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
**Test category:** manual (needs the NPU and `OpenFlowLM/Qwen3-4B-NPU2`); the recipe's arithmetic is unit-tested in `tests/test_qwen3_dense.py`

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
1. `python open_kernels/export_qwen36_kernels.py --model-dir ~/.oflm/models/Qwen3-4B-NPU2` (WSL) → `src/xclbins/Qwen3-4B-NPU2/open_kernels/{dx,ln,lm_head_q4}` + `manifest.json`.
2. `python open_kernels/model/make_decode.py --model-dir ~/.oflm/models/Qwen3-4B-NPU2 --layers 4 --tokens 2 --out open_kernels/model/out_q3`, then `open_kernels/harness/out/run_kernel.exe open_kernels/model/out_q3/run_decode.cfg` and `python open_kernels/model/compare_decode.py --tokens 2 --out open_kernels/model/out_q3`: every layer's residual corr > 0.9999, logits corr > 0.9999, same argmax at both positions.
3. `src\open_qwen36\out\open_qwen36_cli.exe --model <model dir> --kernels src/xclbins/Qwen3-4B-NPU2/open_kernels --ids 151644 --max-tokens 3 --layers 4 --dump-logits <dir>/y` matches step 2's reference logits (the engine's packer, manifest path and attnpos on the dense stream).
4. `python src/open_qwen36/chat.py "Explain what an NPU is in two sentences." --model <model dir> --kernels src/xclbins/Qwen3-4B-NPU2/open_kernels` → a coherent answer ending in `<|im_end|>`.

**Adapters (manual):** every Qwen3-dense adapter class selects the open engine when a
kernel set is installed for its model, and honours `OFLM_QWEN3_ENGINE=open|closed`:
`Qwen3`, `Qwen3_IT`, `Qwen3_TK` and `DeepSeek_r1_0528_8b` (`model_list.json`
families `qwen3`, `qwen3-it`, `qwen3-tk`, `deepseek-r1-0528`). Verify with
`oflm serve <tag>` on a model that has `open_kernels/` installed: the load logs
`<Family> on the open kernels (<dir>)` -- `Qwen3`, `Qwen3-IT`, `Qwen3-TK`,
`DeepSeek-R1-0528` respectively -- and `OFLM_QWEN3_ENGINE=closed` restores the
`qwen3_npu` DLL for all four.

**Result 2026-09-05 (Qwen3-4B):** step 2 logits corr 0.999997 / 0.999994, same argmax and top-5, residual corr ≥ 0.999996 in every layer at both positions; step 3 identical through the engine, request 2 reproduced request 1; step 4 a coherent two-sentence answer ending in `<|im_end|>` at token 58 (272 ms/token). Details: `.claude/plans/open-kernels-phase-b-qwen3-dense.md`.

**Result 2026-09-06 (DynaGuard-4B-NPU2):** a fine-tune whose spec is identical to Qwen3-4B's in every field but `extra` -- step 2 logits corr 0.999999 / 0.999992, same argmax (11619) and top-5, residual corr >= 0.999996 every layer (maxrel <= 1.6e-3); step 3 identical through the engine, request 2 reproduced request 1; step 4 a coherent two-sentence answer ending in `<|im_end|>` at token 62. The SAME check ran first against `src/xclbins/Qwen3-4B-NPU2/open_kernels` with the same numbers, and its own export is byte-identical to that set (3/3 `insts.bin`, xclbins stamps only). Details: `.claude/plans/p0-p1-results.md`.

**Result 2026-09-06 (Qwen3-4B-Thinking-2507-NPU2):** same spec hash as Qwen3-4B (`sha256:602fa1836b21`) -- step 2 logits corr 0.999998 / 0.999994, same argmax (50179) and top-5, residual corr >= 0.999997 every layer; step 3 identical through the engine, request 2 reproduced request 1; step 4 (`--think`) a reasoning chain closed with `</think>` then a coherent two-sentence answer ending in `<|im_end|>` at token 285. Its `--no-build` export is byte-identical to `src/xclbins/Qwen3-4B-NPU2/open_kernels` (3/3 `insts.bin`), and the same slice passed against that directory directly. Details: `.claude/plans/p0-p1-results.md`.

**Result 2026-09-06 (Qwen3-8B-NPU2, and DynaGuard-8B / DeepSeek-R1-0528-Qwen3-8B on its kernels):** the 8B shape's first hardware run. Step 2 logits corr 0.999998 / 0.999997, argmax 104222 / 118063 matching the fp64 replica with identical top-5, residual corr 1.000000 (t0) / >= 0.999998 (t1) in all four layers, maxrel <= 1.1e-3; step 3 through the engine is BIT-IDENTICAL to step 2 (0.000e+00 over all 151936 logits) and request 2 reproduced request 1; step 4 a coherent two-sentence answer ending in `<|im_end|>` at token 63 (540 ms/token, 36 layers, a loaded box). This admits `gemv_q4` K = 12288 to the catalogue; the 4096 norm, the (128, 32, 8, 128, qk-norm, no gate) attention tuple and the K = 4096 q4 head were already in it. DynaGuard-8B (spec hash identical, `sha256:04374f23aede`) passed the same procedure on its own `--no-build` export, byte-identical to Qwen3-8B's in all six artefacts: corr 0.999997 / 0.999997, same argmax and top-5, `<|im_end|>` at token 59. DeepSeek-R1-0528-Qwen3-8B (the same shape, `real_vocab` 151671 instead of 151669) passed on correlation and residuals -- corr 0.999993 / 0.999996, residual corr >= 0.999997 every layer -- but its position-1 ARGMAX differs: the fp64 top two, 102188 and 108204, are 0.004 logits apart on a 14.5-logit scale and the NPU's ~0.015 per-logit deviation flips them, with slots 3-6 unchanged. Harness and engine agree bit for bit, so this is a tie inside q4_1 noise rather than a kernel disagreement, but it is the first time it has crossed `compare_decode`'s "same argmax" bar. `chat.py` handles this tokenizer now (the `<｜Assistant｜>` branch landed this session), answering coherently after a `</think>` chain. Details: `.claude/plans/k-new-points-results.md`.

**Result 2026-09-06 (Qwen3-1.7B-NPU2, and Qwen3-1.7B-NPU2-BASE on its kernels):** the 16/8-head shape's first hardware run, and the first run of the attention TUPLE (128, 16, 8, 128, qk-norm, no gate, pre-RoPE) -- a GQA group of 2 at head dim 128, which the catalogue's per-parameter check had been passing silently and now names. Step 2 logits corr 1.000000 / 0.999993, argmax 1121 / 17764 matching the fp64 replica, top-5 identical at t0 and slots 1-4 identical at t1 (slot 5 differs, 36976 against 78200), residual corr >= 0.999992 in every layer at both positions, maxrel <= 3.9e-3; step 3 through the engine BIT-IDENTICAL to step 2 and request 2 reproduced request 1 (4-layer slice 10-11 ms/token); step 4 a coherent answer ending in `<|im_end|>` at token 50 (139 ms/token, 28 layers). This admits `gemv_q4` K = 6144, `lm_head_q4` K = 2048 and the (128, 16, 8, 128, True, False, False) attention combination. Qwen3-1.7B-NPU2-BASE passed the same procedure identically on a `--no-build` export byte-identical in all six artefacts -- because it IS the same container: its `model.q4nx`, `config.json` and `tokenizer_config.json` have the same sha256 as `OpenFlowLM/Qwen3-1.7B-NPU2`'s, chat template included, so it is the instruct model published under a `-BASE` name rather than a base checkpoint. Details: `.claude/plans/k-new-points-results.md`.

**Result 2026-09-06 (Qwen3-0.6B-NPU2):** the narrowest shape the recipe has produced. Step 2 logits corr 0.999999 / 0.999986, argmax 1121 / 460 matching the fp64 replica with top-5 identical at BOTH positions, residual corr >= 0.999982 in every layer at both positions (the loosest number in the batch, on the narrowest residual in the tree), maxrel <= 4.4e-3; step 3 through the engine bit-identical to step 2 and reproduced; step 4 a fluent answer ending in `<|im_end|>` at token 36 (57 ms/token, 28 layers -- the fastest model in this batch; the answer is factually wrong about NPUs, which is a 0.6B model being a 0.6B model, not a kernel result). This admits `ln` width 1024 -- the first width other than 2048 to take the fused single-core path -- plus `gemv_q4` K = 1024 and K = 3072 and `lm_head_q4` K = 1024. The three geometry firsts the handoff flagged all behaved: the half-used x-stream element (`ELN` 2048 against a 4096-byte element) is read correctly, with no position-independent offset on the first residual; the o-projection GEMV being wider than the layer's own residual (q width 2048, hidden 1024) changes nothing. Details: `.claude/plans/k-new-points-results.md`.

### OPEN-FAMILY-LLAMA3: Llama 3 on the dense recipe
**Applies to:** openflowlm-next (`open_kernels/recipes/dense.py`, `spec.py`, `designs/dense/dx.py`, `src/open_qwen36/`)
**Test category:** manual (needs the NPU and `OpenFlowLM/Llama-3.1-8B-NPU2`); the derivation, the RoPE scaling and the 8B / 3.2-3B / 3.2-1B layouts are unit-tested in `tests/test_llama3.py`

A Llama 3 model (GQA without q/k norms, full RoPE with the llama3 frequency
scaling, eps 1e-5, silu FFN, a q4_1 head) shall run on the open kernels
from its `config.json` alone through the dense recipe: `qk_norm` / `norm_eps`
become the `ATTN_QKNORM` / `LN_EPS` knobs, the scaled inverse frequencies are
computed host side (`ModelSpec.rope_inv_freq`, in the manifest, used by both
position-table builders), and widths that overflow a core's memory are handled
by the recipe (one chunk per weight element; one norm output element per call).

`tie_word_embeddings` is not a refusal. Llama 3.2 (1B / 3B) ties the head to
the embedding table in `config.json`, but every container the recipe packs from
materialises `lm_head.weight` as its own q4 tensor -- OFLM's `.q4nx` does it for
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
kernel set is installed for their model, and honour `OFLM_LLAMA_ENGINE=open|closed`:
`Llama3` (`model_list.json` families `llama3.1`, `llama3.2`) and `DeepSeek_r1_8b`
(family `deepseek-r1`, a Llama-3.1-8B distill). Verify with `oflm serve <tag>` on a
model that has `open_kernels/` installed: the load logs `Llama 3 on the open kernels (<dir>)`
or `DeepSeek-R1 on the open kernels (<dir>)`, and `OFLM_LLAMA_ENGINE=closed` restores
the `llama_npu` DLL for both.

**Result 2026-09-05 (Llama-3.1-8B):** slice logits corr 1.000000 / 0.999993, same argmax and top-5, residual corr ≥ 0.999994 every layer; identical through the engine; a coherent two-sentence answer ending in `<|eot_id|>` at token 79 (203 ms/token). Details: `.claude/plans/open-kernels-phase-c-llama3.md`.

**Result 2026-09-06 (Deepseek-R1-Distill-Llama-8B-NPU2):** spec identical to Llama-3.1-8B's in every field but `extra` -- step 2 logits corr 0.999999 / 0.999987, same argmax (12451 / 37533), residual corr 1.000000 / >= 0.999987 every layer; top-5 identical at position 0 and slots 5-6 swapped at position 1 on a 0.025-logit tie; step 3 identical through the engine, request 2 reproduced request 1; step 4 a coherent answer ending in `<|end_of_sentence|>` at token 429. Its own export is byte-identical to `src/xclbins/Llama-3.1-8B-NPU2/open_kernels` (3/3 `insts.bin`), and the same slice passed against that directory directly. Caveat: `chat.py`'s template probe refuses this tokenizer (`tokenizer lacks ['<|end_of_text|>']`) because R1-Distill keeps Llama 3's `<|start_header_id|>` but renames the EOS tokens -- an engine-external gap, driven through `open_qwen36_cli` with DeepSeek's own template instead. Details: `.claude/plans/p0-p1-results.md`.

**Result 2026-09-06 (Llama-3.2-3B-NPU2 and Llama-3.2-1B-NPU2):** both 3.2 shapes' first hardware run, and both containers materialise `lm_head.weight` despite `tie_word_embeddings: true`, as the acceptance criteria above assume. **3B** (3072 / 28 / 8192, 24 query heads over 8 kv heads -- GQA group 3, the first odd group, `OG_AOUT_ELEMS` 3): step 2 logits corr 0.999998 / 0.999995, argmax 2 / 2 matching the fp64 replica with top-5 identical at both positions, residual corr 1.000000 in all four layers at t0 and >= 0.999996 at t1, maxrel <= 5.3e-3; step 3 through the engine bit-identical to step 2 and reproduced; step 4 a coherent answer ending in `<|eot_id|>` at token 72 (334 ms/token). **1B** (2048 / 16 / 8192, head_dim 64 -- `E_A` 1024, `KV_ROW` 2048, `PTAB_ROW` 1024, one KV band per core, a 768-byte RoPE record in a 1024-byte position row): step 2 logits corr 0.999999 / 0.999994, argmax 1757 / 1757 matching, top-5 identical at both positions, residual corr 1.000000 in all four layers at t0 and >= 0.999996 at t1, maxrel <= 2.4e-3; step 3 bit-identical and reproduced; step 4 `<|eot_id|>` at token 92 (262 ms/token). The 1B is the run that decides head dim 64, since a wrong q/k rotation there gives fluent nonsense rather than a crash: position 0 does not rotate and position 1 does, and both are clean in every layer, so the rotation is right. Together these admit `gemv_q4` K = 3072 and K = 8192, `ln` width 3072, `lm_head_q4` K = 3072 and the attention combinations (128, 24, 8, 128, False, False, False) and (64, 32, 8, 64, False, False, False). Note `chat.py`'s default `--max-tokens 64` truncates both models mid-sentence; 200 is enough. Details: `.claude/plans/k-new-points-results.md`.

**Nanbeige4.1-3B (2026-09-10).** Declares `model_type: llama` and is one for the
recipe: 2560 / 32 / 10752, 20 query heads over 4 kv heads at head_dim 128 (GQA group
5, two q heads per attention element), theta 7e7 with no scaling, eps 1e-5, an
untied 166144-row head, a q4_1 container. Two catalogue points are new: the
attention tuple `(128, 20, 4, 128, False, False, False)` and `gemv_q4` K = 10752
(the widest activation table so far that still keeps two chunks per weight element:
60032 of the core's 61440 bytes). The registry serves it through its own
`Nanbeige` class, which selects the open engine under `OFLM_LLAMA_ENGINE`.

**Acceptance criteria (unit, Nanbeige):** the derivation and layout in
`tests/test_llama3.py::test_nanbeige41_3b_derives_and_lays_out_on_the_llama_recipe`
-- band split `(5, 1, 5, 21, 5)`, `HPE 2`, `H_ELEMS 11`, `PER_CALL 2`, `TAB_BYTES 24192`,
`LMHEAD_BANDS 2596`, `ELN 5120`, `E_A 1024`; `hf_config_check` without `head_dim`
(a llama config may omit it).

**Procedure (manual, Nanbeige):** as OPEN-FAMILY-QWEN3 with `Nanbeige4.1-3B-NPU2`,
`out_nb`, prompt id 166100 (`<|im_start|>`); `chat.py` takes its ChatML template
without injecting think tags (the model opens its own `<think>` block).
`oflm-test --llm --model nanbeige4.1:3b` through `oflm serve`.

**Result 2026-09-10:** slice logits corr 0.999999 / 0.999989, same argmax and top-5 at
both positions, residual corr >= 0.999991 every layer; a 24-token decode chain extends
this to every position 0-23 (logits corr 0.99995-0.999999 throughout, top-5 identical at
21 of 24, a near-tie slot-5 swap at the rest); step 3 bit-identical to the harness,
request 2 reproduced request 1; `chat.py` opens `<think>` and reasons coherently (the
model has no off switch for it); `oflm serve` + `oflm-test --llm` PASS with a real
generation budget (`--gen-lim 600`+; the reasoning chain can outrun a small one, which
reads as an empty answer column rather than a failure).

Nanbeige's adapter originally reached the closed engine's class through a `dynamic_cast`
for checkpoint / restore, null on the open engine -- `oflm serve` segfaulted on the first
request. Fixed to the `causal_lm` virtuals every other open-engine adapter already uses.

### OPEN-FAMILY-GEMMA3: Gemma 3 on the dense recipe
**Applies to:** openflowlm-next (`open_kernels/recipes/dense.py`, `spec.py`, `designs/dense/dx.py`, `designs/ln/ln_nr32.cc`, `harness/stream_patch.hpp`, `src/open_qwen36/`)
**Test category:** manual (needs the NPU and `OpenFlowLM/Gemma3-4B-NPU2`); the derivation, the two RoPE tables, the window's row counts and the 4B layout are unit-tested in `tests/test_gemma3.py`

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
kernel set is installed for their model, and honour `OFLM_GEMMA_ENGINE=open|closed`:
`Gemma3` (`model_list.json` family `gemma3`, e.g. `gemma3:4b`) and `Gemma3_Text_Only`
(family `gemma3-text`, e.g. `gemma3:1b`). Verify with `oflm serve <tag>` on a model
that has `open_kernels/` installed: the load logs `Gemma 3 on the open kernels (<dir>)`
or `Gemma 3 (text) on the open kernels (<dir>)`, and `OFLM_GEMMA_ENGINE=closed`
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
`vegah/OpenFlowLM@feat/kernels` — all eight projection shapes cosine
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
`Atomic-Germ/*-OpenNPU2` and installs through `oflm-add`
(`oflm-add <repo-or-directory> --family granite`), which is the supported path
until it is hosted there. The measurement above is what was run; the oflm-add
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
- `ModelSpec.from_hf_config` on the 9B / 4B / 2B / 0.8B `config.json` (fixtures under `tests/fixtures/`, the models' own files) gives family `qwen35`, `num_experts 0`, `intermediate` 12288 / 9216 / 6144 / 3584, the MoE's layer pattern, 16/4 (or 8/2) heads and `lin_value_heads` 32 (or 16); the nested `text_config` (`qwen3_5_text`) and OFLM's flattened container config derive the same tower; `qwen3_5_moe` still derives to `qwen36moe`, and a config carrying `num_experts` is refused by name.
- Swapping only the FFN moves nothing in the attention half: the 27B spec and a dense twin of it (an FFN narrow enough to keep 10 KB weight elements) give identical DeltaNet / attention / state / KV / lm_head constants, and `qwen35.layout` is `qwen36moe.layout(..., ffn="dense")`, not a copy.
- The 9B layout: `PER_CALL 1` (a 12288-wide activation table leaves no room for two 10 KB weight elements beside the streams), `DN_ROWS 10 / DN_SLICES 13 / DN_PAD 130`, `S_ROWS 130`, `ELN 8192` (so the split `ln_y` / `ln_xn` norm entries), `E_A 2048` with 2 f32 heads per attention element and 4 og heads, `KV_ROW 4096`, `PTAB_ROW 2048`; the pool holds q4-sized `up | gate | down` first, at the same offsets for both layer types.
- No `moe` block, no `rout_idx_off`, no router or shared-expert tensor anywhere in the manifest; each layer type's program is one `run`, with `attnpos` on the full-attention kernel only.
- The manifest fixture parses in `manifest_test.cpp` (a linear-attention layer type with a one-step program and no `moe`); `ssm_out_proj` is a plain `std_perm` with no source-format field, and the two `transpose` ops carry the sizes `pools::apply` needs.
- **One record per value head.** The glue core emits `(NT - VALUE_TILE0) * HEADS_PER_TILE` records and the host drains one per value head; the two are equal only at 32 value heads (4 value conv tiles), so a 16-head model has 2 value tiles against its 4 key tiles. The value head's key head is `h / (lin_value_heads / lin_key_heads)` -- 2 value heads per key head at 32, one at 16.
- **The alpha / beta projection is padded to the accumulator's 32 lanes**, not narrowed: a W element stays 64 rows x 32 bf16 = 4 KB, `AB_ELEMS` is `hidden / 64` whatever the head count, and a 16-head model's `transpose` op carries `dst_rows` so columns 16..31 are zero. `dt_bias` sits at `lin_value_heads` floats inside `small`, not at a fixed 32.
- **The projection is walked in 4 KB halves.** The glue core holds ONE element of the layer-entry norm output, so the alpha and beta projections are re-streamed per half with the accumulator reset passed in (`glue_ab_e.cc`); a half carries `min(2048, hidden - h*2048) / 64` weight tiles, which is 32 and 8 at HID 2560. The side channel's fills are `2 + 4 * ceil(hidden*2 / 4096)` and the recipe refuses a hidden width whose count exceeds `LIMITS["shim_fills"]`, naming the number.
- **Wide glue worker (diagnostic only; fused topology cannot place).** Above 32
  value heads the dense path has `ceil(heads/32)` sequential AB banks and a
  dedicated depth-1 `xn_side` FIFO. Each bank consumes alpha then beta, each
  replaying all xn chunks, then one small-parameter element. Two 32-float
  accumulators are reused; decay/beta span the actual head count. A banked
  helper uses local accumulator indices but global A/dt_bias/output indices
  and processes only the active tail lanes. Existing <=32-head and MoE worker
  paths retain their original streams. The 48-head worker consumes 12 xn
  chunks and continues to six value tiles / 48 records. These contracts are
  tested by executing the actual Python worker with checked FIFO substitutes
  and compiling the actual small-helper indexing loop with host math in
  `tests/test_qwen35_wide_glue.py`. They do not establish IRON placement or
  NPU numerical correctness. The implemented host schedule fails actual IRON
  placement: weights + xn + gact require three input DMA channels, but the core
  has two. Wide recipe dispatch remains explicitly refused; only
  `utilities/probe-qwen35-wide.py` enables the diagnostic topology.
- **Shared WideDeltaNet AB primitive.** `recipes/wide_deltanet.py` describes
  geometry, 32-lane banks/tails, xn chunks and value-to-key grouping without a
  model-name branch. `designs/wide_deltanet/ab.py` uses a separate open dispatch
  with two input DMA channels and reused 32-float accumulators. Both H=5120 and
  H=2560, 48 heads, passed seven NPU dispatches against float64 math over the same
  bf16 inputs on 2026-09-24 (max-relative error <1e-4 and cosine >0.99999).
  The existing fixed-width `transpose_banked` opcode implements the new plan's
  conceptual `transpose_banked32`; both Python and C++ check every active tail
  1..32 and both hidden widths. This establishes AB/decay/beta only, not 48-head
  record emission or recurrent state. No catalogue promotion or model export.
  See [bring-up report](plans/wide-deltanet-bringup.md).

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
is installed for its model, and honours `OFLM_QWEN35_ENGINE=open|closed`: `Qwen3_5VL`
(`model_list.json` family `qwen3.5`, e.g. `qwen3.5:4b`). Verify with `oflm serve <tag>` on a
model that has `open_kernels/` installed: the load logs
`Qwen3.5 on the open kernels (<dir>)`, and `OFLM_QWEN35_ENGINE=closed` restores the
`qwen3_5vl_npu` DLL. Images always need the closed engine -- the open one has no vision
path, and an image payload is refused with
`images need the closed Qwen3.5 engine (OFLM_QWEN35_ENGINE=closed)`.

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
with a one-line warning -- native q8 not implemented yet at this width -- rather than composing
an export that cannot build (OPEN-QUANT-Q8). The other three sizes keep their container's q8
role. The q4_1 path is unaffected and still passes: the shipped
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

### OPEN-FAMILY-PHI3: Phi-3 / Phi-4-mini on the dense recipe
**Applies to:** openflowlm-next (`open_kernels/recipes/spec.py`, `dense.py`, `families.py`,
`designs/attn/attn.h`, `recipes/pack.py`, `model/replica_dense.py`, `model/dense_probe.py`,
`src/open_qwen36/manifest.hpp`, `manifest.cpp`, `pools.cpp`, `src/common/AutoModel/modeling_phi4.cpp`)
**Test category:** manual (needs the NPU and `FastFlowLM/Phi4-mini-Instruct-NPU2`); the
derivation, the longrope tables, the per-row table switch, the layout and the manifest
are unit-tested in `tests/test_phi3.py` and `src/open_qwen36/pools_test.cpp`

A Phi-3 model (`model_type: phi3`; Phi-4-mini is one) shall run on the open kernels
from its `config.json` alone through the dense recipe. Structurally it is Llama 3.2 3B's
layer -- GQA 24 over 8 at head_dim 128 without q/k norms, silu FFN at 3072 / 8192, eps
1e-5, a tied head the container materialises as `lm_head.weight` -- with two things no
family before it had:

- **A partial rotation.** `partial_rotary_factor 0.75` rotates 96 of the 128 head dims
  and leaves the rest alone. The attention core's RoPE loop runs 32 pairs a step, so it
  gains a 16-lane tail and the rule relaxes from "a multiple of 64" to "a multiple of 32"
  (`attn.h`); a family whose rotation is a multiple of 64 compiles the same loop it did.
  The replica and the position table already took `rotary_dim`; nothing else moves.
- **longrope.** Two factor lists over the same theta, one per rotary pair -- the short one
  up to `original_max_position_embeddings` (4096; on Phi-4-mini every short factor is
  1.0), the long one above it -- and one attention scale on cos and sin,
  `sqrt(1 + ln(factor) / ln(original))` with `factor = max_position_embeddings / original`
  (1.190 here), applied regardless of which list is active. HF picks the list per forward
  call from the running sequence length (`seq_len = max(position_ids) + 1 >
  original_max_position_embeddings`); this engine computes one row per token as the
  context grows (`Core::step`), so **both tables are baked into the manifest and the
  position table switches per row** at `original_max_position_embeddings` (`switch_row`
  on the ptab global; `long_inv_freq` beside `inv_freq`) instead of picking one table for
  the whole resident buffer at export time. Row r therefore carries whichever table a real
  forward call at sequence length r + 1 would have picked, and -- matching how a real KV
  cache behaves -- that choice does not move once a row is written even if the
  conversation later crosses the threshold. `original_max_position_embeddings` is
  wherever the container states it (`rope_scaling` or the top level); the export's own
  `--max-ctx` plays no part in the choice, only in how many rows exist. The scale rides on
  the ptab global as `scale` (absent, 1.0, for every other family, whose manifests carry
  neither key and are byte-for-byte unchanged) and both packers apply it unconditionally
  as they write cos and sin.
- **A load-time compatibility check that actually names the RoPE configuration.** Every
  other family bakes `rope_theta` (and Llama 3's scaling) into the manifest without
  checking the container agrees at load -- harmless there, because none of those tables
  vary with anything but the shape fields `hf_config_check` already compares. Phi-3's do:
  two same-shaped containers can be longrope fine-tunes extended to different context
  lengths, with different factor lists and nothing else different, and loading one under
  a kernel set built for the other would run and return plausible garbage. So `phi3`'s
  `hf_config_check` also carries `rope_theta`, `rope_scaling` verbatim,
  `original_max_position_embeddings`, and `max_position_embeddings` when it set the
  attention scale (a `rope_scaling` without its own `factor`). HF lets a config omit
  several of these (`head_dim`, `partial_rotary_factor`, `rope_scaling`), and
  `Manifest::check_model` fails closed on an absent key -- so rather than emit a check
  only when the source config spelled the key out (one-way: a kernel set built from a
  full-rotation config would then accept a 0.75 container), the manifest also carries
  **`hf_config_defaults`**, what an absent key means (`head_dim`: hidden / heads;
  `partial_rotary_factor`: 1.0; `rope_scaling`: none; `original_max_position_embeddings`:
  whatever the sub-object says). The checker compares the expected value against the
  default when the key is absent, so an omitted optional field is accepted exactly when it
  implies what the kernels were built for and refused otherwise, in both directions.
  `hidden_act` other than silu is refused at derivation (the FFN kernel is silu).

Only the HF derivation exists: a Phi-3 GGUF carries the factor lists as tensors
(`rope_factors_{long,short}.weight`), not metadata. The `Phi4` class selects the open
engine under `OFLM_PHI4_ENGINE`.

**Acceptance criteria (unit):**
- `rotary_dim` 96 derives from the factor; without one, the whole head, and the check
  then names 1.0 (`test_refusals_and_defaults`); a non-silu `hidden_act` is refused.
- The short and long inverse-frequency tables equal transformers'
  `_compute_longrope_parameters` (the plain table divided by the list), the short list at
  and below 4096, the long one above; `rope_scale()` equals its attention factor,
  1.1902380714; a Llama spec's is 1.0 and its table ignores the context.
- A `rope_scaling` type other than `longrope` is refused by name.
- The layout: band split `(6, 2, 6, 16, 6)`, `HPE 4`, `OG_AOUT_ELEMS 3`, `ELN 6144`,
  `E_A 2048`, `KV_ROW 4096`, `PTAB_ROW 2048`, `PER_CALL 2`, `TAB_BYTES 18432`,
  `LMHEAD_BANDS 3126`; build dir `dense/build_phi3_h3072`.
- The manifest's ptab global carries `scale`, the short `inv_freq`, `long_inv_freq` and
  `switch_row = original_max_position_embeddings` -- all independent of the export's
  `--max-ctx` (`test_layout_and_manifest`). `hf_config_check` carries
  `partial_rotary_factor`, `head_dim`, `rope_theta`, the raw `rope_scaling`,
  `original_max_position_embeddings` and (when it set the scale) `max_position_embeddings`;
  `hf_config_defaults` carries what an absent `head_dim` / `partial_rotary_factor` /
  `rope_scaling` / `original_max_position_embeddings` means. A container with a different
  rotation, theta, longrope table or `max_position_embeddings` is refused at load by name;
  one omitting `head_dim` is accepted; one omitting `partial_rotary_factor` is refused
  against the 96-dim kernels (`test_the_compatibility_check_is_two_way_through_the_defaults`,
  `manifest_test.cpp`'s phi3 block on `fixtures/manifest_phi4_mini_4b.json`). The
  checked-in `recipes/specs/phi4-mini-4b.json` yields the same checks through
  `export --spec` (`test_a_spec_loaded_from_json_still_emits_the_full_check`). A Llama
  manifest has no `scale` / `long_inv_freq` / `switch_row` key and empty defaults.
- `pack.ptab(..., scale)` multiplies cos and sin; given `long_inv_freq` + `switch_row` it
  reads `inv_freq` for row r < switch_row and `long_inv_freq` for r >= switch_row, in the
  SAME table (not two separate calls that happen to agree); the two are required together.
  `pools::build_ptab` (C++) is byte-identical to `pack.ptab` on the same inputs, checked
  both for a plain scale and for the full switch, by shared FNV-1a hash
  (`src/open_qwen36/pools_test.cpp`'s `ptab_scale_tests` / `ptab_switch_tests`) -- this is
  the actual production function `Core::Core()` calls to build the resident table, not a
  reimplementation. `RowGlobal::switch_row` defaults to `kSwitchNever`, so an unrelated
  family's table is unaffected by the field existing on the struct.
- `replica_dense.rope` rotates only the first `rot` dims and scales the whole rotation;
  `dense_decode` and `dense_probe.py` pick the table from `ctx = pos + 1` per call -- HF's
  own `seq_len` rule -- so positions `original - 1` and `original` straddle the switch
  exactly where the packer's `switch_row = original` does
  (`test_dense_decode_picks_the_table_from_pos_plus_one`).

**Procedure (manual):** as OPEN-FAMILY-QWEN3 with `Phi4-mini-Instruct-NPU2`, `out_ph`,
prompt id 200021 (`<|user|>`; the model has no bos); `chat.py` switches to
`<|user|>...<|end|><|assistant|>` when the tokenizer has `<|user|>` and `<|end|>`.
`oflm-test --llm --model phi4-mini-it:4b` through `oflm serve`. Two catalogue points
enter with it: the attention tuple `(128, 24, 8, 96, False, False, False)` -- the first
partial rotation on the dense design -- and nothing new for the GEMVs (3072 and 8192
are Llama 3.2 3B's).

**What the boundary itself is not covered by:** the 4-layer slice below exercises
positions 0-1, both short-table rows (`original_max_position_embeddings` is 4096), so it
cannot see the switch on real weights directly. What stands in for it: `build_ptab` is
the identical function the engine calls at load, exercised at a row straddling a
synthetic switch and cross-checked against the NumPy packer byte for byte (the
acceptance criteria above); the hardware slice separately proves the engine correctly
consumes whatever the table holds at the positions it was run at. Together these cover
the mechanism end to end without a multi-thousand-token hardware decode.

**Result 2026-09-10:** slice logits corr 0.999998 / 0.999990, same argmax and top-5 at
both positions, residual corr >= 0.999995 every layer (layer 3's residual norm at
position 0, 2807, is the family's attention-sink token; the replica agrees); step 3
bit-identical to the harness, request 2 reproduced request 1; `chat.py` answers
coherently, ending on `<|end|>` at token 50. Fast attention (`attnknobs.FAST_ATTENTION`)
measured against the slow path afterward: identical correlation and residuals, decode
190 -> 84 ms/token (2.3x). Re-run unchanged after the compatibility-check and per-row
longrope fixes below (same corr, argmax, top-5 -- the new manifest fields are additive).

### OPEN-FAMILY-QWEN3VL: Qwen3-VL's decoder is a Qwen3 dense spec
**Applies to:** openflowlm-next (`open_kernels/recipes/spec.py`)
**Test category:** unit (`tests/test_qwen3vl.py`); the end-to-end run is
OPEN-VISION-EMBED's, once the model and a kernel set for it exist

Qwen3-VL's decoder is Qwen3 dense. A config whose `model_type` is `qwen3_vl` or
`qwen3_vl_text` shall derive a `ModelSpec` with `family` `qwen3` and the same
hyperparameters a plain Qwen3 of that geometry derives, so a Qwen3-VL model links to a
Qwen3 kernel bundle rather than building its own. Interleaved M-RoPE changes only the
position table the engine hands the kernels, and the vision tower is read separately by
`VitConfig`; neither reaches the spec. Both config shapes are accepted: the decoder
nested under `text_config` (raw HF) and flattened at the top level (the container OFLM
ships).

**Acceptance criteria:**
- `model_type: "qwen3_vl"` with Qwen3-VL-4B's fields derives `family == "qwen3"`,
  `qk_norm`, no `attn_gate`, `rotary_dim == head_dim`, all layers `dense`.
- The nested and flat forms of the same config give the same `spec_hash()`.
- That hash equals the hash of the same geometry declared as `model_type: "qwen3"`.
- Two configs differing only inside `vision_config` give the same `spec_hash()`.
- A missing decoder field is refused by name, in either config shape.
- Qwen3-VL-4B-Instruct-NPU2's own config.json derives the same `spec_hash()` as Qwen3-4B-NPU2's, so `oflm-add` links it to that bundle with no build of its own. (That is the hash over config.json; `recipes.load.spec_from_model_dir` also folds in the tokenizer's real vocab and the container's per-role weight formats, which need the files.)
- That container is tagged `model_type: "qwen3"`, flat, with no `text_config` and no `vision_config` -- `_qwen3vl_hf` never runs for the model people pull, `_qwen3_hf` does.
- Qwen/Qwen3-VL-4B-Instruct derives the same geometry at a different `rope_theta`.

**The container's rope_theta disagrees with upstream: 1e6 against Qwen's 5e6.** Every
other field matches. A kernel set built from the container rotates at 1e6, and so does
the replica, so the two will agree with each other whether or not that is the right
number -- the same trap OPEN-PACK-Q4-0 walked into. Settle it against text quality or
against transformers with the original weights, not against the replica.

**Adapters:** `Qwen3VL` (`model_list.json` family `qwen3vl`, `qwen3vl-it:4b`) selects the
open engine when a kernel set is installed for its model and honours
`OFLM_QWEN3VL_ENGINE=open|closed`, as the other adapters do. The vision dispatch in
`Engine::prefill` reads a `qwen3vl_image_payload_t` under family `qwen3` -- Qwen3-VL's
decoder derives as plain Qwen3, so its kernel set is a `qwen3` one and the family string
does not say VL.

**Images are not reachable for this family yet**, for three reasons, none of which is the
decoder: the container carries no `vision_config` (OPEN-VISION-VIT-CONFIG), no
`image_token_id` and no `rope_parameters.mrope_section`, which is what
`Engine::prefill`'s existing refusal names; and Qwen3-VL's tower uses deepstack, which
the host tower does not implement. Text-only is the reachable half.

**Result 2026-09-13: the text half runs end to end, and the conditional is settled.**
The hash question the plan left open -- whether the VL container agrees with Qwen3-4B on
the two things a config.json cannot show -- is now answered on the shipped files, not
inferred: pulled `FastFlowLM/Qwen3-VL-4B-Instruct-NPU2` (4.1 GB) and ran
`spec_from_model_dir` on it and on the installed Qwen3-4B-NPU2. Both derive
`sha256:602fa1836b218cfd17b8a11628cde954587cd53ad3345a04ef1d998d23951dfd`: same tokenizer
id count (151669), same per-role quant (q4_1), and the two config.json files differ only
by the three vision file-name keys. So the family-bundle property holds for real -- a
Qwen3-4B open kernel set was exported (WSL, ~1 min) and `oflm-add` linked the VL container
to it by hash with no build of its own, logging
`open kernels from 'Qwen3-4B-open': its manifest spec_hash matches this model's`.

The procedure OPEN-FAMILY-QWEN3 defines, run against this container:

- 4-layer slice at positions 0 and 1 against the fp64 reference: logits corr 0.999995 /
  0.999996, argmax 39161 / 91278 matching, top-5 identical at both, residual corr
  >= 0.999996 in every layer (maxrel <= 3.4e-3).
- Through the engine (`open_qwen36_cli --dump-logits --twice`): BIT-IDENTICAL to the
  harness at both positions (max abs diff 0.000e+00 over all 151936 logits), and
  request 2 reproduced request 1.
- Whole 36 layers through `chat.py`: a coherent two-sentence answer about NPUs ending at
  `[eos]`, 62 ms/token.
- `OFLM_QWEN3VL_ENGINE=open oflm serve qwen3vl-it:4b` logs
  `Qwen3-VL on the open kernels`, and `oflm-test --llm --model qwen3vl-it:4b` PASSES both
  streamed rounds. `OFLM_QWEN3VL_ENGINE=closed` still loads the DLL.

**rope_theta:** the container's 1e6 produces coherent text at these lengths, so it is
what the weights were converted for as far as this evidence goes. That is text quality,
which is the right oracle; it is not a long-context result.

**Result 2026-09-13 (later): images work too.** `oflm-test --vision --model
qwen3vl-it:4b` passes all three rounds through `oflm serve` on the open engine, and the
server says what it did:

    open_qwen36: image 18x88 patches -> 396 tokens + 3 deepstack in 4.37 s
    open_qwen36: image 30x44 patches -> 330 tokens + 3 deepstack in 3.41 s
    open_qwen36: image 32x50 patches -> 400 tokens + 3 deepstack in 4.46 s

The text it extracts from `paris.png` is what that image says, and a following turn
tells a story about all three. Four things had to be settled, and three of them were
recorded here as blockers that turned out to be wrong:

- **The tile order**, OPEN-VISION-VIT-FLAT. Not the 35B's tiling with a collapsed
  header, and not row-major either.
- **The geometry.** Everything but two numbers falls out of the weight file; those two
  are named in the refusal rather than guessed (OPEN-VISION-VIT-FLAT).
- **The injection route.** The design called for `step_gemm_block` and a batched
  `visual_pos_mask`, and noted that no kernel set has a `gemm_block` program -- all
  three log `GEMM-route prefill block size: 0`. It was not needed: `step_embed` already
  walks image rows one at a time and knows which rows are image rows, so feature j is an
  add on `xres` between layer j and j+1 -- a 10 KB sync back, a host add and a sync
  forward, against a tower that costs seconds.
- **Where the missing keys live.** Not `config.json`: see OPEN-VISION-VIT-FLAT's
  sidecar note.

What is still Atomic-Germ's is the durable fix -- `q4nx-build` writing the tower keys,
`image_token_id` and `mrope_section` into the container it converts, so a Qwen3-VL
describes itself the way the other two VLM families do.

### OPEN-FAMILY-QWEN25VL: Qwen2.5-VL's decoder is a Qwen2.5 dense spec
**Applies to:** openflowlm-next (`open_kernels/recipes/spec.py`, `src/common/AutoModel/modeling_qwen2vl.cpp`)
**Test category:** unit (`tests/test_qwen25vl.py`); the tower is OPEN-VISION-VIT-WINDOWED's
and the end-to-end run is OPEN-VISION-EMBED's

Qwen2.5-VL's decoder is Qwen2.5 dense. A config whose `model_type` is `qwen2_5_vl` or
`qwen2_5_vl_text` shall derive a `ModelSpec` with `family` `qwen2` and the same
hyperparameters a plain Qwen2.5 of that geometry derives, so the model links to a Qwen2.5
kernel bundle rather than building its own. M-RoPE changes only the position records the
engine builds and the tower is read separately by `VitConfig`; neither reaches the spec.
Both config shapes are accepted: the decoder nested under `text_config` (raw HF) and
flattened at the top level (the container OFLM ships).

The `Qwen2VL` adapter shall select the open engine whenever a kernel set is installed for
its model and honour `OFLM_QWEN2VL_ENGINE=open|closed`, as every other adapter with an open
path does.

**Acceptance criteria:**
- `model_type: "qwen2_5_vl"` with the shipped 3B's fields derives `family == "qwen2"`, 36 dense layers, hidden 2048, intermediate 11264, 16 heads over 2 kv heads at head dim 128, full RoPE at theta 1e6, and a q/k/v bias.
- That spec's `spec_hash()` equals the one `Qwen2.5-3B-Instruct-NPU2` derives, through `spec_from_model_dir` -- which folds in the tokenizer's id count and the container's per-role weight formats, not only config.json.
- A kernel set exported for either model declares `model_type` `["qwen2", "qwen2_5_vl", "qwen2_5_vl_text"]`, so `Manifest::check_model` accepts both containers.

**Result 2026-09-13 (the shipped 3B): PASS, and it needed no kernel build.** Both
containers derive `sha256:e32bfd7e950c` through `spec_from_model_dir`, so `oflm-add` linked
the installed Qwen2.5-3B set to the VL model by spec hash with nothing rebuilt -- the
family-bundle property, on a second real container rather than in principle. The one thing
that did have to move is the manifest's accepted `model_type` list, which is why the set was
re-exported. Greedy decode through `open_qwen36_cli` on the VL container answers "The capital
of France is Paris." and then emits `<|im_end|>`, at 15.5 tok/s.

Note the two containers do NOT share a weight format: Qwen2.5-3B stores the signed 4-bit
quantiser (every block min exactly zero, OPEN-PACK-Q4-0) and Qwen2.5-VL stores real q4_1.
The packer detects that per tensor, so one kernel set serves both -- which is the property
being claimed, and it would have been invisible had only one of them been tried.

### OPEN-FAMILY-QWEN2: Qwen2.5 is a dense spec with a bias on q, k and v
**Applies to:** openflowlm-next (`open_kernels/recipes/spec.py`, `families.py`, `dense.py`)
**Test category:** unit (`tests/test_qwen2.py`); the hardware run is OPEN-ATTN-QKV-BIAS's

Qwen2.5 is the dense recipe's shape in every respect but one: its q/k/v projections
carry a per-channel bias. A config whose `model_type` is `qwen2` shall derive a
`ModelSpec` with `family` `qwen2` -- GQA, no q/k RMSNorm, no attention gate, full RoPE,
a silu-gated FFN -- and `recipes.families.family_module` shall route it to the dense
recipe, which carries the bias through to the kernels (OPEN-ATTN-QKV-BIAS).

The bias is not a `ModelSpec` field: every Qwen2 has it, which makes it a family
property, and `spec_hash()` covers every field, so adding one would move every shipped
model's hash for no kernel change.

**Acceptance criteria:**
- `model_type: "qwen2"` derives `family == "qwen2"`, `qk_norm` false, `attn_gate` false,
  `activation` `silu`, all layers `dense`.
- `head_dim` absent falls back to `hidden_size / num_attention_heads`; present, it wins.
- A `hidden_size` that is not a multiple of the head count is refused, naming `head_dim`.
- `family_module("qwen2")` is the dense recipe and `"qwen2" in families.FAMILIES`.
- `dense.qkv_bias` is true for `qwen2` and false for every other dense family; `qkv_bias`
  is not a key of `spec.to_dict()`.

### OPEN-PACK-Q4-0: a 5120-byte chunk with no mins is not q4_1
**Applies to:** openflowlm-next (`open_kernels/recipes/pack.py`, `model/q4nx.py`,
`src/open_qwen36/pools.cpp`)
**Test category:** unit (`tests/test_quant_q4_0.py`); the end-to-end run is
OPEN-ATTN-QKV-BIAS's

Some containers store a SIGNED 4-bit quantiser in the same 5120-byte chunk q4_1 uses:
`w = d * int4(q)` with the min written as zero, rather than q4_1's `w = d * q + min`
over an unsigned nibble. Nothing about the chunk's size or the safetensors header
separates the two. Read as q4_1 the tensor comes out one-sided -- every value in a block
shares the sign of its scale -- at about 2.7 times the right spread, and the model
answers with noise while agreeing with the fp64 replica to the bit, because
`dq_chunks_q4_1` misreads it the same way.

A 5120-byte tensor shall therefore be transcoded to q4_1 on the way into the pool when
it is the signed form: flip bit 3 of every nibble, which turns two's complement into
offset binary, then write `min = -8 * d`. `int4(q) == (q ^ 8) - 8`, so both halves are
exact -- the flip is a relabelling and `-8 * d` only moves a bf16 exponent -- and
afterwards the GEMV, the pool and the replica are all already reading the right values.
Nothing downstream learns a new format, the same way a Q4_K container needs no kernel to
know about Q4_K.

Which form a tensor is in comes from the container when the container says: a
`quant_format` stamp in the safetensors `__metadata__`, per tensor or for the file. No
container writes one today, so the fallback is the data -- 256 exactly-zero mins in a
chunk, which a real q4_1 tensor does not manage, because a min is a block's own minimum
and every one of them being 0.0 does not happen to real weights. The stamp wins when it
is there, so a container that declares itself is never second-guessed.

The replica reads through the same rule and the same transcode (`q4nx.dq_tile`), so it
cannot disagree with the pool about a format -- which is exactly how this went unnoticed:
before it, both misread the container identically and every comparison passed.
`.claude/plans/qwen2-q4-0-container.md` carries the evidence.

**Acceptance criteria:**
- A 5120-byte chunk with 256 zero mins reads as the signed form; one non-zero min and it
  is a plain q4_1 tensor, passed into the pool untouched.
- A container that declares `quant_format` wins over the data signal, both ways.
- The transcode gives `min == -8 * d` exactly, and every nibble's `(q ^ 8) - 8` is the
  signed value it stood for.
- `pools.cpp`'s `q4_0_to_q4_1_chunks` agrees with `pack.q4_0_to_q4_1` byte for byte.
- A reader that handles one quant format only names the format it got rather than
  reshaping into it: `Q4NX.lmhead_logits` is the q8 head and refuses a 4-bit one, which
  17 q4_1 chunks would otherwise pass as 10 q8 chunks.

### OPEN-ATTN-QKV-BIAS: a per-channel bias on the q, k and v projections
**Applies to:** openflowlm-next (`open_kernels/designs/attn/attn.h`, `designs/dense/dx.py`,
`recipes/dense.py`, `model/replica_dense.py`)
**Test category:** unit (`tests/test_qwen2.py`, `tests/test_op_range.py`) for the layout and
the refusal; manual (the procedure below, needs the NPU and a Qwen2.5 container) for the
numbers

A family whose q, k and v projections carry a per-channel bias shall have it added to
each projection's output before the q/k norm and the rotation -- `q = q + b_q`, and the
same for k and v -- and nowhere else: `o_proj` and the FFN have none. The bias is a
family property (`recipes.dense.QKV_BIAS_FAMILIES`), so a family without one shall
compile the attention it compiled before, byte for byte.

The three vectors ride in the layer's `consts` buffer as bf16, the dtype the container
stores them in, and stream into the attention core on a fifo of their own, one bias
element per projection element. That lockstep is what makes a second stream cheaper than
either interleaving the fills or holding the whole bias in the core's L1: a projection
element carries `KVH/2` heads as f32 (`E_A` bytes) and the same heads of a bf16 bias are
half that, so `QW*2 / (E_A/2)` is exactly `Q_AIN_ELEMS` and `KVW*2 / (E_A/2)` is exactly
`K_AIN_ELEMS`, for any geometry `attn.h` accepts.

Qwen2.5-3B is also the narrowest attention element any dense family has had -- 2 kv heads
at head dim 128 is 512 bytes -- and two things the design had always got for free stop
being free there. Both are arithmetic over the spec and shall be checked as such
(`tests/test_dense_stream.py`), for every dense family, rather than found on hardware:

- The position record is 1024 bytes, so it spans TWO elements, not one. The core shall
  acquire the whole record and read cos / sin from the element that holds them
  (`ATTN_PTAB_SPLIT`); acquiring one would leave the other half in the stream to be read
  as q, and the leftover would still be there when the next layer started.
- A core emits `NHL / kOGH` output elements of `kOGH = min(NHL, HPO)` heads each, not one
  element of `NHL` heads. The two agree exactly while a core owns one element's worth of
  heads, which every shipped dense family does.

**Acceptance criteria (unit):**
- Every fill into the attention stream -- meta, the position record, q, k, v, a cached KV
  row -- is a whole number of elements, on every dense family, and the count the core
  acquires equals the count the fill delivers.
- Qwen2.5-3B gives `(E_A, PTAB_ROW) == (512, 1024)` and `(PTAB_ELEMS, PTAB_CS_ELEM) ==
  (2, 1)`; Qwen3-4B gives `(1, 0)`, as every family before it.
- A record that is not a whole number of elements, one that spans more than two, and one
  whose cos / sin straddle two are each refused by name.
- `dense.layout` gives a Qwen2.5-3B spec three consts slots, `CD_QB | CD_KB | CD_VB`, sized
  `QW*2 | KVW*2 | KVW*2` and within `CD_BYTES`; a family without a bias gets `-1` for all
  three, not 0 -- 0 is the input norm's own offset, where a stray read would find a real
  tensor instead of failing.
- `dense.pack_plan` emits a `put` for each of `q_proj.bias`, `k_proj.bias`, `v_proj.bias`
  at those offsets, with those caps.
- The bias element count equals the projection element count for q and for k/v.
- The `attn` catalogue combination carries `qkv_bias` as its eighth key; Qwen2.5-3B's
  `(128, 16, 2, 128, False, False, False, True)` is refused by name until hardware has run
  it (OPEN-OP-RANGE).
- `replica_dense.dense_decode` adds the bias for a `QKV_BIAS_FAMILIES` spec and not
  otherwise.

**Procedure (manual):**
1. Compile `designs/attn/*.cc` for a shipped family's flags from the tree before and after
   the change and compare the objects: every one must be byte-identical. The guards
   (`ATTN_BIAS_PARM` / `ATTN_BIAS_ARG`) exist for this, the same way `ATTN_H0_PARM` does --
   an unused parameter changes the generated code.
2. Export the kernel set for the Qwen2.5 container with `OPEN_KERNELS_UNVALIDATED=1`, pack
   it, and run `model/make_decode.py` + `compare_decode.py` at positions 0 and a few
   hundred: logits correlation > 0.9999 and the same argmax against the fp64 replica.
3. `oflm-test --llm` through `flm serve` on the installed model.
4. Then add the tuple to `catalogue.py` and drop the override.

**Result 2026-09-12 (Qwen2.5-3B-Instruct-NPU2):** passes, and admits the attention tuple
`(128, 16, 2, 128, False, False, False, True)` and `gemv_q4` K 11264 to the catalogue.

Step 3 ran the same day and PASSES: `oflm-test --llm --model qwen2.5-it:3b` through
`oflm serve` on the rebuilt engine returns coherent on-topic answers in both stream
rounds, 3862 and 3595 characters, the follow-up round reusing the prompt cache. It
passes again on the fast attention path (OPEN-ATTN-CONTEXT) at 15.9 and 12.6 tok/s
against 2.5 on the shipped kernel, which takes the suite from about forty minutes to
about one.

One caveat that has to travel with that result. The first attempt at the same two
rounds died on the follow-up: the cache-reusing prefill of 16 tokens completed, the
first decode step after it did not, and `Engine::guarded` poisoned the engine and
rebuilt it (`core.cpp`'s per-dispatch wait, 60 s, `OFLM_OPEN_TIMEOUT_MS`). It has not
happened since -- the passing run above plus two non-streamed two-turn repros at 120
and 704 tokens of first-turn context. So it is intermittent, once in four, and NOT
understood.

> **Diagnosed 2026-09-13, and it is not ours.** The dispatch is not slow, and the array
> does not hang: the command is never executed. The NPU driver logs its own view of
> every occurrence as `pci` Event ID 3 against `\Device\NTPNP_PCI0033`, naming the
> process and hardware context --
>
>     PID=1100 Ctx=144 TxnOp=ffffffff CtxPC=28b06005 FeTYPE=0 FeExTYPE=0 FePC=0 FeAM=0
>
> -- and those fields are AMD's own `struct aie2_ctx_health` (`txn_op_idx`, `ctx_pc`,
> `fatal_error_type`, `fatal_error_exception_type`, `fatal_error_exception_pc`,
> `fatal_error_app_module`), which the `amdxdna` driver collects for exactly one reason:
> a command timed out, so it asked the firmware what happened. Every fatal-error field
> is **zero** and `txn_op_idx` is `0xffffffff`: no fault, no exception, nothing in
> flight. A lost command, not a hung kernel.
>
> Eight entries, all inside the 45-minute window holding the three failures; none across
> the following three hours and about 200 clean requests. Ruled out by measurement:
> context position (a sweep at 0 / 256 / 613 / 1024 / 2048), the turn boundary (it hit a
> first-round prefill and a cache-reusing one in the same run), context capacity (40
> requests at the full 32768), extra hardware contexts (three either way), NPU power
> mode (global and persistent), thread handoff (prefill and first decode are back to
> back on one thread), Modern Standby, and this entry's own suggestion below.
>
> **The slow-path theory here was wrong.** Qwen2.5 has been on the fast attention path
> since 2026-09-12 and the failure outlived it; a dispatch that normally takes about
> 2 ms is nowhere near a 60 s ceiling either way. And raising `OFLM_OPEN_TIMEOUT_MS`
> would only hide it. `Core::run` now prints the layer, the dispatch number, the host gap
> before it, waits again to separate a late command from a lost one, and points at the
> driver's event. Full record: `.claude/plans/dx-timeout.md`.

Step 1: 44 attention translation units, built for the qwen3, hunyuan, MoE, llama3 and phi3
flag sets from the tree before and after, every one byte-identical.

Step 2: an 8-layer slice at positions 0-3 gives logits corr 0.999998 / 0.999981 / 0.999979
/ 0.999965 with the same argmax and top-5 at every position, and the whole 36-layer model
gives 0.999965 / 0.999945, argmax and top-5 identical at both. The engine is bit-identical
to the harness at every position of a sixteen-position prefill.

Step 3: `src/open_qwen36/chat.py` through `open_qwen36_cli` on the full model answers the
NPU question in one coherent sentence, ending on `<|im_end|>`.

The first run of step 2 did NOT pass, and what it found is recorded as OPEN-PACK-Q4-0: the
container stores a signed 4-bit quantiser in the chunk q4_1 uses, and both the packer and
the fp64 replica decoded it as q4_1. They agreed with each other to the bit -- engine and
harness identical over sixteen positions, argmax matching at fourteen, 36 layers at
position 0 at corr 0.999944 -- while the model answered with noise. The numbers above are
from after that was fixed. Two things follow for anyone reading a result like it again:
agreement between the device and the replica says nothing about the weights being right,
because they share a dequantiser; and the apparent bf16 KV-cache sensitivity in the first
run was an artefact of the misread weights and is not real. `qwen2` reaches the fast
attention path the way every family does, by measurement, and has not been measured yet.

### OPEN-ROPE-YARN: YaRN inverse frequencies and its attention factor
**Applies to:** openflowlm-next (`open_kernels/recipes/spec.py`)
**Test category:** unit (`tests/test_gptoss.py`)

GPT-OSS extends a 4096-token pretraining context to 131072 with YaRN, which is a third way
of stretching RoPE alongside the Llama 3 and longrope rules the spec already reads. Each
rotary pair is either left alone -- it turns few enough times inside the original context
that the model saw its whole range -- or divided by `factor`, which is plain position
interpolation; the pairs between the two dims where `beta_fast` and `beta_slow` rotations
fit take a linear blend. A `rope_scaling` (or `rope_parameters`) whose type is `yarn` shall
derive that table, and `rope_scale()` shall return YaRN's attention factor,
`0.1 * ln(factor) + 1`, which multiplies cos and sin.

The blend is HF's `_compute_yarn_parameters` including its asymmetry -- the ramp indexes
`rotary_dim / 2` pairs against bounds computed on the `rotary_dim` scale -- because the
position table has to be the one the weights were trained against, not the tidier one.
`mscale` / `mscale_all_dim` (DeepSeek's variant of the attention factor) is refused rather
than ignored.

**Acceptance criteria:**
- GPT-OSS 20B's parameters (theta 150000, factor 32, beta 32/1, truncate false, original
  context 4096, rotary dim 64) reproduce transformers' own table to fp32 precision: pair 0
  is 1.0 (unscaled), pair 31 is `150000^(-31/32) / 32` (fully interpolated), pairs 16 and 17
  are 4.564839e-4 and 1.2931869e-4.
- `rope_scale()` is 1.3465735902799727; an explicit `attention_factor` wins over it; a
  `factor` of 1 or less gives 1.0.
- The branch is keyed on `rope_type`, so linear, longrope, Llama 3 and unscaled families
  derive exactly what they derived before.

### OPEN-FAMILY-GPTOSS: GPT-OSS derives a spec and has no recipe yet
**Applies to:** openflowlm-next (`open_kernels/recipes/spec.py`, `families.py`)
**Test category:** unit (`tests/test_gptoss.py`)

A config whose `model_type` is `gpt_oss` shall derive a `ModelSpec` with `family` `gptoss`:
GQA attention with no q/k norm and no gate, a sliding window on the layers HF marks
`sliding_attention` and none on the ones it marks `full_attention`, YaRN RoPE, and an MoE
FFN with no shared expert. Two keys mean something other than what they say and the
derivation shall read them as they are meant:

- `intermediate_size` is the EXPERT width. There is no dense FFN, so it becomes
  `moe_intermediate` and `intermediate` stays 0.
- `hidden_act` says `silu` and is wrong about it. The experts compute
  `(up + 1) * gate * sigmoid(1.702 * gate)` with `gate` clipped above at 7 and `up` clipped
  both ways, and -- in HF's own safetensors -- `gate` and `up` interleaved down the expert's
  rows rather than split in half. The spec records `activation == "clamped_swiglu"`; 1.702
  and 7.0 are family constants, not fields, and OPEN-GPTOSS-FFN-REF is the arithmetic.

GPT-OSS's `full_attention` is a plain dense layer and shall map to `dense`, NOT to the
spec's `full_attention` layer type -- that one is the full half of Qwen3.6's linear/full
alternation and has nothing to do with this family.

The sink, the biases and the activation's constants are family properties for the same
reason Qwen2's bias is: every GPT-OSS has them and `spec_hash()` covers every field.

`recipes.families.family_module("gptoss")` shall raise `NotImplementedError` naming what is
missing, and `gptoss` shall stay out of `FAMILIES` and `DENSE_FAMILIES`. Routing it to the
dense or the MoE recipe would emit kernels that drop the sink and compute the wrong FFN, and
then report parity against a reference carrying the same gaps -- which is exactly how
OPEN-PACK-Q4-0 went unnoticed.

**Acceptance criteria:**
- gpt-oss-20b's fields derive `family == "gptoss"`, `(num_heads, num_kv_heads, head_dim) ==
  (64, 8, 64)`, `rotary_dim == 64`, `(num_experts, experts_per_tok) == (32, 4)`,
  `moe_intermediate == 2880`, `intermediate == 0`, `shared_expert_intermediate == 0`,
  `qk_norm` and `attn_gate` false, `norm_eps == 1e-5`.
- `layer_types` alternates `dense_local`, `dense` starting at `dense_local`, from the config's
  list or, when it has none, from HF's own default; `sliding_window == 128`. An unknown layer
  type name and a sliding layer with no window are each refused by name.
- `head_dim` is read from the key when present -- GPT-OSS's is 64 while `hidden / heads` is
  45 -- and a `hidden_size` that is not a multiple of the head count is refused naming
  `head_dim`.
- The older `rope_theta` + `rope_scaling` pair and transformers 5's single `rope_parameters`
  object give the same `spec_hash()`. A missing `hidden_size`, `num_hidden_layers`,
  `num_local_experts`, `num_experts_per_tok`, `vocab_size` or `rope_theta` is named.
- `family_module("gptoss")` raises `NotImplementedError` whose message names the sink, the
  clamped SwiGLU and the extra biases; `"gptoss"` is in neither `FAMILIES` nor
  `DENSE_FAMILIES`.
- **The adapter selects the open engine (manual).** `GPT_OSS::load_model` calls
  `_shared_select_open_engine("OFLM_GPTOSS_ENGINE", "GPT-OSS")` like its twelve siblings,
  and `checkpoint()` / `restore()` go through the `causal_lm` base pointer. Until this
  landed, the class built `gpt_oss_npu` unconditionally, so an exported kernel set would
  have been silently ignored, and the `dynamic_cast<gpt_oss_npu*>` behind it returns null
  on the open engine with nothing checking it -- a null dereference on the first prompt.
  Verify with `oflm serve gpt-oss:20b`: with no kernel set installed and
  `OFLM_GPTOSS_ENGINE=open` the load fails with `OFLM_GPTOSS_ENGINE=open but no open
  kernels were found for GPT-OSS-20B-NPU2`, before any weights are read; with the variable
  unset it loads the closed DLL as before.

  **Result 2026-09-14:** both arms confirmed on the shipped container. Forced-open refuses
  by name; unset loads `gpt_oss_npu` and serves. No open kernel set exists for this family
  yet, so the open arm cannot go further than the refusal -- that is the point of wiring it
  now, since without it no future export is testable at all.
- `quant_map_from_chunk_sizes("gptoss", ...)` reads the tensor names `q4nx-build` writes
  (`configs/gpt-oss.json`) and refuses a role at two formats.
- **The real container derives `{"experts": "mxfp4"}`** (done 2026-09-14). The role table
  names the fused `ffn_gate_up_down_exps.weight` the converter actually writes, and
  `container_chunk_bytes` counts U8 as quantized alongside I8 - reading only I8 left the
  expert tensor out of the map entirely, which is how a container with 4-bit-float experts
  derived the everything-is-q4_1 default. Measured on the shipped 14.4 GB container: 97 I8
  projections and 24 U8 expert tensors, all at 2560 bytes, giving exactly that map.
- **2560 is ambiguous by byte count and is refused without dtypes.** GPT-OSS ships q4_1
  projections and MXFP4 experts at the same chunk size, so `CHUNK_FORMAT` deliberately does
  not contain 2560; `AMBIGUOUS_CHUNK` resolves it by dtype (I8 q4_1, U8 MXFP4) and raises
  naming both readings when the caller has no dtypes. Guessing here would silently describe
  a container that does not exist.
- **A tensor whose role the packer PLACES, at a chunk size the reader does not know, is
  refused by name.** It used to `continue`, leaving the role at the q4_1 default. Tensors
  with no placed role - norms, biases - are still ignored, or every model would stop
  loading. Shipped containers are unmoved: the all-q4_1 ones still derive `{}` and
  Ornith-1.5's four q8 roles are unchanged, so no `spec_hash` moves.

**The widths, before any of that.** Hidden 2880 is 45 bands of 64 and shares no factor above
1 with the q width's 64 bands or the kv width's 8, so `dense.cores_for` gives GPT-OSS 20B a
single core -- not the four Gemma 3 12B's 3840 gets, which measured nearly free, but an
eighth of the array. Padding hidden and the expert width to 3072 gives 8, and the same pad
makes the q4_1 chunk's 256 columns tile (2880 is 11.25 of them). See OPEN-WIDTH-PAD.

**The container, which was read after the paragraph above was written.** Every quantized
tensor ships in a 2560-byte chunk covering 32 rows by 128 columns, where every other family
is 5120 over 32 by 256; two adjacent chunks fuse into one pool chunk by byte copies alone.
The container pads K to 2944 and `o_proj`'s output to 3072 itself. The attention projections
and the `lm_head` are ordinary q4_1 -- their `I8` dtype is how every quantized tensor ships,
not a format claim -- while the experts are MXFP4, 4-bit float with a shared E8M0 exponent
per 32, in one fused `ffn_gate_up_down_exps.weight` per layer. `config.json`'s own
`modules_to_not_convert` names that split. The expert biases ship twice: as named tensors and
inside each chunk's padding at byte 128. The projections and the head are packed by
`std_fuse` (OPEN-PACK-CHUNK-FUSE), the experts decode by OPEN-QUANT-MXFP4 and their slab
order is settled by OPEN-PACK-EXPERT-ORDER; what the experts still have no pack op for is
the destination, which waits on the MoE block's layout (OPEN-MOE-WIDE-FF). `.claude/plans/gptoss-bringup.md` has the decoded layouts and the
ordered work.

**What a recipe would still need** (not requirements yet; each earns its own when it is
built): the padded widths above, the sink in the attention core (OPEN-ATTN-SINK), a
clamped-SwiGLU expert kernel and room for the three expert biases and the router's, a bias on
`o_proj` -- which OPEN-ATTN-QKV-BIAS explicitly excludes -- an MoE FFN on sliding-window
layers, which no recipe composes today, and YaRN position tables in the engine. The
arithmetic for all of it is settled and tested (OPEN-GPTOSS-FFN-REF); what is left is
kernels. `.claude/plans/gptoss-moe-and-biases.md` has the element and stream accounting.

### OPEN-GPTOSS-FFN-REF: the fp64 reference for GPT-OSS's experts, router and sink attention
**Applies to:** openflowlm-next (`open_kernels/model/replica_gptoss.py`)
**Test category:** unit (`tests/test_gptoss_ffn.py`; the transformers comparisons skip
without torch)

`replica_gptoss` is the oracle for the four things GPT-OSS does that no family in this tree
does, and it shall agree with transformers' own modules rather than with anything of ours --
the fp64 replica and the kernels read weights through the same dequantiser, so they can agree
perfectly while both being wrong, which is how OPEN-PACK-Q4-0 hid for two sessions.

**The clamped SwiGLU.** `clamped_swiglu(gate, up)` is `(up + 1) * gate * sigmoid(1.702 *
gate)` with `gate` clipped above at 7 and `up` clipped both ways. The asymmetry is
deliberate: a very negative gate still shuts the channel, where a symmetric clamp would floor
it at -7 and leak. `1.702` and `7.0` are family constants.

**The fused row order.** HF stores one `gate_up_proj` per expert with gate at the even output
rows and up at the odd. `split_gate_up` / `fuse_gate_up` are that rule, and they run down a
named axis so a packer holding `[experts, 2 * moe_intermediate, hidden]` can use them. A GGUF
source has the split done already -- `q4nx-build/configs/gpt-oss.json` maps `ffn_gate_exps`
and `ffn_up_exps` as separate tensors -- so which side does the de-interleave depends on the
source, and the reference works from the split pair either way.

**The biases.** `expert_ffn` carries one on each of gate, up and down; `route` carries one on
the router. `gptoss_layer_step` carries one on `o_proj` as well, which OPEN-ATTN-QKV-BIAS
explicitly excludes.

**Sink attention, windowed.** `sink_attention` is the GQA loop around
`replica_dense.sink_softmax`, cut to the window `window_start` gives. HF's sliding rule is
`kv_idx > q_idx - sliding_window`, so a 128-row window admits 128 rows including this token.

**Acceptance criteria:**
- `clamped_swiglu` equals `GptOssExperts._apply_gate` on the fused row order; the clamps are
  asymmetric; an `up` of -1 zeroes the channel and an `up` of 0 passes the gate through;
  `alpha` and `limit` equal transformers' own.
- `split_gate_up` takes the even entries as gate down any axis, and `fuse_gate_up` inverts it;
  a mismatched pair is refused.
- `expert_ffn` from a split pair equals one transformers expert computed from the fused
  tensor; each of the three biases moves the output.
- `route` picks the same experts and the same weights as `GptOssTopKRouter`, including the
  tie-break; softmaxing the top-k equals renormalising the full softmax, which is why the
  existing router core's shape still fits.
- `moe_block` equals `GptOssMLP`; forcing the expert choice keeps the reference's own
  weights and follows the forced order.
- `sink_attention` equals `eager_attention_forward` with the same sinks; a sink far below
  every score gives plain GQA attention; a positive sink scales every channel of the head by
  one factor strictly between 0 and 1; the window agrees with
  `sliding_window_causal_mask_function`.
- Six decode steps through `gptoss_layer_step` land within 1e-5 of the last token of
  `GptOssDecoderLayer`'s own sequence forward, on a sliding layer with YaRN cos/sin. Zeroing
  the sinks, the o_proj bias, the router bias or any expert bias moves it by more than 1e-3
  of the answer, so the tolerance discriminates. Giving `GptOssRMSNorm` an fp64 variance --
  it computes in fp32 whatever the parameter dtype -- closes the gap to 1e-12, which is what
  says the 1e-5 is transformers' rounding and not a missing piece.

### OPEN-WIDTH-PAD: the padded width a recipe may derive, and what it does not settle
**Applies to:** openflowlm-next (`open_kernels/recipes/qwen36moe.py`,
`open_kernels/recipes/pack.py`)
**Test category:** unit (`tests/test_width_pad.py`)

`pad_width(w, n_cores)` shall give the smallest width at or above `w` that a pool can
be built at: a multiple of `lcm(BAND_ROWS * n_cores, 256)`, which satisfies at once the
band law `dense.cores_for` applies and the 256-column k-tile both `q4_bytes` and the
pack index law count in. At a family's OWN core count it is always the identity, so no
shipped kernel set can move.

`BAND_ROWS * n_cores` alone is enough at eight cores and at four, where it is 512 and
256 -- already whole k-tiles -- and that is why one rounding was recorded as sufficient
for as long as every shipped family got four or more. At one and two cores it is 64 and
128, and the rounding lands short: 2880 stayed 2880 at one core and became **2944** at
two, which is the worst width available. `band_bytes(2944)` returns cleanly, and only
then do the two laws underneath disagree with it.

So the two laws refuse for themselves rather than trusting the width they were handed:

- `per_band(K)` shall raise on an odd chunk count. The GEMV runs `rs = 2` -- chunk `i`
  of a band covers row half `i % 2` and k-tile `i // 2` -- so a band is always an even
  number of chunks, and `per_band(2944)` returning 23 is a band no walk can consume.
  The check is inside `per_band`, not in a catalogue entry, so
  `OPEN_KERNELS_UNVALIDATED` cannot soften it.
- The pack index laws -- `pack.std_perm`, `pack.q8_perm`, and their C++ twins in
  `pools.cpp` -- shall raise when `in_dim` is not a whole number of 256-column k-tiles.
  `ncol = in_dim // 256` floors, so at 2944 `std_perm` returned an index array that
  ALIASES -- 1409 distinct file chunks selected for 1472 pool slots -- and nothing in
  `apply_op` or above it looked. A corrupt pool with nothing raised is the outcome this
  guard exists for. `q8_perm` floors identically and gets the same guard: leaving it out
  would keep the corrupt pool reachable through every q8 projection. The C++ interpreter
  gets it because the two must refuse the same things, not only produce the same bytes --
  a manifest carrying a bad width would otherwise pack silently there.

This exists for GPT-OSS-20B, whose hidden and expert widths are both 2880. The
earlier record said 2880 "gets one core of eight". That is true and it understates the
problem by one refusal and overstates what padding fixes by two. Padding to 3072
clears three blockers and leaves a fourth, and the tests name each:

- **Core count.** `cores_for` gives 2880 one core of eight; the padded width gives
  eight. Gemma 3 12B is the precedent for living with fewer: its 3840 also misses
  eight (it gets four) but everything still builds, and 8-vs-4 was measured as nearly
  free. 8-vs-1 has never been measured.
- **Chunk arithmetic.** `band_bytes(2880)` raises before any core count matters -- a
  64-row band of a 2880-wide matrix is not a whole number of 8192-value chunks. This
  is raised inside `q4_bytes`, so `OPEN_KERNELS_UNVALIDATED` cannot soften it. 3072
  is fine. **This was recorded as what stops a build first, and it is not**, because
  the container does not ship 2880 on a quantized axis: it pads K to 2944 itself, and
  `band_bytes(2944)` returns cleanly with `per_band` 23. 23 is odd, which breaks the
  `rs=2` band law, and `pack.std_perm` then floored `2944 // 256` to 11 and returned a
  non-injective index array -- 1409 distinct file chunks for 1472 pool slots, with no
  guard in `apply_op` or anywhere else. The container's own width gave a silently
  corrupt pool rather than a refusal, which makes this requirement's conclusion more
  important, not less. **Both guards landed 2026-09-15** and are stated above; the
  rounding is now an lcm so a low core count cannot produce 2944 in the first place.
- **The norm's validated widths.** 2880 is outside `ln`'s validated set
  {1024, 2048, 2560, 3072, 3840, 4096}; 3072 is in it, on the point Phi-4-mini
  already uses.
- **The MoE core scratch -- NOT fixed by any width.** The main core must hold xm's
  activation table and the expert h's at once, and `tab_bytes` is 2.25K, so at
  3072/3072 the two want 13824 bytes. The reservation they want it against is
  `tab_bytes(wide)`, and `wide` is the 4096 q projection only when `has_full` is set:
  GPT-OSS derives `dense`/`dense_local` layer types, so `has_full` is False and `wide`
  falls back to hidden, making the reservation 6912. Twice over, not the 1.5 times an
  earlier version of this paragraph recorded against 9216. `qwen36moe.common` refuses
  the padded spec either way. The 27B fits only because its expert width is 512
  against hidden 2048; GPT-OSS's expert width EQUALS its hidden, which is the deeper
  problem and breaks three more parts of the MoE block besides this one
  (OPEN-MOE-WIDE-FF, where they are enumerated and measured). **The scratch is no
  longer the FIRST refusal**: since 2026-09-15 the stripe assignment is checked before
  it, and that is the one a padded GPT-OSS hits.

**And padding the norm would be wrong even where it builds.** `designs/ln/ln.h`
divides the sum of squares by the width it was COMPILED at (`LN_N`), so a 3072-wide
norm over 2880 real channels and 192 zeros scales every residual by
sqrt(3072/2880) = 1.0328 -- 3.3% on every layer, a silent wrong answer, not a
rounding difference. So "pad the width and zero the tail" is not on its own a way to
build GPT-OSS: the norm needs either its own divisor knob or a validated `ln` point
at the model's own width.

**Acceptance criteria (unit):**
- `pad_width(2880, 8) == 3072`, and no width between 2880 and 3072 satisfies the band
  law, so 3072 really is the smallest.
- `pad_width(w, cores_for(spec)) == w` for the hidden, dense intermediate, q width and
  kv width of every spec in `recipes/specs/` -- the pad moves no shipped family.
- `cores_for` on gpt-oss-20b's own widths is 1, and 8 once hidden and the expert width
  are padded to 3072.
- `band_bytes(2880)` raises naming the chunk rule; `band_bytes(3072) == 122880`.
- `band_bytes(2944)` does NOT raise -- it returns 117760 -- and `per_band(2944)` does,
  naming the 23 chunks. `per_band(3072) == 24`, `per_band(2048) == 16`.
- `std_perm(1472, 2944)` raises naming 2944; `std_perm(1536, 3072)` returns 1536
  distinct indices. `q8_perm(2944, ...)` raises the same way and `q8_perm(3072, ...)`
  returns distinct (chunk, half) pairs.
- `pools.cpp` refuses `std_perm` and `q8_perm` at `in_dim` 2944, each naming the op and
  the byte count, and packs unchanged at a width that does tile (`pools_test.cpp`).
- `pad_width(2880, n) == 3072` for `n` in 1, 2, 4 and 8, and at each the result's
  `per_band` is even and its `std_perm` is injective over a full band pair.
- `ln` at width 2880 is refused by the catalogue and at 3072 is not.
- `qwen36moe.common` on the PADDED gpt-oss spec still raises, naming the stripe
  assignment; with an expert width that assignment accepts it raises again, naming the
  core scratch. The two are pinned separately so neither hides behind the other.
- sqrt(3072/2880) == 1.0328 to 5e-5, the factor a padded norm would put on every layer.

**Still needs the NPU:** whether 8 cores beats 1 by enough to justify 6.7% more weight
bytes. Nothing in this tree measures 8-vs-1.

**The chunk geometry is settled and was the open question here.**
`q4nx-build/configs/gpt-oss.json` is indeed the only config with `col_block_size` 128
rather than 256. That makes a 2560-byte chunk of 32 rows by 128 columns, and it is
q4_1 in the ordinary layout -- `d[128]` as bf16 at byte 0, `m[128]` at 256, 2048
nibble bytes at 512 -- so two adjacent chunks fuse into one 5120-byte pool chunk by
eight byte-slice copies with no arithmetic. **The packer reads it as of 2026-09-15**,
through its own op rather than through `std_perm`, because the file raster is a
supertile as well as half-width: OPEN-PACK-CHUNK-FUSE. The container also pads K to
2944 on its own, which is what makes `band_bytes(2880)` the wrong refusal to reason
about, and the pad from 2944 to the pool's 3072 falls out of the same op.

### OPEN-MOE-WIDE-FF: the MoE block when the expert width equals hidden
**Applies to:** openflowlm-next (`open_kernels/recipes/qwen36moe.py`,
`open_kernels/designs/layer_x/xcommon.py`)
**Test category:** unit (`tests/test_width_pad.py`) for the refusals and the budget;
`manual` for the rebuilt block, which needs the WSL toolchain and the NPU

Qwen3.6's expert intermediate is a QUARTER of its hidden -- 512 against 2048 -- and
`qwen36moe`'s hand-placed core layout assumes that ratio in more places than it states.
GPT-OSS's expert intermediate EQUALS its hidden, 2880 against 2880, and each of the
assumptions below shall be refused by name until the block is rebuilt, rather than
producing a zero divisor, an over-budget core or a plausible wrong layout.

**The stripe assignment, and it is the first refusal.** `moe_sequence` computes a core's
source offset as `(2 * spp * e + 2 * (c // cps)) * STRIPE + (c % cps) * PAIR`, with
`cps = n_cores // stripes_per_proj`. That splits ONE 128-row stripe across several cores,
which only works while the stripe count DIVIDES the core count. At ff == hidden == 3072
there are 24 stripes over 8 cores, `cps` is 0, and `c // cps` is a division by zero. The
generalisation each core owning `24 // 8 = 3` stripes is a different host sequence, not a
different constant.

**The core scratch.** `tab` is sized `tab_bytes(widest K)`, and the expert hidden's table
sits inside it past xm's. `wide` collapses to the hidden itself on a family with neither
linear-attention nor full-attention layers -- which GPT-OSS is -- so the reservation is
6912 while the two tables want 13824. Exactly twice over.

**The expert hidden's element count.** `moe_sequence` broadcasts `h` as ONE 4096-byte act
element. At ff 3072 it is 12288 bytes, exactly three. The drain side already scales (each
core drains `HID_PC * 4`); only the broadcast fill assumes one.

**The prep kernel's K.** `gemv_q4_prep_f32` builds h's table at the expert width, and the
catalogue's validated `moe` point is `ff: 512`. K = 3072 is a new point and needs its own
compare run.

**The L1 budget, which did not exist for this tail.** Only the DENSE tail computed a main
core's L1; the MoE tail took `PER_CALL = 2` on trust, and a layout that overflowed would
have been found by aiecc, which does not print the shortfall. A main core holds exactly
six things and a stack -- the table, `ms`, `ds`, then depth-2 fifos for the weight, x and
y elements -- so the budget is arithmetic and `core_l1` is now the one place both tails
compute it.

**Measured 2026-09-15, which is what turns this from a wall into a sizing change.** At
the padded 3072/3072 a main core comes to **56,960 bytes of the 61,440 budget, 4,480 to
spare** -- against the shipped 27B's 53,376. But that margin exists only because GPT-OSS
has no linear-attention layers: `DS_FLOATS` is a hard 1280 on the MoE path regardless, and
keeping that 5,120-byte DeltaNet scratch on a core whose kernels never touch it puts the
total at 62,080, **640 bytes over**. So the block fits, and one of the things that makes
it fit is dropping a buffer the family does not use. That is a sizing decision, not a
budget to be discovered during a build.

**Acceptance criteria:**
- `common()` on the padded GPT-OSS spec raises naming the stripes per projection and the
  core count; the 24-over-8 count and `8 // 24 == 0` are asserted, so the message is
  about a zero divisor rather than a tight fit.
- With an expert width the stripe assignment accepts, `common()` raises again naming the
  core scratch, and `tab_bytes(3072) * 2 == 13824`. The two refusals are pinned
  separately so neither hides behind the other.
- `core_l1` on the shipped 27B is 53,376 and is inside `L1_BUDGET`; a spec with twice the
  27B's attention heads is refused naming the budget and the overshoot.
- Both tails compute L1 through `core_l1`; the dense tail's `per_call` result is unchanged
  for every spec in `recipes/specs/`.

**Verification (manual), once the block is rebuilt.** The four assumptions above become
four new catalogue points (`moe` at `ff` 3072 and `hidden` 3072, `gemv_q4` prep at
K 3072), each needing a compare run against `replica_gptoss.moe_block` the way
OPEN-FAMILY-QWEN36MOE's does. Until then this requirement is the refusals and the budget,
which is what a recipe meets first.

### OPEN-EMBED-STRIDE: an embedding row is read at the container's width, not the buffer's
**Applies to:** openflowlm-next (`src/open_qwen36/q4nx_file.cpp`, `core.cpp`,
`open_kernels/model/q4nx.py`)
**Test category:** unit (`src/open_qwen36/pools_test.cpp`, `tests/test_q4nx_embed.py`)

`Q4nxFile::bf16_row` shall take the source row stride from the tensor's own trailing shape
dimension and zero-extend the row to the width the caller asked for, refusing a width
NARROWER than the container's row.

`Core::step_impl` passes the manifest's `hidden` as both the destination width and the
source stride. That is the same number for every family shipped today. It stops being the
same number the moment a recipe derives a padded buffer width (OPEN-WIDTH-PAD puts
GPT-OSS's 2880 at 3072 against a `[201088, 2880]` embedding), and then row `n` is read
`n * (3072 - 2880) * 2` bytes too far. The bounds check was computed at the padded stride
as well, so it did not fire until row 188,519: **94% of the vocabulary came back silently
wrong and the top 6% threw.** The final norm's own size check catches its case loudly; the
embedding had nothing.

**Why the container and not a manifest field.** The obvious fix is to carry the padded
buffer width and the model's own width as two manifest fields and hand `bf16_row` both.
The container already knows its row width, in the header it was read from, and a second
copy in the manifest is a second thing that can disagree with it. Taking the stride from
the tensor cannot be got wrong by a future recipe, and needs no manifest change, so no
shipped manifest or fixture moves.

**The Python replica deliberately does not zero-extend.** `Q4NX.embed` keeps refusing a
`hidden` that is not the table's row width. The engine zero-extends because its residual
buffer is padded; the fp64 replica computes at the model's own width throughout, so a
wider `hidden` there is a caller confusing the two widths rather than a pad. The two never
disagree about a value: both read row `n` at `n * row_width * 2`, and the engine's extra
entries are zeros against weights zero-padded at the same width. `test_q4nx_embed.py` pins
the asymmetry so it is not closed by hand.

**Acceptance criteria:**
- A synthetic `[7, 2880]` BF16 container read into a 3072-wide buffer returns row 5's own
  bytes in `[0, 2880)` and zeros in `[2880, 3072)`. Reading at the padded stride would
  return different values for most columns, and the test asserts that too, so the fixture
  discriminates rather than merely agreeing.
- Row 0 is unchanged by the fix (the offset is `row * stride`, so only rows above 0 moved)
  -- which is why this was invisible to a single-token smoke test.
- The last real row is readable into a padded buffer. Computed at the padded stride the
  bounds check refused it; that is the loud half of the same bug.
- A destination narrower than the container's row is refused, the message naming the
  container's width.
- A row past the end is refused, counted at the container's stride.
- A caller whose width equals the container's reads exactly as before -- the shipped path
  does not move.
- `Q4NX.embed` refuses a `hidden` wider than the table's row, naming the width and the
  stride.

### OPEN-ATTN-SINK: a learned per-head attention sink logit
**Applies to:** openflowlm-next (`open_kernels/designs/attn/attn.h`,
`model/replica_dense.py`, `model/replica_gptoss.py`)
**Test category:** unit (`tests/test_attn_sink.py`, `tests/test_gptoss_ffn.py`) for the math
and the guard; manual (the procedure below, needs the NPU and a GPT-OSS container) for the
numbers

A family with attention sinks carries one learned scalar per head per layer that joins the
softmax denominator and has no value vector behind it, so the head's output weights sum to
less than one and it can decline to attend. Given a head's scores `s_t` (already divided by
`sqrt(head_dim)`) and its sink `c`:

    o_h = sum_t exp(s_t - M) V_t / ( exp(c - M) + sum_t exp(s_t - M) ),  M = max(c, max_t s_t)

The sink is the raw stored scalar: GPT-OSS scales the q.k dots and concatenates the sink
after that, so it is NOT divided by `sqrt(head_dim)`. `replica_dense.sink_softmax` is the
fp64 reference.

That is one more row of an online softmax whose score is `c` and whose V is zero, so on this
codebase's attention core the whole change is the state `attn_init_impl` starts from --
`m = c, l = 1` instead of `m = -1e30, l = 0`, with `oacc` still zero -- and the per-row
kernels, the block kernel and `attn_fin_impl` are untouched. `attn_fin_impl` already divides
by `ml[kMLS + h]`, which now carries the sink's 1.

The sinks shall ride in the meta element after `qn | kn`, as bf16, NOT on a fifo of their
own. They are one value per head for the whole layer -- 128 bytes at 64 heads -- where the
q/k/v bias is one per channel and has to arrive in lockstep with the projection stream; a
second fifo would cost two buffers and a descriptor per layer for data that fits in the meta
element's unused room. `attn_meta_impl` widens this core's own heads into an `sk[NHL]` f32
buffer, so `attn_init_impl` indexes it locally and needs no head offset; only `attn_meta` takes
`h0`, and only under this guard.

A family without sinks shall compile the attention it compiled before, byte for byte:
`ATTN_SINK` defaults to 0 and `ATTN_SINK_IN_PARM` / `ATTN_SINK_OUT_PARM` / `ATTN_SINK_ARG` /
`ATTN_SINK_H0_PARM` expand to nothing, the same trick `ATTN_H0_PARM` and `ATTN_BIAS_PARM`
use -- an unused parameter changes the generated code.

**Acceptance criteria (unit):**
- A sink equal to the scores adds exactly one more equal share to the denominator: two
  positions at score 0 with a sink at 0 give 1/3 each, and a sink at `ln 2` gives 1/4 each.
- The reference equals GPT-OSS's own expression -- append the sink as one more column,
  subtract the row max, softmax, drop the column -- and equals transformers'
  `eager_attention_forward` on a GQA case (skipped where torch is absent).
- A sink far below every score reproduces plain softmax exactly.
- The weights sum to `1 - exp(c - M)/Z`, strictly between 0 and 1, and their ratios are the
  ones plain softmax gives: the sink rescales and contributes no value.
- The online form seeded `m = c, l = 1` equals the closed form over 512 positions with an
  8-sigma score spread, and over a context where the sink holds the max at every row.
- `ATTN_SINK` defaults to 0 in `attn.h`, no recipe emits it, and no recipe geometry carries a
  SINK field.
- `4 * HD + 2 * NH <= E_A` decides whether the sinks fit the meta element, equivalently
  `NH <= HD * (KVH - 2)`; GPT-OSS 20B's `(64, 8, 64)` gives 384 bytes of a 1024-byte element,
  and `attn.h` asserts the same inequality at compile time.
- `replica_gptoss.sink_attention` puts the GQA loop around it and equals
  `eager_attention_forward` given the same sinks; cutting it to a sliding window equals
  running it over the window's rows alone (OPEN-GPTOSS-FFN-REF).

**Procedure (manual) -- step 1 DONE, steps 2-4 outstanding:**
1. Compile `designs/attn/*.cc` for a shipped family's flags from the tree before and after
   and diff the objects: every one must be byte-identical, as OPEN-ATTN-QKV-BIAS step 1 did.

   **Result 2026-09-13: passes.** Every `designs/attn/*.cc` compiled at `e0511bd7^` and at
   `HEAD` under all 11 shipped families' real flag sets, plus two extra sets forcing
   `ATTN_RB` 2 and 4 so `attn_stepb` is covered: **127 objects byte-identical, 0 differing**,
   and 3 pairs that fail to compile on BOTH sides because `attn_stepb.cc:22` raises a
   deliberate `#error` when `ATTN_RB` is unset -- symmetric, so not a difference. Run twice
   on separate trees with the same result. Peano is in WSL at `~/ironenv142`; the earlier
   "nothing here has been near a compiler" note was true when written and is not now.
   The diff shows the rebuild lands on the identical object, NOT that no rebuild happens:
   `recipes/cache.py` hashes the source bytes, so every family's build key did move.
2. Give the recipe a `SINK` knob and a `CD_SINK` consts slot of `NH * 2` bytes (`-1` for a
   family without one, as `CD_QB` does), widen the meta fill from `4 * HD` to
   `4 * HD + 2 * NH` bytes, and pack `self_attn.sinks.weight` into it.
3. Export with `OPEN_KERNELS_UNVALIDATED=1`, pack, and run `model/make_decode.py` +
   `compare_decode.py` at positions 0 and a few hundred against a replica that calls
   `sink_softmax`: logits correlation > 0.9999 and the same argmax.
4. A sink whose weight is zero must reproduce the no-sink answer to the bit, which separates
   a wrong sink from a wrong anything-else.
5. `oflm-test --llm` through `flm serve`, then add the tuple to `catalogue.py`.

### OPEN-VISION-VIT-REF: the vision tower, reference and host port
**Applies to:** openflowlm-next (`open_kernels/model/replica_vit.py`, `replica_deepstack.py`, `src/open_qwen36/vision/`)
**Test category:** unit (`tests/test_vision_vit.py`, needing the container and torch and skipping without them; `tests/test_vision_deepstack.py`, needing only torch); the C++ port is checked by `vit_test.exe` (procedure below)

The shipped `vision_weight.q4nx` -- every linear pre-tiled for the closed
engine's `vision_mm` as `[n/64][k/256][64][256]` bf16, zero-padded -- shall be
un-tiled and run as Qwen3-VL's vision tower: patch embed + bilinearly
interpolated positions, 2-D RoPE attention over the whole image, GELU-tanh
MLP, the 2x2 merger. The numpy forward matches transformers'
`Qwen3VLVisionModel` loaded with the same weights; the host C++ port matches the
numpy forward. The geometry is read as OPEN-VISION-VIT-CONFIG says.

**Acceptance criteria:**
- numpy vs transformers on a random 8 x 8 (unit) / 16 x 16 grid: corr > 0.99999, max error < 1e-3 of max.
- The patch order is merge-block-major; a 48 x 48 grid samples the position table exactly.
- `vit_test.exe <model_dir> <fixture>`: corr > 0.99999, max error < 1e-3 of max against `replica_vit.py --fixture`.

**Result 2026-09-08 (35B tower, 27 blocks):** numpy vs transformers corr
1.00000000, rel 8.7e-6; C++ vs numpy corr 1.00000000, rel 4.0e-6, 16 x 16
patches in 1.02 s (numpy 11.5 s).

**Deepstack.** The forward above is the tower with `deepstack_visual_indexes`
empty, which is the 35B. Qwen3-VL uses deepstack: extra mergers hang off the
blocks that list names (5, 11 and 17 on the 4B), and each one's output is a
second set of image rows for the decoder to absorb, not part of the tower's
output. `replica_deepstack.py` is their fp64 reference. Two things about them
differ from the tower's own merger and shall be reproduced, not approximated:

- A deepstack merger normalises **after** the merge shuffle
  (`use_postshuffle_norm=True`): it reshapes `[n, hidden]` to
  `[n / merge^2, hidden * merge^2]` and runs one LayerNorm across the whole
  merged row. The tower's merger normalises each patch across `hidden` first.
  The two are not interchangeable -- their LayerNorms are different widths.
- An index names the block a feature is taken **after**. Its activation is
  `nn.GELU()`, the exact one, as the tower merger's is.

**Acceptance criteria (deepstack):**
- numpy vs transformers on a random 8 x 12 grid, three taps: the merged output and every deepstack feature at corr > 0.99999, max error < 1e-3 of max.
- Taking each feature one block early gives corr < 0.99 against the same oracle, so the tap position cannot drift unnoticed.
- A deepstack merger's LayerNorm is `hidden * merge^2` wide and the tower merger's is `hidden` wide, read off transformers' own `state_dict`; running either through the other's path raises rather than returning plausible numbers.
- Every merger's output is `out_hidden_size` wide -- the decoder's hidden size, not the tower's.

**Result 2026-09-13 (random 6-block tower, taps at 1, 3, 5, 8 x 12 grid):**
numpy vs transformers corr 1.00000000 on the merged output and all three
features, rel 2.6e-6 to 4.3e-6; the tap-one-block-early control gives at worst
0.9092. No container was involved and none is needed. The C++ port is designed
and not written: `.claude/plans/qwen3vl-deepstack.md`.

### OPEN-VISION-VIT-CONFIG: where the tower's geometry comes from, and which towers are refused
**Applies to:** openflowlm-next (`open_kernels/model/replica_vit.py`, `src/open_qwen36/vision/vit.cpp`)
**Test category:** unit (`tests/test_vision_config.py` for the reference and `tests/test_vision_deepstack.py` for what the weight file can supply instead; `vit_test --configs`, run by `ctest -R OPEN-VISION-VIT-CONFIG`, for the C++ port -- none of them needs a container)

The tower's numbers come from `config.json`'s `vision_config`, in whichever of three
shapes the container carries: OFLM's per-family prefixes `QWEN3_6_MOE_*` and `QWEN3_5_*`,
or the plain transformers keys (`depth`, `hidden_size`, `num_heads`, `intermediate_size`,
`out_hidden_size`, `patch_size`, `temporal_patch_size`, `spatial_merge_size`,
`num_position_embeddings`), which is what a container that kept its source block ships --
Qwen2.5-VL's does. The plain shape has no head dimension and no epsilon: `head_dim` is
`hidden_size / num_heads` and the epsilon is LayerNorm's 1e-6, which is what transformers
uses for this tower.

A `vision_config` describing a tower this code does not implement shall be refused with
the reason named, not run with the extra parts dropped -- dropping one gives image
embeddings that look plausible and are wrong. Refused: a non-empty
`deepstack_visual_indexes` (Qwen3-VL feeds three vision layers through extra mergers into
the first decoder layers), `window_size` / `fullatt_block_indexes` (Qwen2.5-VL's windowed
tower), and a `hidden_act` other than `gelu_pytorch_tanh`.

A container with no `vision_config` at all shall say so and name every key set it looked
for. **Qwen3-VL-4B-Instruct-NPU2 is such a container**: its closed `qwen3vl_npu` engine
hardcodes the tower's numbers in C++ (`src/include/models/qwen3vl/qwen3vl_npu.hpp`, and
the DLL contains none of the `*_VISION_*` key strings the other two do), so the shipped
config.json carries the vision weights' file name and nothing about their shape.

Note what that header does and does not hold: it fixes the *preprocessing* --
`QWEN3_PATCH_SIZE` 16, `QWEN3_TEMPORAL_PATCH_SIZE` 2, the merge sizes, the
rescale mean and standard deviation, the edge limits -- and says nothing about
depth, hidden size, head count or MLP width. Those the closed engine gets from
the weight file and its xclbins.

So the open tower has two possible sources and needs both. `vision_weight.q4nx`'s
tensor shapes give back depth, hidden size, MLP width, output width, position
count, merge factor and channel count, plus how many deepstack mergers there
are (`replica_deepstack.geometry_from_tensors`). Two numbers are not in the
weights at any tiling:

- **the head count** -- the qkv projection is `[3 * hidden, hidden]` for every
  split, so a tower with twice the heads has byte-identical tensor shapes;
- **the deepstack indexes** -- the merger names say there are three, never which
  blocks they hang off.

Those shall come from a `vision_config`, which means `q4nx-build` writing one
into the containers it converts (the source block carries all of it, and
`inject_oflm_keys` now keeps it). For a container that already shipped without
one, the refusal stands: a guessed head count gives image embeddings that look
plausible and are wrong, which is the failure this requirement exists to
prevent. Reading a tiled linear back gives a bound, not a width -- the 35B's
4304-wide MLP reads as 4352 -- so the derivation reports whether its numbers are
exact.

**Acceptance criteria:**
- `geometry_from_tensors` over transformers' own `state_dict` shapes recovers depth, hidden, MLP width, output width, position count, merge factor and channel count, and the deepstack merger count.
- It returns no head count and no deepstack indexes, and two towers differing only in those have identical tensor shapes.
- Over the closed engine's `[n/64][k/256][64][256]` tiling it reports its widths as inexact.
- The 35B container's `QWEN3_6_MOE_*` block reads as 27 x 1152, 16 heads x 72, MLP 4304 -> 2048, patch 16, 2304 positions; a Qwen3.5 container's `QWEN3_5_*` block reads with `hidden == heads * head_dim`.
- Qwen3.5-0.8B's HF config and its shipped container give the same tower (12 x 768, 12 heads x 64, MLP 3072 -> 1024), the HF one deriving `head_dim` and using eps 1e-6.
- Qwen3-VL-4B-Instruct-NPU2's config.json is refused with a message naming `vision_config`.
- Qwen/Qwen3-VL-4B-Instruct's config.json is refused with a message naming `deepstack`.
- Qwen2.5-VL-3B-Instruct-NPU2's config.json is refused with a message naming `window`.

### OPEN-VISION-VIT-WINDOWED: Qwen2.5-VL's tower attends inside windows
**Applies to:** openflowlm-next (`open_kernels/model/replica_vit_qwen25.py`, `src/open_qwen36/vision/`)
**Test category:** unit (`tests/test_vision_vit_windowed.py`; the oracle needs torch and skips without it, the window math and the container arithmetic do not. The C++ window permutation is `vit_test --window-index`, run by `ctest -R OPEN-VISION-VIT-WINDOWED`, which needs neither torch nor a container; the C++ forward is checked numerically by `vit_test --windowed`, procedure below)

Qwen2.5-VL's vision tower shall be reproduced as a numpy fp32 forward and as a host C++
port beside the full-attention tower OPEN-VISION-VIT-REF covers. It is not
the tower OPEN-VISION-VIT-REF covers: most blocks attend only within a square
window of `window_size / spatial_merge_size / patch_size` merge units, the
blocks named by `fullatt_block_indexes` attend over the whole image, and the
tower permutes its merge units so that windows are contiguous, runs the entire
stack in that order, and restores the original row order after the merger. The
rotary table is permuted with the tokens. The blocks use RMSNorm, a SwiGLU MLP
with biases, and split q/k/v/o projections; there is no learned position table
and the patch embed has no bias.

The reference is checked against transformers' own
`Qwen2_5_VisionTransformerPretrainedModel`, which builds the weights that the
reference then reads -- nothing in this repo sits on both sides. The oracle's
tower is initialised at `initializer_range = 0.4`, because at HF's default the
residual stream dominates the output and a forward with the windows wrong still
agrees to 2e-4; a test asserts that the windows-ignored forward fails, so that
sensitivity cannot regress unnoticed.

The C++ port reads the tower's geometry through `VitConfig::qwen25_from_config_text`, which
is separate from the full-attention reader OPEN-VISION-VIT-CONFIG describes -- that one
still refuses a windowed `vision_config`, because the tower it configures cannot run one.
It is checked against `replica_vit_qwen25.py --fixture`, which writes a synthetic
`vision_weights.q4nx` in the shipped layout (the converter's names, both dims of every
vision_mm matrix padded to 256 and tiled) from weights transformers built, so the port runs
end to end with no container on the box. That checks the forward, not the names: only
opening a shipped container can do that.

**The container holds one tensor the tower does not read.** Its published size did not
reconcile: with transformers' geometry and the tiling every other shipped container uses,
`vision_weights.q4nx` should be 1,377,729,112 bytes and it is 1,430,158,096. The
difference, 52,428,984, is exactly 1600 vision_mm tiles (50 MiB, one more 5120 x 5120 bf16
matrix) plus 184 bytes of header, and could not be padding, because a tiled weight grows
only by whole 32,768-byte tiles. Reading the header settled it: the file holds **519**
tensors, the 518 this tower reads plus one named `identity`, a 5120 x 5120 bf16 identity
matrix stored tiled like any weight -- the same shape as `merger.mlp.0.weight`. The closed
engine presumably multiplies by it to move data through `vision_mm` where the arithmetic is
a copy. Both loaders skip it and both refuse any other unaccounted tensor, because a
missing piece reads as plausible numbers rather than as an error.

**Acceptance criteria:**
- numpy vs transformers on a random 12 x 10 grid: corr > 0.99999, max error < 1e-4 of max, both with four windowed blocks and with every block full-attention.
- A forward in which every block attends over the whole image gives corr < 0.9 against the same oracle.
- `window_index(12, 10, merge=2, window=112, patch=14)` is the hand-derived permutation with segment boundaries `[0, 64, 80, 112, 120]`, in the numpy reference and in the C++ port.
- A grid that divides the window size evenly (8 x 8 patches) still pads a whole empty window, which collapses: index `0..15`, boundaries `[0, 64]`.
- The merger's activation is the exact GELU, `x * Phi(x)` against the standard normal CDF, not the tanh approximation (which the oracle comparison cannot distinguish).
- The container size model reproduces Gemma3-4B's, Qwen3.5-0.8B/9B's and the 35B's `vision_weight.q4nx` exactly, accounts for Qwen2.5-3B's `model.q4nx` data to the byte, and reproduces Qwen2.5-VL-3B's `vision_weights.q4nx` exactly at 1,430,158,096 once `identity` is counted.
- Every one of the 519 names, dtypes and shapes the model predicts is in the shipped file's own header, and `identity` is the only tensor outside the `model.visual.` prefix.
- `load_weights` never reads `identity`, and a container with any other tensor count is refused by name.
- `vit_test --windowed <fixture> <fixture>`: corr > 0.99999, max error < 1e-3 of max against `replica_vit_qwen25.py --fixture`.

**Result 2026-09-13:** the C++ port lands. `vit_test --windowed` against the numpy
reference on 12 x 10, 16 x 24, 6 x 18 and 8 x 8 patch grids: corr 1.00000000, rel 5.0e-6 to
8.8e-6, the reference itself corr 1.0000000 rel 6.0e-6 against transformers on the same
towers. `ctest -R OPEN-VISION-VIT-WINDOWED` (the permutation, no container, no torch)
passes. Suite 452 passed / 1 skipped.

**Result 2026-09-13 (host arithmetic): the tower is 3.8x faster and slightly more
accurate.** The ~35 s a 40 x 56 grid took was never an NPU problem. `linear()` and
`attention()` in `vision/vit.cpp` had three defects between them, all host-side:

- `linear()` swept every activation row once per 8-wide output block, so the whole
  activation matrix -- 11 MB at that grid -- came back out of L3 160 times per
  projection. It now runs a 128-row panel against all output blocks, which is under
  a megabyte and stays in L2.
- `attention()` accumulated each QK dot into ONE float, a serial dependency chain at
  three or four cycles a multiply where `linear()` had eight independent ones.
- `attention()` read V straight out of the interleaved qkv buffer at a stride of
  3 x hidden floats -- 15 KB -- and the AV loop walks the whole segment once per
  query, so a full-attention block re-read it n times with nothing able to prefetch.
  q and k were already being re-laid contiguous for exactly this reason; v had been
  left behind. It is re-laid now too.

All three sites also take an AVX2 + FMA path chosen at RUNTIME. The binary's baseline
ISA is unchanged: raising one translation unit's `/arch:` is the ODR hazard
`src/CMakeLists.txt` records for `npue_embedding.cpp`, and `vit.cpp` shares inline
headers with `pools.cpp` and `core.cpp`. A machine without AVX2 runs the scalar loops.

Measured with `vit_test --windowed` on the shipped Qwen2.5-VL-3B container against the
numpy reference, same binary options, same fixture:

| patch grid | before | after | correlation | rel error before / after |
|---|---|---|---|---|
| 24 x 32 (192 rows) | 12.15 s | 3.23 s | 1.00000000 | 5.34e-05 / 1.94e-05 |
| 40 x 56 (560 rows) | 37.09 s | 9.63 s | 1.00000000 | 8.19e-06 after (1.65e-05 before) |

The accuracy improves because multiple accumulators round better than one serial
chain, so the gate this requirement already sets is met more comfortably, not less.
`ctest` in `src/open_qwen36/build_cli` stays 3 of 3.

The tensor names are now confirmed from outside this repo: the closed
`src/lib/hrx/libqwen2vl_npu.so` contains `model.visual.`, the four `attn.*_proj`, the three
`mlp.*_proj`, `rmsnorm1` / `rmsnorm2`, `merger.ln_q` / `merger.mlp.0` / `merger.mlp.2` and
`patch_embed.proj.weight`, with no `blocks.` segment and no other vision tensor name. One
shape was wrong and is fixed: `merger.ln_q` is `[hidden]`, since it normalises before the
2 x 2 concat.

**Result 2026-09-13, on the shipped container.** It was pulled and opened. Every predicted
name, dtype and shape matches the file's own header, all 519 of them, and the size model
is exact. Two things only a real container could show, both now fixed: the 50 MiB is the
`identity` tensor above, and `replica_vit_qwen25.vision_config` was reading OFLM's prefixed
key names (`*_VISION_NUM_LAYERS`) where this family's container keeps the plain transformers
block (`depth`, `hidden_size`, `window_size`, `fullatt_block_indexes`, ...) -- the C++
reader had it right and the numpy one did not. With the real weights loaded, the numpy
tower agrees with transformers' `Qwen2_5_VisionTransformerPretrainedModel` at **corr
1.00000000** (rel 1.7e-6 to 4.4e-5) on 12 x 10, 8 x 8 and 6 x 18 grids, the oracle holding
this container's own weights rather than random ones; and the C++ port agrees with the numpy
tower on the same container at **corr 1.00000000**, rel 5.9e-6. So the chain from
transformers to the shipped bytes to the host port is closed.
`.claude/plans/qwen25vl-container-size.md`, `.claude/plans/qwen25vl-windowed-vit.md`.

### OPEN-VISION-EMBED: the open engine takes an image payload
**Applies to:** openflowlm-next (`src/open_qwen36/engine.cpp`, `core.cpp`, `pools.cpp`, `src/common/AutoModel/modeling_qwen3_6_moe*.cpp`, `modeling_qwen3_5vl*.cpp`)
**Test category:** e2e (`utilities/flm-test --vision --model <vlm>` through `flm serve` with the open engine)

`Engine::prefill(ids, payload)` with an image payload shall run the vision
tower on each image and step each merged patch's row through the model as a
hidden vector (`Core::step_embed`) at its M-RoPE position (t, h, w) = (c, c +
row, c + col), text tokens after an image continuing from the same counter
(`rope_parameters.mrope_section`, interleaved), later prefill chunks and
generated tokens inheriting it; a request without images is the unchanged
text path. The model classes read their preprocessing constants from
`config.json` and no longer require the closed engine for images.

**Deepstack.** A tower with `deepstack_visual_indexes` produces one extra set of
image rows per index, and those are added into the decoder's residual stream
rather than stepped through it. Feature `j` shall be added **after** decoder
layer `j` has run -- transformers gates this as
`layer_idx in range(len(deepstack_visual_embeds))` and applies it to
`hidden_states` after the layer, so three features cover layers 0, 1 and 2. The
add touches the image tokens' rows only, in prompt order, one feature row per
image token; text rows are untouched, which is what keeps a text-only request
after an image request unchanged.

Folding feature 0 into the input embedding instead is a different computation
and shall not be done: it would pass the feature through layer 0's attention
and MLP before the residual stream ever sees it.

This is host-side work for the first feature only. Features 1 and 2 land between
decoder layers, and the dense layer program has no input for a per-row addend,
so the open engine cannot run a deepstack model on the NPU without either a new
kernel input or a break in the layer loop. `.claude/plans/qwen3vl-deepstack.md`
weighs the two.

**Acceptance criteria (deepstack, unit -- `tests/test_vision_deepstack.py`):**
- `deepstack_layer_map(3) == [0, 1, 2]`.
- The injection leaves every non-image row bit-identical and does not mutate its input.
- A mask whose True count differs from the feature's row count is refused by name, as is a feature whose width is not the decoder's hidden size.

**Acceptance criteria (e2e):**
- `flm-test --vision --model qwen3.6-moe:35b` (and a Qwen3.5 VL size) passes on the open engine with the answer on the fixed test image matching the closed engine's.
- A text-only request after an image request answers as before (the position records are restored on `clear_context`).
- Qwen2.5-VL-3B answers correctly about a decodable image and about that image in a following text-only turn.

**Result 2026-09-13 (Qwen2.5-VL-3B, the windowed tower): PASS.** `oflm serve
qwen2.5vl-it:3b` loads on the open kernels -- 36 pools resident, the tower in 1.4 s -- and a
689 x 480 JPEG becomes a 40 x 56 patch grid, 560 tokens, in about 35 s on the host CPU.
Asked what is in it, the model answers "The image shows a seagull standing on top of a
lamppost", which is what the photograph shows; a text-only follow-up in the same
conversation answers "Blue" to a question about the sky, so the image rows and the M-RoPE
counter carry across the turn as this requirement says they must. The decoder runs on a
kernel set built for Qwen2.5-3B-Instruct (OPEN-FAMILY-QWEN25VL), and its logits match the
fp64 reference at corr 0.99999125 and 0.99999362 on the two scored positions, argmax and
top-5 identical.

Two failures in `oflm-test --vision` were NOT the engine's, and the closed engine was run on
the same box to say so: it fails all three rounds with a bad allocation while applying the
chat template and returns nothing, where the open engine answered the first. The suite's
other two images never reached either engine, because they are PNG and the reader could not
decode one.

> **Fixed 2026-09-13 (OPEN-VISION-IMAGE-READ).** The diagnosis above is half right and the
> half it gets wrong changes whose problem it is. It is not what OFLM ships: upstream's
> `avcodec-61.dll` decodes PNG and a copy is checked in at `src/lib/`. It is what we BUILD
> against -- vcpkg's ffmpeg leaves `zlib` out of its default features, so the `avcodec-63.dll`
> beside `src/build/oflm.exe` is configured `--disable-zlib` and has no png decoder or
> encoder at all. That is every fresh Windows clone, not this machine. The reader now decodes
> PNG itself and `oflm-test --vision` passes all three rounds; see that requirement.

**Result 2026-09-08:** runs end to end through `flm serve` on Qwen3.5-0.8B and
on the 35B (tower resident in 2.2 s, a 30 x 44-patch image -> 330 tokens in
14.8 s on the CPU, prefill 516 tokens, the answer describes the image
correctly, follow-up turns continue from the same (t, h, w) counter). The
suite's other two images are dropped by the app's own reader before either
engine (fixed 2026-09-13, OPEN-VISION-IMAGE-READ). **The closed-engine comparison did not run**: the closed 1.0.4 DLL
segfaults on the local 1.0.2 / 0.9.45 containers (it expects the Q4_K branch),
so on this box only the open engine can serve these files. Log:
`.claude/plans/issue-16-hw-results.md`.

### OPEN-VISION-IMAGE-READ: a PNG decodes whatever FFmpeg is linked
**Applies to:** openflowlm-next (`src/common/image/png_decode.cpp`, `image_reader.cpp`)
**Test category:** unit (`src/common/image/png_decode_test.cpp`, `ctest -R OPEN-VISION-IMAGE-READ`
in `src/build`); the wiring into the reader is `e2e`, procedure below

The image reader shall decode PNG without depending on the linked FFmpeg having a
png decoder. `image_png::decode_rgb24` takes the file bytes and returns tightly
packed RGB24 -- colour types 0, 2, 3, 4 and 6, bit depths 1, 2, 4, 8 and 16, all
five row filters, all three deflate block types -- and refuses by name anything it
does not implement, interlaced (Adam7) files in particular, at which point the
reader still offers the file to FFmpeg. Alpha is dropped rather than composited
and 16-bit samples are truncated to their high byte, which is what the FFmpeg path
does through `sws_scale`.

It carries its own inflate. Linking zlib would give `oflm.exe` a load-time import
the Windows installers do not ship -- `src/inno/oflm.iss` and `src/wix/oflm.wxs`
enumerate DLLs by name and list `zlib1.dll`, while vcpkg's zlib is `z.dll` -- so a
decoder whose whole point is "works whatever is linked" would have added a link
dependency to get there.

**Why this is a requirement and not a build fix.** vcpkg's ffmpeg port has a
`zlib` feature and it is not a default one, so the avcodec a fresh Windows clone
links is configured `--disable-zlib` and has no png decoder. Upstream's shipped
`avcodec-61.dll` does have one. The failure was invisible because a PNG that fails
to decode is SKIPPED, not refused: `modeling_*.cpp` logs a line and prefills the
remaining images, so the model answers confidently about images it never saw and a
test can pass on one it imagined. Two of the three images `oflm-test --vision`
sends are PNG.

**Acceptance criteria (unit):**
- Every generated fixture in `specs/open-engine/tests/fixtures/png/` decodes to its
  `.rgb` byte for byte. The fixtures are written by `make_png_fixtures.py`, which
  computes the expectation from the source pixels it encoded -- not from this
  decoder -- and cross-checks every case against Pillow, an independent decoder.
  Pillow agrees on all of them but 16-bit grayscale, where its own I;16 conversion
  saturates at 255 instead of scaling.
- Each generated fixture cycles all five row filters down its rows, and between them
  the set covers stored, fixed-Huffman and dynamic-Huffman deflate blocks.
- `paris.png` and `spectrogram.png`, the two `oflm-test` sends, decode to the sha256
  Pillow gives, recorded in `fixtures/png/bundled.json`. They are read where they
  already live rather than copied.
- An interlaced file is refused with a message naming "interlaced"; a truncated file
  and a JPEG are refused.
- Hostile inputs are refused by name, because images arrive base64 inside an HTTP
  request and none of these needs an attacker to do anything unusual: a header
  declaring 65535 x 65535, a header declaring 2^31 x 2^30, a header declaring
  16384 x 16384, a decompression bomb (200 MB of zeros in 200 KB), a palette image
  with no PLTE, a palette index past the end of PLTE, and a zero dimension. The bomb
  cannot work by construction -- the output buffer is sized from the header, so
  inflate stops the moment it would exceed it -- and the test says so rather than
  leaving it to be re-derived.
- Two ceilings on the IHDR, and the three size headers above are one test each
  (added 2026-09-16, `vegah` on PR #92). **No side over 16384**, refused before
  anything is computed from it, and **no more than 2^26 pixels**. The side cap is
  arithmetic, not taste: PNG allows 2^31-1 a side and the header says what it says
  whatever the file holds, so at 2^31 x 2^30 with 16-bit RGBA `(stride + 1) * h` is
  exactly 2^64, wraps to a small number, and passes a budget that is checked after
  the multiply -- after which `unfilter` sizes its row buffer from the same wrapped
  product, gets zero, and the first scanline writes past the end of it. A side above
  2^31 would also be reported back through an `int` as a negative width. The pixel
  cap is taste, and generous: every vision path resizes to 12.8 megapixels before the
  tower sees anything, and past the two together no surviving header costs more than
  512 MB of filtered rows and 201 MB of RGB24. 65535 x 5000 -- legal sides, an inch
  under the old 1 GB byte budget, and a gigabyte of allocation -- is refused on its
  width.
- 40,000 mangled inputs (byte flips, truncations, spliced noise over ten valid seeds)
  compiled with MSVC `/RTC1` produce no crash, and every input that DOES decode is
  self-consistent: `rgb.size() == width * height * 3`, so a caller sizing from the
  reported dimensions cannot walk off the buffer.
- The test links no FFmpeg and no zlib, so it passes on a machine whose ffmpeg has
  no png decoder -- which is the machine the bug is about.

**Verification (e2e), for the reader wiring the unit test cannot reach:**
1. `OFLM_QWEN2VL_ENGINE=open oflm serve qwen2.5vl-it:3b`
2. `oflm-test --vision --model qwen2.5vl-it:3b`
3. The server logs `Total images: 3`, not 1, and no `Skipping image that failed to load`.
4. The first round's text-extraction check passes -- it can only pass by reading
   `paris.png`, whose text is the answer.

**Result 2026-09-16:** `ctest -R OPEN-VISION-IMAGE-READ` passes 25 of 25, the two added
cases being `adv_wrap` (2^31 x 2^30) and `adv_gigapixel` (16384 x 16384); `adv_huge`
now answers on the side cap rather than the byte budget, which no longer exists.

The crash was reproduced before it was fixed, because the fixtures alone do not
reach it: the decoder's inflate refuses a stream that runs out early, so a 68-byte
`adv_wrap` is turned away on the short IDAT and never gets to the arithmetic. Filling
it does reach it -- 1 GiB of zeros, which deflates to a 4.7 MB PNG, small enough to
base64 into a request. On the pre-fix decoder that file takes `decode_rgb24` to
`0xC0000005`, an access violation, inside the first scanline's `memcpy`: `unfilter`
had sized its row buffer to `height * stride`, which is 2^30 * 2^34 and so zero, and
copied 17 GB into it. The same file on the fixed decoder returns false with
"2147483648 x 1073741824 is past the 16384 pixel side limit" and allocates nothing.
`adv_wrap` stays small and checked in: post-fix it is refused on the side, pre-fix on
the short IDAT, so it still fails if the check is removed.

The fuzz criterion above was not re-run. The change only refuses more headers, and
refuses them before any allocation, so it cannot reach an input the corpus could not
before.

**Result 2026-09-13:** `ctest -R OPEN-VISION-IMAGE-READ` passes 23 of 23, and the fuzz run above found nothing. End to end on
Qwen2.5-VL-3B through the open engine, `oflm-test --vision` passes all three rounds
for the first time on this box -- text extraction, seagull and spectrogram -- where
before the fix the same suite reported `Total images: 1`, failed text extraction, and
passed the spectrogram check on a spectrogram the model had invented. The extracted
text is "The capital of France is Paris...", which is what `paris.png` says.

**Blast radius, and it is checked, not asserted.** `ImageReader` is shared, so this
changes what `modeling_gemma3`, `gemma4e`, `gemma4_12b`, `qwen2vl`, `qwen3vl`,
`qwen3_5vl`, `qwen3_5_omni` and `qwen3_6_moe` do with a PNG, on the closed engine as
well as the open one: they prefill image rows where they used to prefill nothing. The
CLOSED engine was run on Qwen3-VL-4B to confirm it benefits too -- `Total images: 3`,
no `Skipping image that failed to load`, and all three `--vision` rounds pass, where
the same binary before the fix failed text extraction and the model answered "you have
only provided one image". No kernel, manifest, xclbin or spec hash moves.


### OPEN-VISION-VIT-FLAT: Qwen3-VL's container tiling, geometry and deepstack
**Applies to:** openflowlm-next (`open_kernels/model/replica_deepstack.py`,
`src/open_qwen36/vision/vit.cpp`, `core.cpp`, `engine.cpp`, `utilities/oflm-add`)
**Test category:** unit (`tests/test_qwen3vl_container.py`, and `vit_test --deepstack`
for the C++ port); the end-to-end run is OPEN-VISION-EMBED's

Qwen3-VL-4B-Instruct-NPU2's vision tensors are declared two-dimensional as
`[elements / 32768, 32768]`, and the order inside shall be read as tiles of 64 output
rows by 512 input columns, row-major within a tile and row-major over the tiles, with
nothing padded. A tensor stored at its natural shape (`pos_embed.weight`, every bias and
norm, the 5-D patch embed) is read directly; the discriminator is the 32768 row width,
not the rank.

**This was settled against an independent oracle, not by inspection.** The earlier record
had it as unsettled between "the 35B's tile order with a collapsed header" and "plain
row-major", with a column-norm correlation test unable to separate them -- and a wrong
choice gives image embeddings that look plausible and are wrong. All 315 tensors were
compared element for element against `Qwen/Qwen3-VL-4B-Instruct`'s own safetensors: 314
match under the rule above and the 315th matches directly. The bf16 values are identical,
so the container is upstream's weights reordered, not requantised.

**The geometry then falls out of the weight file**, because nothing is padded and
`patch_embed.proj.weight` keeps its natural shape, so `hidden` is known and every other
width divides out exactly. Two numbers never do, and shall be refused rather than
defaulted: the attention head count (qkv is `[3 * hidden, hidden]` at any split) and
which blocks the deepstack mergers hang off (the names say how many, never which).

**Where those two live, and why not `config.json`.** `pull` compares every
registry-listed file against a REMOTE manifest's byte count and treats any difference as
a truncated download, so a key added to the installed `config.json` is silently replaced
-- on this model a 4 GB re-pull. They go in `vision.json` beside the model, which is not
in that list; `oflm-add` writes it at install time. `image_token_id` needs neither: it is
`<|image_pad|>` in the container's own tokenizer, which is where the engine reads it.

**Acceptance criteria (unit):**
- 104 of the container's tensors are in the flat form and 211 at their natural shape;
  `pos_embed.weight` is 2-D and NOT flat, so the discriminator is the row width.
- `untile_flat` inverts the tiling rule, and refuses a shape that is not whole tiles and
  an element count that does not match.
- Every linear's element count is exactly `out * in` -- nothing is padded.
- The derived geometry equals upstream's `vision_config` on all ten numbers.
- `geometry_from_flat` returns no `heads`, no `head_dim` and no `deepstack` -- a count of
  mergers is not their indexes.
- It refuses a container with no `patch_embed`, and one whose patch embed is not three
  channels.
- Against transformers' `Qwen3VLVisionModel` loaded with UPSTREAM's weights while the
  replica reads the CONTAINER: merged, last_hidden and all three deepstack features at
  corr > 0.99999. Taking each tap one block early must drop deepstack[0] below 0.9, or
  the tap positions are not being tested.

**Verification (the C++ port):** `vit_test --deepstack <model dir> <fixture> 16 5,11,17`
against `replica_deepstack`'s fixture -- corr > 0.99999 and rel < 1e-3 on the merged
output and every feature.

**Result 2026-09-13:** numpy off the container against transformers with Qwen's weights
-- merged corr 1.00000000 rel 1.10e-05, last_hidden 1.00000000 / 1.38e-05, the three
features 1.00000000 at 4.46e-07, 8.80e-06 and 7.84e-06; the one-block-early control
0.6866. The C++ port against that numpy reference on an 8 x 12 grid -- merged
1.00000000 / 4.48e-06, features 1.00000000 at 3.07e-07, 1.05e-06 and 7.17e-07. Nine
tests, eight of which need neither the 830 MB container nor the 3.9 GB upstream shard.

### OPEN-FAMILY-LFM2: LFM2 replaces attention with a short convolution in most layers
**Applies to:** openflowlm-next (`open_kernels/recipes/spec.py`, `families.py`)
**Test category:** unit (`tests/test_lfm2.py`); the kernel and its hardware run are
OPEN-SHORT-CONV-KERNEL's

LFM2 is a hybrid: some layers are GQA attention, the rest replace the attention
block entirely with a three-tap depthwise causal convolution. That makes the
convolution a LAYER TYPE, not a parameter on an existing one, so `short_conv`
joins `LAYER_TYPES` beside `linear_attention`, `full_attention`, `dense` and
`dense_local`. A config whose `model_type` is `lfm2` shall derive a `ModelSpec`
with `family` `lfm2`, `full_attention` at the indices the config names and
`short_conv` everywhere else.

Nothing about the convolution is a new `ModelSpec` field. Its width is the
hidden size on every LFM2 that ships and its tap count is `conv_kernel`, the
field gated DeltaNet already has; a config where either stops being true is
refused by name rather than given a field, because `spec_hash()` covers every
field and a new one would move every shipped model's hash for no kernel change.
Adding a member to the `layer_types` tuple moves nothing, because no shipped
spec uses the new value.

The FFN width follows transformers' own rule (`Lfm2Config` lets `block_ff_dim`
override `intermediate_size`, `Lfm2MLP` then takes two thirds of it and rounds
up to `block_multiple_of`), not the raw `intermediate_size` key.

The short-conv block's fused input projection and its output projection take
the `linear` and `linear_out` quant roles a DeltaNet layer already has, so no
role is added to `QUANT_ROLES`.

`recipes.families.family_module("lfm2")` shall resolve to `recipes/lfm2.py`,
its own recipe, and `lfm2` shall be in `FAMILIES`. Routing the family to the
nearest existing recipe would emit kernels that drop the convolution and then
report parity against a replica making the same mistake. (Until the design
landed on 2026-09-12 it raised `NotImplementedError` naming the gap instead;
that is the standing rule, and `gptoss` is what the table holds now.)

**Acceptance criteria:**
- `model_type: "lfm2"` with LFM2-1.2B's config gives 16 layers, `full_attention` at 2, 5, 8, 10, 12, 14 and `short_conv` at the other ten; hidden 2048, 32 heads over 8 kv heads at head dim 64, full RoPE, `qk_norm` true, no gate, intermediate 8192, `conv_kernel` 3, eps 1e-5.
- `layer_types: ["conv", "full_attention", ...]` derives the same `spec_hash()` as `full_attn_idxs`; a name the family does not have is refused by name.
- `conv_dim` or `conv_dim_out` that is not `hidden_size` is refused naming the key; `conv_bias: true` likewise.
- `block_ff_dim` 12288 gives intermediate 8192, which is what the container's gate / up projections hold; with `block_auto_adjust_ff_dim` false the width is taken as written.
- Every checked-in spec under `recipes/specs/` hashes to what it hashed before `short_conv` existed, and no key named for the convolution appears in `to_dict()`.
- `family_module("lfm2")` is `recipes.lfm2`, and `lfm2` is in `FAMILIES`; the dense and qwen35 recipes refuse an lfm2 spec by family.
- `quant_map_from_chunk_sizes("lfm2", ...)` maps `shortconv.in_proj` to `linear` and `shortconv.out_proj` to `linear_out`; the installed container, all 4-bit, gives an empty map.

**Container, read 2026-09-12** (`LFM2-1.2B-NPU2/model.q4nx`, 149 tensors): ten
layers hold `shortconv.{in_proj, conv, out_proj}` and six hold
`self_attn.{q,k,v,o}_proj` plus `q_norm` / `k_norm` at width 64; every layer
holds `input_layernorm`, `post_attention_layernorm` and the three MLP
projections. `shortconv.conv.weight` is bf16 `[2048, 3]` -- one row of three
taps per channel, a depthwise `Conv1d` weight with its singleton input-channel
axis squeezed out. The embedding is `model.token_embd.weight` (bf16) and the
head is 4-bit, not q8. All 93 four-bit tensors are the signed quantiser the
packer transcodes (OPEN-PACK-Q4-0).

### OPEN-SHORT-CONV-REF: the fp64 reference for the short-conv block
**Applies to:** openflowlm-next (`open_kernels/model/replica_lfm2.py`, `model/lfm2_forward.py`)
**Test category:** unit (`tests/test_lfm2.py`)

The reference for one token through a short-conv block shall be, in float64:

    h = W_in @ x                             # 3 x hidden rows, in the order [B | C | u]
    Bx = B * u
    conv[c] = sum_k state[c, k] * w[c, k]    # state[:, taps-1] is this token's Bx
    out = W_out @ (C * conv)

with no bias and no normalisation inside the block, and with the conv cache
holding `Bx` -- the gated product the convolution reduces over -- rather than
the block input. The newest token pairs with the LAST tap; a transposed weight
is the one orientation error that still produces plausible numbers, so it is
asserted directly. An attention layer is the dense recipe's block unchanged and
goes through `replica_dense.dense_decode`, not a second copy of the same math.

**Acceptance criteria:**
- Three hand-computed tokens through a two-channel, three-tap block reproduce exactly, output and cache both.
- The cache after the first token holds `B * u`, not `x` and not `B`.
- With only the first tap non-zero the first token's output is zero; with only the last tap it is the ungated-conv value.
- The padded-convolution form over a whole prompt equals the one-token-at-a-time form bit for bit, and appending a token cannot change an earlier output.
- A conv state of the wrong depth and an `in_proj` that is not three times hidden are refused by name.

**Result 2026-09-12 (LFM2-1.2B, CPU, no NPU):** the whole model runs in fp64
straight out of the container through `model/lfm2_forward.py` and answers
coherently -- "What is the capital of France?" gives `The capital of France`
greedily, with `Paris` second at the first position. A wrong tap order, a wrong
layer schedule, a wrong `[B | C | u]` split or a misread weight format all
produce noise here, so this is the offline evidence that the geometry above is
right. About 30-40 s per token.

### OPEN-SHORT-CONV-KERNEL: the short-conv layer on the NPU
**Applies to:** openflowlm-next (`open_kernels/designs/short_conv/`, `recipes/lfm2.py`)
**Test category:** manual (the procedure below; needs the NPU, a Linux kernel build and
the LFM2 container)

The design, the element accounting and the reasoning are in
`.claude/plans/lfm2-short-conv.md`; this requirement carries the procedure
that verifies it, written down before the kernel was.

What the engine needs is only data: `manifest.cpp` looks a layer's type up by
NAME (`layer_types.at(layers[layer])`) and a fixed-size state buffer is already
the `"linear"` state kind, so a short-conv layer type and its 16 KB conv state
need no C++ change. The whole gap is one design directory and one recipe
module.

**Verification procedure (manual):**
1. Build the kernels in WSL with `PATH=~/xrt-tools/bin` and `LD_LIBRARY_PATH=~/xrt-tools/lib`: `OPEN_KERNELS_SPEC=<lfm2 spec> python build_design.py designs/short_conv/cx.py designs/short_conv/build_lfm2_cx_h2048`, and the attention layer's `dx`-style build beside it.
2. `python -m recipes.manifest --model-dir <LFM2 dir> --out manifest.json`; check `layers` names the two types where the config's `full_attn_idxs` says, and that `layer_types.short_conv.buffers.state` is `{"kind": "linear", "bytes": 16384}`.
3. Pack and run a slice: `open_kernels/model/make_decode.py --model-dir <dir> --layers 2` (one short-conv layer then one attention layer) against `open_kernels/model/lfm2_forward.py --layers 2`. The conv layer alone is the first thing to look at: a tap-order error shows as a residual that is right at position 0 and wrong from position 1.
4. Whole model, position 0 and positions 1-3: logits corr against `lfm2_forward.py`, argmax and top-5 identical. The bar the other families cleared is corr > 0.9999 with identical argmax.
5. `utilities/flm-test --llm --model lfm2:1.2b` through `flm serve`, plus a coherence read of the answer.
6. Only then add `(64, 32, 8, 64, True, False, False, False)` to the `attn` combination set and the short-conv points to `catalogue.py`, with the date and this requirement's name.

**Acceptance criteria:**
- Steps 3 and 4 pass at the bar above, on the container at `LFM2-1.2B-NPU2`.
- `catalogue.py` holds the attention tuple and the `short_conv` point `(taps 3, width 2048)`, so an LFM2-1.2B export needs no `OPEN_KERNELS_UNVALIDATED`.

**Result 2026-09-13 (LFM2-1.2B-NPU2, Strix): PASS.** The first build ran on the
NPU and produced noise; the fault was in `cx.py`, not the conv core. It handed
the GEMV a per-band count divided by the chunks-per-element count (8 instead
of 16 at hidden 2048), and `gemv_q4_pool_group_rt` derives the table width
from that argument, so every projection read a 1024-wide activation table and
B, C and u came out around 1e37. `per_band` now lives in
`recipes/qwen36moe.py` beside `band_bytes` and `tests/test_lfm2_recipe.py`
pins the round trip. With that fixed, one token from a zeroed state through
layer 0 (`--dump-act`) matches the fp64 reference at every stage: `xn`
0.9999992, B / C / u 0.999999, the conv output `y` 0.999998, `out` 0.999999,
`res` 0.9999999, `h` and `out2` 0.9999996, and the layer-0 logits 0.9999988
with the same argmax. Layer 0 alone over positions 0-3 (the conv state
carrying across tokens): corr 0.999998-0.999999, argmax and top-5 identical
at every position. Whole model, positions 0-3: corr 0.99995, 0.99999,
0.99999, 0.99999, argmax and top-5 identical. Greedy decode of the France
prompt gives `The capital of France is Paris` and the fp64 reference produces
the same six tokens. The kernel set was then exported cleanly with
`export_qwen36_kernels.py`, installed by `oflm-add` (which found it by spec
hash, `sha256:fd500fa0be38`), and `oflm-test --llm --model lfm2:1.2b`
through `oflm serve` PASSED both rounds with coherent answers
(`utilities/oflm-test/results/20260913_082356/windows/`): 789 and 746
tokens, both ending on the model's own stop token. Serving it needed one
adapter change: `modeling_lfm2.cpp`'s `LFM2` class now selects the open
engine through `_shared_select_open_engine` under `OFLM_LFM2_ENGINE`, the
way every other family with an open path does; `LFM2_5_TK` (the thinking
variant, which casts its engine to the closed class for checkpoint /
restore) stays on the closed DLL. Decode through the server ran at 6.8 and
2.7 tok/s against 18-25 tok/s in the standalone CLI, which is not a server
overhead but the context position: those CLI runs were at position 25 and the
server's two rounds averaged 410 and 1180. The six attention layers were still
on the slow path. LFM2 joined the fast one later the same day and the sweep is
flat -- OPEN-ATTN-CONTEXT carries the numbers.
- Until they do, `families.family_module("lfm2")` keeps raising and the geometry stays out of `catalogue.py`.
### OPEN-PREFILL-BATCH: the block prefill route
**Applies to:** openflowlm-next (`open_kernels/recipes/{qwen36moe,dense}.py`, `designs/gemm_q4_prefill/`, `designs/dense/dx_attn.py`, `designs/layer_x/mx.py`, `src/open_qwen36/{manifest,core,block_host,engine}.cpp`)
**Test category:** manual (needs the NPU; `Qwen3.6-35B-A3B-NPU2` for the MoE kinds, a dense family's own kernel set for `dense`); the recipe emission, the manifest schema, the host stages and the GEMM operand helpers are unit-tested in `tests/test_prefill_batch.py`, `tests/test_gemma3.py`, `src/open_qwen36/manifest_test.cpp` and `src/open_qwen36/block_host_test.cpp`

A kernel set may carry a block prefill route: per layer type a `gemm_block`
naming, by kind, the GEMM dispatches that replace the layer's projections for
T = 256 tokens at once -- `dense` (0167/#32): the five-step chain with T
single-token attention dispatches, run through the kernel and buffer args its
own `attn_kernel` / `attn_args` name (default `dxB`, so every dense layer
type shares one attention kernel and position table unless it says
otherwise); a layer type with its own sliding window (Gemma 3's
`dense_local`) names its own kernel entry -- the same instruction stream,
patched with its own window, per `stream_patch`'s existing `attnpos`
mechanism -- and its own table, exactly as the sequential path's `dx` /
`dx_local` already do. The `dense` kind also carries `sandwich` and `act`, so
a family whose residual chain norms the attention and FFN outputs before they
join the residual (Gemma 3's sandwich norms) and whose FFN gates a
GeGLU-tanh rather than a SiLU is served by the same route and the same
five-step program; `linear`: qkv|z then out, with the DeltaNet
recurrence on the host between them; `full`: q|k|v|gate then o, with attention
over the KV rows on the host -- the weight buffers those dispatches read as
contiguous runs of the layer type's pack ops, and, for the MoE kinds, the
MoE-only dispatch (`mx`, the second half of the whole-layer core program as its
own xclbin) that runs the ROUTED experts for one token from the router record
the host wrote. The shared expert is not in that dispatch: it is the same
weights for every token of the block, so the route runs it once as two more
GEMMs (up|gate, contiguous and band-law in the pool, then down) with silu and
the sigmoid gate on the host, folds it into the residual the dispatch is
handed, and `mx` closes on `xres + acc` (`moe_accfin`'s slot < 0). Hardware contexts are shared: ONE GEMM xclbin
for the whole route -- the core program depends on neither N nor K, the band
count K/256 reaching each core as a runtime parameter the instruction stream
writes -- and one `mx` xclbin for both layer
types. `OFLM_OPEN_GEMM_BLOCK=1` (read through `getenv_oflm`, so the pre-rename
`FLM_` export still works) selects the route; off by default so every existing
measurement is unaffected. With it on the engine takes the route whenever the
set carries it and the prompt has at least the crossover length (64 tokens;
`OFLM_OPEN_GEMM_BLOCK_MIN` overrides), never for a prompt that has had an image. Only the real tokens of a padded block touch the
state, write KV rows, run the MoE or advance the position, and the conv state,
S and the KV rows leave the buffers as the sequential path would (bf16 where
the kernels keep bf16). A projection streamed at q8 has no route -- the GEMM
dequantises the q4_1 band law -- and such a manifest is the sequential one
unchanged.

On a MoE kernel set the blocks are run **layer-major**: every T-wide block of
the prompt goes through layer l's projections and host stages before layer
l + 1 starts, and the layer's expert block then runs ONCE over every token of
the prompt instead of once per block. An expert is therefore streamed for all
of its tokens in the prompt rather than for the eight or so it owns in one
block. Nothing about a layer's arithmetic changes -- the device state (KV rows,
DeltaNet S and conv rows) is still written block by block in position order,
and layer l still reads the residual layer l - 1 produced for the same block,
because layer l - 1 finished every block first -- so the ordering is exactly
the block-major one per layer; what the result does depend on is which tokens
share an expert's dispatch, since `OPEN-MOE-BATCH` is not invariant to that
(see its own result). The schedule also lets a layer read its DeltaNet state
off the device once per prompt instead of once per block, and stops the
full-attention layers pulling the whole KV window back for every block.
`OFLM_OPEN_LAYER_MAJOR=0` (or `--block-major` on the CLI) takes the
block-at-a-time loop; a `dense` layer type has no layer-major form and stays
block-major.

**Acceptance criteria (unit):**
- The 35B emission as `test_prefill_batch.py` asserts it: `linear` runs `gemm_n12288_k2048` (qkv|z, pool ops 5 and 6) then `gemm_n2048_k4096` (out, consts op 10); `full` runs `gemm_n9216_k2048` (q, k, v, gate: pool ops 5-8) then `gemm_n2048_k4096` (o, pool op 9); one context `gemm` for every GEMM shape, plus `mx`; kernels `mx_linear` / `mx_full` with the moeroute2 patch; globals `gemm_x_k{K}` = K·T·2 and `gemm_y_n{N}` = N·T·4 bytes; a spec with `attn`, `linear`, `linear_out` or `shared` at q8 emits none of it and its manifest equals the sequential one.
- The parser holds a route to its kind (`manifest_test.cpp`): 5 steps for dense, 2 for linear / full, every step a 3-argument run naming a declared weight buffer, weight ops inside the pack plan, `moe_kernel` declared with the moeroute2 patch, each refused by name otherwise.
- The host stages equal `open_kernels/model/replica_block.py` on its random fixture (`block_host_test.cpp`): og and S within 1e-3 of the reference's scale, the conv state bit-exact in bf16, the KV rows within a bf16 ulp, rows before the block and past `t_real` untouched, the top-k ids exact; the tiler and the transpose equal the plain loops. The numpy reference equals its own one-token-at-a-time form with the state carried, and padding past `t_real` changes nothing.
- The 35B's shared expert emits `shared_program` = `gemm_n1024_k2048` (up|gate) then `gemm_n2048_k512` (down) with `shared_ff` 512 on both MoE layer types, its weight buffers naming the contiguous pool ops; the parser refuses a MoE route without two such steps, or one whose shared step names a buffer `shared_weights` does not define (`manifest_test.cpp`).
- The build key covers `designs/gemm_q4_prefill/*` (`test_prefill_batch.py`).
- The dense kind's recipe (`dense.py gemm_route`) emits one `gemm_block` per layer type present, sharing the five-step program, weight map and widths (`layout`/`geometry` don't vary by layer type) but each naming its own `attn_kernel` / `attn_args`: a family with one layer type gets the schema's defaults (`dxB` / `...,"ptab"`); Gemma 3's two get `dxB` / `ptab` for `dense` and a second kernel entry `dxB_local` / `ptab_local` -- the SAME `dx_attn/insts.bin` stream, patched with the family's sliding window -- for `dense_local`, plus `sandwich: true` and `act: "gelu_tanh"` on both (`manifest_test.cpp`'s qwen3 and gemma3 fixture blocks, `tests/test_gemma3.py`). A family combining sandwich norms with a non-gelu-tanh activation, or the reverse, gets no route (`tests/test_gemma3.py`) -- the host chain only implements the two pairings above. Nor does a family whose q/k/v carry a bias or whose position record is wider than one attention element (Qwen2, both): `dx_attn` has neither stream and refuses to build, so the manifest stays the sequential one (`tests/test_qwen2.py`). The parser refuses a dense route whose `attn_kernel` runs the same instruction stream as any sequential `program` step (`dx`, `dx_local`) -- `attnpos` alone does not make a kernel the attention-only dispatch, and the sequential stream would run the whole layer per token (`manifest_test.cpp`).
- Layer-major is refused where it has no meaning: `layer_major_ok()` is false without a block
  route, with `OFLM_OPEN_LAYER_MAJOR=0`, and for any layer type whose kind is not `linear` or
  `full`; `step_gemm_prompt()` throws rather than silently running the wrong schedule, and
  throws for a prompt that would run past the context capacity.
- The engine says out loud when the OpenMP wait policy could not be applied. vcomp reads
  `OMP_WAIT_POLICY` when it loads, so the static initialiser that sets it only works if vcomp
  is delay-loaded (`/DELAYLOAD:VCOMP140.DLL`); a build without that runs the host workers
  spinning through every dispatch and is ~20 % slower at prefill with nothing to show for it.
  `Core`'s constructor checks whether vcomp was already in the process when the initialiser
  ran and prints a named WARNING if it was, so the flag cannot be dropped silently. The flag
  is necessary and NOT sufficient: in `oflm.exe` eight implicitly linked closed model DLLs
  import vcomp themselves, so the warning fires there and the variable has to come from the
  environment (2026-09-21 result below). The criterion is that the warning is accurate, not
  that it never fires.

**Procedure:**
1. `python open_kernels/export_qwen36_kernels.py --model-dir ~/.flm/models/Qwen3.6-35B-A3B-NPU2` (WSL) builds `gemm_n12288_k2048`, `gemm_n2048_k4096`, `gemm_n9216_k2048`, `mx_linear` and `mx_full` beside the sequential set and writes the manifest with the route.
2. Each GEMM shape through the harness: `python make_test.py --shape nN_kK --tokens 256`, `run_kernel.exe run_nN_kK_t256.cfg`, `python compare.py nN_kK_t256` -> PASS.
3. `open_qwen36_cli --layers 4 --prefill-logits --dump-logits <dir>/y` with and without `--gemm-block` on a 19-token prompt: the same argmax and top-5 at every position except documented near ties (reference margin < 0.05 logits), corr > 0.999 per position.
4. All 40 layers on a ~1000-token prompt with `--max-tokens 8`: the same greedy continuation, the last position's argmax and top-5 equal; TTFT with and without `--gemm-block` recorded.
5. `flm-test --llm` through `flm serve` on `qwen3.6-moe:35b-a3b` with the route on.
6. Layer-major against block-major, same binary and same kernels: a multi-block prompt through
   a prefix that contains a full-attention layer, `--prefill-logits --dump-logits` both ways,
   diffed position for position. With `OFLM_OPEN_MOE_BATCH=0` (the per-token expert dispatch)
   the two must be **byte-identical** -- that is what says the schedule itself changed nothing.
   With the batched expert kernel they are not, and the gate is instead that both sit the same
   distance from the per-token run: equal max |diff| and equal argmax-flip count against it.

**Result 2026-09-11 (Qwen3.6-35B-A3B-NPU2, steps 2-5):** every GEMM shape PASSes the harness at rel_fro 2.2e-3 (gate 5e-3), 0.8 ms (512 x 2048) to 14 ms (12288 x 2048) per dispatch. Step 3: argmax 18/19 and top-5 19/19 against the sequential path, the one flip a 0.003-logit tie, corr >= 0.99996 per position; against the fp64 replica the block route is 18/19 (corr >= 0.9995) where the sequential path is 19/19 (>= 0.9998) -- the bf16 GEMM's rounding, not a stage. Step 4 on a 1020-token prompt, all 40 layers, nothing else on the NPU: prefill **121.4 s -> 41.5 s (119 -> 41 ms/token, 2.9x)**, and **40.0 s (39 ms/token)** once the shared expert moved out of the per-token dispatch (2026-09-12: 11.3 % off the route against the 11.1 % of the stream it is; the same greedy token, and step 3 improves to argmax 19/19, top-5 19/19, corr >= 0.9993), the 8-token greedy continuation identical, last-position argmax and top-5 equal, corr 0.9985 at full depth. Per 256-token block: the GEMMs 0.77-0.84 s (8 %), the host stages 1.5-2.0 s (17 %; attention grows with the window), the per-token MoE dispatches 7.1-7.9 s (73 %) -- what the token-batched expert kernel (`OPEN-MOE-BATCH`, the plan's stage 2) removes. (An earlier reading taken with another process serving on the NPU, 173 -> 61 ms/token, had the same ratio.) Step 5: `flm-test --llm` passes through this tree's `flm serve` (v1.0.4; the load log shows `Qwen3.6-MoE on the open kernels` and `block prefill route: T = 256`), both answers coherent; through the server the route prefills 972 tokens in 41.8 s (43 ms/token) and 2582 in 119.3 s (46 ms/token). For scale, the closed `qwen3_6_moe_npu` kernels in stock FLM 1.0.2 prefill the same two prompts in 14.3 s and 21.9 s (14.7 and 8.5 ms/token): the open path is still 3-5x behind them at prefill, and its per-token cost rises with length where theirs falls. Serving a 1.0.2 container from this tree needs the registry gates disarmed (`OFLM_CONFIG_PATH` at copies of `model_list.json` / `model_info.json` carrying `flm_min_version` 1.0.2 and the real file sizes) and a scratch model copy under `OFLM_MODEL_PATH`, or the app re-pulls the 22 GB file. Details: `.claude/plans/prefill-batch-35b.md`.

**Result 2026-09-13 (the host stages):** the block line now splits `mid` into
the DeltaNet's two halves and the attention's, which contradicted the standing
assumption about where the host time went. At 256 tokens `mid` was 1040 ms a
block and **856 of it was the DeltaNet's per-token half** -- the conv, the q/k
norms and the alpha/beta projection -- against 111 ms for the delta rule on S
that every plan had named as the expensive part. The per-token half was
single-threaded because the conv was written as a ring buffer with a shift,
which looks like a recurrence and is not: the conv reads a fixed four-tap
window of the projection's own rows, so row r of token t's window is qkv row
t - 3 + r and every token is independent. Over tokens under OpenMP it is
**856 -> 165 ms**, bit-exact (`block_host_test.cpp` still matches
`replica_block.py` on og, S and the conv state rows).

Two host stages around the GEMMs went with it. `gemm()` returned its output by
value, so each of the 160 dispatches a block allocated a vector and
value-initialised it before the transpose overwrote every element -- 716 MB a
block of zeroes written for nothing; the four buffers now live across layers
(transpose stage **240 -> 126 ms**). And both layer kinds transposed the widest
GEMM into a 12 MB buffer only to memcpy it apart immediately, so
`transpose_parts` writes the column ranges into their destinations in one pass.

Per 256-token block the host half went **1306 -> ~500 ms**: `mid` 1040 -> ~360
(DeltaNet 165 + 111, attention 70-115 growing with the window), transpose 115,
tail 75, shared expert 40. With `OPEN-MOE-BATCH`'s two changes of the same day,
`open_qwen36_cli --gemm-block` on 2582 tokens went **57.7 -> 41.6 s (22 -> 16
ms/token)** with the identical eight-token greedy continuation, and 512 tokens
9.98 -> 7.50 s. `oflm-test --llm` passes through this tree's `oflm serve` on
the route. Details: `.claude/plans/moe-stage-cost.md`, raw data in
`.claude/plans/decode-run/logs/`.

**Result 2026-09-20 (the q4_1 dequant, in the pool's own orientation):** the
block route's GEMM was spending **39 % of every dispatch turning q4_1 into
bf16**, in two passes over the band: a gather that transposed raw nibble bytes
into a row-major uint8 scratch, then a dequant that read that scratch sixteen
elements at a time with the row's `d` and `m` arriving as scalar loads and
broadcasts. Ablated with the design's own `GQP_NULL_GATHER` / `GQP_NULL_DEQUANT`
on the checked-in n8192 k2048 T256 fixture, minima of three runs: the whole
dispatch 5.977 ms, the gather 0.498 of it, the dequant 1.867, leaving 3.621 for
the matmul and the streams.

It is far cheaper in the orientation the pool is already packed in. A 64-byte
contiguous read of the nibble area is 8 k x 8 row-pairs, and in that orientation
the eight rows' scales are eight strided entries of one 16-wide load -- a
vector, not eight scalars. That is exactly the shape `designs/moe_batch` works
in, and the scale builder is now the same function (`gqd_scale`, `mb_scale`'s
twin). So the fused pass masks, converts, scales and adds in 64-lane vectors
with no scalar work at all, then `aie::transpose`es the RESULT into row order
and `interleave_zip`s the even and odd rows back together -- which lands four
whole (4 rows x 8 k) A-operand blocks, 32 contiguous bf16 each, for four stores
instead of sixteen. The gather pass disappears entirely, and the scale is
applied as one `mac` seeded with `m` so that `n * d + m` rounds to bf16 once
instead of twice.

**The dispatch goes 5.977 -> 4.474 ms, 25.1 %; the q4_1 -> bf16 cost itself
2.356 -> 0.853 ms, 2.8x.** It is also more accurate, from the single rounding:
rel_fro against the fp64 reference 2.202e-3 -> **1.684e-3** (gate 5e-3), maxrel
2.205e-3 -> 1.769e-3. On the 35B at 2582 tokens, kernel sets run one after the
other, the **GEMM stage goes 10.56 -> 7.88 s (24 %)** and the whole prefill
28.75 -> 25.79 s. Against the sequential path the two dequants are the same
distance out (worst per-position corr 0.9974 either way over 120 positions, 3
argmax flips for the old and 2 for the new), so the route's own spread is
unchanged. `GQD_TWO_PASS` rebuilds the old path for an A-B; the uint8 scratch
it needs is still allocated, and can go with it.

Two things tried and not kept: **hoisting the scale builds** into a k-block loop
around the column-block loop, although `d` and `m` only change every 32 k, was
slightly SLOWER (4.589 against 4.501 ms before the `mac` change) because the
compiler already eliminates the repeats and the narrower unrolled window costs
more than the hoist saves; and the remaining body is 352 bundles against a
Peano resource floor of 266, so there is perhaps 20 % more in scheduling.

**Result 2026-09-21 (the wait policy the binary never applied, and the host stages):**
the engine has set `OMP_WAIT_POLICY=PASSIVE` from a static initialiser since
2026-09-20 and **never once applied it**. vcomp reads the variable when the DLL
LOADS, and an implicitly linked DLL loads before any static initialiser in the
exe runs, so the initialiser set a variable the runtime had already read past --
every measurement since was taken with the workers spin-waiting through each
dispatch, and the 38.55 -> 29.93 s the plan credits to A.1 came from setting the
variable in the shell. What proved it: the same trick for `OMP_NUM_THREADS` put
10 in the environment and `omp_get_max_threads()` still returned 24; then, same
binary at 2582 tokens, **20.5 s with the variable unset against 17.1 s with it
set from outside**. `/DELAYLOAD:VCOMP140.DLL delayimp.lib` moves vcomp's load to
the first call into it, which is after the initialiser: three alternating runs
at 2582 tokens give **16.8 s mean against 21.1 s with `OFLM_OPEN_OMP_PASSIVE=0`**.
Both build paths carry the flag, and the engine now checks at construction
whether vcomp was already loaded when the initialiser ran and warns by name if
it was, so a link line that drops it cannot cost 20 % of the prefill in silence.

The "in-block penalty" -- every dispatch at 1.4-1.6x its `--bench` minimum -- is
the array's clock, and the probes say so three ways. Cycling all thirty linear
layers' weights costs 0.13 ms on `mb_s256`, so it is not cold memory. Two
thousand back-to-back `mb_s256` dispatches hold 12.36-12.48 ms over 25 s with no
drift, so it is not thermal. `bench_dispatch` grew a SLEEPING 30 ms gap beside
its busy one, and under `--pmode performance` the two cost the same (`mb_s256`
12.4 -> 19-21 ms, `gemm_n12288` 6.5 -> 10.3) while under `--pmode turbo` both
cost 0.1 ms -- so an idle NPU loses clock whatever the CPU is doing. The rest is
the host cores: with the workers spinning, `mb_s256` inside a prefill is 17.5 ms
and **the same dispatch repeated immediately, with no host work in front of it,
is still 17.5**, which is the package and not the work before it. With PASSIVE
actually in force the host thread count stops mattering the way it appeared to
(2582 tokens: 17.6 s at the runtime's own count against 19.5 at ten), so the
count stays the runtime's own and `OFLM_OPEN_OMP_THREADS` is only a knob.

Five host stages, each bit-exact and each measured as alternating pairs: the
per-(layer, block) scratch is reused rather than freshly allocated and
zero-filled (30 MB a call, 440 calls; **26.5 -> 25.4 s**); the [N, T] -> [T, N]
transpose is an AVX2 8x8 register network instead of a four-byte-at-a-time
blocked loop (**`gemm tr` 1448 -> 583 ms**, 23.4 -> 22.6 s); DeltaNet's causal
conv runs its taps as contiguous passes over the channels with the
carried-state test hoisted out of the channel loop, so it vectorises
(**`dn conv` 1.75 -> 1.45 s**); the shared expert stops copying 8 MB of its
input per call to satisfy a `std::vector` parameter and its silu, its output
gate and both layer kinds' residual add stop being serial (**shared 550 ->
62 ms**, 21.5 -> 21.0 s); and the delta rule stores S once a token instead of
twice, recomputing `Si[j] * dc` in the second pass rather than writing 64 KB in
the first (**stage 1.40 -> 1.25 s**). The gate throughout is a 600-token,
40-layer prefill diffed position for position against the previous binary:
**600 of 600 identical, max |diff| 0.0** after each change and at the end.

**End to end, `open_qwen36_cli --gemm-block`, three repeats on one resident
weight load, best of three:**

| tokens | `--pmode performance` | `--pmode turbo` | closed, same box, same day |
|---|---|---|---|
| 662 | 4.75 s | 4.63 s | 12.39 s |
| 1122 | 7.95 | 7.63 | 15.20 |
| 2102 | 14.52 | 13.73 | 18.92 |
| 2582 / 2593 | 18.27 | 16.69 | 21.34 |
| 3642 | -- | ~22.8 (fitted) | 23.05 |
| 4096 | 28.91 | 25.75 | ~24.6 (fitted) |

2582 tokens went **26.70 -> 16.69 s** across the session and 1122 went
**13.55 -> 7.63**. Fitted, the open prefill is 0.6 s + N / 164 tok/s against
closed's 10.0 s + N / 280 (re-taken today over five lengths x three cycles,
best of three per length; the 2026-09-20 fit was 9.93 + N / 280.3, so the line
reproduces across days even though the same box's decode rate doubled between
them). They cross at about **3600-3700 tokens**: below that the open route is
ahead -- 2x at 1122 -- and at 4096, the context capacity, it is about 5 %
behind. Two caveats on that comparison: closed's figure is the server's own
prefill timer and carries a ~10 s fixed term whatever the length (662 tokens
cost it 12.4 s where its own marginal rate says 2.4), where ours is a CLI
prefill with the weights already resident, so the open side through
`oflm serve` is the remaining like-for-like check; and closed sets
`performance` itself at startup, so the `performance` column is the
like-for-like one -- it wins at every length below 4096 as well. Details:
`.claude/plans/prefill-parity-2026-09-21.md`.

**Result 2026-09-21 (the dispatch log re-taken with PASSIVE in force):** every
per-kernel in-block figure recorded before the delay-load fix above — including
the 1.4-1.6x "in-block penalty" — was measured with the workers spinning.
Re-taken on the fixed binary, `OFLM_OPEN_DISPATCH_LOG=1 --gemm-block` on a
2582-token prompt against `--bench 20` on a linear and a full-attention layer of
the same binary, the penalty is **gone on the kernels that carry the run**:
`mb_s256` 13.407 ms mean against a 12.447 solo minimum (**1.08x**, 400 calls,
5.36 s of the 19.46 s prefill), `gemm_n12288_k2048` 7.351 against 6.357
(**1.16x**), `gemm_n9216_k2048` 5.539 against 4.816 (1.15x). The old figure
survives only on the three narrow GEMMs — `gemm_n2048_k4096` 3.200 against
2.027 (1.58x), `gemm_n1024_k2048` 1.45x, `gemm_n2048_k512` 1.46x — which
together are 2.06 s of the 5.09 s of GEMM dispatch. Summed over every dispatch,
**1.88 s of the 19.46 s run sits above the bench floor, 9.7 %**, where the
2026-09-20 reading put it near 6 s.

Two thirds of what remains is one avoidable thing. The layer-major schedule
amortises a hardware context switch across all eleven blocks for the expert
kernel and the qkv GEMM, which is why those are at 1.08x; it does not for the
ten full-attention layers, where each (layer, block) runs `gemm_n9216_k2048` ->
2x `ag_s` + 2x `ag_pv` -> `gemm_n2048_k4096` and so leaves the `gemm` context
and re-enters it 110 times each way. This binary's context-switch probe prices
a switch into `ag` at +2.55 ms and staying inside `gemm` at +0.03, which
accounts for the whole `ag_*` excess (356 ms) and ~308 ms of
`gemm_n2048_k4096`'s 516: **~0.66 s, 3.4 % of the prefill, recoverable by
hoisting the attention dispatches out of the per-block loop** (all eleven
blocks' `q|k|v|gate`, then all the attention, then all the `o` GEMMs — two
switches a layer instead of twenty-two). Causality needs no new mask, since
`ag_s` is dispatched with the window length `(b + 1) * 256` and so reads
exactly blocks 0..b whatever later KV rows are already written; the cost is
~23 MB of scratch to hold each block's attention output until its `o` GEMM. Not
done.

Also from the re-taken log: dispatch logging itself costs ~1.2 s at this speed
(19.46 and 20.37 s on two repeats against 18.27 best-of-three without it), so a
logged run is no longer a timing run; and **host compute is now the larger half
of the prefill** — 11.46 s of the 19.46 is dispatch and the other 8.0 s is CPU,
3.13 s of it DeltaNet, with nothing overlapping.

**Result 2026-09-21 (step 5, and the wait policy the SERVER cannot apply):**
`oflm-test --llm` passes 5 of 5 on `qwen3.6-moe:35b-a3b` through this tree's
`oflm serve` with `OFLM_QWEN36_ENGINE=open` and `OFLM_OPEN_GEMM_BLOCK=1` — both
modes, both prompts, the context-retention check — on a server carrying the
layer-major schedule, which had never been exercised through the server path
before. The load logs `Qwen3.6-MoE on the open kernels`,
`block prefill route: T = 256` and `prefill schedule: layer-major`.

It also prints the delay-load WARNING, and it is right to: **`oflm.exe` carries
`/DELAYLOAD:VCOMP140.DLL` and still cannot apply `OMP_WAIT_POLICY` from inside
the process**, because eight of the closed NPU model DLLs it links implicitly
(`qwen3_6_moe_npu`, `qwen3_5vl_npu`, `qwen3_5_omni_npu`, `qwen3vl_npu`,
`gemma_npu`, `gemma4e_npu`, `gemma4_12b_npu`, `gpt_oss_npu`) import vcomp
themselves, so it is in the process before any code in the exe runs. Delay-loading
vcomp from the exe is necessary and not sufficient; the CLI has no such dependency,
which is why the same flag works there. Measured on the same binary and kernels,
with the variable set in the environment before launch against not set, four
rotating cycles per length, the server's own `prefill_speed_tps`: **2593 tokens
17.43 s against 24.14, and 1.34-1.39x at every length from 182 up** — more than
the 20 % the CLI saw.

**With PASSIVE the server matches the CLI to within 2 %** (662 tokens 4.79 s
against 4.75, 1122 7.84 against 7.95, 2102 14.18 against 14.52, 2593 17.43
against 18.27 at 2582), which answers the standing caveat that the open figures
were CLI-only while the closed ones came from a server timer: they were not
flattering themselves, and the capacity the server loads at (32768 against the
CLI's 4096) costs nothing measurable. Fitted, the open prefill through the
server is **0.46 s + N / 153 tok/s**. Read off the same timer as closed, the
open route is **2.6x faster at 662 tokens, 1.94x at 1122, 1.33x at 2102, 1.22x
at 2593**, level at 3062 (20.76 s) and 12.6 % behind at 3642 (25.96 against
23.05) -- **a crossover near 3100 tokens**, not the ~3700 the CLI-against-server
comparison implied. Without PASSIVE, which is what a default build of the server
does today, the fit is 0.39 s + N / 109 and the crossover falls to **about 1700
tokens**. The closed column is the same day's measurement on this box through
stock FLM's own server and was not re-taken; note also that the closed engine's
host work is OpenMP too, so it was measured with its own workers spinning.

Not fixed. The candidates are delay-loading those eight DLLs as well (which
changes how the closed path fails when one is missing, and they are upstream's),
a guarded re-exec at the top of `main` with the variable set, or documenting the
variable and having the launcher export it.

**Result 2026-09-21 (both engines back to back, one afternoon):** the comparisons
above pair an open reading from one run with a closed reading from another, which
is what `measuring-closed-engine` warns against, so both were then measured
back to back: same container (stock FLM 1.0.2 serving hardlinks to the same
`model.q4nx`), same prompts, same script, 48 generated tokens, four rotating
cycles per length, best of four off each server's own `usage` timer. Prefill,
open against closed: **1.94 s against 14.19 at 182 tokens (7.3x), 8.95 against
18.85 at 1122 (2.1x), 20.96 against 24.15 at 2593 (1.15x)**. Decode, open against
closed: **8.38 against 14.17 tok/s at 182 tokens of context, 7.87 against 14.83 at
1122, 7.00 against 14.46 at 2593 — closed is 1.7-2.1x ahead and the gap widens
with context.** Decode has had no work since 2026-09-13.

Fitted on this pair, closed prefill is **14.8 s + N / 279 tok/s** against the
10.0 + N/280 of the earlier run the same day. **The marginal rate reproduces to
within half a percent; the fixed term moved by half again** — and since the
crossover is set almost entirely by that fixed term it is not a stable number
(3100 tokens against one reading, past 4800 against the other). The short-prompt
ratios are the robust statistic; a crossover should be quoted as "past 3000
tokens" or not at all. Our own open figure also moved within the afternoon
(20.96 s at 2593 against 17.43 two hours earlier, same binary and configuration,
the only difference being 48 generated tokens per request instead of 8), which the
pairing absorbs but which is worth knowing before quoting any single number.

**Result 2026-09-13 (one GEMM context for the whole route):** the GEMM core
program used to bake K in as the trip count of its band-group loop, so the route
carried three GEMM xclbins and therefore three hardware contexts. The band count
K/256 now reaches each core as a runtime parameter the instruction stream writes,
and all five of the 35B's projection shapes are streams over **one** xclbin. All
five build to the same 203231-byte image (equivalent under
`export_qwen36_kernels.py`'s own `xclbin_equivalent`, build stamps only), pass
the harness at rel_fro 2.17e-3 to 2.25e-3 against the fp64 reference, and are
**bit-exact against the previous compile-time-K kernel** on the same vectors --
the change moves where K comes from, not the arithmetic. The runtime bound costs
64 bytes of program memory per core (+2048 B over 32 cores) against a 16 KB
budget.

A hardware context change costs 2.47 ms into the GEMM context, 2.49 into the
attention one and 2.93 into the expert kernel's, measured with an interleaved
baseline over three runs (`Core::bench_dispatch`'s context-switch probe takes its
own baseline in the same loop, since subtracting one measured minutes earlier put
box drift straight into the delta). It does **not** scale with the kernel: across
the `mb_s*` ladder, 0.57 to 14.1 ms of work and 32x the streamed bytes, it is
2.82-3.14 ms with no trend. A 256-token block made 210 changes and now makes 100.

End to end on 2582 tokens, `open_qwen36_cli --gemm-block`, three runs each side:
the **GEMM stage goes 1489 -> 1096 ms a block, a 393 ms saving** (spread 19 ms
across the three baseline runs, 5 ms across the two clean collapsed runs), and
prefill **43.1 -> 38.9 s** against the fastest baseline, with the identical
eight-token greedy continuation. Per dispatch the mechanism is visible directly:
every projection whose context change was removed drops 2.75-3.39 ms, while the
two that still follow the expert dispatch are unchanged (-0.17 and +0.06). Note
that a switch costs ~3.37 ms inside a block against the 2.47 the isolated probe
reports, so the block-level saving is larger than a per-switch model predicts
(393 against 273); the isolated probe understates it.

The core reads its band count immediately after acquiring the first weight band,
so the dataflow orders the read: that acquire cannot complete until the runtime
has issued the fill, which it issues after the parameter writes. A
`WorkerRuntimeBarrier` -- the idiom `npu_offload/gemm_rtp/gemm_pretiled.py` uses
for the same job -- **deadlocks here**, because the runtime releases it once per
dispatch while this worker body runs once per weight row-block group;
`designs/attn_block` survives it only because its body runs exactly once per
dispatch. Verified on mlir-aie 1.4.2 that the generated core keeps the
`AcquireGreaterEqual` ahead of the parameter load; nothing in the source forces
that ordering, so a toolchain that reordered it would hang the dispatch outright
rather than return wrong numbers, and re-checking it is worth a moment on a
toolchain bump. Details: `.claude/plans/gemm-context-collapse.md`, raw data in
`.claude/plans/decode-run/logs/`.

**Result 2026-09-18 (the dense kind, seven of eight families; Gemma 3 completes the set in code):**
`recipes/dense.py` now emits the `dense` kind's route for every dense-shaped family in the
catalogue (Qwen3, Llama 3, Granite, Phi-3, Qwen2, HunYuan, and Gemma 3's two layer types) --
the C++ side (`step_gemm_block_layer`) already implemented it end to end; the recipe was the
only missing piece for the first six. Paired against the closed engine on one box, one sitting,
a 981-token Qwen3-4B prompt: TTFT **1.98 s closed / 64.55 s open sequential (32.8x) -> 38.82 s
with the route (19.6x)**, argmax and top-5 agreeing 19/19 against the sequential path at
correlation >= 0.9999559, decode unaffected. The route's own instrumentation says why the gain
is 1.66x and not more: the five projection GEMMs are 17 % of prefill and flat per block, exactly
as designed; 74 % is the T single-token attention dispatches, unbatched here (`OPEN-PREFILL-ATTN`
is the fix, and it is now the largest term in dense prefill as well as in decode). Gemma 3 needed
two additions the other six did not: a per-layer-type `attn_kernel` / `attn_args` (its sliding
window layers get their own `dxB_local` sharing `dxB`'s stream, per the acceptance criterion
above) and the `sandwich` / `act` fields for its sandwich norms and GeGLU-tanh, both unit-tested
(`manifest_test.cpp`'s qwen3 and gemma3 fixture blocks, `tests/test_gemma3.py`, all 602 spec
tests and the manifest/block_host/vit C++ tests passing). Details: `.claude/plans/closing-the-kernel-gap.md` §9.

**Result 2026-09-18 (Gemma 3 on hardware, and the dx_attn hang it found):** the first
`dxB` dispatch on Gemma3-4B timed out (ERT state 8), reproducibly, with or without the
sliding-window kernel -- so not the per-layer-type plumbing. `designs/dense/dx_attn.py` had
been copied from `dx.py` before dx.py's og handling changed: it still put core 0's og on the
KV-row fifo and emitted `NHL // HPO` og elements per core, which is **zero** when a core owns
fewer heads than an og element carries. Gemma 3 (NHL 2, HPO 4) and Phi-4-mini (4, 8) are
both such geometries; the six families the route had run on all have NHL >= HPO. dx_attn.py
now matches dx.py (kOGH = min(NHL, HPO) heads per element, one og fifo per core), and every
dense family's dx_attn build key moves. Gates, same box, same sitting, a 1290-token random-id
prompt so the sliding window is past its 1024 rows, block route against the sequential path:
Gemma3-4B all 34 layers **8/8 greedy tokens identical after the prefill, 66.8 s vs 86.6 s
(1.30x)**; 6 layers over all 1290 positions argmax 1274/1290, top-5 1230/1290, min corr
0.99927, 32/32 greedy. The control on the same prompt, Qwen3-4B (hd 128) on its re-exported
set: argmax 1276/1290, top-5 1205/1290, min corr 0.99989, 32/32 greedy, 10.8 s vs 19.9 s at 6
layers -- the same flip profile (top-2 margins of hundredths of a logit, the route's bf16
GEMM), so no regression on the hd-128 path and Gemma's deviation is in family. Phi4-mini, whose
route had never run, now does: 16/16 greedy at 4 layers, 6.0 s vs 12.1 s. The decode gate is
the one that matters for a sliding-window family -- the block route writes the KV rows
`dx_local` decodes against afterward -- and it passes at full depth. The 1.30x against
Qwen3-4B's 1.66x is expected: past 1024 rows the sequential path's local layers already attend
over a capped window, so the route has less to win there. Details:
`.claude/plans/gemma3-block-prefill.md`.

### OPEN-MOE-BATCH: the token-batched expert kernel
**Applies to:** openflowlm-next (`open_kernels/designs/moe_batch/`, `open_kernels/recipes/qwen36moe.py`, `src/open_qwen36/{manifest,core}.cpp`)
**Test category:** manual (needs the NPU; the harness measurement and the full-model check are the artifact, `tests/test_moe_batch.py` documents the procedure); the band offsets, the recipe emission and the manifest schema are unit-tested in `tests/test_moe_batch.py` and `src/open_qwen36/manifest_test.cpp`

The block route's routed experts run on a token-batched kernel instead of
one dispatch per token: a dispatch streams every slot's expert once for up
to `nt` of its tokens, one expert per column with the four rows splitting
its output rows, taking the product transposed (eight tokens as the
mmul's A rows, the weight as its B operand, whose 8 k x 8 rows block is a
q4_1 chunk's raw nibble layout) so a weight tile costs a mask and a convert
and the per-row scales ride along as vectors. `nt` is a whole number of those
eight-row mmul tiles and is declared in the manifest, so the design's `MB_NT`
and the driver's gather / scatter cannot drift apart; every buffer, every core
loop and the `mb_x` / `mb_h` / `mb_y` globals scale with it. The expert pools are read as
packed: a 128-row stripe is two 64-row bands interleaved at k-tile
granularity, so a band is a strided read and no repack exists. A kernel set
carries one `mb` xclbin and one instruction stream per dispatch length
(`gemm_block.moe_batch.kernels`, a binary ladder down from the expert count:
256, 128, 64, 32, 16 and 8 slots for 256 experts), each slot compiled as its
own expert and patched per dispatch (`moebatch`, moeroute2's table with every
expert a placeholder). The route gathers each expert's tokens eight at a time
into `mb_x`, runs the shortest stream that holds the experts still owed tokens,
scatters `mb_y` back with the router weights, and goes round again until every
token is served. The ladder has to be fine because an unused slot is not free:
it streams a real expert's weights and the result is discarded, so the rungs
decide how much of a dispatch is wasted; the shared
expert stays outside it (`OPEN-PREFILL-BATCH`). A set without `moe_batch`,
or `OFLM_OPEN_MOE_BATCH=0`, runs `mx` per token as before.

**Acceptance criteria (unit):**
- The up / gate band tap is sizes [8, 10240] strides [20480, 1] at `(8 e + 2 (b // 2)) STRIPE + (b % 2) BAND`, the down band tap two elements at `POOL_DOWN + e 655360 + (b // 2) 40960 + (b % 2) BAND`, derived from `stripe_transpose`, `std_perm` and `down_perm` themselves (`test_moe_batch.py`).
- The 35B emission: both MoE layer types carry `moe_batch` = streams `mb_s256 / mb_s128 / mb_s64 / mb_s32 / mb_s16 / mb_s8` on context `mb`, args `pool, mb_x, mb_h, mb_y`, `nt` 8; the builds pass `MB_SLOTS`, `MB_NT` (the same number the manifest declares), `MB_HID`, `MB_FF`, `MB_EXPERTS`, `MB_POOL_DOWN`, `MB_POOL_BYTES`; the globals are sized for 256 slots at that `nt` (`test_moe_batch.py`).
- The parser (`manifest_test.cpp`): a stream a `moe_batch` names must exist with patch `moebatch`, its slot count a positive multiple of 8, its x / h / y declared globals; the fixture parses to those six streams on both kinds.

**Procedure:** as `tests/test_moe_batch.py` documents -- the 64-expert harness run (`make_test.py --slots 64`, `compare.py s64`, gate rel_fro <= 5e-3 on y) and the full-model checks of `OPEN-PREFILL-BATCH` steps 3 and 4 with and without `OFLM_OPEN_MOE_BATCH=0`.

**Result 2026-09-12 (Qwen3.6-35B-A3B-NPU2):** the harness at 64 experts PASSes at rel_fro 4.4e-4 (gate 5e-3; every slot's cosine >= 0.999995, every token column's >= 0.99997), 4.27 ms per run = 33 GB/s over the 143 MB streamed. Full model: the 4-layer check against the sequential path is argmax 19/19, top-5 19/19, corr >= 0.9993 per position, and against mx per token corr >= 0.99999. The 1020-token prompt at 40 layers, nothing else on the NPU: prefill **40.0 s -> 24.0 s (39 -> 23 ms/token)**, the same 8-token greedy continuation as mx per token (first token 248068). Per 256-token block the expert stage went 7.1-7.5 s to 1.78-1.87 s: two dispatches per layer (256 slots, then ~95 of the 128-slot stream; 342-363 visits per layer), the 256-slot dispatch 27-32 ms (19 GB/s against the harness's 33: the difference is the context switch and the host memory churn between dispatches, measured by `--bench` and written up in `.claude/plans/prefill-gap.md`, "What a dispatch actually costs"). The block is now GEMM 1.4 s, host 1.8-2.4 s (growing with the window), experts 1.8-2.1 s. At 2582 tokens (the length the closed kernels were measured at) the same run is 64.5 s, 25 ms/token: the expert stage and the GEMMs are flat per block (1.7-1.9 s and 1.38 s) while the host stage grows from 1.36 s in block 0 to 3.72 s in block 9 -- 262 ms per 256 rows of window, and by the last block half of it. Against stock FLM 1.0.2's closed `qwen3_6_moe_npu` (14.3 s at 972 tokens, 21.9 s at 2582) the open path is 1.6x behind at ~1000 tokens and 2.9x at 2582, where before this kernel it was 2.9x and 5.4x. The host stages are the next step (`.claude/plans/prefill-gap.md`). **Through `flm serve` (2026-09-12, kernel set with the two DeltaNet decode changes as well, `k35v4`):** `flm-test --llm` passes -- both answers coherent and on-topic, the follow-up served from the prompt cache, no dispatch error; the same two long prompts prefill in 21.8 s at 972 tokens (22 ms/token, from 41.8) and 71.2 s at 2582 (28 ms/token, from 119.3), client-side time to first token, decode about 8 tok/s. The closed kernels' 14.3 s and 21.9 s make that 1.5x and 3.3x behind.

**Result 2026-09-20 (the core program's shape: one parity at a time):** the
expert dispatch was spending a third of its inner loop spilling registers, and
the fix is a loop order, not an algorithm. A 64-lane f32 mmul accumulator is
256 B and the accumulator file holds about two of them. The kernel kept the
even and odd output rows' accumulators live together and dequantised each
k-block into a third (`se.from_vector(m); mac(se, n, d)`), so the allocator
spilled a whole accumulator per k-block. In the emitted code (`kernel_remarks.py`
and the disassembly's spill count) the inner loop was **88 bundles of which 40
carried a spill or a reload**, against an MII floor of 54.

Taking one parity at a time leaves one output accumulator live, the dequant
temporary fits beside it, and the same loop is **41 bundles with zero spills**,
MII 39. The second pass re-reads the nibbles and the activations out of L1,
which is far cheaper than the spill traffic it replaces. Fully unrolling the
k-block loop on top is worth another 1 %. At 16 slots (31.5 MB of expert
weights), minima of three runs:

| | ms |
|---|---|
| interleaved parities, no loop hints (what shipped) | 0.979 |
| one parity per pass | 0.903 |
| + the k-block loop unrolled | **0.892** |
| the weight stream alone (`MB_NULL_MM`) | 0.779 |

The exposed core time falls **0.190 -> 0.116 ms, 39 %**, and the dispatch is
within 15 % of its own DMA cost. The output is **bit-identical**: a 600-token,
4-layer run matches the previous kernel at every one of 600 positions, because
nothing about the arithmetic or its order changed, only which registers hold it.

On the 35B at 2582 tokens, the two kernel sets run one after the other, the
**expert dispatches themselves go 10.06 -> 8.98 s, 10.8 %**, with every other
stage of the run unchanged to within noise (the host gather 470 -> 465 ms, the
scatter 446 -> 448). Whole prefill 30.1 -> 28.4-29.5 s.

Three things this rules out, each measured rather than argued. **The band
stride is not a lever**: reading the pool contiguously (`MB_CONTIG`) moves the
stream floor 0.779 -> 0.768, 1.4 %. **Unrolling the k-tile loop as well is
not shippable**: it is 3 % faster at 16 slots and then crashes Peano at the
real slot count (`Register not in mBMs`, the aie2p code emitter refusing a
register the allocator picked for the bigger body) -- a compiler crash that
only appears at one `MB_SLOTS` is not worth 3 %. **And the width still does
not pay**: `MB_NT` 16 shares one dequantised weight across two sub-tiles but
needs two accumulators, and the emitted code is the same bundle count per
token as `MB_NT` 8, so the DMA halves and the dispatch does not.

What is left of the core is the mmul: aie2p's native bf16 `mac_dims` is
(4, 8, 8), so `mmul<8, 8, 8>` is emulated and each mac carries eight
`vextbcst` operand builds -- 32 of the inner loop's 154 operations were
operand broadcast. The native shape is the next lever, and it only pays
alongside a wider `MB_NT`, which only pays once the core is cheap.

**Result 2026-09-20 (the slot width, and what this kernel is actually bound by):**
the token slot width became a parameter (`MB_NT`, a whole number of 8-row mmul
tiles; the manifest's `nt`) so that an expert's 1.97 MB could be streamed for
sixteen of its tokens instead of eight. It builds, it is numerically clean, and
**it is not worth turning on**, for a reason worth writing down: this kernel is
core-bound, not DMA-bound, and halving its DDR traffic therefore buys nothing.

The design's own ablations say it directly. At 16 slots (31.5 MB of expert
weights, the same bytes at either width), minima of three runs:

| | `MB_NT` 8 | `MB_NT` 16 |
|---|---|---|
| the streams alone (`MB_NULL_MM`) | 0.862 ms | 0.906 ms |
| plus the nibble unpack and the product (`MB_NULL_DQ`) | 0.871 | 1.522 |
| the whole dispatch | **1.105** | **1.874** |

The weight stream costs 0.86-0.91 ms whichever width, which is the 36 GB/s this
kernel has always read; the whole dispatch at `MB_NT` 8 is already past it. So
at 8 the core work is about balanced with the stream and at 16 it is plainly
exposed: 1.70x the time for 2x the tokens, i.e. 15 % better per token, and the
arithmetic rate is 0.86 TFLOPS (1.06 with the scales dropped) against the 2.2 the
plain-bf16 attention GEMM reaches on the same array.

Sharing the dequantised weight across the sub-tiles is the obvious answer and
**both ways of arranging it are slower**, because two accumulators per parity is
already what the accumulator file holds: innermost sub-tile loop with the
parities interleaved (4 accumulators live) 2.036 ms, one parity per pass (2
live, two passes over the nibbles) 2.506 ms, against 1.874 for redoing the
dequant per sub-tile. All three produce the same output to the last bit.

End to end on the 35B at 2582 tokens the expert stage went 11.24 -> 10.83 s
(4 %) and total prefill did not move (31.09 s against 30.57 at `MB_NT` 8,
inside the box's own drift across three repeats). So the recipe stays at 8 and
the parameter stays: raising it is one line once the core rate is up, which is
the thing actually worth working on.

Two things checked along the way. Rebuilding the six streams at `MB_NT` 8 after
all this gives **byte-identical instruction streams** to the shipped set and a
bit-identical 600-token, 4-layer run, so the parameterisation is a genuine
no-op at the old width. And the harness at 16 experts passes at `MB_NT` 16 with
rel_fro 4.5e-4 (gate 5e-3), every token column's cosine >= 0.99998 -- the wider
slot is correct, it is simply not faster. One trap: every kernel in this design
sizes its buffers off `MB_NT`, so every one of them needs the `-D`. With only
the two product kernels getting it, `mb_zero_ug` cleared half the accumulator it
used to clear and h came back NaN from exactly the bands it had stopped
touching. Details: `.claude/plans/prefill-parity-results.md`.

**Result 2026-09-13 (the read-back and the ladder):** two things were being
thrown away, both found by logging what each dispatch actually carried
(`OFLM_OPEN_MOE_BATCH_LOG=1`) against `--bench` for what it costs alone.

The read-back staged the whole dispatch's `y` -- 20 MB a layer -- into a second
buffer to un-interleave the C tiles, then read all of it back to scatter the
rows into their tokens. A column belongs to exactly one token, so the two
passes are one: reading the tiles straight into the token's row is the same
arithmetic in the same order and drops 40 MB a layer. The read stage went
**185 -> 31 ms** a block and the expert stage 2035 -> 1763, most of the second
number being the host traffic that was slowing the next dispatch.

The ladder was 256, 128, 32, 8. A 256-token block leaves ~306 visits in a layer
(min 292, max 333 over 2048 expert-token pairs), so the driver ran 256 and then
a 128 for the last ~50 -- and a padded slot streams a real expert's 2 MB for
nothing. That was **20.3 %** of every block's expert traffic, about 6 GB. With
64 and 16 in the ladder the same layer closes on a 64: padding **7.2 %**,
expert stage 1763 -> 1625 ms a block. The streams ride the xclbin that is
already there and took 30 s each to build.

The stream rate itself is not the problem it was read as: `mb_s256` alone is
16.3 ms for 256 slots = 63.6 us a slot = **30.9 GB/s**, flat from 32 slots up,
against the lm head's 44 GB/s on contiguous q8. What a dispatch costs beyond
that is ~2.9 ms of context switch and the host traffic around it, unchanged in
shape from 2026-09-12. `.claude/plans/moe-stage-cost.md` has the decomposition
and what is left.

**Result 2026-09-13 (why the stream sat at 31 GB/s):** it was never the strided
band reads. A build whose A taps read the pool contiguously -- same bytes, same
footprint, wrong arithmetic -- comes back at 16.31 ms against the shipped
16.33: no difference. Nulling the core's work through the *same strided taps*
gives 11.36 ms = **44.3 GB/s**, the rate the lm head gets on contiguous q8, so
the DMA was never the limit and the shim split, the two input channels and the
fifo depths are all cleared with it. The kernel is core-bound.

Most of what the core spent was one helper: `mb_rep8` built the mmul's 64-lane
B operand out of 8-lane pieces, which the compiler lowers to a scalar extract
and a push **per lane** -- about seventy per 32-k block, and most of the inner
loop. Building the same vector by halving a 512-bit register and doubling back
up (`concat`, `filter_even` / `filter_odd`) takes `mb_step_ug` from 258 to 168
instructions. 256 slots alone **16.33 -> 14.21 ms (30.8 -> 35.4 GB/s)**, and in
the engine a uniform 1.13-1.16x across the whole ladder: `mb_s256` 16.38 ->
14.26, `mb_s8` 0.64 -> 0.58. About 110 ms off a 256-token block's expert stream.
Output is byte-for-byte identical (`cmp` on the harness's y and h at 16 slots,
not a tolerance), so no requirement moves.

Applying the scales to the summed product instead of to every weight -- the
change this started as, and algebraically the better one -- is a measured no:
the raw partial and the running C need two accumulators per parity and four do
not fit the file, so the kernel has to walk one parity at a time and re-read
the nibbles. 30.4 ms against 16.3. It is ~24x more accurate (rel_fro 2.0e-05
against 4.7e-04) and is recorded in `.claude/plans/moe-stream-rate.md` in case
the register pressure is ever solved. 14.21 ms still stands against an 11.55 ms
DMA floor; the remaining ~2.3 ms is the scale application.

### OPEN-PREFILL-ATTN: the block attention's products on the NPU
**Applies to:** openflowlm-next (`open_kernels/designs/attn_block/`, `open_kernels/recipes/qwen36moe.py`, `src/open_qwen36/{manifest,block_host,core}.cpp`)
**Test category:** manual (needs the NPU; the harness measurement and the full-model check are the artifact, `tests/test_prefill_attn.py` documents the procedure); the recipe emission and the manifest schema are unit-tested in `tests/test_prefill_attn.py` and `src/open_qwen36/manifest_test.cpp`

The block route's full-attention layers run their attention as two bf16 GEMM
dispatches per kv head instead of on the host: the scores `Q K^T` (the kv
head's group of query heads times the block's tokens as rows, the window's
cached K rows as columns) and then `P V`, with the causal row softmax on the
host between them. The kernel is the whole-array bf16 GEMM the embedding
models run on (`npu_offload/gemm_rtp/gemm_pretiled.py`), built with its
runtime loop bounds so one xclbin (`ag`) carries an instruction stream per
256 rows of window up to `attn_block.l_max` for each product
(`kernels_s`, `kernels_pv`); a window wider than that is taken in chunks with
the running max and sum merged across them. The host keeps the q / k norms,
RoPE, the KV-cache write, the causal mask, the softmax and the output gate;
1/sqrt(hd) is folded into Q before the bf16 rounding (exact: a power of two at
every head dim here), and the denominator counts the bf16-rounded P the kernel
multiplies. A set without `attn_block`, or `OFLM_OPEN_ATTN_BLOCK=0`, runs
`host::attention_block` as before.

**Acceptance criteria (unit):**
- The 35B emission: the full-attention type carries `attn_block` = `m` 2048, `hd` 256, `l_max` 4096, args `ag_a, ag_b, ag_c`, streams `ag_s<L>` / `ag_pv<L>` for L = 256 .. 4096 by 256 on context `ag`; the builds pass `AG_M`, `AG_K`, `AG_N` (K = hd, N = L for the scores; K = L, N = hd for the values); the globals are sized for the widest window; the linear type carries none (`test_prefill_attn.py`).
- The parser (`manifest_test.cpp`): `m`, `hd` and `l_max` positive multiples of 256, three args that are declared globals, every window a positive multiple of 256 within `l_max`, `kernels_s` reaching `l_max`, and `kernels_s` / `kernels_pv` covering the same windows; the fixture parses to 16 windows of each on the full type only.

**Procedure:** as `tests/test_prefill_attn.py` documents -- the harness run at L = 2048 (`make_test.py --L 2048`, the two builds, `compare.py s2048` / `pv2048`, gate rel_fro <= 5e-3) and the full-model checks of `OPEN-PREFILL-BATCH` steps 3 and 4 on a prefix with a full-attention layer, with and without `OFLM_OPEN_ATTN_BLOCK=0`, then `oflm-test --llm` through `oflm serve` with `OFLM_OPEN_GEMM_BLOCK=1`.

**Result 2026-09-12 (harness, Qwen3.6-35B-A3B shapes):** both products PASS at L = 2048 -- rel_fro 1.1e-7 (scores) and 6.9e-7 (values) against fp64, 0.95 ms per 2.15 GFLOP dispatch (2.2 TFLOPS), the two shapes' `final.xclbin` 72 bytes apart (the UUID). Forty dispatches a block, about 40 ms, for the products `host::attention_block` spent about 2.5 s on at 2582 tokens. **Full model, 2026-09-12 (Qwen3.6-35B-A3B-NPU2, the set with the 32 attention streams, 40 layers, clean box):** the 8-layer prefix on the 19-token prompt agrees with the host attention on argmax 19/19 and top-5 19/19, corr >= 0.99999 per position (max |diff| 4e-2: bf16 products and a bf16 P, not bit-exact by design). The 1020-token prompt: **21.8 s either way** (21 ms/token; the host stage 1.44 -> 1.47 s per block on the NPU path against 1.37 -> 2.13 s on the host -- attention is only a fifth of that stage at this length), the same 8-token greedy continuation. The 2582-token prompt: **66.0 s -> 54.6 s (26 -> 21 ms/token)**, the same 8-token continuation, the host stage flat at 1.23-1.37 s per block where the host attention grew it from 1.37 to 3.81 s; the GEMM column grows 1.33 -> 1.48 s with the attention dispatches and their context switches. The route is now flat per token with length; what remains per block is the per-token MoE dispatches (~2.0 s), the projection GEMMs (~1.45 s) and the host DeltaNet (~1.0 s of the host stage). Logs: `.claude/plans/decode-run/logs/long{1020,2582}_attn{0,1}.log`, `gate_attn_v5.log`. **Through `oflm serve` (2026-09-13, `OFLM_OPEN_GEMM_BLOCK=1`):** `oflm-test --llm` passes (both answers coherent, the follow-up from the prompt cache); the two long prompts prefill in 20.5 s at 972 tokens (from 21.8) and 59.0 s at 2582 (from 71.2), client-side time to first token. **Both engines re-measured paired on a quiet box, 2026-09-13**, the same script and the same two prompts back to back (`.claude/plans/decode-run/logs/serve_closed.log`, `paired_open.log`): open 19.3 s and 56.0 s (19.9 and 21.7 ms/token), decode 8.0 tok/s; stock FLM 1.0.2's closed kernels 11.9 s and 18.5 s (12.2 and 7.2 ms/token), decode 15.4 tok/s -- **1.6x behind at 972 tokens, 3.0x at 2582, 1.9x at decode**. The closed engine is faster than the 2026-09-11 figures recorded elsewhere in this spec (14.3 s, 21.9 s, 12 tok/s), so those were taken under load and every ratio computed against them flatters the open path; use the paired numbers.

### OPEN-REQUEST-ISOLATION: the same request on a reused engine gives the same tokens
**Applies to:** openflowlm-next (`src/open_qwen36/core.cpp`: `read_back`, `Core::reset`, `seek`, `route`, `arm_route_records`, `det_step`; `cli.cpp` `--repeat`, `--det-step`)
**Test category:** manual (needs the NPU and a resident 35B)

`oflm serve` keeps one resident `Core` across requests. Run twice on it, the same
request must produce the same tokens and bit-identical logits: nothing a request
leaves behind (recurrent state, KV rows, act scratch, patched instruction
streams, position records) may reach the next one, and no host read may return
a buffer's contents from before the dispatch that wrote it.

The host reads one thing back in the middle of a decode step: the router's
top-k (`idx`, 32 bytes of the record the layer's first dispatch writes into
`act`), which it patches into the second dispatch's expert fills (`moeroute2`).
A completed wait on that dispatch did not guarantee the host saw the record: in
some periods about one read in 3000 returned the previous step's `idx`, while a
re-sync microseconds later returned the new one. The stale list is a valid one,
so nothing failed -- the step ran the wrong experts and the tokens diverged a
few steps later.

What a read is on this driver. `xrt::bo::sync(FROM_DEVICE)` on the Windows NPU
driver (its user-mode shim, `xrt_core.dll`; the XRT headers do not say) is,
for any range smaller than the L3 cache, a CLFLUSH loop over every 64-byte line
the range touches -- the start rounded down, the partial last line included --
with no fence after it; larger ranges go to the kernel driver. Two things
follow. Rounding a sync out to whole cache lines changes nothing (the loop
already covers them). And CLFLUSH is not ordered against later loads (AMD's
manual requires an MFENCE after it for that), so a load that follows the sync
may still be served from the stale cached line. Which of two mechanisms
produced the stale reads -- that missing fence, or the dispatch's final write
reaching memory after its completion is reported -- is not settled: the fault
did not recur on demand (see the result). The engine therefore closes both:

1. Every device-to-host read in `core.cpp` goes through `read_back()`: the
   sync, then an MFENCE. `OFLM_OPEN_READ_FENCE=0` drops the fence, for the A/B
   only.
2. Each step arms every `idx` slot with a sentinel (`0xFFFFFFFF`, with no
   dispatch outstanding), and `route()` re-syncs until the dispatch's record has
   replaced it, throwing if it has not after 1 s. This is what turns any
   remaining late read into a wait (or a hard error) instead of wrong experts.
   `OFLM_OPEN_ROUTE_SENTINEL=0` restores the old single read, for the A/B only.

**Verification (manual):**
1. `open_qwen36_cli --kernels <set> --ids 248045 --at-position 1024 --max-tokens 24 --repeat 3`:
   every request must print `REPRODUCED` (exit 0).
2. The tighter probe: `--ids 248045,846,220,95772,2005,95815,828,95726,110004,95849,3709,96674,101697,97785,2005,95815,828,95999,2005,95933,828,95726,104062,248046 --at-position 1024 --det-step 100`
   (one resident engine, the 24-step request forced 100 times, every step's
   logits compared bit for bit with a reference pass): `0 of 100 reps
   differed`. It prints `late router reads caught: N of M`; N > 0 means the
   fault happened and the sentinel absorbed it.
3. The failure is intermittent -- it came in periods (hours of zero, then about
   a third of the probe's reps). A clean run of step 2 with
   `OFLM_OPEN_ROUTE_SENTINEL=0 OFLM_OPEN_READ_FENCE=0` means the box is in a
   quiet period, and says nothing about the fix until one without it fails.
4. A dispatch in flight when Windows enters modern standby does not complete
   (`ERT state 8`, a `pci` Event ID 3 at wake). Discard any run whose window
   contains a Kernel-Power 506/507 pair; it is not this requirement's failure.

**Result 2026-09-22 (Qwen3.6-35B-A3B-NPU2, `k35main`, 40 layers).** In a failing
period, the old single read: 68 of 220 probe reps diverged, first at steps 1-23,
never at step 0. Widening the read to the whole 1088-byte record, alternated
with it in the same period: 0 of 100. `OFLM_ROUTE_CHECK=1` caught one read
changing on a re-read (layer 29, position 1043: an old expert list, then the new
one on an immediate re-sync).

**Result 2026-09-23 (same model and set, upstream main + this change).** The
fault did not recur. `--det-step 40` runs: the old read (sentinel and fence
off) 0 of 200 reps diverged -- 40 on a quiet box, 80 under four memory-copy
processes, 80 under a loop of four parallel WSL kernel builds; the fence alone
0 of 120; the full fix 0 of 160, with 0 late reads caught in 161,280. So the
2026-09-22 period is still the only evidence the fault exists, and the fix is
justified by it and by the mechanism above, not by a live catch. Bit-exact
against main (19 ids, 4 layers: max |diff| 0.0 at all 19 positions), `--repeat
3` at position 1024 reproduced, decode unchanged (three alternated pairs at
position 1024: main 125 / 121 / 120 ms/token, this change 123 / 121 / 122).
Detail: `.claude/plans/decode-gap-2026-09-22/pr1-route-read.md`.

### OPEN-DECODE-PIPELINE: the decode route keeps a dispatch in flight across the layer boundary
**Applies to:** openflowlm-next (`src/open_qwen36/core.cpp`: `Core::step_impl`, `start_run`, `wait_run`, `bench_step`)
**Test category:** manual (needs the NPU and a resident 35B; the two bit-exact dumps and the paired step timings are the artifact)

A decode step issues one hardware command per half-layer, and serially the array
is idle for a host turnaround at each of them. The route instead starts layer
l + 1's FIRST dispatch before waiting on layer l's LAST one, so the device takes
it up the instant the previous command retires. The rule it must not break:
**a kernel's instruction stream is patched only when no run on that kernel is
outstanding.** Instruction BOs are per kernel NAME and shared by every layer, so

- the router's top-k patch (`moeroute2`, on `lx1` / `ax1`) happens only after the
  previous layer's second dispatch -- the same kernel object whenever the two
  layers share a type -- has been waited on, and after this layer's first
  dispatch has been waited on, because the patch reads the router record that
  dispatch wrote into `act`;
- the per-token `attnpos` patch (on `ax0`) happens once at the top of the step,
  with nothing outstanding at all.

`xres` is the one buffer every layer writes and the next layer's first dispatch
reads, and nothing on the host orders those two commands. `xrt::run::start()` is
documented as asynchronous, XRT's own `xrt::fence` exists precisely "to
synchronize operations between run objects", and no ordering of two outstanding
runs is promised anywhere in `xrt_kernel.h`; the ordering is therefore an
EMPIRICAL property of one hardware context's command queue, and the gate below
is its proof -- a reorder would read a stale `xres` and every logit after it
would move.

`OFLM_OPEN_SUBMIT_AHEAD` selects the schedule: `1` (the default) queues ahead
only when the two dispatches run in the SAME hardware context, `0` restores the
serial `start(); wait();` route, and `2` -- which crosses contexts -- **hangs the
array** and exists only as the probe that established that. A layer whose program
is a single dispatch (the dense families) queues that dispatch itself behind the
previous layer's; the deepstack path (Qwen3-VL reads `xres` back between layers)
runs serially.

**Verification (manual):**
1. **Bit-exact, 19 ids, 4 layers, sequential.** `open_qwen36_cli --model <35B>
   --kernels <set> --pmode performance --layers 4 --ids
   248045,846,198,760,28758,8427,4821,303,411,20012,369,264,2526,1287,314,4471,34523,440,836
   --max-tokens 1 --prefill-logits --dump-logits <prefix> --quiet`, once with
   `OFLM_OPEN_SUBMIT_AHEAD=0` and once with the level under test, compared with
   `decode-run/cmp_exact.py`: **max |diff| 0.0 at every position.**
2. **Bit-exact, a 64-token greedy continuation.** The same binary on
   `gap-table/ids_1122.txt` with `--gemm-block --max-tokens 64 --dump-logits`,
   the `_t<i>.bin` dumps compared position by position: the same 64 token ids and
   **max |diff| 0.0** at every one.
3. **Timing.** `--bench-decode N` prints the serial per-kernel table and the sum
   of dispatch minima; `--bench-step N` then sweeps the schedules INTERLEAVED
   (rep i takes each level in turn) over the real step and prints min / median /
   mean of its wall time. The step's wall time and the sum of its dispatches'
   own times are reported separately: once dispatches overlap, a queued-ahead
   command's start-to-done includes the time it waited behind the one in front,
   so the two stop being the same number. Real decodes are run ALTERNATED
   (off, on, off, on, ...), never one run each.
4. The engine logs `host threads: N (OMP_WAIT_POLICY=PASSIVE)` at construction;
   a WARNING on that line invalidates every timing above.

**Result 2026-09-22 (Qwen3.6-35B-A3B-NPU2, 40 layers, `sets/k35main`).** Both
gates PASS: the 19-id 4-layer dump and the 64-token continuation are **max |diff|
0.0 against the baseline's reference dumps**, with and without the pipeline, and
the 64 token ids are identical. On a quiet box, `--bench-step 20` over two
alternated rounds puts the pipeline **~3 ms of median step time** ahead of the
serial route (at positions 1 and 1024; about 105 against 108 ms at position 1).
Level 1 ran a soak (three requests on one resident engine, ~1,400 pipelined
layer boundaries), every gate run and the integration benches with no `ERT state`
line anywhere.

**What it cannot reach.** The host gap between a dispatch returning and the next
starting is 0.02-0.07 ms, so the host time a step spends outside its dispatches
is a few ms in total: that is the ceiling for any submit-ahead scheme. The larger
term is the lx <-> ax hardware context change, 22 per step. Queueing ahead across
that change (level 2) hung the array three times in three runs
(`ERT_CMD_STATE_TIMEOUT`), so no host schedule hides it; one xclbin carrying both
layer types does.
