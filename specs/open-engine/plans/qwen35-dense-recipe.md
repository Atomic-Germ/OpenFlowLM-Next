# Plan: Qwen3.5 dense on the open kernels — a sixth family, composed from two we have

**Status:** designed 2026-09-06 (Fable), implementation delegated. Not built, not run.
**Spec impact:** one new requirement `OPEN-FAMILY-QWEN35`; two new pack ops
extend `OPEN-PACK-PLAN`; new catalogue points under `OPEN-OP-RANGE` after the
hardware pass; `lm_head_q8` gains a K knob.
**Why it is first in line:** 19 published models, ~23.5 k downloads/month
(`.claude/plans/open-kernel-model-priority.md`, rows 5, 6, 8), including the
single most-downloaded model on Atomic-Germ's Hugging Face.

## The architecture, and what we already have

A Qwen3.5 dense layer is a Qwen3.6-MoE layer with the MoE block replaced by a
gated dense FFN. Every other field matches the validated 35B-A3B exactly:
`layer_types` (3 linear_attention : 1 full_attention), `attn_output_gate`,
head_dim 256, partial RoPE 64, 16/32 linear key/value heads of 128, conv
kernel 4, vocab 248320, q8 lm_head. So:

| block | design that has it | validated at |
|---|---|---|
| layer-entry norm, gated DeltaNet (qkv/z GEMV → glue → S update → post) | `designs/layer_x/lx.py` part 0 | HID 2048 |
| gated full attention (q/gate/k/v GEMV → attn helper → o GEMV) | `designs/layer_x/ax.py` part 0 | HID 2048, 16/2 heads |
| gated dense FFN (up \| gate per band → act → down) | `designs/dense/dx.py` main body, steps 6–7 | HID 2560 / 4096, FF 9728 / 14336 |
| norm + residual helper without a router | `designs/dense/dx.py` `ln_body` | 2560 / 4096 |
| q8 lm_head | `designs/lm_head_q8` | K = 2048 only (hardcoded) |

The composition: `lx` / `ax` with their MoE tail swapped for the dense FFN tail,
their ln+router helper swapped for the plain ln helper, and — since nothing is
routed — **one instruction stream per layer type** instead of two (the cores
never knew about the part boundary; `dx.py` already runs a whole layer in one
dispatch). Program per layer: `run lx` (linear) or `run ax` (full, `attnpos`
patch). The engine needs no C++ change for the programs; it needs the two pack
ops below.

## Sizes, in the order to do them

| shape (hid / L / ffn / heads / kv) | models | HF 30d | new things |
|---|---|---|---|
| **9B** 4096 / 32 / 12288 / 16 / 4 | Qwen3.8-Distilled-9B, Ornith-1.0-9B, Qwen3.5-9B-Claude-*, Qwopus, Qwythos, Qwable, … (12) | 14 620 | K = 12288 GEMV; lm_head_q8 K = 4096; attn 16/4 HD 256 gated; `PER_CALL 1` in the layer_x fabric |
| **4B** 2560 / 32 / 9216 / 16 / 4 | Qwen3.8-Distilled-4B, NuExtract3-4B, Qwen3.5-4B (3) | 8 347 | HID not a multiple of 4 KB elements in lx/ax (dx.py's element-index prep, ELN-sized ln/glue elements); K = 2560 / 9216 (validated in dense); lm_head_q8 K = 2560 |
| **2B** 2048 / 24 / 6144 / 8 / 2 and **0.8B** 1024 / 24 / 3584 / 8 / 2 | Qwen3.8-Distilled-2B, Qwen3.5-2B, Qwen3.5-0.8B (3) | 574 | **16 linear value heads** (qkv 6144, `ssm_a[16]`) — a new DeltaNet point (catalogue has heads = 32 only); attn 8/2; lm_head_q8 K = 2048 / 1024 |

9B first: HID 4096 is a whole number of 4 KB elements everywhere (xn = 2
elements, VW = 4096, og = 2 elements), so it is the *cleanest* composition and
the biggest prize. The 4B then adds the fractional-element machinery, the small
ones the 16-head DeltaNet.

## What the container actually holds (probed 2026-09-06, all sizes)

Names are `model.layers.N.` (the MoE container uses `model.layer.N.`). Every
projection is q4_1 / 5120 **except three tensors per linear layer**:

| tensor | dtype / chunk | note |
|---|---|---|
| `linear_attn.ssm_out_proj.weight` | **I8 / 8704 (q8)** — 9B: [128, 16, 8704] = 4096×4096 | the MoE stores this one as q4_1 |
| `linear_attn.ssm_alpha_proj.weight`, `ssm_beta_proj.weight` | I8 / 8704 (q8) [1, hid/256, 8704] | 32 × hid values |
| `linear_attn.ssm_alpha_proj.bf16.weight`, `ssm_beta_proj.bf16.weight` | BF16 **[32, hid]** | the MoE stores `[hid, 32]` (what `glue_ab` reads) |

Everything else — `qkv_proj`, `self_attn.{q,k,v,o,gate}_proj`, `mlp.{up,gate,down}_proj`,
the norms, `ssm_conv1d [4, qkv_dim]`, `ssm_norm [128]`, `ssm_a`, `ssm_dt.bias`,
`embed_tokens` BF16, `lm_head` q8 — is the layout the two existing recipes
already pack.

**Decision: the q8 out projection is re-quantised to q4_1 on the host at pack
time** (`requant_q4_1`, a new pack op in both packers). Reasons: it is the only
q8 GEMV in the model, the main cores have no q8 GEMV entry point, and the 35B
already runs this exact projection at q4_1. The quality cost is measured, not
assumed (risk R3 below), and a main-core q8 GEMV is the follow-up if it matters.
Alpha/beta come from the `.bf16.weight` copies through a `transpose` pack op
(host-side, like `conv_transpose`), so the glue kernel is untouched.

## Risks to retire before building anything (all without hardware)

- **R1 — `PER_CALL 1` in the layer_x fabric.** The 9B's down GEMV table is
  2.25 × 12288 = 27.6 KB; with 10 KB weight elements the main core's L1 sums
  past 60 KB (`dense.per_call`), so the design must use 5 KB elements like
  Llama 8B. In `layer_x` the DeltaNet S slices are streamed on the same w fifo
  (`DN_ROWS = CALL_BYTES / (dim*4)` — 20 rows today, 10 with 5 KB), and
  `moe_sequence` / `dn_sequence` / `gen_kernels.py` derive their counts from
  `CALL_BYTES`. Confirm nothing in `dnx_*.cc` / `dnx.h` hardcodes 20 rows.
  If something does, the fallback is to keep 10 KB elements and split the
  down GEMV's K into two accumulating passes — say so before building.
- **R2 — 16 KB program memory** on the main core. The FFN tail has fewer call
  sites than the MoE tail (gms + act + gy vs hdr/gup/silu/prepf/gdown/accfin/out),
  so it should shrink; only the build tells.
- **R3 — re-quantisation cost.** In fp64 on one 9B linear layer: `out_proj·og`
  with the q8 weights vs the re-quantised q4_1 weights, over a few hundred
  random `og` vectors — report the correlation and relative error. Also report
  it on the whole 4-layer slice reference (q8 vs requant) so we know what the
  acceptance threshold is measuring.
- **R4 — alpha/beta orientation.** Dequantise the q8 `[1, hid/256, 8704]`
  tensor, compare with the `.bf16.weight [32, hid]` copy both ways; confirm which
  is `[head][hid]` so `transpose` writes exactly the `[hid, 32]` layout the 35B
  packer copies verbatim.
- **R5 — `ln` helper element sizes.** `lx.py` / `ax.py` hardwire 4 KB elements
  for the norm helper and the glue's xn copy (`u8_4k`, `fxn`); `dx.py` sizes
  them `ELN = HID*2`. At HID 4096 the f32 x is 16 KB = four 4 KB elements —
  check the fused `ln_fn(x0, x1, a0, a1, w, …)` five-input form fits (Llama 8B
  hit exactly this and split the outputs: "8 KB norm elements … one element
  per call"). Follow whatever `dense.py` decided for 4096.

## New requirement

### OPEN-FAMILY-QWEN35: Qwen3.5 dense on the open kernels
**Applies to:** openflowlm-next (`open_kernels/recipes/qwen35.py`, `spec.py`,
`designs/layer_x/lx.py`, `ax.py`, `xcommon.py`, `designs/lm_head_q8`,
`recipes/pack.py`, `src/open_qwen36/pools.cpp`, `manifest.cpp`, `model/replica_qwen35.py`)
**Test category:** manual (needs the NPU and a Qwen3.5 container); the
derivation, the composed layout and the two pack ops are unit-tested in
`tests/test_qwen35.py`

A Qwen3.5 dense model (gated DeltaNet linear-attention layers with a gated
full-attention layer every fourth, a silu-gated dense FFN, q8 lm_head, `model_type`
`qwen3_5` or `qwen3_5_text`) shall run on the open kernels from its
`config.json` alone: the `qwen35` recipe composes the qwen36moe recipe's
attention half (Layout / Common / Linear / Attn of `qwen36moe.py`, minus the
MoE, router and shared-expert fields) with the dense recipe's FFN half
(`POOL_UP / POOL_GATE / POOL_DOWN`, `A_XM / A_H / A_OUT2`, `MS_U / MS_G`,
`per_call`), and `lx.py` / `ax.py` build with the FFN tail and the plain norm
helper selected by the recipe (`R.ffn == "dense"`), one instruction stream per
layer type. The q8 `ssm_out_proj` is re-quantised to q4_1 by the packer
(`requant_q4_1`); alpha/beta are read from their bf16 copies through
`transpose`. Images are refused as on the other VLM families.

**Acceptance criteria (unit):**
- `ModelSpec.from_hf_config` on the 9B / 4B / 2B / 0.8B configs (fixtures under
  `tests/fixtures/`) gives family `qwen35`, `num_experts 0`, `intermediate`
  12288 / 9216 / 6144 / 3584, the MoE's layer pattern, 16/4 (or 8/2) heads,
  `lin_value_heads` 32 (or 16); `qwen3_5_moe` still derives to `qwen36moe`.
- The composed layout for the 9B: every DeltaNet / attention constant equals
  `qwen36moe.recipe()`'s for a 35B spec with `hidden = 4096` (the attention half
  is the MoE's, not a re-derivation); the pool holds q4-sized `up | gate | down`
  at the offsets `dense.layout()` would give; `PER_CALL 1`; no `moe` block, no
  `rout`, no router consts in the manifest; programs are one `run` per layer
  type with `attnpos` on the full-attention kernel only.
- `requant_q4_1` on a synthetic q8 chunk set: the output is 5120-byte q4_1
  chunks whose `nib·d + m` reading is the optimal per-32-block q4_1 of the
  dequantised q8 values (max-abs error ≤ d/2 per block, d = (max−min)/15), and
  the NumPy and C++ packers produce identical bytes.
- `transpose` on a `[32, hid]` bf16 tensor gives the `[hid, 32]` bytes the MoE
  packer's `put` copies for the 35B; NumPy and C++ agree.
- The manifest fixture for the 9B parses in `manifest_test.cpp` (the C++ reader
  accepts a linear-attention layer type with a one-step program and no `moe`).

**Procedure (manual):** as OPEN-FAMILY-QWEN36MOE with `Qwen3.8-Distilled-9B-NPU2`
(`--model-dir`), `out_q35`, an 8-layer slice (six linear, two full), 3 greedy
tokens from `[248045]`; then the engine CLI, then `chat.py` (the Qwen template).
Thresholds as the 35B: logits corr ≥ 0.99999 vs the replica **fed the same
re-quantised out_proj**, same argmax and top-5, residual corr ≥ 0.9999 every
layer. Separately reported: the same slice's logits corr vs the replica fed the
q8 out_proj (the quality cost of R3).

## Changes

| file | change |
|---|---|
| `recipes/spec.py` | `_qwen35_hf` for `qwen3_5` / `qwen3_5_text` (reuse `_qwen36moe_hf`'s field reads; `num_experts 0`, `intermediate = intermediate_size`, family `qwen35`); GGUF `qwen3_5` if the metadata keys are known, else refuse by name. `HF_FAMILIES`, `hf_model_types`. |
| `recipes/qwen36moe.py` | factor `common()` / `layout()` so the DeltaNet and attention geometry can be built without MoE fields (a `ffn="moe"|"dense"` parameter or a split into `attention_layout()` + `moe_layout()`), **without moving a single 27B offset** — `test_recipe_layout.py`'s frozen dicts are the guard. |
| `recipes/qwen35.py` (new) | `recipe / layout / pack_plan / programs / builds / hf_config_check / manifest_layout / KERNEL_SOURCES / GEN_KERNELS` — the family surface `families.py` expects. Pack plan: MoE-style ops for qkv/z/gate/q/k/v/o and the linear consts, dense-style `std_perm` for up/gate/down, `requant_q4_1` for `ssm_out_proj` into `C_WOUT`, `transpose` for alpha/beta, `lmhead_q8` for the head. `_check` requires the points listed above (new ones fail until validated; `OPEN_KERNELS_UNVALIDATED=1` for the hardware pass). |
| `recipes/families.py`, `catalogue.py` | register `qwen35`; catalogue entries only after the NPU pass (the handoff lists them). |
| `designs/layer_x/lx.py`, `ax.py`, `xcommon.py` | read `R.ffn`; when dense: `main_body` tail = `X.dense_ffn_body` (port dx.py's up/gate/act/down loop into xcommon, using the same `gms` / act / `gy` entry points dx generates), helper = `X.ln_body` without the router stage, host sequence steps 7–8 = dx.py's steps 5–7 (norm + residual → FFN → residual out), `PART` ignored (one stream), build dirs `layer_x/build_{family}_{lx|ax}_h{hid}`. Element sizes for the helper / glue from the recipe (`ELN`), prep by element index (`f_prep(xe[i], tab, HID, i)`) when HID is not a 4 KB multiple. The MoE path must build byte-identical to today (`--check` against `src/xclbins/Qwen3.6-35B-A3B-NPU2/open_kernels`). |
| `designs/layer_x/gen_kernels.py`, `designs/dense/gen_kernels.py` | generate the FFN TUs for the layer_x fabric (act kernel, `gms`, prep kernels at the new K set). One entry point per `.cc` (duplicate-symbol trap). |
| `designs/lm_head_q8/lm_head_q8.py` (+ `.cc`) | `K` from `LMHEAD_K` (default 2048); `qwen36moe.builds` passes it too. |
| `recipes/pack.py`, `src/open_qwen36/pools.cpp`, `manifest.cpp` | `requant_q4_1` (q8 chunks → q4_1 chunks → `std_perm` order into the destination; a q8 chunk holds 8192 values in the same 32-row × 256-col tile as a q4 chunk, so the row/col law is unchanged) and `transpose` (`rows`, `cols`, element bytes). The parser refuses a `requant_q4_1` without `nch`/`in_dim` like it refuses a bare `std_perm`. |
| `model/q4nx.py`, `model/replica_qwen35.py` (new), `model/make_decode.py`, `compare_decode.py` | general `dq_chunks_q8`; the replica composes `replica.linear_decode` / `attn_decode` with `replica_dense`'s FFN, taking the out_proj either re-quantised (the acceptance reference) or q8 (the quality reference) by flag; make_decode picks the replica by family. |
| `src/open_qwen36/chat.py`, `manifest_test.cpp`, `tests/test_qwen35.py`, `tests/fixtures/manifest_qwen35_9b.json`, `make_fixtures.py`, `spec.md`, `README.md` | the usual family additions (see commit `e971ccc4` for the file set). |
| `src/common/AutoModel/modeling_qwen3_5*.cpp` | engine selection for family `qwen3.5` (the app's `Qwen3_5VL` class) — **after** the wiring stream (W) lands its helper; not part of this plan's first cut. |

## Order

1. R1–R5 (a few hours, Windows python + the 9B container downloaded to
   `~/.flm/models/Qwen3.8-Distilled-9B-NPU2`); write the findings at the top of
   the handoff file. A red R1 stops here.
2. `spec.py` + `qwen35.py` + the qwen36moe refactor, TDD against
   `test_qwen35.py` and the frozen `test_recipe_layout.py`; manifest fixture;
   `manifest_test.cpp`.
3. Pack ops in both packers with the byte-equality test.
4. The design changes (`lx` / `ax` / `xcommon` / gen_kernels / lm_head_q8), kept
   buildable for the MoE path — `--check` byte-identity on the 35B is the
   regression test the hardware stream runs before anything else.
5. Replica + make_decode + compare_decode.
6. Handoff for the hardware stream: the exact export / slice / CLI / chat
   commands for the 9B, the catalogue lines to add on success, the R3 number
   to record beside the result.

## Not in this plan

Vision (the models are VLMs; images route to the closed engine, issue #16),
prefill, MiniCPM-V-4.6's tower (its text tower is the 0.8B shape; the family is
not in `all_models.hpp`), a main-core q8 GEMV (follow-up if R3 says the
re-quantised out_proj costs real quality).
