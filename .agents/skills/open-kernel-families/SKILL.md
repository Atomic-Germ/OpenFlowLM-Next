---
name: open-kernel-families
description: Reference for the open-kernel family recipes and designs (dense qwen2/qwen3/llama3/gemma3/hunyuan/granite/phi3/k2, qwen35, qwen36moe, lfm2), their specs and geometry traps. Use when a build in one family fails, when adding a size whose geometry is unusual, or when touching recipes/*.py or designs/dense|layer_x.
---

# Families, recipes, and designs

Read `open-kernel-pipeline` first. That is the path: geometry decides, a spec
names it, a recipe turns it into a manifest, and `oflm add` / `build-new-spec`
run the build and the gates. **This file is the reference for what is
different inside each family** — the durable traps, not a list of commands to
run by hand.

## The map

| family | recipe | designs | specs | shape |
|---|---|---|---|---|
| qwen2, qwen3, llama3, gemma3, hunyuan, granite, phi3, k2 | `recipes/dense.py` | `designs/dense/dx.py`, `dx_attn.py`, `ln`, `lm_head_q4` | `qwen25-3b`, `qwen3-*`, `llama3*`, `gemma3-*`, `hy-mt2-*`, `granite42-*`, `phi4-mini-4b`, `nanbeige41-3b`, `k2-horizon-*` | one dispatch per layer, GQA + gated FFN |
| qwen35 | `recipes/qwen35.py` | qwen36moe's `layer_x` + dense FFN tail | `qwen35-*` | 3 linear : 1 full, gated DeltaNet, dense FFN |
| qwen36moe | `recipes/qwen36moe.py` | `designs/layer_x/{lx,ax,ux}.py`, `ln`, `lm_head_q8` | `qwen36-35b-a3b` | MoE, two dispatches per layer |
| lfm2 | `recipes/lfm2.py` | `designs/short_conv` + dense's attention block | `lfm2-*` | 10 short-conv : 6 full-attention |

`recipes/families.py` is the registry: `FAMILIES` lists what has a recipe, and
`NOT_IMPLEMENTED` names the architectures that deliberately do not (with the
reason). A family not in either has no recipe and cannot be routed to the
nearest one — that is how a model gets kernels that drop a whole stage of its
layer and still "match" a replica making the same mistake.

## Dense recipe (`recipes/dense.py`)

Covers `qwen2`, `qwen3`, `llama3`, `gemma3`, `hunyuan`, `granite`, `phi3`
and `k2` (`DENSE_FAMILIES`). The layer is:

```
ln -> gemv q | k | v -> attention ([q/k RMSNorm,] RoPE, no gate) -> gemv o
   -> [post-attention ln] -> +residual -> ln -> gemv up | gate -> act(gate) * up -> gemv down
   -> [post-FFN ln] -> +residual
```

- `designs/dense/dx.py` is the fused full layer. One xclbin, one instruction
  stream per layer, 8 main cores plus the `ln` helper and the attention helper.
- `designs/dense/dx_attn.py` is dx.py's attention phase as its own dispatch,
  the T-token prefill split. It carries `ATTN_QKV_BIAS` (Qwen2) and
  `ATTN_PTAB_SPLIT` (a position record wider than one element). Non-bias,
  single-record families take the byte-identical pre-port path.

Family differences live in the recipe/spec, not in a fork of the design:

| family | difference |
|---|---|
| qwen3 | q/k RMSNorm, eps 1e-6 |
| llama3 | no q/k norms, eps 1e-5, llama3 RoPE frequency scaling |
| gemma3 | GeGLU-tanh, sandwich norms, two layer types (`dense_local` sliding window + global) |
| hunyuan | q/k RMSNorm applied **after** RoPE (`QKNORM_POST_ROPE`) |
| qwen2 | per-channel q/k/v bias (`QKV_BIAS_FAMILIES`, `ATTN_QKV_BIAS`) |
| granite | four scalars folded into the weights by `q4nx-build` (see below) |
| phi3 | 96-of-128 rotation and longrope |
| k2 | GroupRMSNorm and split-half RoPE |

**The re-export contract.** `dx.py` reads `chunk_bytes(quant)` and
`band_bytes(K, quant)` off the *family module*, not through `dense`. A new
dense-family recipe must re-export them:

```python
band_bytes = D.band_bytes
chunk_bytes = D.chunk_bytes
```

Forgetting this is the `AttributeError: module 'recipes.lfm2' has no attribute
'chunk_bytes'` that kills every `[dx]` / `[dx_attn]` panel in the export.

**Prefill routes.** A dense family may carry a block prefill route in its
manifest (`gemm_block` / `attn_block`):

- `BLOCK_ATTN_QKNORM_ROPE = ("qwen3",)` — q/k RMSNorm then half-split rotation.
- `BLOCK_ATTN_ROPE = ("k2",)` — rotation only.
- The 8-bit `dit_gemm` route (`DIT_T = 1024`) is the `fast` variant; the q4_1
  route (`T = 256`) is `lean`. A shape that does not fit (`K % 512`,
  `N % 1024`) or a family with no `attn_block.prep` gets no `dit` route.

## Qwen3.5 dense (`recipes/qwen35.py`)

A Qwen3.5 layer is a Qwen3.6-MoE layer with the MoE block replaced by a dense
silu FFN. The module composes the MoE's `layout` / `common` / `linear` / `attn`
with `ffn="dense"` and adds the FFN pool block and `h` / `out2` act stages.
What is the MoE's stays the MoE's: the 3 linear : 1 full pattern, gated
DeltaNet, the attention output gate, head_dim 256 with a 64-dim partial RoPE,
16 linear key heads of 128, conv kernel 4, the q8 lm_head.

Container facts that matter:

- `ssm_out_proj` is q8 / 8704 in the container but is requantized to q4_1 on
  the host (the main cores have no q8 entry point).
- `ssm_{alpha,beta}_proj` are q8 with a bf16 `[heads, hidden]` copy beside them.
  The MoE stores `[hidden, heads]`; `glue_ab` reads the transposed copy.
- The 27B has 48 value heads over 16 key heads. llama.cpp tiles the value heads
  (`GGUF head r*num_k + kh = HF head kh*grp + r`). A wrong until with `grp = 2`
  is right for 32-over-16 and wrong for 48-over-16. `q4nx/models/qwen35.py`
  (`v_untile`, `untile_qkv`, head counts from `qwen35.ssm.*`) pins it, and
  `tests/test_qwen35_vheads.py` covers it.
- Before any chat test on a new container, check the container against the
  author's safetensors:
  `python open_kernels/model/container_vs_hf.py --model-dir <container> --hf-shard <shard>`
  (`ALL MATCH`, quantized projections >= 0.99). A value-head scramble shows as
  every head-indexed DeltaNet tensor near 0 and everything else ~0.997.
- The 27B walls, all handled by the recipe: down GEMV split (K 8192 + 9216),
  streamed norm (`ln_add2` / `ln_nr` / `ln_add3`), 64-lane glue for 48 value
  heads, half-outer glue walk (11 fills), and one-element-at-a-time
  xn/xm/og prep. `OFLM_KEEP_FAILED=1` keeps `final.prj` so `llvm-size -A`
  can name the core that overflowed.
- The GGUF f32 twin (`spec.quant == "q4_1_f32"`) is the GGUF-direct build.

## Qwen3.6-MoE (`recipes/qwen36moe.py`)

The whole-layer designs are `designs/layer_x`:

- `lx.py` — linear-attention (DeltaNet) layer.
- `ax.py` — full-attention layer.
- `ux.py` — the merged one-context image (`OPEN_LAYER_ONE_CTX=1`, default when
  the spec has both layer types and no q8 projection role). A q8 model keeps
  `lx`/`ax` because the merged main-core program has room for one GEMV entry.

The router's top-k output patches the second dispatch's expert fills; the host
re-points them, then runs `lx1` / `ax1`. The q8 lm_head is `lm_head_q8`. Pool
chunks are q4_1 5120 bytes (`CHUNK`), GGUF-direct 6144 (`CHUNK_F32`).

GGUF-direct for this family is **ready** in this tree:
`GGUF_EXPERT_OPS_READY = True`, and `expert_stripes_gguf` / `expert_down_gguf`
read the expert tensors from the GGUF source. A set built before that carries
no `gguf` section and is refused by `oflm add`.

## LFM2 (`recipes/lfm2.py`)

Ten of sixteen layers replace attention with a short depthwise causal conv.
The attention layers are the dense block unchanged; the new layer is:

```
h  = W_in @ x                 # 3 x hidden rows, [B | C | u]
Bx = B * u
conv[c] = sum_k state[c, k] * w[c, k]
out = W_out @ (C * conv)
```

- The cache holds `Bx`; this token pairs with the last tap.
- The conv weight ships `[hidden, taps]` and goes into consts **tap-major**
  `[taps, hidden]`, because a core loading tap k for 32 channels wants them
  contiguous. A transposed tap weight still produces plausible numbers — that
  is the one error to check against `model/replica_lfm2.py`.
- The fused `shortconv.in_proj.weight` is `[3*hidden, hidden]` with B, C, u
  down the rows. The plan emits three `std_perm` ops at source chunk offsets
  `0, n, 2n`, so no new `gemv_q4` point is needed.

## Family trap catalogue

### Granite: the fold

Granite is Llama plus `attention_multiplier`, `embedding_multiplier`,
`residual_multiplier`, `logits_scaling`. `ModelSpec` does not express them and
`attn.h` hard-codes `1/sqrt(HD)`. **`q4nx-build` folds all four into the
weights**: `q_proj *= attention_multiplier * sqrt(hd)`, `o_proj` / `down_proj`
`*= residual_multiplier`, `embed_tokens *= embedding_multiplier`, `lm_head /=
logits_scaling`. The container keeps the originals under
`q4nx_folded_multipliers`.

An unfolded container is refused twice — by `recipes/spec.py::_granite_scale_check`
at generation, and by `hf_config_check` at load. It does not crash; it returns
plausible garbage. Granite also has no closed engine fallback
(`llama_npu` refuses hidden 2560), so a missing set is a named refusal.

### Phi-3 and Nanbeige

- Phi-3 rotates 96 of 128 dims (48 pairs). `designs/attn/attn.h` has a 16-lane
  RoPE tail for `(ROT/2) % 32`; the rule is "a multiple of 32". Families whose
  rotation is a multiple of 64 compile the identical loop.
- longrope bakes both `inv_freq` and `long_inv_freq` on the ptab global and
  switches per row at `original_max_position_embeddings`. `rope_scale()`
  (1.190 on Phi-4-mini) rides the same global; `pools.cpp`, `pack.ptab` and
  `replica_dense.rope` all multiply cos/sin by it.
- The load check names `rope_theta`, the longrope config and
  `max_position_embeddings`; two same-shaped containers can differ only in
  their factor lists.
- Nanbeige is `model_type: llama` at 20 heads over 4 — no new kernel, just the
  catalogue points. Its prompt opens `<|im_start|>`; it always thinks, so give
  it a real generation budget.
- A Phi export can die in Peano on `unable to rename temporary ... final.prj/...o`
  on a `/mnt/c` mount. That is the mount, not the code; retry the export.
- `oflm serve` segfaults on the first request if an adapter reaches its closed
  engine through `dynamic_cast<xxx_npu*>`. Use the `causal_lm` virtuals
  (`this->lm_engine->checkpoint()`), not the closed class.

### K2-Horizon: the q/k row order

K2 is GQA 32/8 at head_dim 128, no q/k norm, gate or bias, split-half RoPE,
silu FFN, untied 250624-row head, and **GroupRMSNorm**
(`layernorm_num_groups` contiguous groups each normalized on its own: 3.7B is
2 x 1280, 7B is 4 x 1024).

- A `k2-horizon` GGUF already holds HF's split-half q/k rows. `q4nx-build -f k2`
  used to apply Llama's un-interleave anyway; a scrambled pack is still fluent,
  so check a likelihood, not the text (7B NLL 2.44 correct vs 2.96 scrambled).
  Fixed in `q4nx/models/k2.py::_gguf_qk_interleaved`.
- The tokenizer contains `<|start_header_id|>`, which `chat.py` used to take
  for Llama 3. Build ids with transformers' `apply_chat_template` on IFM's
  `chat_template.jinja` and feed `open_qwen36_cli --ids`.
- `ln` at this family is grouped: set `LN_GROUPS`, and do not pool statistics
  across groups.

### Qwen3-8B: `fast` against `lean`

The dense set carries both prefill routes. `fast` (default) makes a 9-bit
copy of the five projections at load (~7.8 GB, ~6 s longer) and swaps it for
the q4 route's per-layer buffers; peak 13.3 GiB against 9.8 GiB for `lean`.
Measured 981-token prefill: 4.7 s (`fast`) against 11.1 s (`lean`). On a box
that cannot spare the memory, `--prefill-mode lean`.

The 8-bit route refuses `OFLM_OPEN_ATTN_BLOCK=0`, `OFLM_OPEN_HOST_ATTN` and
`OFLM_OPEN_HOST_ATTN_DECODE` at load; the message names `--prefill-mode lean`.
Accuracy gate when the route changes: teacher-forced NLL over ~2048 tokens,
`fast` within 0.5 % relative of `lean`; report KL and top-1 agreement, do not
gate on correlation. Time prefill on a quiet box under the shared timing lock.

### Gemma 3 1B: a design wall, not a catalogue point

`hidden 1152` is not a multiple of the 256-wide q4 k-tile. No gate can fix it;
`open_kernels/recipes/candidates/gemma3-1b.json` records the wall. `build-new-spec`
moves such a spec to `candidates/` and names the reason.

### GPT-OSS: no recipe, by name

`recipes/families.py::NOT_IMPLEMENTED` lists `gptoss`: the arithmetic is
settled and tested (`model/replica_gptoss.py`) but no kernel computes it —
clamped SwiGLU, biased o_proj/router/expert projections, sliding-window MoE
composition, and expert intermediate == hidden, which the qwen36moe layout
refuses by name (OPEN-MOE-WIDE-FF). Do not route it to the nearest recipe.

### Qwen3.5-VL

The text arch is `qwen35`; the vision tower is Qwen3-VL's
(`clip.projector_type: qwen3vl_merger`, wrapper
`Qwen3_5ForConditionalGeneration`). The open engine has no vision path, so an
image request routes to the closed engine; a text-only install works.

### LFM2-MoE (`lfm2moe`)

One family serves LFM2.5-24B-A2B (40 layers, 64 experts, top-4, moe 1536, dense 11776) and
LFM2.5-8B-A1B (24 layers, 32 experts, top-4, moe 1792 -> padded 2048, dense 7168). Mixers are
short-conv (`SHORT_CONV`) and full attention (`FULL`), no gate, no DeltaNet, no window, no sinks.
FFN is dense on the first `dense_lead` layers (`SHORT_CONV_DENSE` / `FULL_DENSE`, encoded in
`layer_types` so the pack plan dispatches per type) and routed MoE after.

Traps already paid for (do not rediscover next time):
- The router is logits.sigmoid, not softmax. Selection is on `p + exp_probs_b.bias`, weights are
  the unbiased `p` renormalised over top-k (`norm_topk_prob`) times `routed_scaling_factor` (1.0).
  `designs/router/router.h` takes ROUTER_E / ROUTER_TOPK / ROUTER_SIGMOID / ROUTER_BIAS /
  ROUTER_NORM macros; the default E=256/topk8/softmax keeps every existing build identical.
  The bias is a 4 KB element after the router W in consts (`CA_BIAS`), read by `router_fin_bias`.
  `rout_idx_off` is `E*4` (was 1024 at E=256): `moe_hdr2` and `Core::route` only read idx and `w`.
- Expert width is wide: 1536 is 12 stripes over 8 cores (spp > n), 1792 padded to 2048 is 16.
  `qwen36moe.common` refuses it (OPEN-MOE-WIDE-FF). The law here is whole 64-row halves per core:
  `HALVES_PER_CORE = padded_ff // (64 * 8)`, padded with zero chunks to a multiple of 512.
  Ms layout (`HID_PC`, `MS_FLOATS`) and `tab` (hidden + expert hidden tables) are sized for it.
  `gen_kernels.py` emits a 3-arg `gup(..., sub)` for the wide case; `moe_body`/`moe_sequence`
  index halves as `glob = c*hpe + h`. The pack ops take `src_rows` and zero-fill padded stripes
  (up) and padded k-tiles (down); `expert_down_gguf` infers the pad from `B` vs the tensor.
- GGUF arch `lfm2moe` writes `attention.head_count_kv` as a per-layer array (0 on conv layers);
  the Python spec takes the max as `num_kv_heads` and derives mixers from it. The C++ `derive_config`
  does the same and emits `layer_types` with the dense prefix, `num_dense_layers`, and the
  nested `rope_parameters`. `hf_config_check` carries `num_dense_layers`, `moe_intermediate_size`,
  `intermediate_size`, `norm_eps` and the nested rope; `hf_config_defaults` supplies the latter two.
- Tensor names: gguf_name needs no new dots -- shortconv and `exp_probs_b.bias` already map by
  remaining string after the `blk.` rewrite. `shared_expert_gate` is deliberately not packed (no
  shared expert; the sgw element stays zero, the hdr sigmoid on it is unused).
- Attention is GQA 32/8, head_dim 64, full RoPE, q/k RMSNorm pre-RoPE, ATTN_GATE=0 (fin_ng path).
  Do not re-add the gate region to `w_regions`/`y_regions` for this family. The no-gate
  `full_sequence` in `xlayer.py` must skip the v-stage fill (q/k/v are regions 0/1/2 already
  filled by the first loop) and read `o` from region 3, not 4.
- Whole-layer designs: `sx.py` (conv+MoE, 2 parts) and `ax.py` no-gate (2 parts). A short-conv
  core kernel (`short_conv_step`) MUST be passed as a Worker `fn_arg`, not captured by closure:
  as a closure it goes through `ExternalFunction.__call__`'s numpy dtype validation, where the
  fifo's MLIR-typed element (`F32Type`) never equals `np.float32` and every build fails
  `Argument 0: expected (512,)/<class 'numpy.float32'>, got (512,)/f32`. cx.py passes it as a
  fn_arg for the same reason. Also: `TaskGroup` must be imported in sx, and every created group
  finished (`tg_o` was left unclosed).
- `GEN_KERNELS` is one path, so a family reusing two design trees (layer_x + dense for the
  leading dense layers) uses a wrapper (`designs/lfm2moe/gen_kernels.py`) that generates both;
  the dense side gets a shim carrying `dense.geometry(spec)` since dense's gen reads only
  `R.geo`/`R.spec`. The `lfm2` recipe guard was widened to accept `lfm2moe` specs *with* dense
  layers (cx.py/dx.py reuse); its build dir must be distinct (`..._lfm2moe`) or the build key
  collides with lfm2's same-geometry set.
- Router bias pack: the `put` cap must be the exact `E*4` (the GGUF tensor is F32[E]); a cap of
  the padded ELEM slot (4096) fails in `pack_norm` ("is X B of f32, the slot holds Y B of
  bf16"). The consts blob is zero-memset before packing, so the slot's tail stays zero.
- `make_test_mx.py` top-k picks must be `< E` (the [3,11,23,41] list breaks for E=32) and the
  run cfg's build dir must come from `MX_BUILD_DIR` (op-gate passes its geometry-tagged dir;
  a hard-coded `build_mx_full_lfm` silently ran the 24B xclbin against 8B buffers).
