# Open Gemma3-1B Text Engine and Hybrid-Quantized NPU Format

This plan establishes the open replacement for the closed `gemma_text_npu`
engine and sets the quantization and kernel methodology for **all text model
families**. Vision towers are explicitly deferred to later passes.

Target model: **`google/gemma-3-1b-it`** (chosen small, representative of the
Gemma3 text family).

---

## Decisions (agreed)

| # | Decision | Rationale |
|---|---|---|
| 1 | Ship **bf16** safetensors, not fp32 | 2x smaller download/load; no appreciable accuracy loss. |
| 1b | **Embedding stays at highest practical precision** | Minimize accuracy deviations; embeddings feed every token. |
| 2 | **Own both ends** (format + packer + kernel) and design for community ownership | Do not inherit the closed Q4NX 32x256 row-parallel geometry. Co-design a layout that suits the AIE. |
| 3 | **Decode on CPU first**; port to GEMM only if it reduces complexity | Otherwise go straight to a dedicated small-M kernel in Phase 2. |
| 4 | **Replace the closed path entirely**, including remnants, once they are understood | Same approach as the embedding replacement. |

---

## Ground truth

### Gemma3-1B architecture (from `google/gemma-3-1b-it/config.json`)

```
model_type:             gemma3_text        hidden_size:           1152
num_hidden_layers:      26                 intermediate_size:     6912
num_attention_heads:    4                  num_key_value_heads:   1
head_dim:               256                vocab_size:            262144
sliding_window:         512                sliding_window_pattern: 6
rope_theta:             1e6                rope_local_base_freq:  1e4
rms_norm_eps:           1e-6               query_pre_attn_scalar: 256
hidden_activation:      gelu_pytorch_tanh  torch_dtype:           bfloat16
max_position_embeddings: 32768             eos_token_id:          [1, 106]
```

Derived: Q = 4x256 = **1024**, KV = 1x256 = **256**, GQA groups = **4**,
attention scale = `1/sqrt(256)` = **0.0625**. KV cache =
26 layers x 2 x 256 x 2 B = **26,624 B/token** (bf16) -> ~872 MB at 32768 ctx.

**Tied embeddings**: no `lm_head.weight` tensor; the LM head reuses
`model.embed_tokens.weight` (262144 x 1152, ~302 M params, the single largest
tensor).

**Hybrid attention**: `sliding_window_pattern: 6` -> every 6th layer
(`(L+1) % 6 == 0`) is global/full attention; the rest are sliding with a
512-token window. `config.json` has **no `layer_types` key** — it must be
derived, not read.

### Closed Q4NX block layout (recovered, for reference only)

5120-byte block = 32 rows x 256 cols at 5/8 byte/weight:

```
[ scale      512 B  bf16 ]   index = g*32 + r   (g = col/32, r = row)
[ zero_point 512 B  bf16 ]   same indexing
[ nibbles   4096 B        ]   byte = 1024 + (R*256 + C)*8 + k
                              R = row/16, k in [0,8)
                              low nibble = row 16R+2k, high nibble = +1
dequant:  w = (nibble - zero_point) * scale
```

We are **not** required to keep this geometry. See Decision 2.

### NPU constraints (verified)

- AIE2P has **no native int4 MAC**. `kernels.mm` takes a single `input_dtype`
  for both operands and offers no int4 (`--dtype-in {bf16,i16,i8}`).
- Therefore "never unpack to fp" is achievable as **unpack in-core,
  tile-local, fused into the GEMM** — a dequantized tile lives only in L1/L2
  and is immediately MAC'd. No full float weight matrix exists anywhere, and
  nothing is unpacked on the host. This is exactly what the closed stack's
  fused `dequant_mm.xclbin` does for Gemma4-12B and Qwen3.6.
- The payoff is **memory bandwidth and capacity** (4x smaller weights), not
  MAC throughput.
- Existing open design (`npu_offload/matmul/matmul_whole_array.py`) requires
  `M % 256 == 0`; **decode is M=1**, and the closed stack needed dedicated
  skinny kernels (`mv.xclbin`, `short_seq_mm.xclbin`).
- The closed Q plane is **row-parallel** (8 bytes = 16 rows x 1 column) while a
  GEMM consumes **K contiguous** — a friction point the new layout should fix.

---

## Phases

### Phase 0 — Weight pipeline and oracle (no engine) — **COMPLETE**

- [x] Confirmed `google/gemma-3-1b-it` availability (already in the HF cache).
- [x] `q4nx/open_causal.py`: open **bf16 safetensors** export for the Gemma3
      text family, mirroring `open_embedding.py`. Handles single-shard and
      sharded layouts. Produces config + tokenizer sidecars +
      `model.safetensors` + `weights_manifest.json` (relative paths,
      per-tensor dtype, tied-embedding map) + `model_info_entry.json`.
- [x] CLI flag `--open-causal-lm` (+ `--npu-assets` later).
- [x] **Reference oracle** (`q4nx/reference.py`): independent NumPy
      implementation producing prompt -> logit/token fixtures.

**Results** (built at `Models/Gemma-3-1B-OpenNPU2`, git-ignored):

| Item | Value |
|---|---|
| Tensors | 340, all BF16 |
| Model size | **2.0 GB** bf16 (vs ~4 GB fp32) |
| Tied embeddings | `lm_head.weight` -> `model.embed_tokens.weight` |
| Derived global layers | `[5, 11, 17, 23]` (from `sliding_window_pattern: 6`) |
| `embed_scale` | `sqrt(1152)` = 33.9411 |
| `attn_scale` | `1/sqrt(256)` = 0.0625 |
| GQA groups | 4 |
| Fixtures | 6 prompts, token counts `[6, 10, 7, 7, 11, 2282]` |

Byte-reproducible: two independent builds produced identical trees.

**Oracle validated by output coherence** (see "Environment" below for why
`transformers` was not used):

- "The capital of France is" -> ` Paris.` / " Paris.\n\nThe largest city..."
- "In a distant galaxy, a small robot discovered" -> " a strange artifact - a
  shimmering, pulsating orb of pure energy."
- "def fibonacci(n):" -> "\n    \"\"\"\n    This function calculates the nth
  Fibonacci number.\n    \"\"\""
- "The three laws of robotics are" -> " a set of principles that govern the
  behavior of robots. These laws, developed by..."

The 2282-token prompt exercises the hybrid sliding-window path.

**Corrected assumption:** Gemma3 text **does** scale embeddings by
`sqrt(hidden_size)` (as `Gemma3TextModel` does in HF), matching the open
embedding engine. An earlier reading suggested it did not; the oracle's
coherent output confirms scaling is correct. Always verify against the oracle.

### Environment: AMD ROCm constraints (important)

This workspace is AMD. Two hard constraints shaped Phase 0 and will shape the
rest of the work:

| Constraint | Impact | Resolution |
|---|---|---|
| `transformers` + ROCm torch **segfaults** in `from_pretrained` (even with `HIP_VISIBLE_DEVICES=""` and `device_map='cpu'`) | Cannot use HF as the oracle generator | Wrote an independent NumPy oracle instead. Bonus: no torch dependency, so the community can run it anywhere. |
| NumPy has **no bfloat16**; `safetensors.numpy` fails with `data type 'bfloat16' not understood` | Cannot load the bf16 checkpoint with the plain NumPy path | Decode bf16 as `uint16 << 16` viewed as `float32` — the same bit trick the C++ engine uses, so oracle and engine agree at the bit level. |

Also note `/tmp` is a RAM-backed tmpfs (~4.7 G free). Large model artifacts must
go on real disk; use the git-ignored `Models/` directory.

Next validation step: cross-check the NumPy oracle against the closed
`gemma_text_npu` engine (still functional) to confirm both agree.

### Phase 1 — CPU causal engine from bf16 safetensors — **COMPLETE**

Progress: forward pass validated, KV cache validated, `causal_lm` adapter
validated, wired into `AutoModel`, closed `gemma_text_npu` unlinked, and
end-to-end generation verified through the CLI.

New `src/open_gemma3/engine.{hpp,cpp}`; the forward pass is complete and matches
the oracle exactly. Remaining: the `causal_lm` adapter (KV cache, incremental
decode), wiring into `AutoModel`, and removing the closed path.

**Validated result** — `src/test/gemma3_text_open/`, a standalone target (it
does not use `src/test/CMakeLists.txt`, which unconditionally links the closed
`q4_npu_eXpress`/`mha`/`dequant`/`gemm`/`lm_head` stack):

```
open_gemma3: loaded 340 tensors (hidden=1152 layers=26 heads=4 kv=1
             head_dim=256 vocab=262144, global layers: 5 11 17 23)
[PASS] prompt 0 (6 tok):    argmax ok  top-100 100/100  worst|diff| 0.0001  greedy 16/16
[PASS] prompt 1 (10 tok):   argmax ok  top-100 100/100  worst|diff| 0.0000  greedy 16/16
[PASS] prompt 2 (7 tok):    argmax ok  top-100 100/100  worst|diff| 0.0001  greedy 16/16
[PASS] prompt 3 (7 tok):    argmax ok  top-100 100/100  worst|diff| 0.0000  greedy 16/16
[PASS] prompt 4 (11 tok):   argmax ok  top-100 100/100  worst|diff| 0.0001  greedy 16/16
[PASS] prompt 5 (2282 tok): argmax ok  top-100 100/100  worst|diff| 0.0002
6/6 prompts matched the NumPy oracle
```

The worst logit deviation is 0.0002 — fp32 rounding noise. Crucially, the
**2282-token prompt passes**, which exercises the hybrid sliding/global attention
path, and **greedy continuation is 16/16**, validating multi-step decode.

Because the oracle is an independent implementation, this is strong evidence that
embedding scaling, dual-base RoPE, GQA with a single KV head, the tied LM head,
bf16 decoding, and the sliding-window mask are all correct.

**KV cache validated.** Incremental decode through the cache reproduces the
full-recompute path **bit-exactly** (cosine 1.000000) on all short prompts,
covering cache write/read, the sliding-window lower bound, and causal bounds.

**Adapter + end-to-end generation validated.** `Gemma3TextOpen`
(`src/open_gemma3/gemma3_text_open.{hpp,cpp}`) implements `causal_lm` and is
wired into `Gemma3_Text_Only::load_model`. Through the adapter + runtime
tokenizer, argmax and continuation match the oracle exactly:

- "The capital of France is" -> ` Paris.`
- "In a distant galaxy, a small robot discovered" -> ` a strange artifact - a
  shimmering, pulsating orb of pure energy. It was unlike`

CLI generation works: `flm run gemma3:1b` answers ` Paris` (surrounding
`<mask>` tokens are chat-template/special-token decoding artifacts, not an
engine issue).

**Closed path removed from the build:** `gemma_text_npu` is no longer in
`target_link_libraries` and no longer appears in `flm`'s `NEEDED` entries.
`automodel.hpp` includes the open adapter instead of
`models/gemma_text/gemma3_text_npu.hpp`.

**Bugs found and fixed along the way** (each would have been painful to
rediscover):

| Bug | Fix |
|---|---|
| `buffer(std::vector&&)` is a **shallow mapping** that takes no ownership — returning `buffer<bf16>` built from a local vector left the logits pointing at freed memory (garbage, argmax always 2). | Allocate an owning `buffer<bf16>(n)` and fill it in place. |
| HTTP failures were never checked: curl returns `CURLE_OK` for a 401/404 and the error page was written to disk as the model file. | Check `CURLINFO_RESPONSE_CODE`; any non-2xx fails and deletes the partial file. A server error is categorically broken, unlike a benign hash mismatch. |
| The runtime needs `bos_token_id` (int) and `eos_token_id` (array) in `tokenizer_config.json`; HF checkpoints ship only the token strings. | `ensure_runtime_tokenizer_ids()` in the builder backfills ids from `config.json`, resolving token text via `tokenizer.json`. |
| `model_family` string `gemma3-text-open` was not in the dispatch table -> `map::at` threw. | Reuse the existing `gemma3-text` family; `gemma3:1b` now points at the open model so existing user tags keep working. |

**Still outstanding (Phase 5 / cleanup):**

| Item | Note |
|---|---|
| Closed remnants | `gemma_text_npu` XRT/HRX/Windows binaries still in `src/lib/`, plus installer entries and `src/test/gemma_text_npu/` wiring. |
| Re-upload patched tokenizer | The `tokenizer_config.json` on HF/ModelScope predates the token-id backfill. Re-run the builder and re-upload that one file. |
| Preemption | `checkpoint()`/`restore()` and `get_k_cache`/`get_v_cache` are explicit "not implemented" stubs; nothing on the Gemma3 text path calls them. |
| KV cache memory | fp32, ~1.7 GB at 32768 context. bf16 would halve it. |

Reuse from `src/open_embedding` (validated): rmsnorm (`x*scale*(1+w)`), GQA
reduction, dual-base RoPE, `rotate_half`, fp32 max-subtracted softmax,
`gelu_tanh`, `matmul_t`, and `NpuMatmul` as a separate TU. Extract into a
shared `open_common` only after both engines are green, to protect the
validated embedding path.

Must build (gap list versus the embedding engine):

| Gap | Note |
|---|---|
| Causal masking | Embedding's `full_attention` is bidirectional — wrong for LM. Need `kpos <= t`, plus sliding band. |
| KV cache | None exists. Two fill policies: ring at 512 (sliding), append (every 6th). |
| Incremental decode | M=1 QKV, RoPE at absolute position, attention over `[0..p]`. |
| `layer_types` synthesis | Engine hard-fails without a `layer_types` key; derive from `sliding_window_pattern`. Gemma3-1B does NOT scale embeddings — verify against the oracle, do not assume. |
| LM head | 1152 -> 262144, returns `buffer<bf16>`. Tied from embeddings. |
| bf16 loading | Current loader assumes fp32; needs dtype-aware reading per manifest. |

Gate: logit/token agreement with the Phase 0 oracle.

### Phase 2 — NPU offload (bf16 weights)

- Family xclbin bundle at `src/xclbins/Gemma3-1B-OpenNPU2/`, built from
  `matmul_whole_array.py` (bf16 -> f32).
- Offload prefill (large M) and the LM head (dominant per-token cost).
- Decode: CPU first (Decision 3). Add a dedicated small-M kernel only if
  simpler than shoehorning the padded GEMM.
- Gate: NPU vs CPU agreement; `flm-test --llm`.

### Phase 3 — Hybrid quantization groundwork (CPU)

- Publish the **open format specification** as the shared contract (packer +
  runtime + kernel), owned with the community.
- Open **reader + CPU oracle** for the new layout, validated against the
  packer's own output.
- Per-tensor sensitivity sweeps: Q4 on big projections (q/k/v/o/gate/up/down);
  norms stay bf16; embedding stays high-precision (Decision 1b); lm_head and
  embedding decided by measurement, not by default.

### Phase 4 — Fused int4 dequant+GEMM on NPU

- Stage 1: single-core Iron `ExternalFunction` dequant core for one block
  (precedent: `matmul_i16.py` bypassing `kernels.mm`).
- Stage 2: fuse with the validated bf16 -> f32 whole-array MM.
- Frictions the new layout should remove: K-contiguous nibble order, and
  scale-group granularity aligned to the k-tile.
- `zero_point` handling: fold `(q - zp)` into the activation domain, or keep a
  second accumulator `sum(x)`.

### Phase 5 — Packaging and methodology capture

- Family xclbins + `Atomic-Germ/Gemma3-1B-OpenNPU2` repo.
- Registry wiring: `model_list.json` **and** `model_info.json` (the latter
  silently skips files — it is the real gate).
- Remove closed remnants (Decision 4): `gemma_text_npu` link, its
  XRT/HRX/Windows binaries, installer entries, and standalone-test wiring.
- Generalize the builder so the next text family is config-only; record the
  methodology as a new skill.

---

## Risks

| Risk | Mitigation |
|---|---|
| bf16 vs fp32 accuracy on a 1B model | Measure with the Phase 0 oracle before committing. |
| KV cache size (872 MB at full context) | bf16 cache; cap `MAX_L`; document the tradeoff. |
| No int4 MAC on AIE2P | Accept tile-local unpack; the win is bandwidth, not FLOPs. |
| Decode M=1 does not fit the existing `M % 256` design | CPU decode first; dedicated kernel in Phase 2/4. |
| `q4nx-build` Gemma3 HF path is currently broken (`hidden_size` unset) | Fix as part of Phase 0/3; it blocks the HF export. |
| Replacing the closed path loses a fallback | Keep the closed library until NPU+CPU gates pass, then remove. |
