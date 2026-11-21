---
name: open-gemma4-text
description: Build, verify and run Gemma 4 E4B / E2B (text) on the open XDNA2 kernels from Google's GGUF -- the gemma4e recipe (four kinds: sliding / global x own / shared KV cache, one gx.py image), the attn.h switches it adds, E2B's one kv head and double-wide FFN, the engine's KV sharing, per-layer inputs and soft cap, the block prefill route, the guard that keeps FLM's official container off these kernels, and the oflm serve path. Use when exporting a gemma4e kernel set or debugging a Gemma 4 E-series load or serve. The 12B is a different recipe (gemma4.py, dx.py).
---

# Gemma 4 E-series text on the open kernels

Spec: `specs/open-engine/spec.md` OPEN-FAMILY-GEMMA4, OPEN-PREFILL-G4, OPEN-GGUF-BF16-POOL. Recipe:
`open_kernels/recipes/gemma4e.py` (family `gemma4e`). Reference: `open_kernels/model/replica_gemma4.py`
(fp64, from the GGUF), held to transformers by `gemma4_transformers_check.py`.

The 12B is family `gemma4` (`recipes/gemma4.py`, two `dx.py` builds, K = V, `ATTN_QLOCAL`). It shares the
attention switches `ATTN_VNORM`, `ATTN_SCALE` and `ATTN_ROPE_PAIR` with the E-series and nothing else.

## What the model is (read off the GGUF, not the plan)

- E4B: 42 layers (5 sliding then 1 global, x7), hidden 2560, ff 10240, 8 heads over 2 kv, head dims
  256 (sliding) / 512 (global), window 512, per-layer input 256, layers 24-41 KV-shared (sources 22 and
  23), softcap 30, vocab 262144, tied head. `use_double_wide_mlp` is FALSE on E4B (E2B: true).
- E2B: 35 layers (4 + 1), hidden 1536, ff 6144 (12288 in its 20 shared layers), ONE kv head.
- Scores are unscaled (`scaling = 1.0`); v gets a weightless RMSNorm; norms are stored as `w` (no `1 + w`).
- Global RoPE is `proportional`: pairs (i, i + 256) for i < 64 at theta 1e6 ^ (-2i/512). llama.cpp stores
  it as `rope_freqs` (64 ones, then 1e30).
- Google's QAT GGUF differs from the bf16 HF checkpoint (norms, layer scalars): validate against the
  GGUF's own weights, never the HF safetensors.
- GGUF names (llama.cpp gemma4): `attn_norm` (input), `post_attention_norm`, `ffn_norm` (pre-FFN),
  `post_ffw_norm`, `inp_gate` / `proj` / `post_norm` (per-layer input), `layer_output_scale`,
  `per_layer_token_embd` (Q6_K), `per_layer_model_proj` (F16), `per_layer_proj_norm`, `rope_freqs`.
  The recipe names them directly (`gemma4e.py` T_*), not through `GgufFile::gguf_name`.
- llama.cpp calls every Gemma 4 `gemma4`. `spec._gemma4_any_gguf` sends a file with per-layer inputs or
  shared KV layers to `gemma4e`; config.json's `gemma4` / `gemma4_text` model type is `gemma4e` too.

## The kernel side

- Four builds, `G4_PART` = local | global | local_shared | global_shared, four parts of
  `designs/dense/gx.py`: one image, one context `dx`, the per-kind counts in RTP words, one attention
  core set per kind, and the per-layer input inside the layer program. `OPEN_G4_PLE_NPU=0` keeps the
  per-layer input on the host (threaded, bit-exact); it is in the build key. There is no four-context
  rollback: `dx.py` does not build the E-series kinds.
- `gx.py` builds from `designs/dense/dxkit.py`, a copy of `dx.py`'s attention cores, norm helper and host
  sequence as of 2026-10-20. `dx.py` (the 12B and every other dense family) does not use it, so an edit
  to dxkit moves no other family's kernels.
- attn.h switches the E-series sets, all off for every other build: `ATTN_SCALE=1.0f` and `ATTN_VNORM`
  (shared with the 12B), `ATTN_ROPE_PAIR` (partner distance, global kinds), `ATTN_PTAB_CS_OFF` (the sliding
  kinds' 1024-byte element cannot hold 512 + 1024 bytes of record, so cos / sin move to the second
  element), `ATTN_KVH` 1 (E2B), `ATTN_KV_SHARED` (E2B's shared kinds), `ATTN_NORM_SHARED` (q/k and v
  norms share one 1/rms, which keeps the attention cores inside program memory) and `ATTN_INTFP`.
- Decode (2026-10-19, `--bench-step`): E2B ~60 ms/token at position 1, ~70 at 512; E4B ~93 and ~115.
  A gate run against the host route is bf16 vs fp64: judge by greedy tokens, argmax (near ties on the
  soft-capped logits flip) and corr >= 0.999, not bit-exactness.
- Hand-written kernels go in `designs/ln/` (or another tracked dir), never `designs/dense/*.cc`: those are
  generated per export and gitignored (`ple_ln.cc` lives in `designs/ln/`).
- Pools are GGUF-direct at 5120-byte chunks (bf16 scales narrowed at pack time); no `*_f32` twin.

## Export (native Windows, under `dibs run build`)

In PowerShell: `Set-Location C:\dev\mlir-aie; . .\iron_env.ps1`, then from the worktree's `open_kernels`:

```
python export_qwen36_kernels.py --spec recipes/specs/gemma4-e4b.json --out <dir> --force -j 2
python ../utilities/core_sizes.py designs/dense/build_gemma4e_h2560_gxp_global   # program memory per core
```

The E4B and E2B points are in the catalogue, so no unvalidated switch is needed. E2B:
`--spec recipes/specs/gemma4-e2b.json`. A gx part takes ~10-12 minutes at `-j 2`. Export straight into the
installed model's `open_kernels` dir and pass that dir to `oflm-add --open-kernels` ("already in place").
oflm-add copies the GGUF to `model.gguf`; on a tight disk, replace the copy with a hard link. Where a set
that passed on the NPU exists, add `--check <that dir>`.

## E2B (one kv head)

- Its shared layers' 12288-wide FFN fits one chunk per weight element, the others two; the generated
  dense TUs are per export, so `gemma4e.per_call_cap` pins every kind to one (E4B: no cap).
- On the image every kind uses 8 main cores; the sliding kinds' 4-band kv projection splits as one k band
  on cores 0-3 and one v band on cores 4-7 (`dense.qkv_segments`).
- A KV-shared kind at one kv head reads this position's row back as one element: `ATTN_KV_SHARED`
  selects the one-pointer `attn_step_new`.
- Its GGUF lists `feed_forward_length` per layer; the engine's GGUF reader skips arrays, so
  `derive_config` takes the FFN width from layer 0's `ffn_up` tensor.
- The slice run: `open_qwen36_cli --model <dir> --kernels <set> --layers N --ids ... --dump-logits
  <prefix>`, then `model/compare_gemma4.py <prefix>_t0.bin <replica.npz> --key npu_logits_lN`.

## The official container must not pick up these kernels

The set is exported under the official model's name (`Gemma4-E4B-IT-NPU2`), where FLM's own `.q4nx`
container also looks. Its one view packs GGUF-direct (`std_perm_gguf`), so `Engine::find_kernels`
declines it unless the model directory reads a GGUF (`Manifest::reads_gguf_only`, `Core::reads_gguf`),
and the official tag stays on the closed engine.

## Serving (`oflm serve`)

- Google's tokenizer_config.json names `<bos>` with no ids, and oflm exits at load without
  `bos_token_id` / `eos_token_id`: oflm-add merges them from the source directory's
  generation_config.json (bos 2, eos [1, 106, 50]).
- Google's config.json nests the text model under `text_config`; the adapter lifts absent keys to the
  top level on the open path, or the sampler is built with vocab 0 and the server crashes on the first
  token.
- Scratch serving: install with `oflm-add <dir> --tag <tag> --config <root>\model_list.json --models-root
  <root>\models --open-kernels <set> --no-xclbin`, then `oflm serve <tag> --port <p>` with OFLM_MODEL_PATH
  = `<root>` and OFLM_CONFIG_PATH = `<root>\model_list.json`. Start it with PowerShell `Start-Process
  -PassThru` writing the PID to a file, not through `$(...)`: the server inherits the pipe and the
  substitution waits for it to exit. Stop it by that PID.
- Garbled arithmetic in an answer ("$64 - $6", an answer that stops mid-sum) points at the sampler
  first: `Gemma4e::load_model` sets frequency and presence penalties of 1.0 over the last 64 tokens,
  which push down digits already written. Replay with `"frequency_penalty": 0, "presence_penalty": 0`
  in the request before suspecting the engine.
- Images and audio: the open path refuses them, and `OFLM_GEMMA4_ENGINE=closed` cannot load a GGUF
  install. They need a curated `gemma4-it` tag (FLM's `.q4nx` container). The refusal is a
  `request_error`: the client gets 400 "Invalid request" and the reason is in the server log.
- Run a server session as one script under `dibs run npu -- bash <script>`.
- Never rewrite a shell script while a bash instance is still running it: bash reads scripts as it
  goes, and a half-read old instance ran the new text, restarted a server and overwrote its logs.
