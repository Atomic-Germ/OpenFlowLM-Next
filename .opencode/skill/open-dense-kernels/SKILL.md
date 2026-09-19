---
name: open-dense-kernels
description: Build, verify and ship the open XDNA2 kernel sets for the dense recipe -- the families routed through recipes/dense.py (qwen2/qwen3/llama3/gemma/lfm2, one dispatch per layer) and its two attention designs (dense/dx.py full layer, dense/dx_attn.py attention-only prefill dispatch). Use when rebuilding a dense-family set, adding a new dense family recipe, or hitting "has no attribute 'chunk_bytes'", "no bias stream", or "position record is wider" in a dense export.
---

# The dense recipe (recipes/dense.py) and its designs

The dense recipe drives the qwen2 / qwen3 / llama3 / gemma / lfm2 family sets:
`designs/dense/dx.py` (the fused full layer) plus `designs/dense/dx_attn.py`
(dx.py's attention phase as its own dispatch, the T-token prefill split) and
`ln` / `lm_head_q4`. Specs export into
`src/xclbins/<model>/open_kernels/{dx,dx_attn,dx_f32,lm_head_q4(+_f32),ln,gemm_*}`.

Build the whole export set with the `*-default` CMake presets
(`cmake --preset fedora-default -DOFLM_KERNEL_SPECS=` then `ninja -C build`);
`*-debug` presets build engine-only. Export is incremental by `build_key`
(design source + `attn/*.cc*` + `recipes/*.py` hashes), so a failed spec is
retried without redoing the cached ones.

## The recipe re-export contract (lfm2's bug)

`dx.py` reads `QR.chunk_bytes(quant)` / `QR.band_bytes(K, quant)` **off the
family module**, not through `dense`. `dense.py` re-exports them from
`.qwen36moe` (line 44); a new family recipe must too. `lfm2.py` forgot, so its
`[dx]` build died with `AttributeError: module 'recipes.lfm2' has no attribute
'chunk_bytes'`. Re-export `band_bytes` and `chunk_bytes` from `.dense` in every
new dense-family recipe.

## dx_attn.py now carries bias and split position records

The attention-only dispatch was a verbatim copy of dx.py's pre-bias version and
_raised_ for families with biased q/k/v (Qwen2) or a multi-element position
record. Both are ported across (this shipped Qwen2.5-3B):

- `ATTN_QKV_BIAS=1` + the `abias` fifo: `u8_ab` element (`E_A // 2` bytes),
  `bz` on `attn_q/attn_k/attn_v`, one acquire/release pair per projection
  element, 4 shape-selected worker bodies (bias x block), `CD_QB/KD/VB` fills,
  and a QKVB `sequence_attn` with `abias_p` as its own Runtime ABI arg (bias
  prod on `Tile(0,0)`, the one free shim column in this design's map).
- `ATTN_PTAB_SPLIT=1` + the `NPTAB, CSE` case: `pz` second element on
  `attn_meta`, `acquire(1 + NPTAB)`, `f_meta(e[0], e[1], e[1 + CSE], ...)`.
  The record fill already delivers all `NPTAB` elements in one `PTAB_ROW`-byte
  tap (one record = `max(1024, KVW*2)` bytes; Qwen2.5-3B at 2 KV heads / hd 128
  has `e_a` 512, so two 512-byte elements per record).

Non-bias, single-record families still take the exact pre-port paths (byte-identical
kernels, same flags, same ABI shape). If a new family adds both a bias **and** a
wider record, only the `BNEWQKVB`/`NPTAB` paths in `dx_attn.py` change.

## Verify

- `ninja -C build` must end `kernel export ok: 12 open_kernels spec(s) + BERT
  design sets`.
- `ctest` in `build/`: oflm_smoke, openai_compat, OPEN-VISION-IMAGE-READ,
  bench_embed.
- NPU harness: `open_kernels/gemv_q4/make_test.py --layout gguf` on the QKV and
  share_down regions (cos once, fp64 reference).

## Rules

- Never commit xclbins (`src/xclbins/*/open_kernels` is git-ignored); the
  distributed package ships them pre-built.
- A new dense-family recipe must re-export `band_bytes`/`chunk_bytes` before it
  ships, or every `[dx]`/`[dx_attn]` panel in its export fails.
- Run the full preset export, never `build_design.py` by hand; the preset is the
  shipped-package authority.