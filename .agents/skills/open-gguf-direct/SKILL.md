---
name: open-gguf-direct
description: Direct GGUF loading for the open engines (supersedes q4nx for compatible quants). Use when adding GGUF support to another engine, debugging a GGUF model that refuses to load, rebuilding the f32-scale kernel variants, extending to MoE, or changing the pack ops.
---

> **Branch scope (2026-10-25):** this document describes the *whole*
> GGUF-direct feature, which is spread across branches. On **this branch
> (`feat/gguf-direct`)** some cited files do **not exist** — they live on
> `origin/feat/fedora-docker` and are unmerged to `main`. Verify a path
> before assuming it is here.
>
> **NOW ON THIS BRANCH** (ported from fedora 2026-10-25, commit `88fef13`):
> the qwen35 **f32-scale twin** — `open_kernels/recipes/gguf.py` (the shared
> `section()`/`twin_builds()` helper), `recipes/qwen35.py` + `recipes/qwen36moe.py`
> (twin key, `lx_f32`/`ax_f32` names, `pool_quant`/`pool_chunk`, the f32 law in
> `common()`), `designs/layer_x/layout.py` (`GEMV_SCALES_F32` flip +
> re-derivation) and `xcommon.py` (`F32` → `-DGEMV_Q4_SCALES_F32=1`). The Qwen3.5
> and Qwen3.8-27B sets now emit a `gguf` manifest section and build their
> `*_f32` twins (verified by re-exporting Qwen3.5-0.8B: manifest byte-identical
> to fedora's shipped one modulo build_key).
>
> **ALSO NOW ON THIS BRANCH** (2026-10-26): `ModelSpec.kernel_hash`
> (`1d1571c` — oflm-add's geometry match; without it every fine-tune/GGUF
> `oflm add` crashed), the **build-new-spec workflow** (`b72c2e7`:
> `utilities/build-new-spec.py` + the attention gate in
> `designs/attn/{probe_dense,probe_support,make_test,compare,attn_reference}.py`
> + `recipes/dense.attn_flags()` as the single attn.h flag list), and the
> entire **rope-pairing** feature (`c85d185` — llama3 GGUF twins now rotate
> correctly; the gate's fp64 oracle learned the interleaved law too, which
> fedora's has not). Spec count is now **26** (added qwen3-0.6b, qwen3-1.7b, qwen3-14b,
> lfm2-2.6b, llama32-1b, llama32-3b, nanbeige41-3b; gemma-3-1b-it is parked in
> `recipes/candidates/` — a recipe wall, same as fedora). 24 of the 26 sets
> carry a `gguf` section; the two that do not are qwen3-14b (a split FF has no
> f32 twin) and qwen36-35b-a3b (the MoE expert ops are not written). Registration of a
> new spec in the frozen test lists is **scripted**:
> `utilities/register-spec.py` (build-new-spec step 6) — never hand-edit the
> hashes in test_lfm2.py / test_spec_model_names.py.
>
> Still **absent** on this branch (present on fedora-docker):
>
> - `specs/open-engine/tests/test_gguf_twin.py`
> - `utilities/lx-probe.cpp`
> - the MoE **expert GGUF ops** — `GGUF_EXPERT_OPS_READY = False` in
>   `qwen36moe.py`, so the qwen36-moe family still ships no `gguf` section.
>
> Everything else referenced here (the `src/gguf/` parser,
> `open_kernels/gguf_pool/`, `open_kernels/designs/gguf/`,
> `manifest_supports_gguf()` in oflm-add) **is** on this branch.

## Scope of this phase: TEXT models only (2026-10-26)

Every GGUF-direct claim here is about text LLMs. Deliberately NOT in this
phase, and not to be "fixed" opportunistically:

- image generation (FLUX.2 klein VAE/DiT kernels), multimodal (VLM towers),
  embedding models (bge / nomic / gte), audio (whisper) and the MTP path --
  each has its own engine and its own kernel sets.

The goal is that `oflm add` works for each kind AND that the instrumentation
(utilities/build-new-spec.py, op-gate.py, register-spec.py, spec-gaps.py) covers
them the same way; doing it one kind at a time is intentional.

Some models still ship CLOSED pre-built xclbins inherited from AMD/FLM
(Qwen3.6-MoE, GPT-OSS, ...). Those are read-only reference: the `rea` MCP can
inspect them (container layout, per-op streams) when an open implementation
needs to match a legacy layout -- for example the MoE expert GGUF ops, whose
pack layout the closed 35B set already contains. Do not ship against them; use
them as the reference an open kernel is verified against.

## The design (decided 2026-09-07)

- **Pool layout for GGUF-direct models** (`spec.quant == "q4_1_f32"`): the
  same 32x256 tiles as q4nx, but the `d`/`m` planes hold the GGUF block
  scales/mins **widened EXACTLY fp16 -> f32** (mantissa <<13, exponent +112 —
  lossless; no requant, no bf16 narrowing in the file). q4 chunk = **6144 B**
  (d @0:1024, m @1024:2048, codes @2048:); q8 lm_head chunk = **9216 B**.
  Codes keep GGUF values, permuted into the kernel's 16-lane interleave.
- The kernel converts f32 scales -> bf16 with **integer RNE** ((u + 0x7FFF +
  ((u>>16)&1)) >> 16; `load_scale` in gemv_q4.h / lm_head_q8.h) — so a GGUF
  model's numerics are bit-identical to its q4nx twin. Do NOT convert the
  scales with the accum-roundtrip float path in lm_head_q8: the f32-emulated
  `from_vector/to_vector<bfloat16>` mis-schedules in that kernel's kb loop
  (it was fine in gemv_q4, but integer ops are cheap and safe — keep them).
- **Q4_0** expands to d+m with m=0 at pack time (lossless). **K-quants
  (Q4_K/Q6_K/...) are refused** for matmul tensors with a pointer to
  q4nx-build; the *embedding row* dequant supports F32/F16/BF16/Q4_0/Q4_1/
  Q8_0/Q4_K(_S/_M)/Q6_K on the host (GgufFile::embed_row, llama.cpp-faithful
  loops).
- **One export ships both layouts**: `builds()` gains `*_f32` twins (dx_f32,
  lm_head_q4_f32); the manifest carries a **complete nested `gguf` manifest**
  (contexts/kernels/layer_types/tail/globals/pack/layout) and the engine
  swaps the whole view when the model dir has `model.gguf` (q4nx wins if both
  exist).
- **Config hybrid**: config.json from the model dir wins; without one the
  engine derives the checked fields from GGUF KV (`derive_config` in
  core.cpp, llama-family `<arch>.embedding_length` keys). GGUF-invisible
  fields (Granite's folded multipliers) still need a config.json.
- **Tensor names**: GgufFile::gguf_name maps HF-style manifest names
  (model.layers.N.self_attn.q_proj.weight) to llama.cpp names
  (blk.N.attn_q.weight); pools' `std_perm_gguf` packs Q4_0/Q4_1 byte-exactly
  and requantizes Q8_0 / Q4_K / Q6_K to the pool's q4 layout (everything else
  is refused at pack time, pointing at q4nx-build);
  `put`/`pack_norm` convert f32/f16 small weights to the bf16 consts blob.
- **oflm-add**: repos with llama.cpp-style GGUFs install one compatible file
  as `model.gguf` (preference Q4_1 > Q4_0 > Q8_0; multi-part and other quants
  refused with reasons). Registry entries keep format "NPU2" and add
  `details.weights = "gguf"`; q4nx models stay *-NPU2, GGUF installs are
  *-GGUF directories, so one of each kind can coexist.
- **The user's HF cache is a first-class source.** A model fetched with
  huggingface_hub / llama.cpp / ollama is on disk under
  `~/.cache/huggingface/hub` (or `HF_HOME`/`HF_HUB_CACHE`). The installer
  (`hf_cache_snapshot` -> `copy_from_dir`) copies a cached GGUF instead of
  downloading, and verifies it from the local file (no network). The C++ pull
  (`src/pull/download_model.cpp download_from_hf_cache`) reuses any resolve-URL
  file from the cache before downloading, so `oflm pull` and the app's
  auto-download never re-fetch a model the user already has.

## Integrated into main (2026-10-07, `feat/gguf-direct`)

The branch merged into current main (commit `b3e8f1f`). Main had moved a lot
since the branch forked (sandwich norms, qknorm_rope, the alpha/beta
transpose, `split_q4_1_chunks`, the OpenMP wait policy and read fence,
`det_step`/`run_split`/`start_run`, the block-prefill route). The GGUF work
was adapted to that, not the reverse:

- **`WeightFile` is the seam** (`weight_file.hpp`). Extend it with the read
  methods main's host stages call -- `bf16` / `f32` / `meta` / `bf16_row` --
  and implement them on `GgufFile` (`meta` returns the row-major [rows, cols]
  shape; GGUF stores dims fastest-first). `Q4nxFile` implements the same
  interface. `TensorMeta` moved into `weight_file.hpp`. So a polymorphic
  `file_` serves both containers, and main's `file_->bf16(...)` host reads
  compile unchanged. Do NOT re-minimise the interface: main's dense-route host
  code calls those four by name.
- **`core.cpp`**: every model/layout field goes through the weights view `w_`
  (`man_` or `man_.gguf.get()`). Only `man_.gguf`, `man_.family` and
  `man_.spec_hash` are read from the primary manifest. `derive_config()`
  reconstructs the checked fields from GGUF KV. The q4nx container still wins
  when both `model.q4nx` and `model.gguf` exist.
- **`pools.cpp`**: main's q8 split (`split_q4_1_chunks`) and the branch's
  `std_perm_gguf` / `pack_norm` coexist inside `apply`; the q4nx-only ops
  `dynamic_cast<const Q4nxFile*>` (and the split branch does too, since only a
  q4nx container stores q8 chunks to split).
- **`dx_attn.py`**: main's og-element geometry (OGH/N_OG, one og fifo per
  core) combined with the branch's q/k/v bias stream and split position
  record. Four worker-body shapes (QKVB x RB); the bias fifo is the worker's
  SECOND argument.

### Validation (2026-10-07, this box, NPU present)

- `open_qwen36` standalone build + `GGUF-PACK` / `OPEN-MANIFEST` ctests pass;
  the full `oflm` app builds; `gguf_pool.py` self-tests bit-exact;
  `oflm-add` tests pass; the 798-test spec suite passes.
- **End-to-end**: `qwen3-8b` recipe spec added (`open_kernels/recipes/specs/`),
  exported (both layouts) into `src/xclbins/Qwen3-8B-NPU2/open_kernels/`, and
  run with `open_qwen36_cli --model <dir with model.gguf>`:
  - `mradermacher/Qwen3-8B-i1-GGUF` **Q4_1 and Q4_0** (exact packs) both load,
    all 36 layers resident, and answer
    `<think>\nOkay, the question is asking for the capital of` -- byte-identical
    token streams from the two quants.
  - `Qwen3-8B.Q8_0.gguf` and `Qwen3-8B.Q4_K_M.gguf` (requant paths) load and
    answer ` thinking\nOkay, the user is asking` -- identical to each other.
- **`oflm run`** (the app, `model_backend` + tokenizer + chat template) loads
  the same GGUF through the open kernels and generates the same coherent
  reasoning. Set `OFLM_MODEL_PATH` / `OFLM_OPEN_KERNELS_DIR` /
  `OFLM_CONFIG_PATH` / `OFLM_MODELINFO_PATH`; the model dir needs
  `config.json` + `model.gguf` + `tokenizer.json` + `tokenizer_config.json`
  (use `extract_tokenizer_from_gguf` for the last two).
- The id-level CLI needs only `model.gguf` (config derived from KV, no
  tokenizer).

### Q6_K dequant bug fixed (2026-10-07)

The Q6_K `embed_row` advanced `qh` 16 B per 128-value half instead of 32,
skewing every Q6_K tensor (typically a Q4_K_M GGUF's `output.weight`). The
exact packs and Q8_0 worked but Q4_K_M gave word salad. Root-caused with a
`dq_dump` harness (dumps one dequantized row to compare against the gguf
Python reference; `gguf_pack_test` had no K-quant coverage). Fixed
(`qh += 32`), matched the reference (maxabs 5e-10), and a self-contained Q6_K
`embed_row` regression now lives in `gguf_pack_test`. The Q8_0/Q4_K/Q6_K
*dequant* is what needed the K-quant unit coverage; the shared re-quantizer
was already covered for Q4_0/Q4_1.

### The block-prefill route is NOT f32-scale (fixed 2026-10-07)

`gemm_q4_prefill` / `attn_block` read the pool by a hardcoded
`GQD_CHUNK_BYTES = 5120` bf16-scale law, so a f32-scale (GGUF) pool of 6144 B
chunks would be misread. `recipes/dense.py programs()` therefore emits **no
`gemm_block` for the `q4_1_f32` spec** (`r = None if f32 else gemm_route(...)`):
a GGUF kernel set prefills through `step()` (decode-as-prefill), which is
exact, and the engine's `layer_major_ok()` correctly reports the route absent
(`block prefill route: T = 0`). The q4nx primary keeps its block route. To
bring the route back for GGUF, add `SCALES_F32` twins of those GEMMs (their
GQD scale loads and chunk size) -- a perf job, not a correctness gate.

## Toward deprecating q4nx

The pieces to drop the q4nx container for a family, in order:

1. **Export the family with the f32 twins.** `recipes/dense.py builds()` already
   emits `dx_f32` / `lm_head_q4_f32` beside the q4nx pair, and the nested
   `gguf` manifest. `utilities/build-all.sh` or
   `python open_kernels/export_qwen36_kernels.py --spec <spec> --out src/xclbins/<Model>-NPU2/open_kernels`
   produces a set that serves BOTH layouts.
2. **Point the catalogue at a GGUF source repo.** A `model_list.json` entry can
   name a llama.cpp-style repo (e.g. a `mradermacher/*-i1-GGUF`); `oflm add`
   installs the best compatible file as `model.gguf` and links the family
   `*-NPU2` kernel set by the same `details.family`/size. It refuses to install
   unless the linked kernels carry a `"gguf"` manifest section
   (`manifest_supports_gguf`), which step 1 guarantees.
3. **The embedding path is already there**: `embed-gemma:300m` ships as a GGUF
   (`embeddinggemma-300M-Q8_0.gguf`) and the open embedding engine loads it.

Once a family's catalogue entry is a GGUF repo, `model.q4nx` is dead weight for
that family. q4nx stays only for K-quants the pack ops refuse (Q5_K/IQ/etc.),
which still go through `q4nx-build`.

## Build + verify (this machine)

**`utilities/build-all.sh` does all of it**: toolchain venv check -> export
every spec (all 6 build, both layouts, ~10 min total) -> `cmake --preset
linux-default` build of flm -> `cmake --install` into a prefix (default
`/opt/fastflowlm` if writable, else `./install`). Logs per spec in
`build-logs/`. Flags: `--prefix`, `--specs a,b`, `--skip-kernels`,
`--skip-app`, `--harness` (adds FLM_BUILD_OPEN_KERNELS_HARNESS=ON),
`--force`. Verified end-to-end 2026-09-07: 6/6 specs export, install tree
runs (`install/bin/flm --version` -> FLM v1.0.4 with open_kernels sets
present for every dense family).

```
uv venv --python 3.13 ironvenv && source ironvenv/bin/activate
pip install -r ironvenv-requirements.txt          # mlir_aie 1.4.2 + llvm-aie wheel
export PATH="$PWD/ironvenv/lib/python3.13/site-packages/llvm-aie/bin:/opt/xilinx/xrt/bin:$PATH"
# xclbinutil + aiebu-asm live in /opt/xilinx/xrt/bin (no ~/xrt-tools here)

python open_kernels/gguf_pool/gguf_pool.py                              # pool-law self-tests (bit-exact)
GEMV_SCALES_F32=1 python open_kernels/build_design.py open_kernels/designs/gemv_q4/gemv_q4.py <out>
LMHEAD_N=... LMHEAD_SCALES_F32=1 python open_kernels/build_design.py open_kernels/designs/lm_head_q8/lm_head_q8.py <out>
# full export builds both layouts: python open_kernels/export_qwen36_kernels.py --spec ...

cmake -S src/gguf -B build-gguf -DXRT_INCLUDE_DIR=/opt/xilinx/xrt/include -DXRT_LIB_DIR=/opt/xilinx/xrt/lib
cmake --build build-oq36 && ctest --test-dir build-oq36        # GGUF-PACK + OPEN-MANIFEST

# NPU harness (an accel0 NPU is present on this box):
python open_kernels/designs/gemv_q4/make_test.py --layout gguf --bands 4
cd open_kernels/designs/gemv_q4 && /tmp/opencode/harness/run_kernel run_qkv_gguf.cfg && python compare.py qkv_gguf
```

## Validation results (2026-09-07, this box)

- `open_kernels/gguf_pool/gguf_pool.py` self-tests: pool bytes dequantize **bit-exactly**
  to the GGUF blocks (Q4_1/Q4_0/Q8_0, several shapes, both band splits).
- C++ `std_perm_gguf` (GGUF-PACK ctest): bit-exact vs the reference dequant;
  **byte-identical to the Python packer** on the synthetic file AND on the real
  `mradermacher/Peach-2.0-9B-8k-Roleplay.i1-Q4_1.gguf` (196608 B compared).
- NPU: gemv_q4 f32-scale build **PASS cos=1.0, maxrel 3.9e-6, nbad=0** —
  numerics identical to the q4nx build (the reference narrows the f32 scales
  to bf16 first; make_test.py --layout gguf does this).
- lm_head_q8 f32-scale: all-ones and per-row-scale probes pass exactly;
  random-code fixtures show **maxrel ~5e-3 vs the q4nx build's 4.5e-6**
  (cos 0.999995). Suspect: the f32-emulated part/accum path under this TU's
  register pressure. KNOWN ISSUE — blocks the MoE family only (the dense
  recipe's lm_head is a q4 GEMV). Debug entry points: probe fixtures in
  designs/lm_head_q8 (w_probe*.bin / run_probe*.cfg), compare vs
  build_b8_syn (q4nx synthetic), and the epilogue vmac.f/vsrs.2x sequences in
  the two builds' .o files.
- OPEN-MANIFEST + fixture tests pass (nested gguf manifests for
  qwen3/gemma3); full pytest suite 65 passed.

### Gemma 3 12B f32-scale build (2026-09-15)

The 15360-wide activation table makes the f32-scale `dx_f32` main tile 768 B
too large with a 6 KiB stack. Keep the x FIFO at depth 2: lowering it to 1 is
invalid because the worker acquires two x elements. The verified fix is a 5
KiB stack on f32-scale main workers only; ordinary q4nx builds remain at 6 KiB.

```sh
PATH="$PWD/ironvenv/lib/python3.11/site-packages/llvm-aie/bin:/opt/xilinx/xrt/bin:$PATH" \
OPEN_KERNELS_SPEC="$PWD/open_kernels/recipes/specs/gemma3-12b.json" \
GEMV_SCALES_F32=1 ./ironvenv/bin/python open_kernels/build_design.py \
  open_kernels/designs/dense/dx.py \
  open_kernels/designs/dense/build_gemma3_h3840_f32
```

Verified with Python 3.11.15, mlir-aie 1.4.2, and Peano
21.0.0.2026080301+c9c5ecb7. Build completed in 42.5 s:

- `final.xclbin`: `5e00ea6231bdfff503942b4a9d3c82e39ad6e784411df515a8e6a7fb4208face`
- `insts.bin`: `9315d04c8a86995b493c09d6308c5e508d0e8677b4472a0f1401b7082f4d1ceb`

### Qwen2.5 / LFM2 estate (2026-09-18)

The dense recipe's Qwen2.5 and LFM2 spec exports now build cleanly, which
closes the last gap in the GGUF-direct end-to-end story for the dense
families: `dx_attn.py` streams the q/k/v bias and the 1024-byte split position
records (`ATTN_QKV_BIAS` / `ATTN_PTAB_SPLIT`), and `lfm2.py` re-exports
`band_bytes`/`chunk_bytes` from the dense recipe. The f32-scale (`*_f32`)
GGUF twins export in the same build, so a Qwen2.5-3B or LFM2 container shipped
as `model.gguf` (see `open-dense-kernels/SKILL.md`) has its full kernel set.

## Fine-tunes of a supported model (2026-10-21)

The GGUF-direct path is for FINE-TUNES as much as for the base models, and
the steps that used to stand between them are gone (branch `feat/fedora-docker`):

- **The gate is the manifest, not a family-name list.** The list of
  "GGUF-capable families" was generated from the recipes and compared with
  the WRONG names: the recipes say `qwen35` / `qwen36moe`, the registry (and
  every `details.family`) says `qwen3.5` / `qwen3.6-moe`. So the gate
  refused exactly the families it was written for. The capability is a
  property of the kernel set (its manifest carries the f32-scale `gguf`
  section), so the gate reads the manifest of the set the install links.
  `utilities/generate-gguf-families.py` is gone; on this branch the gate is
  `manifest_supports_gguf()` in `utilities/oflm-add/oflm_add/__init__.py`
  (reads the linked set's manifest for its `gguf` section). The deeper test
  `specs/open-engine/tests/test_gguf_twin.py` (the section may only name
  kernels the manifest's builds make) lives on `feat/fedora-docker`, **not
  in this tree** — do not look for it here.
- **The family of a fine-tune is derived from what the model IS.** Two
  routes, both from what is on disk: the *geometry* (the installed kernel
  set whose `ModelSpec.kernel_hash()` matches -- the depth and the
  tokenizer's id count are the manifest's business, the kernels' are not --
  names the official model, and the registry says that model's family) and
  the *base chain* (the README's base model, when it is an official model).
  "Ornith-1.5-9B" says nothing; it is a Qwen3.5-9B and the installed set
  says so.
- **The f32 twin is the recipes' business, in one place**
  (`open_kernels/recipes/gguf.py`): the section and the `*_f32` builds
  cannot disagree any more.
- **What a GGUF of the hybrid (qwen35) family needs** (the dense families
  were already there): llama.cpp's names for the linear attention
  (`ssm_*`, `attn_qkv`, `attn_gate`) and `post_attention_norm` for the
  pre-FFN norm; the SSM parameters f32 (or quantized) where the container
  keeps them bf16 (the consts ops decode and convert); the config.json may
  be the VLM's form (the text tower's fields under `text_config`); the
  eos ids come from the config.json when the chain has no
  generation_config.json.

### Still open: the hybrid family's f32 numerics (RESOLVED — see below)

**Superseded by "The lx_f32 wedge (FIXED 2026-10-22 — the mixed-law build)"**
and the sections after it: the NaN residual was the mixed-law build (recipe
not re-derived after the `GEMV_SCALES_F32` flip), the deadlock the same f32
build's non-integral state pieces, and the "reply garbage" the host sampler
zero-vocab crash plus the lm-head container gap below. The bisect knob
`OFLM_GGUF_POOL_Q41=1` (commit 5182da0) remains the law bisect for any new
hybrid-GGUF math question: with the same GGUF run through the *q4_1* twin —
the builds every shipped q4nx model answers on — the residual after the
layers is finite (nan=0; the packing is proven against the Python reference
by `gguf_pack_test`, so math questions narrow to the f32 chunk handling
itself). The layer_x's `make_test.py` + `compare.py` chain (fp64 replica,
spec from `OPEN_KERNELS_SPEC`) is still the tool of record for localising
any f32-law math fault to a stage (qkv, xres, xm, S, out).

### The geometry comes from the weights

A fine-tune's repo rarely carries its own config.json: the one the
installer fetches is the BASE's. The Qwen3.8-9B-Distill's GGUF says 33
blocks where the stack is 32 -- the extra block is the **MTP head**
(next-token prediction), whose tensors carry `nextn.eh_proj / enorm /
hnorm` and whose *regular* tensors look exactly like a full attention
layer's. llama.cpp's block_count counts it; the stack does not. So the
depth and the layer types are read off the file's tensors, with every
block that carries a `nextn` tensor excluded:

- the recipes' GGUF spec reads the depth and the layer types off the
  file's tensor names (a linear layer carries the fused `attn_qkv`, a
  full one the separate `attn_q` / `attn_k` / `attn_v`), minus the MTP
  block;
- the engine's derived config does the same (whole `nextn` blocks
  excluded), and for a GGUF the file's geometry wins over the
  config.json's claims (the model_type comparison is for the q4nx
  container's config only);
- oflm-add derives the model's spec from the GGUF too.

### A new depth is a new manifest, not a new build

The kernels are width-keyed; the depth is the manifest's business (the
layer loop it drives). So oflm-add, when no installed set was built for
the model's shape exactly, looks for a set that serves the same
*geometry* (`ModelSpec.kernel_hash()`) and writes the **sibling set**:
the base's kernel files (hard-linked) with this model's own manifest.
The Qwen3.8-9B-Distill (33 layers) runs on the Qwen3.5-9B's kernels
with its own manifest -- no export, no build, nothing for the user to
run.

## Gotchas learned here

- **designs/dense/gen_kernels.py regenerates gemv_q4_gy.cc / gemv_q4_gms.cc
  on every export** — hand-editing those two files is silently undone at the
  next export (cost an hour; looks like a mysterious file revert). The
  symbol-prefix fix lives in the GENERATOR templates now
  (GEMV_Q4_WRAP(GEMV_Q4_PREFIX, y/ms)). NOTE: designs/layer_x/gen_kernels.py
  still emits UNPREFIXED names, and **that is correct — layer_x does NOT need
  the prefix macro** (resolved 2026-10-25: `lx_f32`/`ax_f32` of Qwen3.5-0.8B
  built cleanly with the unprefixed `gemv_q4_gy` + the `-DGEMV_Q4_SCALES_F32=1`
  define). The dense recipe needs the `gemv_q4s32_*` prefix only because its
  **mixed-law** gemm/q8-split route puts a q4_1 gemv and an f32 gemv in the
  SAME binary, so the two symbols must not collide. A layer_x build is never
  mixed — `layout.py` flips the whole `SPEC.quant` wholesale, so exactly one
  `gemv_q4_gy` exists per binary and the compile define alone selects the body.
  Do not add the prefix macro to layer_x templates; it is dead weight there.

- `##` in a macro suppresses expansion of its operands: symbol prefixes need
  the two-level `WRAP(PFX, NAME) -> WRAP__(PFX, NAME)` pattern (see
  gemv_q4.h GEMV_Q4_ENTRY_, the wrapper .cc files, lm_head_q8.cc).
- GGUF tensor offsets are RELATIVE to the (aligned) data section; GGUF v3
  removed the alignment field (fixed 32). The Python parser in
  open_kernels/gguf_pool.py handles v2/v3; the C++ GgufFile mirrors it.
- The lm_head pool's band law is the **128-row supertile**
  (pool k <- file (4*(k/32)+(k%4))*8 + (k%32)/4), NOT the q4 band law —
  gguf_pool.pack_q8_pool_lmhead vs pack_q8_pool.
- Probe cfgs hardcode the build dir; after rebuilding a design, check WHICH
  build the cfg runs (a stale binary cost an hour here).
- harness taplib rejects a tap whose offset == tensor size (zero-band cores
  at tiny N): test with >= 1 band per core.

## The qwen35 f32 twin ported to `feat/gguf-direct` (2026-10-25, commit `88fef13`)

Fedora's qwen35 GGUF-twin work was ported **verbatim** (the whole branch diff
was f32-only), so the Qwen3.5 and Qwen3.8-27B sets on THIS branch emit a `gguf`
manifest section and build their `lx_f32`/`ax_f32` twins:

- `recipes/gguf.py` (new, byte-identical to fedora): `F32_QUANT`, `section()`,
  `twin_builds()`.
- `recipes/qwen36moe.py`: `pool_quant()`/`pool_chunk()`, `chunk`/`stack` params
  through `core_l1()`/`down_split()`, the f32 law in `common()`/`_common_dense()`,
  the `gemm_route` f32 early-return, and `GGUF_EXPERT_OPS_READY = False` (the
  MoE family stays off — no expert GGUF pack ops yet).
- `recipes/qwen35.py`: the twin key in `programs()`, `lx_set`/`ax_set` =
  `*_f32` when `f32`, the `GGUF.section` gate, `builds()` twin gate.
- `designs/layer_x/layout.py`: the `GEMV_SCALES_F32` spec flip
  (`replace(SPEC, quant="q4_1_f32")` + recipe re-derivation).
- `designs/layer_x/xcommon.py`: `F32` → `-DGEMV_Q4_SCALES_F32=1`.

Deliberately NOT ported at the time: rope-pairing (`spec.py` `ROPE_INTERLEAVED_FAMILIES`),
the export cache/race fixes, `LX_STOP` debug, and the `piece_tap`/`ln_split`
design hunks (those modules do not exist here). (Rope-pairing and the export
cache fixes have since landed — `c85d185`, see the banner.) `GGUF_EXPERT_OPS_READY` stays
`False`.

**Verification (this box):**
- Regression vs pre-change snapshots (`/tmp/opencode/gguf_baseline/`): every
  non-f32 spec is byte-identical except `build_key` (which hashes
  `recipes/*.py`, so the new `gguf.py` moves every key — expected); the qwen35
  specs gain `gguf` + `builds +[lx_f32, ax_f32]` and nothing else changes.
- Re-export `Qwen3.5-0.8B`: `python open_kernels/export_qwen36_kernels.py --spec
  open_kernels/recipes/specs/qwen35-0.8b.json -j 4` → only `lx`/`ax`/`lx_f32`/
  `ax_f32` rebuild (57–63 s each; program memory `lx_f32` 15712/16384,
  `ax_f32` 13536/16384), `ln`/`lm_head_q8` reuse. The exported manifest is
  **byte-identical to fedora's shipped one modulo `build_key`**: `gguf` chunk
  6144, `kernels.lx.build=lx_f32`, `kernels.ax.build=ax_f32`, nested
  `gguf.builds=[ax, lm_head_q8, ln, lx]`.
- `utilities/check-settings-ci.sh` OK; the export output under
  `src/xclbins/*/open_kernels*/` is git-ignored (built kernels are not
  committed — only the 5 recipe/design files are).

**Box-environment gotcha (this harness):** the shell carries
`OFLM_ROOT=.../OpenFlowLM-Next` (fedora's tree) and our tree has **no
`ironvenv`** — the only mlir-aie+Peano install is fedora's. `env.sh` honors a
pre-existing `OFLM_ROOT` (line 180-181), so without overriding it every path
points at fedora. The split that works: `OFLM_ROOT` → our tree, `OFLM_VENV_DIR`
→ fedora's `ironvenv`, unset the stale `OFLM_*` path vars (see
`/tmp/opencode/env_ours.sh`). The export writes repo-relative, so it lands in
our tree regardless.



## GGUF-direct on the qwen35 family: the shipped-manifest bug (found 2026-10-22)

Every qwen3.5-family set (0.8/2/4/9/27B) shipped with its `gguf` section naming
`lx/final.xclbin` and `ax/final.xclbin` — the **q4_1-law** builds — while a
GGUF model's pool is packed at the **f32 law**. The q4_1 kernels consumed
f32-law chunks (scale planes read as bf16 code words -> `~3.4e38` finite
garbage + uniform `±NaN(1)` `0x7F81`), which is the full NaN signature. The
dense sets got this right (`Qwen3-8B-NPU2` names `dx_f32/`), so a 3-minute
shipshape check is: **list `gguf.contexts` across `src/xclbins/*/manifest.json`
and expect the projection contexts to carry `_f32`** — any set naming `lx/` or
`ax/` in its gguf section will run bf16 kernels on f32 pools.

- Fix: `recipes/qwen35.py programs()` now derives `lx_set = "lx_f32" if f32
  else "lx"` / same for `ax_set`, mirroring `dense.py`'s `dx_set` switch
  (`programs(replace(spec, quant=q4_1_f32))` is what the gguf section runs, so
  the context paths must follow the quant). `recipes/dense.py` programs()
  ~line 697 is the reference pattern.
- Re-export is enough (`--no-build`): all five sets re-exported 2026-10-22,
  manifests regenerated, no rebuild needed (the twins were already built and
  correct — they were just orphaned by the stale names). The 9B repro build is
  byte-identical to the shipped `lx_f32/insts.bin` (`cmp` = OK).
- A misread that cost time: a dumped `XN` region full of NaN is NOT the norm's
  output — `lx.py` `(d)` dense_sequence stage 9 overwrites the spent A_XN
  region with a "junk xn" at layer end; the first norm's xn is invisible in
  end-of-run act dumps.

### The lx_f32 wedge (FIXED 2026-10-22 — the mixed-law build)

The genuine `lx_f32` build deadlocked at its first dispatch (ERT state 8
timeout) while the byte-adjacent q4_1 twin ran fine. Root cause: **the f32
build was mixed-law**. `designs/layer_x/layout.py` loaded the recipe
(`R = current_recipe()`) BEFORE the `GEMV_SCALES_F32` flip of
`SPEC.quant -> q4_1_f32`, so the layout constants (`_L`, `S_HEAD_BYTES` …)
were f32-law but every other recipe-derived law stayed q4_1: `xcommon`'s
`C = R.common` kept `CALL_BYTES/TILE = 5120`, so the `@w0..@w7` object fifos
compiled as `memref<5120xui8>` while the DeltaNet state piece is f32-law
67,584 B (= 11×6144, but 13.2×5120 — not an integral multiple). A piece that
is not a multiple of its object fifo's element can never complete its shim
acquire counts → deterministic deadlock at the first state piece. Forensics
that nailed it: `aiebu-dump -d insts.bin` on a fresh build — every shim
BLOCKWRITE `len` must be `k ×` its fifo element; per law
`67,584 / 6,144 = 11` ✓ vs `67,584 / 5,120 = 13.2` ✗.

- **Fix** (`designs/layer_x/layout.py`, one site): after the flip,
  re-derive the recipe — `SPEC = replace(SPEC, quant="q4_1_f32"); R =
  for_spec(SPEC).recipe(SPEC)` — exactly what `designs/dense/dx.py` (the
  working f32-law dense tail) already did; layer_x just never picked the
  pattern up. All recipe consumers (`C = R.common` in xcommon, `D =
  R.linear`/`R.attn` in lx/ax/ux) then agree with the layout.
- **Now verified on the NPU at every size**, all ten f32 twins
  (`lx_f32` + `ax_f32` of 0.8B/2B/4B/9B/27B) complete (probe state 4) and the
  fresh q4_1 builds are byte-identical (`cmp` ✓) to the shipped streams — the
  flip is inert without the env, so nothing else moved. The generated kernel
  TUs are law-stable (`PER_CALL = 2` is a module constant and the baked band
  quotients survive the ×6/5 chunk scaling), so the exporter's single
  law-blind `generate()` stays correct.
- `aiebu-dump`-how-to: 8-word shim descriptors `[len, addr_lo, 0, 0, ctrl,
  colbase, 0, colbase]`; per-column BD tables at `0x1d000 + bd*0x20`, column
  prefixes step 0x2000000. `txn2mlir.py` core-dumps on these streams; use
  `aiebu-dump -d` + the build's `final.prj/aie.mlir` instead.
- `utilities/lx-probe.cpp` gained a **generic N-buffer mode**: `lx-probe -g
  <xclbin> <insts.bin> <out_path> <wait_s> <in_bytes...>` — all-zeros inputs,
  the last buffer is dumped out on completion (`paths` empty ⇒ `n_in = 0`; an
  earlier version read past the vector and segfaulted). The ax stream's run
  arguments are `pool, xres, consts, KV, act, ptab` (manifest
  `full_attention.program[0].args` — passing KV/act/ptab in any other order
  segfaults the probe on a 24 MB-writing BD aimed at a small BO).
- End-to-end: prefill on `ornith-gguf:9b-i1` ran with `OFLM_OPEN_LAYER_NAN=1`
  — all 32 layers `xres nan=0`, decode began, and the engine then segfaulted
  in the host sampler (`Sampler::sampler_topk_apply`) — fixed separately, see
  "the sampler zero-vocab crash" below.

### The sampler zero-vocab crash (fixed 2026-10-22)

`oflm run ornith-gguf:9b-i1` segfaulted in `Sampler::sampler_topk_apply`
after a clean prefill: a Qwen3.5 GGUF pulled straight from HF keeps the
model's **nested config** (`text_config` holds the LM hyperparams
including `vocab_size`; only `hidden_size`/`eos_token_id` sit at the top).
`LM_Config` reads top-level only, so `AutoModel::set_sampler` built the
sampler over **zero features** — the `sample()` copy loop no-ops and
`sampler_topk_apply(top_k=20)` then calls
`std::partial_sort(begin, begin+20, begin)` on an empty vector (middle
beyond last) → UB → segfault on the first decode token. The q4nx converter
(`inject_oflm_keys` in utilities/q4nx-build) flattens `text_config` at
conversion time, but a pulled GGUF ships the model's raw config.
Fix: `LM_Config::from_pretrained` now lifts `text_config` into the canonical
top-level key set with the same semantics as the converter (`model_type`
always follows the text model; every other key only fills a missing top-level
hole; flat configs without `text_config` are untouched).

### The lx_f32 wedge (historical, superseded above)

With the manifest fixed, the real f32 twins actually run — and the **genuine
`lx_f32` build deadlocked at its first dispatch** (ERT state 8 timeout), while
the byte-adjacent q4_1 twin ran fine. Reproduced WITHOUT the engine:
`utilities/lx-probe.cpp` dispatches one context with the engine's exact call
sequence against file fixtures (the dumped `/tmp/opencode/f32_pool0.bin`,
`f32_consts0.bin`; missing or `-` files start as zeros — the wedge does not
need valid data). q4_1 completes in ~150 ms; lx_f32 timed out. Not staleness:
a warm rebuild was `cmp`-identical. Root cause and fix above.

- `utilities/lx-probe.cpp`: usage + fixture sizes in its header comment
  (manifest `gguf.layer_types.linear_attention.buffers` gives consts/state
  bytes; `globals.xres` etc.). Compiles with the XRT C headers from
  `$XILINX_XRT` (the dual-mode headers give the `xrt::` classes; there are no
  `.hpp` wrappers in this install).
- `utilities/act-region-report.py`: region census of a `--dump-act` file
  against the design's layout (XN/QKV/Z/VEC/O/OG/OUT/RES/XM nan + maxabs).
  Pass the model's spec json; `GEMV_SCALES_F32=1` (default) reads the f32 law.
- `OFLM_OPEN_DUMP_POOLS=<dir>` dumps layer 0's packed pool+consts per law
  (`{f32,q41}_pool0/consts0.bin`), `OFLM_OPEN_LAYER_NAN=1` does the per-layer
  xres NaN/maxabs census. Both are bring-up knobs, classified in
  `generate-registry.py` `NON_SETTING_ENV_VARS` (the registry check fails
  otherwise). `OFLM_GGUF_POOL_Q41=1` is the older law bisect; it now swaps
  kernel contexts through `std_perm_gguf`->`std_perm` pairing in core.cpp.
- **The LX_STOP truncation shim is a dead end**: `LX_STOP=1` (truncate part 0
  after the glue, in `lx.py` dense_sequence) wedges at BOTH laws — a truncated
  stream leaves the workers' fifo-acquire counts unsatisfied, so the run never
  completes without core-body cooperation. Stage isolation by stream
  truncation does NOT work here.

### The lm-head container gap (closed 2026-10-22)

With the sampler fixed, decode ran — and produced ~4.8 tok/s of garbage: ids
248068/271/248069/271 (the think template, correct) then a fixed cycling
of ascii-range ids 22–37 (`'7' ':' '<' '=' '>' '?' '@' 'A' …`). The dumped
logits (the CLI's `--dump-logits`) explained it: **47,695 of 248,320 logit
rows are NaN**, all within the first 191,736 ids, ~25% of them, no row
periodicity (mod 2/3/4/5/8 flat) — a garbage scale read, not a per-group
bug. Everything past file coverage recovers the memset zeros.

Root cause: the head op `lmhead_q8` (`src/gguf/gguf_file.hpp` / `open_kernels/gguf_pool/`) ~773) was a
plain memcpy in the kernel's supertile order — it assumed `lm_head.weight`
(= gguf `output.weight`) already held the engine's q8 chunk raster: 32-row ×
256-column chunks, 256 bf16 scales then 8192 int8 codes, at pool chunk
k ← file `(4*(k/per_band) + k%4)*nk + k%per_band/4`, `per_band = 4*nk`
(nk = in_dim/256). Only a q4nx-build conversion writes that (Q8NX layout).
A raw llama.cpp GGUF stores the head row-major in its own quant; Ornith-1.5's
`output.weight` is **Q6_K** (4096 × 248320), `Qwen3.8-9B-Distill-GGUF`'s
is whole-file **Q8_0** (byte count even lands exactly on the chunk grid —
1,080,688,640 = 124,160 × 8,704 — but row-major raster + fp16 scales,
still wrong). The op's `bounds()` silently passed a short count and filled
the pool with misread bytes.

Implemented (`lmhead_q8`, GGUF branch): the same transcode the pool ops do.
  - Q8_0: pure remap — the source already carries per-32-block int8 codes
    and an fp16 scale, so widen fp16→bf16 (RNE, matching the converter's
    cast) and write into the chunk raster at the plain (row-major) index
    `(row/32)*ncol + kt` — a raw GGUF is row-major, so no supertile
    permutation (the memcpy path permutes because the q4nx container stores
    the head in supertile order). Bit-identical to a q4nx conversion of the
    same Q8_0 tensor.
  - anything else (Q6_K, Q4_K, Q4_0, ...): dequant each row to f32 (the
    same llama.cpp-faithful path `std_perm_gguf` uses for the pool tensors)
    and requant to int8 code + bf16 scale per 32-value group
    (`bf16_rne(mx/127)`, clamp to [-127,127]), at the same plain index.
Verified: a fresh CLI run on ornith-gguf:9b-i1 now reports **0 NaN logit
rows** (was 47,695) with a real argmax; `gguf_pack_test` + the registered
ctest (GGUF-PACK, OPEN-MANIFEST) pass; the q4nx path is untouched
(`qwen35-npu2:0.8b` still answers a full coherent "**Paris**." turn).

Open: the 9B hybrid GGUF path **hangs at prefill layer 30 (full_attention)**
on every run — it reproduces with both the f32 and the q4_1 law, so it is
not the head and not the law; it is a per-layer slice issue on the 9B hybrid
path (the 0.8B native and the dense GGUFs run cleanly) that needs its own
investigation. Until that is resolved, hybrid GGUF-direct does not complete
a turn on this box.

## First size of a family that only had f32 twins: Qwen3-14B (2026-10-23)

The native (`qwen3-0.6b` .. `8b`) and Qwen3.5 sizes were in the catalogue;
the qwen3-14b spec built everything EXCEPT the `lm_head_q4` head — its
catalogue point at K=5120 (hidden 5120) had never been fixture-run, so the
export died on `lm_head_q4: K=5120 is outside the validated set`. The
attention tuple (128, 40, 8, 128) was already in (entered by the attention
gate, qwen3-14b earlier).

The clean run, entirely through `utilities/`:

1. **Fixture.** The `lm_head_q4` design had no fixture of its own
   (`build-new-spec.py`'s OP_FIXTURE named `designs/lm_head_q8/make_test.py`
   as the model to port). Ported to
   `open_kernels/designs/lm_head_q4/make_test.py` + `compare.py`: fp64
   reference from the pool bytes, the band law IS `q4_1_pack`'s standard one
   at rs=2 (64-row bands; the uneven 8-core band split is taps, not bytes),
   `--layout q4nx|gguf` for the bf16-scale and f32-scale (GGMF-direct)
   chunk layouts, `--k <width>`. NPU run at K=5120 (4096 rows, 64 bands,
   8 x 6+2 over the cores): q4nx **PASS cos=1.0, maxrel 4.3e-06**; gguf
   **PASS cos 1.0, maxrel 3.9e-06**.

   Fixture trap worth remembering: the f32-scale kernel NARROWS the pool's
   f32 scales to bf16 in flight (`gemv_q4.h load_scale`, documented as
   "numerics bit-identical to the q4nx chunks; the pool keeps the GGUF scale
   values exactly"). The reference must narrow the scales the same way
   (`deq_chunk`, scale dtype bf16) — comparing against the exact f32 gives
   maxrel ~2.3e-3 / 335 bad rows: a fixture bug, not a kernel bug. The
   `lm_head_q8` fixture keeps the same discipline via
   `dequant_pool_lmhead(..., scale_dtype=bfloat16)`.

2. **Catalogue.** `utilities/catalogue-add-point.py` now takes
   `--param K --value 5120` next to `--point ...` (a values() Param is not a
   combination() tuple; both writes are AST-located, comment-placed, and
   verified after writing; a params dict can be a POSITIONAL child of the
   Template, so the locator checks both shapes). The insert is
   character-offset based — appending to a values() call whose args sit
   inline with its `)` on the same line must NOT emit a new source line, or
   `5120,` lands outside the call (the first attempt produced
   `Template(...) got multiple values for 'note'`; reverted, then fixed to
   char-offset insertion with the note at the params dict's line end).

3. **Export.** `utilities/export-kernels.py --specs qwen3-14b --skip-bert`
   through the validated path: `ok: qwen3-14b`. Notably the exporter prints
   **"no GGUF-direct twin (dx_f32): the f32-scale build does not fit a core
   at this shape"** — the 14B set ships only the q4nx head; a GGUF-direct
   install of a Qwen3-14B-geometry model rides the lm-head q8 requant path
   (`lmhead_q8`, pools.cpp) instead, so the missing twin is not a blocker.

Commit: the fixture pair, `recipes/catalogue.py` (the K=5120 point),
`utilities/catalogue-add-point.py` (--param/--value), and `OP_FIXTURE` in
`utilities/build-new-spec.py` updated to name the new fixture.

## Filling out the missing sizes (2026-10-23)

`utilities/build-new-spec.py --repo <hf-repo> [--spec-name <stem>]` run over
the official registry's gaps. What landed:

- **llama32-1b, llama32-3b, lfm2-2.6b** — full chains (spec → gate PASS →
  catalogue → build → export). Shipped as the canonical set names
  `Llama-3.2-1B-NPU2` / `Llama-3.2-3B-NPU2` / `LFM2-2.6B-NPU2` (extra.model
  must be the registry name — the name test enforces it; export writes
  `src/xclbins/<extra.model>/open_kernels`, so fix extra.model BEFORE the
  first export, not after).
- The engine's match key is `ModelSpec.kernel_hash()`
  (`open_kernels/recipes/spec.py`). `gen-kernel-spec.py` now matches existing
  specs and generates candidates THROUGH `ModelSpec.from_hf_config` /
  `kernel_hash()` instead of its old family+hidden shortcut, which produced
  two false links this session: Llama-3.2-1B ↔ minicpm5-2b (same family+hidden,
  head_dim 64 vs 128 — different compiled attention) and — the subtle one —
  LFM2-2.6B ↔ lfm2-1.2b (same family+hidden, FF 10752→adjusted **7168** vs
  the 1.2B's **8192**). After the fix LFM2-2.6B correctly got its OWN kernels.
- Geometry links that are genuine (kernel-hash equal, link-only, no artifact —
  build-new-spec deletes fresh link files): LFM2.5-1.2B → lfm2-1.2b (same
  conv/full pattern [2,5,8,10,12,14], FF 12288→adjusted 8192, rope 1e6),
  DeepSeek-R1-0528-Qwen3-8B → qwen3-8b, DeepSeek-R1-Distill-Llama-8B →
  llama31-8b (its config carries the llama3 rope_scaling).
- **gemma3-1b is a recipe wall, not a spec**: head_dim 256 with ONE kv head
  (cos/sin straddle two 512-byte attention elements in the dense design) and
  hidden 1152 (not a multiple of 256). Candidate parked at
  `open_kernels/recipes/candidates/gemma3-1b.json` — needs a design change,
  do not re-attempt without one.
- LFM2-2.6B numbers worth remembering: hidden 2048, 30 layers (layer_types
  derive from its config's `layer_types` list, same conv/full mix), FF
  10752 → **7168** after the swiglu auto-adjust; the registry's "2.6B" is a
  deeper LFM2-1.2B, NOT a wider one.
- Tests: the frozen spec_hash map in `specs/open-engine/tests/test_lfm2.py`
  needs an entry for every new shipped spec (asserts the set of specs.json
  names first, then each hash); the `manifest_qwen35_9b` fixture was stale
  from the GGUF-direct switch and regenerated via `make_fixtures.py` (IRON
  python) — its only diff was the gguf section's f32 build dirs
  (ax/lx → ax_f32/lx_f32), the expected divergence.
- `cmake --build build --target export_kernels`: **25 specs + BERT**, ok.
- Pre-existing environmental failure on this box only:
  `test_qwen35_wide_glue.py::test_small_bank_pointer_arithmetic_in_compiled_cpp`
  — the test links with `-fsanitize=undefined,bounds` and Fedora's sanitizer
  runtime (`/usr/lib64/libubsan.so.1.0.0`) is not installed; test/fixture/
  dn_glue are byte-identical to origin/main.

## Adding a size: what the script does now, and what it still does not (2026-10-26)

One command, no hand steps:

```bash
source utilities/env.sh
python utilities/build-new-spec.py --repo Qwen/Qwen3-0.6B        # or --spec-name <stem>
```

| Step | Automated? | Where |
|---|---|---|
| the spec JSON from HF config (or a `link_to` when the base already has kernels) | yes | `utilities/gen-kernel-spec.py` |
| the tokenizer's **real_vocab** (the logits mask; a generated spec used to ship none) | yes, since `c72d08d` | `gen-kernel-spec.py` (`--refresh` re-derives an existing spec) |
| the model name (`extra.model`, the `src/xclbins` directory) | yes, derived; `register-spec.py` records it when the derived name differs | `test_spec_model_names.py` |
| the **attention** point: run the production worker at the geometry vs an fp64 oracle | yes | `designs/attn/probe_dense.py` (step 3b) |
| the **other op** points (gemv K, ln width, lm_head K): run each op's own fixture | yes for `gemv_q4`, `ln`, `lm_head_q4`; other ops abort with a named reason | `utilities/op-gate.py` |
| the catalogue edit (values() AND the whole-tuple combination) | yes | `utilities/catalogue-add-point.py` |
| the build, the manifest, the `gguf` twin | yes | `utilities/export-kernels.py` |
| the frozen test lists (spec_hash, EXPECTED name) | yes — never hand-edit them | `utilities/register-spec.py` (step 6) |
| **what is missing at all** (registry + another checkout) | yes | `utilities/spec-gaps.py` |

`python utilities/spec-gaps.py --probe <hf-repo>` (repeatable) says whether a
SIZE would be an easy add, without writing a spec: "needs attention gate +
gated: gemv_q4" / "RECIPE WALL: an FF of 18944 does not fit ... a three-way
split is not implemented" / "no recipe for this config (<the wall>)". Plan an
expansion from that, not from a guess. Known verdicts (2026-10-26): Qwen3-32B
and Qwen2.5-14B automatable; Qwen2.5-7B needs a 3-way FF split; Gemma 3 27B is
a kernel wall (query_pre_attn_scalar 168 vs head_dim 128).

`python utilities/spec-gaps.py --reference ../OpenFlowLM-Next` answers "what does
that tree have that this one does not" with a classification per spec (builds
here / needs N points, gated or not / recipe wall), and
`--reference`-less mode checks the official registry. A model that shares a
family and parameter count with a shipped spec is reported as a `link` (it rides
that set by `kernel_hash`); a `*-candidate.json` is the generator's default
output name, not a spec.

What is still a design step (the script refuses, with the op and the value):

- **`lm_head_q8` at a new K** — the fixture compiles K as a module constant (2048).
- **`gemv_q8`, `deltanet`, `short_conv`, `gemv_q4_prep_f32`, `router`, `moe`** —
  no per-geometry fixture; each was validated through a whole-layer run.
- **recipe walls** (a width that does not fit L1, cos/sin straddling the
  attention element): `build-new-spec` moves the spec to
  `open_kernels/recipes/candidates/` instead of shipping a set that cannot build.

`utilities/op-gate.py` also runs standalone, which is the fix for "the build
failed on one off-grid width":

```bash
python utilities/op-gate.py --spec qwen3-8b            # list + gate every missing point
python utilities/op-gate.py --spec qwen3-8b --op ln    # one op
python utilities/op-gate.py --spec qwen3-8b --dry-run  # what it would run
```

Reading a missing point: `recipes/catalogue.py` is a list of what has actually
been RUN on the hardware. `catalogue.collecting()` (used by both scripts) means
one `recipe()` pass reports EVERY missing point, not just the first that would
raise — so a new size never has to be discovered one failure at a time.

## The config.json trap in `oflm add` (found 2026-10-26, `108391f`)

`oflm add empero-ai/Qwen3.8-4B-Distill-GGUF` installed fine (geometry matched
`Qwen3.5-4B-NPU2`) and then `oflm bench` refused it:

    open_qwen36: Qwen3.8-4B-Distill-GGUF: config.json lacks 'layer_types'

The repo's config is a MULTIMODAL wrapper -- `hidden_size`, `num_hidden_layers`,
`layer_types`, `full_attention_interval` under `text_config`, the top level
holding the vision keys (Qwen/Qwen3.5-4B's own config is the same shape). The
engine's checks are top-level, and it merges the GGUF's DERIVED metadata only
for keys config.json lacks: that supplies `hidden_size` and friends, but the
GGUF has no `layer_types`, so the check walks all the way to it and refuses.

`reconcile_config_json()` now makes the installed config self-contained:
flatten `text_config`/`llm_config`/`language_config` into the top level (the
wrapper is kept), fill anything still omitted from the kernel set's
`hf_config_check` (skipping keys with a manifest default), and REPORT a key the
config contradicts. The engine flattens at load too, so an install from before
this fix still loads.

Generalisable lesson: a GGUF install is only as good as the config.json beside
it, and neither the repo's nor its base's config is guaranteed to be in the
shape the engine validates. Anything `hf_config_check` names must be present
after the install -- `config.json` is part of the kernel set's contract, not
commentary.

## The spec batch: one easy, three named pieces (2026-10-27, `5e95f7f`)

Ran the sizes that "only lacked a spec". Result:

| model | outcome |
|---|---|
| **Llama-3-8B** (plain rope) | **shipped** — `Llama-3-8B-NPU2`, spec derived from the GGUF (`gen-kernel-spec.py --gguf`, the HF repo is gated), then `build-new-spec.py --spec llama3-8b` |
| granite-4.2-8b (FF 12800) | gate: `gemv_q4 K=12800`. The FF fits the *recipe's* L1 model so it is not split, and the standalone fixture cannot hold a K above ~12288 |
| Hy-MT2-1.8B (128, 16, 4, post-rope qk norm) | the **attention probe** does not cover this shape: its k output is wrong even for a ZERO input (k rel 0.68), while `groups` passes — look at attn_meta/attn_k element slicing at kvh=4 before suspecting the kernel |
| K2-Horizon-0.9B | `ln width=1536 groups=2` does not build; gemv K=1536 and lm_head K=1536/vocab 64256 ungated; and its config nests rope under `rope_parameters.rope_theta`, which the k2 family builder refuses (the spec is the template fallback) |

All three parked in `recipes/candidates/` with `extra.blocked_on` recording the
above, so nobody re-derives it.

Two workflow additions made this batch possible and are worth remembering:
`gen-kernel-spec.py --gguf <file>` (a gated/absent HF config, geometry from the
file) and `build-new-spec.py --spec <stem>` (build/gate/register an existing spec).

Also fixed: the GGUF **arch** map and the HF **model_type** map had been mixed
together, so `k2_horizon` / `hunyuan_v1_dense` stopped deriving a family. They are
separate vocabularies (k2-horizon vs k2_horizon); a test now pins both.

## The two "impossible" sizes share ONE enabler (2026-10-27)

Measured, not guessed:

* **Qwen2.5-7B** (FF 18944): `_ffn_split` offers (9216, 9728) and the recipe
  refuses -- "does not fit a main core's L1 in two pieces of whole f32 elements
  ... a three-way split is not implemented".
* **Qwen3-32B** (FF 25600): the two-way split is (12288, 13312), and 13312
  cannot be gated (the standalone gemv fixture holds the K-wide table + the
  K-wide activation in one core: ~59 KB of 64 KB).

Both want the SAME feature: a **three-way FF split**. 25600 = 9216 + 8192 + 8192
and 18944 = 6400 + 6272 + 6272 put every piece at or below 12288, i.e. inside
what the gate can measure. What it needs beyond `down_split`/`ln_add3` is the
norm helper's third partial output (`ln_add4`) and the split's 3-piece DMA walk.
So "impossible" here is one kernel piece, not two models' worth of work.

**Granite 4.2's rambly output is (very likely) the chat template**, not the
numbers: Granite 4.2 wraps a thinking block, and a fixed Qwen3.x template (used
for tool handling) does not fit it. Retest with Granite's own template before
suspecting the fold.

## Vision from GGUF: the projector is the whole tower (2026-10-27, `669aba8`)

Every non-text modality was closed-kernel before this: of the 54 sets in `src/xclbins`,
28 have `open_kernels/` and all 28 are text. Vision is the one that needs **no NPU
kernel work at all**, because the tower is host CPU (`open_qwen36/vision/vit.cpp`) and
a llama.cpp projector is plain (not pre-tiled for the closed engine's `vision_mm`).

What a `mmproj-*.gguf` actually contains (read from
`mradermacher/Qwen3-VL-4B-Instruct-GGUF`, the NON-i1 sibling -- an i1 repo does not
carry it):

```
general.architecture = clip            clip.projector_type = qwen3vl_merger
clip.vision.block_count 24   embedding_length 1024   attention.head_count 16
clip.vision.feed_forward_length 4096   projection_dim 2560
clip.vision.patch_size 16    spatial_merge_size 2    attention.layer_norm_epsilon 1e-6
clip.vision.is_deepstack_layers [ ..., 5, 11, 17 ]
tensors (316 = 12*24 + 6*3 + 10):
  v.blk.N.{attn_qkv,attn_out,ffn_up,ffn_down,ln1,ln2}{,.weight,.bias}
  v.patch_embd.weight + .weight.1 + .bias   v.position_embd.weight
  v.post_ln.{weight,bias}                   mm.0 / mm.2 {weight,bias}
  v.deepstack.{5,11,17}.{norm,fc1,fc2}
```

Three consequences, each of which saved a piece of work:

1. **The file states the geometry.** `clip.vision.*` carries what the container made
   `oflm add` write into a side `vision.json` (block count, widths, heads, the deepstack
   indexes). `VitConfig::from_gguf` reads it and cross-checks the shapes, so that
   hard-coded side file is no longer needed for a GGUF model.
2. **The deepstack indexes are in the tensor NAMES** (`v.deepstack.5.*`), so they
   survive every writer even when the metadata array does not.
3. **The temporal split is the slice count**: `v.patch_embd.weight` + `.weight.1` is how
   llama.cpp splits the Conv3d, so 2 tensors = temporal patch 2.

The mapping (`load_vit_gguf`), and the traps:

* every matmul is ggml `[in, out]` (fastest dim = input) -> tower `[out, in]`; get this
  backwards and the numbers still look plausible;
* the fused `attn_qkv` maps straight onto the tower's existing `B.qkv` (the container
  path already reads fused qkv);
* the patch embed becomes `(C, T, P_h, P_w)` -- the order `reorder_patches_inplace`
  emits (`imageproc.cpp`), which is HF's Conv3d input order;
* `v.post_ln` normalises over `hidden` (before the 2x2 concat) but a deepstack merger
  normalises over `hidden * merge^2` -- the widths differ, which is what stops them
  being swapped silently.

**`GgufFile::f32()` is not for this.** It assumes `[rows, cols]` with `rows = dims[1]`;
on the 4-D patch embed that decodes 256 of 1,572,864 values. `f32_all()` is the any-rank
form (`dequant_row` refactored out of `embed_row`).

Verification, both cheap and committed:

* `vit_gguf_test` (`ctest OPEN-VISION-VIT-GGUF`): writes its own projector with
  position-revealing values, checks the patch split, the qkv/merger transposes, the
  deepstack norm width, and the three refusals (not a projector; a projector this
  reader does not run; one extra tensor). No device, no weights, no network.
* `utilities/check-mmproj.py --mmproj <url> --safetensors <repo>`: compares a REAL
  projector against the checkpoint it was converted from using byte ranges only. On
  Qwen3-VL-4B's mmproj-f16 every mapping agrees to max|d|/rms ~1e-6, the position table
  exactly. **Use RMS-relative error, not per-element relative**: near-zero weights make
  the latter read 3% on a tensor that is bit-exact.

### VISION FROM GGUF IS DONE (2026-10-27) -- and what it took

End-to-end on the NPU, no environment, `oflm add mradermacher/Qwen3-VL-4B-Instruct-i1-GGUF`
then `oflm serve qwen3-vl:4b`:

```
open kernels: <model>/open_kernels      vision tower resident (mmproj.gguf, 24 blocks) in 0.7 s
image 18x88 patches -> 396 tokens + 3 deepstack in 5.63 s
REPLY: The image displays a black background with white text stating that Paris is the
       capital of France and serves as its center for finance, commerce, culture, arts,
       fashion, and science.
```

An i1 repo has no projector: strip `-i1` and take `*mmproj*.gguf` from the sibling (f16 --
the tower is host bf16, so f16 is lossless and needs no dequant). `oflm add --text-only`
skips it, which is the same code path a text-only checkpoint takes; the engine then simply
has no tower.

Three pieces made it work beyond the reader:

1. **The arch and the kernels disagree, on purpose.** Qwen3-VL's GGUF arch is `qwen3vl`,
   and its decoder IS qwen3's (hidden 2560, 36 layers, FF 9728, 32/8) -- the `qwen3`
   builder reads `qwen3vl.*` keys unchanged. So the SPEC hashes as family `qwen3` (which
   is what finds the kernels) while `details.family` stays `qwen3vl` (which is what picks
   the vision adapter, whose `preprocess_image` + `prefill(ids, payload)` do the images).
   Its rope base is 5e6, not qwen3-4b's 1e6, and `kernel_hash` keeps `rope_theta` (the
   ptab's `inv_freq` is manifest data), so it needs its OWN set -- `qwen3vl-4b` exports
   as `Qwen3-VL-4B-Instruct-NPU2`, the official name, which is also how the registry
   names the family (`family_of_official`).
2. **The projector describes config.json too.** A GGUF model has no `vision_config`
   anywhere: `projector_vision_config()` translates `clip.vision.*` back into it (depth,
   hidden, heads, FFN, projection, patch/merge, in_channels, temporal = the patch_embd
   slice count, `num_position_embeddings` from the table's own shape, and the deepstack
   indexes from the tensor names), and the image/vision special-token ids come from
   tokenizer.json. Without it the adapter's patch size is 0 and no image can be prepared.
3. **M-RoPE was already done.** Every manifest carries `globals.ptab = {per_row,
   inv_freq}`, `Core` allocates `max_ctx x per_row` and builds it, and `write_record()`
   rewrites the rows an image touches with its 3 positions. The only missing piece was
   `mrope_section`, which the engine reads from config.json -- so the install writes
   `rope_parameters` from `qwen3vl.rope.dimension_sections` ([24,20,20,0] -> [24,20,20],
   interleaved, which is Qwen3-VL's law).

### Two bugs that only showed up at load (both pinned by tests now)

1. **`core.cpp` skipped the GGUF section when the chunk laws matched.** It compared
   `w_->chunk_bytes == man_.chunk_bytes` and, if equal, used the BASE section -- but the
   base's pack ops read a `.q4nx` (`std_perm`) and only the section has the GGUF-reading
   ones. Since the section moved to the q4_1 law, "equal" is now always true, so EVERY
   GGUF model died with "lm_head.weight is not in a q4nx container". The merge must always
   happen; the shortcut is gone.
2. **Two recipe decisions still keyed on `q4_1_f32`,** the old twin's name: the lm head's
   pack op and Gemma3's embedding scale (a container pre-scales token_embd by
   sqrt(hidden) at conversion; a GGUF's is raw). Both were silently wrong for a
   `q4_1_gguf` section -- the head refused, and Gemma3 would have answered nonsense.
   `ModelSpec.reads_gguf` is now the predicate for "these weights are a GGUF FILE" (either
   law), and `specs/open-engine/tests/test_gguf_section.py` pins both plus the predicate.

Rule of thumb for the next family: `q4_1_f32` means the f32 CHUNK law; `reads_gguf` means
the weights came from a GGUF. Any `== "q4_1_f32"` that meant the latter is a bug.

## The install+serve matrix (2026-10-27): what the shipped build needed

A full matrix on an installed build (`cmake --workflow --preset linux-default` +
`sudo cmake --install build`) found five product-level gaps. All fixed in
`e2cd48a` / `e0b8fe4` / `04dd123`:

1. **The GGUF section is now the q4_1 law, always.** It used to be the f32-scale
   twin and was omitted when that twin did not fit -- so `oflm add
   unsloth/Qwen3-14B-GGUF` refused with "no GGUF-direct open kernel set found"
   while `Qwen3-14B-NPU2` existed. `GGUF.section()` now returns the quant name the
   nested manifest is built under (`q4_1_gguf`: the base chunk law, base kernels,
   GGUF-reading pack ops), and its argument became an explicit "is this family
   GGUF-ready" (the MoE still passes False). Verified: 14B links again.
   The f32-scale twins are no longer built at all -- exports are smaller/faster,
   and `OFLM_GGUF_POOL_F32` has no kernels to reach (kept for old sets).

2. **A GGUF install always ends with a usable config.json.** A gated base
   (meta-llama/*) left a stub `{"model_type": "mistral"}`, which derives no spec
   (hence "No open-kernel spec ... closed kernels only" for `MaziyarPanahi/
   Llama-3.2-3B-Instruct-GGUF` and `mradermacher/gemma-3-4b-it-i1-GGUF`) and built
   the sampler over zero features (the `Meta-Llama-3-8B` segfault after prefill).
   `repair_config_from_gguf()` writes the geometry the GGUF itself names, and
   `spec_from_gguf_source()` derives the spec from the file when the config yields
   none -- so the geometry match runs and the sampler has its vocab.
   (Also: the two `from_gguf_metadata(md, names)` calls passed an argument our
   `spec.py` does not take, so *every* GGUF spec derivation silently failed.)

3. **A baked rope table makes rope_scaling irrelevant.** llama.cpp writes
   `rope_freqs.weight` when it baked Llama-3's scaling in, and the GGUF's metadata
   then has no rope_scaling -- which no set could match. `rope_ignore()` drops
   `rope_scaling` from the comparison for exactly those files. Llama-3.2-3B now
   links; Meta-Llama-3-8B (no tensor, plain rope) correctly does NOT link the
   llama3.1-8B set -- it needs a plain-rope spec of its own.

4. **Family names.** `k2-horizon` (K2-Horizon-0.9B said "could not determine
   details.family"), `hunyuan-dense`, `qwen35`, `qwen3next` added to the GGUF arch
   and config maps. A missing set now says "no open kernel set matches this
   model's geometry" instead of blaming the nearest official size.

5. **Load time.** The requant loop is parallel now. It is still slow for a
   K-quant source (a Q4_K_M 8B: 71 s, warm cache -- CPU, not I/O) because
   `embed_row` decodes the WHOLE row for every 256-wide chunk: a 4096-wide
   projection decodes each row 16 times. Decoding a 32-row band once and writing
   its ncol chunks is the fix that takes it to a few seconds. Quant preference
   itself already prefers Q4_1 (byte-exact) over the K-quants.

**Sizes still without a set** (all "no open kernel set matches this geometry",
and each needs `build-new-spec`): Qwen2.5-7B (recipe wall: FF 18944 needs a
three-way split), Qwen3-32B (K 13313 cannot be gated by the standalone gemv
fixture), granite-4.2-8b, Hy-MT2-1.8B, K2-Horizon-0.9B, Meta-Llama-3-8B (plain
rope), Llama-3-8B-Instruct (same). Phi-4-mini still needs the fused-qkv plan.

## Dense: 7 of 8 shipped (2026-10-26, `298ed19`)

Verified end-to-end on the NPU with no environment, one prompt, temperature 0:

| family | model | answer |
|---|---|---|
| qwen3 | Qwen3-4B-Base Q4_0 | " Paris. Can you provide ..." |
| llama3 | Llama-3.2-3B-Instruct Q4_0 | "Paris." |
| gemma3 | Gemma3-4B-IT Q4_1 | "The capital of France is **Paris**." |
| k2 | K2-Horizon-7B-i1 | coherent reasoning |
| qwen2 | Qwen2.5-3B-i1 | "The capital of France is Paris" |
| granite | granite-4.2-3b-i1 | "Okay, the user said ... Paris is" |
| hunyuan | Hy-MT2-7B-i1 | "La capitale de la France est" |
| phi3 | Phi-4-mini-Instruct | **not yet** -- see below |

What each of the three hard ones needed, because the next family will hit one:

* **granite**: `attention_multiplier` must be folded into q_proj. A container is
  folded by q4nx-build; a GGUF is not. The spec records the PRE-fold value in
  `extra.granite_attention_multiplier` (from the config, or from
  `q4nx_folded_multipliers` when a container states it); dense.py puts `fold` on
  the q_proj op of the **std_perm_gguf** plan only; `std_perm_gguf` forces the
  requant path when `fold != 1` and scales the decoded block; check_model accepts
  a LIST of valid values, so one set serves the folded container (0.125) and the
  unfolded GGUF (0.015625).
* **hunyuan**: nothing had shipped one, so `model_family_map()` had no entry and
  the server said "not a chat model". It maps to the Granite adapter (dense,
  open engine, template-driven). Then `bos_token_id` was `null` with the token
  string absent from tokenizer.json: oflm-add resolves ids from tokenizer.json OR
  from the GGUF's `tokenizer.ggml.*_token_id`, treats null as missing, and writes
  before the "nothing to merge" early return.
* **qwen2.5**: the HF FF is 11008 and 11264 is the converter's pad. The padded
  container entry is gone and the spec is the true 11008, whose `gemv_q4` K was
  gated on the NPU. Build-new-spec skips the attention gate when the tuple is
  already validated (the probe refuses q/k/v-bias families by design).

**phi3 is the last one**, and it is recipe work, not a spec: llama.cpp's phi3 arch
stores `blk.N.attn_qkv.weight` FUSED, while our phi3 plan expects separate
q/k/v. The Instruct GGUF therefore fails at "no tensor ... self_attn.q_proj".
Needs a fused-qkv plan (split in_dim / chunk0, like the qwen35 gated-q split).

## The five dense families that were never run: what each one needs (2026-10-26)

Pulled the user's mradermacher i1 repos (Q4_1 first) and served each:

| family | repo | result | what it needs |
|---|---|---|---|
| k2 | K2-Horizon-7B-i1 | **works** — coherent reasoning output | nothing |
| hunyuan | Hy-MT2-7B-i1 | installs and packs; the server refuses "not a chat model" | the engine's `model_family_map()` has no `hunyuan`, and there is no adapter class: nothing has ever shipped it as a chat model. Needs a family entry + prompt format |
| qwen2 | Qwen2.5-3B-i1 | geometry mismatch (FF) | HF/GGUF `intermediate_size` is **11008**; our spec (and AMD's container) are **11264** — the container is padded by one k-tile. A GGUF needs a spec at 11008, or the padded container entry goes and the spec is rebuilt at the true FF |
| phi3 | Phi-4-mini-reasoning-i1 | geometry mismatch (rope) | the **reasoning** variant's longrope `short_factor` equals its long factors; our phi4-mini spec is the **Instruct** one (short = 1.0). Same kernels, different ptab table -> its own spec/set. Phi-4-mini-**Instruct** GGUFs ride the existing set |
| granite | granite-4.2-3b-i1 | refused: our family builder rejects it | `attention_multiplier = 0.015625` != 1/sqrt(head_dim 64) = 0.125. AMD's container is **folded** (q_proj pre-scaled); a llama.cpp GGUF is **not**. Needs a pack-time fold of q_proj by multiplier/ (1/sqrt hd) |

Dense verified end-to-end: **qwen3, llama3, gemma3, k2** (4 of 8). Two of the four
remaining are just specs to generate (qwen2 at 11008, phi3-reasoning); the other
two are engine pieces (the hunyuan adapter, the granite fold).

## DENSE IS DONE; the f32-scale twin is the trap (2026-10-26, `31d3636` + default flip)

A GGUF now runs on the **q4_1 twin by default** (`OFLM_GGUF_POOL_F32=1` opts into
the f32 twin as a probe). With no environment at all, temperature 0, one prompt:

| model | dialect | answer |
|---|---|---|
| Qwen3-4B-Base (Q4_0) | half-split rope + qk_norm | " Paris. Can you provide ..." |
| Llama-3.2-3B-Instruct (Q4_0) | interleaved rope | "Paris." |
| Gemma3-4B-IT (Q4_1) | sandwich norms, partial rope, hd 256, window 1024 | "The capital of France is **Paris**." |
| Nanbeige4-3B-Base (Q4_1) | llama3, 20/4 heads, FF 10496 | "ital of France is Paris." |

How we got there, because the next family will need the same forensics:

1. `OFLM_OPEN_DUMP_POOLS=<dir>` writes layer 0's packed pool + consts. Under the
   q4_1 law the engine's chunk for gemma3-4b's layer-0 q_proj is **byte-identical**
   to AMD's container chunk (0/5120 bytes differ), consts **0/24576**, and the
   embeddings match the container within bf16 rounding. So the packing, the name
   map and the quant conversion are right.
2. `OFLM_OPEN_LAYER_NAN=1` (which also disables the decode pipeline so each layer
   completes before it is read) showed the **f32 twin** turns layer 0 into all-NaN
   for gemma3-4b and Nanbeige, while the q4_1 twin reproduces the container's
   per-layer numbers (13058.38 vs 13058.63 ...).
3. Therefore the fault is the f32-scale twin's kernels for those shapes, not the
   pack, not the load path, not the family. The q4_1 twin is what every q4nx
   container runs, so it is the safer default and numerically the same model.

`OFLM_GGUF_POOL_Q41` is gone; the knob is `OFLM_GGUF_POOL_F32` (probe only). The
bisect merge also swaps the HEAD and the globals now -- `lmpool` is the head
pool's offset, so it and everything after it move with the chunk law; without
that the bisect ran a q4_1-packed head through the f32 kernel and looked broken.

**Still open**: why the f32 twin NaNs at layer 0 for those shapes (gemma3-4b,
Nanbeige) -- either fix it or stop shipping f32 twins. And Qwen3.5 is unchanged
by the law (soup on both), so the hybrid family's fault is in its own packing.

## GGUF-direct NUMERICS: the load path is now fedora's, and what that proved (2026-10-26)

`39b1c26` replaces our older GGUF load path with the one fedora exercised on the
NPU: Q5_K/IQ4_NL/IQ4_XS block sizes + the IQ4 codebook, the qwen35 ssm_* name
identities, `post_norm_` (the pre-FFN norm is whichever name the file has),
`consts_as_bf16`, the f32-slot `pack_norm`, `std_perm_gguf` serving both pool
laws keyed off `chunk_bytes` with per-row requant of Q8_0/Q4_K/Q5_K/Q6_K/IQ4,
the `lmhead_q8` Q6_K requant, `check_model`'s text_config lookup, and
`lm_config`'s text_config lift (without which a Qwen3.5 GGUF's sampler is built
over zero features and segfaults right after prefill).

Measured after that port ("The capital of France is", temperature 0):

| model | source | result |
|---|---|---|
| Qwen3-4B-Base | Q4_0 + tied Q6_K head | **coherent** |
| Qwen3.5-4B-i1 | Q4_1 | generates, multilingual soup (was a segfault) |
| Qwen3.8-4B-Distill | Q4_K/Q6_K | generates, soup |
| Nanbeige4-3B-Base-i1 | llama3, Q4_1 + Q6_K head | repeats `\x0b` |

`OFLM_GGUF_POOL_Q41=1` (the f32-numerics bisect above) leaves Nanbeige
unchanged, so its fault is not the f32 chunk law.

**AMD's Nanbeige container was a different model, and is gone (`2e9be98`).**
`oflm pull nanbeige4.1:3b` refused to load on our set: the container's config
says `intermediate_size = 10752`, the HF checkpoint says **10496**. The
container's own chunk arithmetic confirms 10752 (`mlp.down_proj` is
3360 x 5120 chunks = 80 bands x 42 k-tiles), and its tokenizer is a different
one (vocab 166107, `<|im_start|>`; the 3B-Base GGUF has 166102, `<s>`). So no
HF-derived spec could serve it. The registry entry is removed rather than our
spec bent to it -- users pull GGUF now. The set stays and a Nanbeige GGUF links
it by geometry; `oflm-add` takes the family from the manifest's own spec when
the registry does not list the set.

That LOOP (Nanbeige GGUF emitting `\x0b`) happens with BOTH the old and the new
engine, so the container was not its cause -- the container only explains why
the container could never load. Open question, and the numeric gate is the tool.

## GGUF-direct NUMERICS: one combination verified (2026-10-26)

Measured on the NPU by serving each installed model and asking a question:

| model | quant | result |
|---|---|---|
| Qwen3-4B-Base-GGUF (Antigma) | Q4_0 | **coherent text** |
| Nanbeige4-3B-Base-i1 | Q4_1 | loads, emits `\x0b` repeats |
| Qwen3.8-4B-Distill (empero) | Q4_K/Q6_K | loads, garbage |
| Qwen3.5-4B-i1 | Q4_1 | **segfaults in prefill, layer 0** |

The pack path runs for all of them (pools pack, layers go resident), so this is
not a mapping/plumbing failure any more -- it is arithmetic. The Qwen3.5
hybrid tensors needed the name identities and the size-adaptive side reads
(`e90246c`); they are necessary but not sufficient.

**The missing instrument is a numeric gate.** Nothing compares a GGUF model's
logits with its q4nx twin's (or with a CPU reference). Every other geometry in
this repo has one -- the attention gate, the per-op fixtures, `make_fixtures.py`
-- and GGUF-direct has none, which is why four families shipped "installed and
linked" without anyone noticing that only one of them runs. Build that gate
before more GGUF breadth: one layer, one prompt, `OFLM_OPEN_LAYER_NAN=1` /
`OFLM_OPEN_DUMP_POOLS=<dir>` to compare the packed pool against a NumPy pack of
the same GGUF tensor, then bisect per op.

Useful debugging facts found on the way:

- `OFLM_OPEN_LAYER_NAN=1` prints per-layer xres stats; a crash **before** the
  first layer trace is in layer 0's dispatch, not in the model loop.
- `OFLM_OPEN_DUMP_POOLS=<dir>` dumps layer 0's packed pool + consts, which is
  the comparison point for a NumPy pack of the same tensors.
- A GGUF's dims are FASTEST FIRST: `ssm_conv1d.weight` [4, 8192] is row-major
  [channels][taps] (HF's layout), while the q4nx container stores the filter
  taps-major. `transpose` / `conv_transpose` now pick by the tensor's shape.

## The wrong-kernels trap in `oflm add` (found 2026-10-26)

`resolve_official()` ends with "no exact size match for X B; using Y B kernels",
and that last resort is a WRONG link for a size the registry does not carry: a
`Qwen/Qwen3-14B-GGUF` install resolved to `Qwen3-8B-NPU2` and would have
"succeeded". The geometry match already existed -- `family_from_kernels()`
walks every root for a manifest whose `spec.kernel_hash()` equals the model's --
but only its FAMILY was used and the set name was discarded.

Now: the geometry-matched set is preferred as the xclbin source, and a GGUF
install whose set does not match the model's geometry is REFUSED
(`_require_same_geometry`) with the `build-new-spec` hint. Checked both ways:
`Qwen3.5-9B-i1-GGUF` still links `Qwen3.5-9B-NPU2`; `Qwen3-14B-GGUF` stops
(that shape has no f32 twin, so no GGUF-direct set can exist).

Limitation, and why the refusal is the safe end state: the geometry at add time
comes from the *peeked config*, so a GGUF repo with no usable config.json cannot
be geometry-matched before the download. Those need a registry entry (or
`--xclbin-from`).

## Adding a size that needs a WIDE FF: the dense split (2026-10-26, `940e450`)

Qwen3-14B (hidden 5120, FF **17408**) hit `dense: a 17408-wide activation table
does not leave room for the streams in a core's L1`. The layer_x / qwen35 path
has had the down-GEMV split since the 27B; the dense path had not. Ported from
fedora, one implementation shared with layer_x:

- `open_kernels/ln_split.py` — the streaming norm bodies (`ln_split_body`,
  `ln_split_body3`). The fused helper holds five 10 KB elements; at HID 5120
  that does not fit the norm core. It adds the residual half by half
  (`ln_add2`), writes the sum to DDR, norms it back (`ln_nr`).
- `open_kernels/ironutil.py` — `piece_tap`: one K piece of a banded weight pool.
- `recipes/dense.py` — `_ffn_split` (`down_split`), per-call measured against
  the widest PIECE, `DOWN_SPLIT`/`NORM_SPLIT` on the geometry, the `AD_OUT2B`
  bounce buffer, `DENSE_DOWN_SPLIT`/`DENSE_NORM_SPLIT` probes in the build key,
  and `GGUF.section(..., twins_built=f32_twin_fits(spec))`: **a split FF has no
  f32-scale twin** (table + wider elements overflow a core), so the twin is not
  built and the manifest advertises none.
- `designs/dense/dx.py` — the down GEMV in K pieces, `ln_body3` (res + out2 +
  out2b via `ln_add3`), the streaming-norm worker, and the step 6/7/8 DMA
  branches.

Only a spec with `DOWN_SPLIT` changes the design: `of_x`'s depth becomes 3 at
HID 5120 (2 everywhere else), so the existing dense sets' generated code is
unchanged. `build_key` moves anyway — it hashes `recipes/*.py`.

Also: `lm_head_q4` got a fixture (`designs/lm_head_q4/{make_test,compare}.py`,
from fedora) and a gate, so its K wall is validated by running the kernel:
K=5120 PASS cos=1.0 maxrel=4.27e-06.

**xclbins are not byte-reproducible** build-to-build (same source, different
hash — that is why the export cache keys on content, not on the artifact), so
"did my change alter the design?" is answered at the recipe/geometry level or by
re-running the model, never by hashing final.xclbin.

## Still open (Phase-MoE / later)

- qwen36moe recipe f32 twins for the OTHER designs of the family: dx/dx_f32
  and lm_head law-combos where the family ships them, plus the
  expert_stripes/expert_down GGUF ops (3D [experts, n, k] tensors) and the
  singular/plural `model.layer(s)` name mismatch.
- open_gemma3 / open_embedding engines untouched (they read safetensors).
- End-to-end `flm serve` with a dense-family GGUF (vocab must match a shipped
  spec exactly; none in the HF cache qualifies today — Peach is llama-arch
  with vocab 64000 vs the llama31-8b spec's 128256). **When the blocker is a missing
  spec/catalogue entry for the model's geometry, generate it with
  `utilities/gen-kernel-spec.py` / `utilities/build-new-spec.py` (see
  "Filling out the missing sizes") rather than trying to force a link. Spec
  files are tracked exceptions (do not derive them at runtime).**



## Llama3 rope pairing (interleaved) — ON THIS BRANCH since `c85d185` (2026-10-26)

> **Scope note:** ported from `origin/feat/fedora-docker` commit `4066987`
> on 2026-10-26. All symbols below (`ROPE_INTERLEAVED_FAMILIES`,
> `rope_pairing`, `ATTN_ROPE_IL`, `rope_il`) are in this tree. One deliberate
> divergence from fedora: the attention gate's fp64 oracle
> (`designs/attn/attn_reference.py`) implements the interleaved law and
> `make_test.py` passes `spec.rope_interleaved` — fedora's oracle is still
> half-split-only, so a fedora gate run on a llama geometry would fail.
> Llama-3.2-1B/3B and Nanbeige4.1-3B were gated with it (all PASS).

Root cause of llama3-family GGUF garbage: the engine applied half-split pairs `(j, j+ROT/2)` for all families, but llama3-family GGUF expects interleaved pairs `(2j, 2j+1)` (llama.cpp NORM rope law). Qwen3 works with half-split.

**Plumbing (both kernel design + host):**
- `open_kernels/recipes/spec.py`: `ROPE_INTERLEAVED_FAMILIES = {"llama3"}`, `ModelSpec.rope_interleaved` property.
- `open_kernels/recipes/dense.py`: passes `ROPE_IL` in geometry/flags and writes `layout.rope_pairing` (`"interleaved"` or `"halfsplit"`) into the manifest.
- `open_kernels/designs/dense/dx_attn.py`: threads `ATTN_ROPE_IL` flag to `dx_attn` kernel build.
- `open_kernels/designs/attn/attn.h`: adds `ATTN_ROPE_IL` path using `aie::interleave_unzip`/`aie::interleave_zip` over the head dim (identical pair-ordered cos/sin tables).
- `src/open_qwen36/manifest.cpp`: `Manifest::parse` parses `layout.rope_pairing` → `rope_interleaved` (default half-split).
- `src/open_qwen36/block_host.{hpp,cpp}`: `AttnGeom.rope_il` propagated; interleaved `rope()` matches kernel pairing.
- `src/open_qwen36/core.cpp`: wires `rope_il` from manifest to host paths (block prefill/host attention) and keeps layer dump plumbing intact.

**Validation (on that branch):** rebuilt `Llama-3.2-1B-NPU2` kernels with `rope_pairing: interleaved` (export/kernels regen), rebuilt `open_qwen36_cli`, and NPU greedy decode of the test prompt produced coherent text matching the reference. Layer-wise dumps (`OFLM_OPEN_LAYER_DUMP`) show correct alignment after the fix.
