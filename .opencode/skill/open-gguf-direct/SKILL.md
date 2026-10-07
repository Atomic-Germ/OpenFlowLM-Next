# Skill: open-gguf-direct

Direct GGUF loading for the open engines (supersedes q4nx for compatible
quants). Use when: adding GGUF support to another engine, debugging a GGUF
model that refuses to load, rebuilding the f32-scale kernel variants,
extending to MoE, or changing the pack ops.

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

python open_kernels/gguf_pool.py                              # pool-law self-tests (bit-exact)
GEMV_SCALES_F32=1 python open_kernels/build_design.py open_kernels/designs/gemv_q4/gemv_q4.py <out>
LMHEAD_N=... LMHEAD_SCALES_F32=1 python open_kernels/build_design.py open_kernels/designs/lm_head_q8/lm_head_q8.py <out>
# full export builds both layouts: python open_kernels/export_qwen36_kernels.py --spec ...

cmake -S src/open_qwen36 -B build-oq36 -DXRT_INCLUDE_DIR=/opt/xilinx/xrt/include -DXRT_LIB_DIR=/opt/xilinx/xrt/lib
cmake --build build-oq36 && ctest --test-dir build-oq36        # GGUF-PACK + OPEN-MANIFEST

# NPU harness (an accel0 NPU is present on this box):
python open_kernels/designs/gemv_q4/make_test.py --layout gguf --bands 4
cd open_kernels/designs/gemv_q4 && /tmp/opencode/harness/run_kernel run_qkv_gguf.cfg && python compare.py qkv_gguf
```

## Validation results (2026-09-07, this box)

- `open_kernels/gguf_pool.py` self-tests: pool bytes dequantize **bit-exactly**
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

## Gotchas learned here

- **designs/dense/gen_kernels.py regenerates gemv_q4_gy.cc / gemv_q4_gms.cc
  on every export** — hand-editing those two files is silently undone at the
  next export (cost an hour; looks like a mysterious file revert). The
  symbol-prefix fix lives in the GENERATOR templates now
  (GEMV_Q4_WRAP(GEMV_Q4_PREFIX, y/ms)). NOTE: designs/layer_x/gen_kernels.py
  still emits UNPREFIXED names — correct for today's MoE recipe (no f32
  twins yet), but the MoE f32 phase MUST add the macro to those templates
  or lx/ax f32 builds will fail with undefined gemv_q4s32_* exactly like
  the dense ones did.

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

## Still open (Phase-MoE / later)

- qwen36moe recipe: f32 twin builds (lx/ax/dx/lm_head_q8 *_f32), the
  expert_stripes/expert_down GGUF ops (3D [experts, n, k] tensors), the
  singular/plural `model.layer(s)` name mismatch, q8 f32 numerics bug above.
- open_gemma3 / open_embedding engines untouched (they read safetensors).
- End-to-end `flm serve` with a real GGUF (needs a dense-family GGUF matching
  a shipped spec's vocab exactly; none in the HF cache qualifies today —
  Peach is llama-arch with vocab 64000 vs the llama31-8b spec's 128256).
