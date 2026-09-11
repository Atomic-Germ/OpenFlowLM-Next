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
  (blk.N.attn_q.weight); pools' `std_perm_gguf` accepts Q4_0/Q4_1 only;
  `put`/`pack_norm` convert f32/f16 small weights to the bf16 consts blob.
- **flm-add**: repos with llama.cpp-style GGUFs install one compatible file
  as `model.gguf` (preference Q4_1 > Q4_0 > Q8_0; multi-part and other quants
  refused with reasons). Registry entries keep format "NPU2" and add
  `details.weights = "gguf"`; q4nx models stay *-NPU2, GGUF installs are
  *-GGUF directories, so one of each kind can coexist.

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
