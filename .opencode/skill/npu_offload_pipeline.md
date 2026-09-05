# NPU Offload Pipeline Skill

**Purpose**: Reusable workflow for offloading dense GEMM operations from transformer models to AMD NPU2 via mlir-aie/iron, with validated CPU reference comparison.

---

## Overview

This skill captures the end-to-end pipeline proven on `google/embeddinggemma-300m`:

1. **Model analysis** → identify GEMM shapes, data types, weight layouts
2. **Design creation** → mlir-aie whole_array topology (4×N cores, tiled 64×K×N)
3. **Host/backend** → XRT dispatch with hw_context + register_xclbin
4. **Engine integration** → route `matmul_t` through NPU with bf16→f32 kernel
5. **Validation** → E-suite + oracle cosine threshold (≥0.999)

**Key achievement**: bf16→f32 output dtype gives bit-exact FP32 reference match (E8 cosine 0.999993 vs threshold 0.999).

**Integration status**: Open embedding fully integrated into FastFlowLM repo, replacing closed `libgemma_embedding.so`. Auto-manifest generation handles both FLM and HuggingFace cache layouts. Server verified working: `./flm serve -e 1` → `/v1/embeddings` returns valid 768-dim vectors. **NPU verified**: 12 xclbins compiled (6 shapes × 2 M values), bf16→f32 kernel, cosine similarity 1.000000 vs CPU reference.

---

## Prerequisites

- AMD NPU2 (Strix) at PCI `0000:c2:00.1`, `/dev/accel/accel0`
- XRT at `/opt/xilinx/xrt`, `export XILINX_XRT=/opt/xilinx/xrt`
- ironvenv: Python 3.11–3.13, **mlir-aie 1.4.2**, Peano `llvm-aie 21.0.0.2026080301+c9c5ecb7`
  - `pip install -r ironvenv-requirements.txt` (release wheels; no eudsl-python-extras — it
    shadows mlir_aie's `aie/` package and breaks `import aie.iron`)
  - Runtime API is 1.4.2's: `tg = TaskGroup()`, `fifo.prod().fill(src, tap=, group=tg)`,
    `fifo.cons().drain(dst, tap=, wait=True, group=tg)`, `tg.finish()`,
    `Runtime(sequence_fn, [types..., handles...])`, `Program(dev, rt, workers=[...])`.
    The 1.3.4 form (`rt.task_group()`, `rt.fill(fifo, …)`, `with rt.sequence(...)`) is gone.
  - Also needs `xclbinutil` and `aiebu-asm` on PATH (from an XRT install).
- Model weights in FP32 safetensors + weights_manifest.json

---

## Step 1: Model GEMM Inventory

For your target model, enumerate every dense matmul in the forward pass.

### Required Information Per GEMM

| Field | Description | Example (EmbeddingGemma) |
|-------|-------------|--------------------------|
| `name` | Unique identifier | `layers.0.self_attn.q_proj` |
| `M` | Batch×SeqLen (dynamic) | `T` (≤2048) |
| `K` | Input dimension | 768 |
| `N` | Output dimension | 768 (Q), 256 (K/V), 1152 (gate/up) |
| `weight_layout` | Stored as [N,K] or [K,N]? | `[N,K]` (out-major) |
| `data_type` | FP32/BF16/INT8 | FP32 stored → bf16 NPU |
| `batch_dim` | Does M=B×S or M=S? | M=S (B=1) |
| `dynamic_M` | Does M vary per call? | Yes (seq len) |

### Worksheet Questions

Before proceeding, answer:

1. **What are the distinct (K,N) pairs** across all GEMMs? (Deduplicate identical shapes)
2. **What is max sequence length** (max M)? → determines pad sizes needed
3. **Weight storage** — are projection weights stored out-major `[N,K]` (needs transpose to `[K,N]`) or row-major `[K,N]`?
4. **Data types** — FP32 stored? Any INT8/INT4 quantized weights?
5. **Batch support** — does engine process B>1 (M=B×S) or only B=1?
6. **Special ops** — any batched GEMMs (attention scores, attn×V) with different layout?

---

## Step 2: Design Creation (`matmul_whole_array.py`)

Adapt the template for your shapes.

### Compile-Time Parameters

```python
@iron.jit(aiecc_flags=["--alloc-scheme=basic-sequential"])
def whole_array(
    A: In, B: In, C: Out,
    *, M: CompileTime[int], K: CompileTime[int], N: CompileTime[int],
    tile_n: CompileTime[int], n_aie_cols: CompileTime[int],
    dtype_in_str: CompileTime[str] = "bf16",
    dtype_out_str: CompileTime[str] = "f32",   # ← CRITICAL: f32 output for bit-exact
):
    m = k = 64
    n = tile_n
    n_aie_rows = 4
    # ...
    matmul_kernel = kernels.mm(
        dim_m=m, dim_k=k, dim_n=n,
        input_dtype=iron.str_to_dtype(dtype_in_str),
        output_dtype=iron.str_to_dtype(dtype_out_str),
        vectorized=True,
    )
```

### Compilation Matrix

For each distinct (K,N):

| M_pad | tile_n | n_aie_cols | dtype_out | Use case |
|-------|--------|------------|-----------|----------|
| 512 | 32 | 4 | f32 | Short sequences (≤512) |
| 2048 | 32 | 4 | f32 | Long sequences (≤max_pos) |

**Why tile_n=32 for f32?** C buffer = m×n×n_aie_rows×4 bytes. 64×64×4×4=64KB per core L2. tile_n=32 → 32KB fits. tile_n=64 OOM.

**Why n_aie_cols=4?** 16 cores (4 rows × 4 cols) balances throughput vs overhead. 8 cols (32 cores) slower for these shapes (per-tile overhead dominates).

### Compilation Commands

```bash
# From ironvenv, in npu_offload/matmul/
for M in 512 2048; do
  for K N tile_n in \
    "768 768 32" "768 256 32" "768 1152 32" \
    "1152 768 32" "768 3072 32" "3072 768 32"; do
    python matmul_whole_array.py \
      -M $M -K $K -N $N --tile-n $tile_n --n-aie-cols 4 \
      --dtype-in bf16 --dtype-out f32 \
      --xclbin-path $ASSET_DIR/m${M}_${K}x${N}.xclbin \
      --insts-path $ASSET_DIR/m${M}_${K}x${N}.insts \
      --dev npu2
  done
done
```

**Asset naming**: `m{M}_${K}x${N}.{xclbin,insts}` in model_dir/npu_matmul_f32/

---

## Step 3: Host/Backend (`npu_matmul.cpp`)

### Modern XRT Load Pattern (tested working)

```cpp
auto xcl = xrt::xclbin(xclbin_path);
device.register_xclbin(xcl);
auto ctx = std::make_unique<xrt::hw_context>(device, xcl.get_uuid());
auto kernel = std::make_unique<xrt::kernel>(*ctx, "MLIR_AIE");
```

### Buffer Allocation (per shape)

```cpp
bo_instr: size = insts.size()*4,   flags=CACHEABLE,       group_id(1)
bo_a:     size = Mpad*K*2,         flags=HOST_ONLY,       group_id(3)
bo_b:     size = K*N*2,            flags=HOST_ONLY,       group_id(4)
bo_c:     size = Mpad*N*4,         flags=HOST_ONLY,       group_id(5)  // FP32!
```

### Dispatch (opcode=3)

```cpp
// A/B: bf16 (uint16_t), C: fp32 (float)
memcpy(am, a, M*K*2); memcpy(bm, b, K*N*2);
memset(cm, 0, M*N*4);  // zero-pad FP32 output
bo_a.sync(TO_DEVICE); bo_b.sync(TO_DEVICE); bo_c.sync(TO_DEVICE);
auto run = (*kernel)(3, *bo_instr, insts.size(), *bo_a, *bo_b, *bo_c);
run.wait();
bo_c.sync(FROM_DEVICE);
memcpy(c, cm, M*N*4);  // FP32 output
```

**Critical**: `bo_c.sync(TO_DEVICE)` before launch — prevents nondeterministic garbage.

---

## Step 4: Engine Integration

### Weight Transposition (once at load)

```cpp
// Engine stores weights as [N,K] fp32. NPU needs [K,N] bf16.
for (auto& [name, wvec] : w_) {
  if (!is_npu_projection(name)) continue;
  size_t N = shape[0], K = shape[1];
  std::vector<uint16_t> bf(K * N);
  for (size_t n=0; n<N; n++)
    for (size_t k=0; k<K; k++)
      bf[k*N + n] = f32_to_bf16(wvec[n*K + k]);  // transpose [N,K]→[K,N]
  w_bf16_[name] = std::move(bf);
}
```

### Routing in `matmul_t_npu`

```cpp
void Engine::matmul_t_npu(name, x, M, K, N, y) {
  if (npu_ && M>0 && M<=2048) {
    int mpad = npu_->m_pad_for(K, N, M);
    if (mpad > 0 && w_bf16_.count(name)) {
      // Convert x[M,K] fp32 → a_pad[mpad,K] bf16 (zero-pad)
      // Dispatch → c_pad[mpad,N] fp32
      // Truncate to y[M,N] fp32
      return;
    }
  }
  matmul_t(x, w, M, K, N, y);  // CPU fallback
}
```

### Selective Offload

Only offload numerically-tolerant large projections:

```cpp
// NPU: q, o, gate, up, down (large, tolerate bf16)
// CPU: k_proj, v_proj (small N=256), contrastive head (sensitive)
static bool is_npu_projection(const string& name) {
  return ends_with(name, "q_proj.weight") ||
         ends_with(name, "o_proj.weight") ||
         ends_with(name, "gate_proj.weight") ||
         ends_with(name, "up_proj.weight") ||
         ends_with(name, "down_proj.weight");
}
```

---

## Step 5: Build System (CMake)

```cmake
# In main CMakeLists.txt
if(FLM_USE_OPEN_EMBEDDING AND NOT FLM_USE_HRX)
  target_compile_definitions(flm PUBLIC FLM_USE_OPEN_EMBEDDING_NPU=1)
  target_sources(flm PRIVATE open_embedding/npu_matmul.cpp)
endif()
```

### Runtime Flags

| Env Var | Purpose |
|---------|---------|
| `FLM_NPU_DISABLE=1` | Force CPU path |
| `FLM_NPU_DEVICE_ID=0000:XX:XX.X` | Override NPU PCI address |
| `FLM_CONFIG_PATH=/path/model_list.json` | Model list file |
| `FLM_XCLBIN_PATH=/path/to/xclbins` | Closed stack assets (if needed) |

---

## Step 6: Validation (E-Suite)

```bash
# Start server with NPU
export XILINX_XRT=/opt/xilinx/xrt
export FLM_CONFIG_PATH=/path/model_list.json
export FLM_XCLBIN_PATH=/path/xclbins
./flm serve -e 1

# Run embedding tests
flm-test --embedding --port 52625
```

### Required Thresholds

| Check | Threshold | NPU Target |
|-------|-----------|------------|
| E8 Reference Agreement | cosine ≥ 0.999 | 0.999993 (bf16→f32) |
| E2 Repeatability | cosine = 1.0 | 1.000000 |
| E6 Cross-Path | cosine = 1.0 | 1.000000 |
| E7 Batch Ref | cosine = 1.0 | 1.000000 |

**If E8 fails**: check `dtype_out` is f32 (not bf16), verify `bo_c.sync(TO_DEVICE)`, confirm weight transpose correctness.

---

## HF Cache & Auto-Manifest Integration

When integrating into a fresh repo, the engine needs to handle models pulled via `flm pull` (which downloads to `~/.config/flm/models/<name>/`) AND models already cached in HuggingFace cache (`~/.cache/huggingface/hub/`).

### Auto-Manifest Generation

The engine auto-generates `weights_manifest.json` from safetensors files when:
- The manifest file doesn't exist, OR
- It exists but is incomplete (missing dense head tensors like `2_Dense`, `3_Dense`)

**Detection logic** (`engine.cpp:load_weights()`):
```cpp
bool need_regen = false;
if (!read_file(mpath, mtext) || !(manifest_ = json::parse(mtext, ...)).is_object()) {
    need_regen = true;
} else {
    auto& tensors = manifest_["tensors"];
    if (!tensors.contains("2_Dense.linear.weight") || !tensors.contains("3_Dense.linear.weight")) {
        need_regen = true;  // incomplete manifest
    }
}
```

**Search order for dense head weights** (per `2_Dense`, `3_Dense`):
1. FLM layout: `<model_dir>/weights/<name>.safetensors`
2. HF layout: `<model_dir>/<name>/model.safetensors`
3. HF cache: `<hf_cache_snapshot>/<name>/model.safetensors`

**HF cache lookup** (`find_hf_cache_snapshot()`):
- Reads `~/.cache/huggingface/hub/models--<org>--<model>/refs/main` for snapshot hash
- Returns `<hub>/snapshots/<hash>/` if directory exists
- Handles both `HF_HOME` env var and default `~/.cache/huggingface`

### Manifest Format

```json
{
  "config": "<absolute_path>/config.json",
  "tokenizer": "<absolute_path>/tokenizer.json",
  "tensors": {
    "embed_tokens.weight": {"file": "<abs_path>/model.safetensors", "offset": 0, "shape": [768, 128256]},
    "layers.0.input_layernorm.weight": {"file": "...", "offset": ..., "shape": [768]},
    "2_Dense.linear.weight": {"file": "<abs_path>/2_Dense/model.safetensors", "offset": 0, "shape": [768, 768]},
    "3_Dense.linear.weight": {"file": "<abs_path>/3_Dense/model.safetensors", "offset": 0, "shape": [768, 768]}
  }
}
```

### `safetensors` Header Parser

For auto-generation, parse the safetensors binary header to extract tensor metadata:

```cpp
// Format: [8 bytes: header_length][header_json][tensor_data...]
// Header JSON: {"tensor_name": {"dtype": "F32", "shape": [N,K], "data_offsets": [start, end]}, ...}
static std::unordered_map<std::string, std::unordered_map<std::string, nlohmann::json>>
safetensors_index(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    uint64_t hdr_len = 0;
    f.read(reinterpret_cast<char*>(&hdr_len), 8);
    std::string hdr(hdr_len, '\0');
    f.read(&hdr[0], hdr_len);
    auto j = nlohmann::json::parse(hdr);
    // Skip "____metadata" key, return tensor entries
}
```

### model_list.json Entry for HF Model

```json
{
  "name": "embed-gemma:300m",
  "url": "https://huggingface.co/google/embeddinggemma-300m",
  "files": ["config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json",
            "2_Dense/model.safetensors", "3_Dense/model.safetensors"],
  "run": "flm serve -e 1",
  "type": "embedding",
  "size": "300m",
  "mode": "fp32",
  "backend": "cpu",
  "min_npu_gen": 2,
  "download_size_mb": 2500
}
```

**Key**: Subdirectory files like `2_Dense/model.safetensors` are downloaded correctly because `download_model.cpp` calls `create_directories(path.parent_path())`.

---

## Troubleshooting Checklist

| Symptom | Likely Cause | Fix |
|---------|--------------|-----|
| `load_axlf: Operation not supported` | Using deprecated `load_xclbin(path)` | Use `register_xclbin(xclbin)` + `hw_context` |
| Nondeterministic output / first rows zero | Missing `bo_c.sync(TO_DEVICE)` | Sync output BO to device before launch |
| E8 cosine ~0.98 | bf16 output quantization | Use `dtype_out=f32` (bf16_f32 kernel) |
| M=2048 OOM | tile_n too large | Reduce `tile_n` (32 for f32, 64 for bf16) |
| "allocated buffers exceeded available memory" | C buffer too large | Reduce `tile_n` or `n_aie_cols` |
| Server crash on model load | `find_model_list` returns dir not file | Set `FLM_CONFIG_PATH=.../model_list.json` (file) |
| `unordered_map::at` on model load | `weights_manifest.json` missing dense heads | Check if manifest is incomplete (missing `2_Dense`/`3_Dense` entries) |
| HF cache lookup fails silently | `c = '--'` assigns multi-char literal to char | Use string concatenation: `safe_id += "--"` not in-place `c = '--'` |
| Model dir has Q4NX but no safetensors dense heads | Old closed model files mixed with open | Delete stale `weights_manifest.json`, engine regenerates from safetensors |
| `HOME` env var not set in container | HF cache path defaults to `./.cache/huggingface` | Set `HF_HOME` or `HOME` env var |
| `cannot read *.insts` | Symlinks to blob storage (broken) | Recompile xclbins with `matmul_whole_array.py`, copy real files |
| NPU not used despite xclbins present | `FLM_USE_OPEN_EMBEDDING_NPU` not defined | Add `target_compile_definitions(flm PUBLIC FLM_USE_OPEN_EMBEDDING_NPU=1)` in CMake |

---

## Compilation Lessons Learned

### Asset Naming
Engine expects `m{M}_{K}x{N}.{xclbin,insts}` — NOT the whole_array default naming.
Use `--xclbin-path` and `--insts-path` flags to control output names.

### Symlink vs Real Files
When copying assets from another repo, verify with `file` command.
Broken symlinks cause silent "cannot read" errors. Always copy real files.

### Compilation Parameters for f32 Output
- `--dtype-in bf16 --dtype-out f32` (bf16 input, f32 output)
- `--tile-n 32` (required for f32 output: C buffer = 64×32×4×4 = 32KB fits L2)
- `--n-aie-cols 4` (16 cores: 4 rows × 4 cols)

### M Value Selection
M must be divisible by `m * n_aie_rows = 64 * 4 = 256`.
For max_position_embeddings=2048: compile M=512 and M=2048.
Engine selects smallest m_pad >= actual M at runtime.

---

## Extending to New Model Types

### For Each New Model

1. **Run inventory** (Step 1) — document all (K,N) pairs
2. **Compile asset matrix** — generate xclbins for each (K,N) × {512,2048} × f32
3. **Add weight patterns** to `is_npu_projection()` — match new layer names
4. **Verify weight layout** — transpose if stored differently
5. **Run E-suite** — confirm E8 ≥ 0.999
6. **Profile** — kernel latency vs CPU (expect 2.5-3× speedup on 16 cores)

### Upstream Integration Checklist

When merging open embedding into the main flm repo:

- [ ] Copy `open_embedding/` sources into `src/open_embedding/`
- [ ] Copy `npu_utils_matmul.hpp` into `src/include/npu_utils/`
- [ ] Create adapter class (e.g., `open_gemma_embedding.hpp`) wrapping `open_embedding::Engine`
- [ ] Update `auto_embedding_model.hpp` — remove closed deps (Q4NX, `embedding_model.hpp`)
- [ ] Update `all_embedding_model.hpp` — always instantiate open adapter
- [ ] Remove closed files (modeling_gemma_embedding, gemma_embedding, embedding_model)
- [ ] Remove closed xclbin directories
- [ ] Update `CMakeLists.txt`:
  - Add open_embedding sources
  - Remove closed library link
  - Add `FLM_USE_OPEN_EMBEDDING=1` and `FLM_USE_OPEN_EMBEDDING_NPU=1` defines
  - Add `tokenizers` link (needed for tokenizer)
- [ ] Update `model_list.json` — point URL to HF model repo
- [ ] Update `model_info.json` — add correct HF file metadata (oid, lfs_oid, size)
- [ ] Verify `flm pull` downloads all files including subdirectory paths (`2_Dense/model.safetensors`)
- [ ] Verify auto-manifest generation handles both FLM layout and HF cache layout
- [ ] Test: `./flm serve -e 1` → embedding endpoint returns valid vectors

### Correct Open-Only Adapter Boundary

Keep model loading and text tokenization inside `open_embedding::Engine`. Do
not partially reuse the closed embedding shared loader: `_shared_load_model()`
constructs Q4NX, `npu_xclbin_manager`, and `gemma_embedding`, so merely leaving
that function compiled retains closed link dependencies even if no factory
calls it.

For an open-only replacement:

1. Reduce `AutoEmbeddingModel` to shared identity/state (`model_path`,
   `is_model_loaded`, `current_model`, device pointer) and its virtual API.
2. Remove the closed Q4NX/tokenizer/NPU-manager fields and `_shared_*` methods;
   never replace `_shared_embed()` with a dummy result.
3. Use the known-good header-only `OpenGemma_Embedding` adapter from
   `docs/ExampleNPU/src/include/AutoEmbeddingModel/open_gemma_embedding.hpp`.
   It calls `engine_.load(model_path)` and `engine_.embed_with_prefix(...)`
   directly, preserving every official task prefix.
4. Instantiate only `OpenGemma_Embedding` in `all_embedding_model.hpp` and
   remove `gemma_embedding` from `target_link_libraries`.

The engine enum is namespace-level (`open_embedding::task_type_t`), not nested
under `Engine`. Prefer `embed_with_prefix()` in the adapter because it also
preserves clustering, classification, code retrieval, similarity, and
summarization prompts that cannot be represented by the engine's two-value
query/document enum.

### Correct CMake Wiring

Use explicit sources rather than extending broad globs:

```cmake
list(APPEND SOURCES "${CMAKE_SOURCE_DIR}/open_embedding/engine.cpp")
if(NOT FLM_USE_HRX)
    list(APPEND SOURCES "${CMAKE_SOURCE_DIR}/open_embedding/npu_matmul.cpp")
endif()

target_include_directories(flm PUBLIC
    ${CMAKE_SOURCE_DIR}          # resolves open_embedding/engine.hpp
    ${CMAKE_SOURCE_DIR}/include
)

target_compile_definitions(flm PUBLIC FLM_USE_OPEN_EMBEDDING=1)
if(NOT FLM_USE_HRX)
    target_compile_definitions(flm PUBLIC FLM_USE_OPEN_EMBEDDING_NPU=1)
endif()
```

`npu_matmul.cpp` includes XRT directly and must not be compiled for HRX. The
CPU engine remains available in HRX builds.

Verification used for this integration:

```bash
cmake -S src -B src/build \
  -DFLM_VERSION=0.9.24 -DNPU_VERSION=1 -DFLM_USE_HRX=OFF
cmake --build src/build -j4
```

This completed successfully on 2026-09-02. Existing AVX512 and deprecated
`wstring_convert` warnings are unrelated to the open embedding integration.

### Builder Reuse Rule

New model conversion/build support should extend `utilities/q4nx-build`
instead of creating an independent converter. Reuse its architecture
detection, tensor-name mapping, and tensor-layout knowledge, then add an open
output backend for safetensors manifests and generated kernel/build metadata.

### Open Embedding Distributable Route (2026-09-02)

EmbeddingGemma is distributed as an **open, unquantized HF repo**
(`Atomic-Germ/Embedding-Gemma-300M-OpenNPU2`), not as Q4NX. Rationale:
embedding needs no quantization, and the distributable route is the same
pipeline future open families will use, so it stays extensible. Pointing
`flm pull` at the upstream `google/embeddinggemma-300m` repo is not viable: it
is gated, `flm pull` uses plain curl with no HF token, and it lacks
`weights_manifest.json`.

One-shot reproducible build:

```bash
q4nx-build --open-embedding -i google/embeddinggemma-300m \
  -o ~/Embedding-Gemma-300M-OpenNPU2 \
  --npu-assets <dir of m{M}_{K}x{N}.{xclbin,insts}>
```

Registry wiring is mandatory and easy to miss:

- `src/model_list.json` supplies `files`, `url`, and `name` (the `name` field
  is the on-disk directory name under `<models_root>/models/`).
- `src/model_info.json` is the authoritative manifest for `flm pull`.
  `build_download_list` **silently skips** any file absent from it, then
  reports success. Always merge the builder's `model_info_entry.json` into
  `src/model_info.json`, or the pull downloads nothing and the engine fails
  later at load time.
- Subdirectory paths such as `2_Dense/model.safetensors` download correctly;
  `download_file` creates parent directories.

Engine portability requirements for any distributable:

- `weights_manifest.json` records paths **relative to the model dir**.
  `Engine::resolve_path()` joins them against `model_dir_`, and absolute
  legacy manifests still work.
- `Engine::ensure_manifest()` regenerates the manifest when missing or when
  the dense heads are absent, scanning `model.safetensors` plus
  `weights/<head>.safetensors`, `<head>/model.safetensors`, or
  `<head>.safetensors`. Shipping the manifest is optional.
- Dense-head tensor names are `2_Dense.linear.weight` and
  `3_Dense.linear.weight` (the head files contain a single `linear.weight`).

Verified end to end: built repo loads through the real `src/model_list.json`
entry, and deleting `weights_manifest.json` reproduces identical embeddings
(cosine 0.997711 against the bf16 reference array in
`src/test/gemma_embedding/test.cpp`).

Remaining manual step: uploading the built directory to HuggingFace is not
automated; a human creates the repo and pushes (`git lfs` or `huggingface-cli
upload`). `ms_url` is intentionally empty until a ModelScope mirror exists.

### NPU Kernel Distribution Policy (2026-09-02)

Compiled kernels use the same two locations as the closed-source stack, but
with an explicit escape hatch for new models.

**Established families ship kernels with the application.**

- Source of truth: `src/xclbins/<Model-Dir>/npu_matmul_f32/`, installed to
  `<xclbin_prefix>/xclbins/<Model-Dir>/npu_matmul_f32/`.
- These are built at build time from open Iron/mlir-aie designs, not
  downloaded, and are not listed in the model's `files` list or
  `model_info.json`. Model repos for established families contain weights and
  configuration only.
- `src/CMakeLists.txt` installs the whole `xclbins` tree, so adding a family
  directory is enough; no install-rule change is needed.

**New or prototype models may ship their own kernels.**

- `q4nx-build --open-embedding --npu-assets <dir>` places kernels in the model
  directory's `npu_matmul_f32/`, so a brand-new model works end to end before
  it is promoted to a maintained family. Promote it later by moving the
  kernels into `src/xclbins/`.

**Lookup order in `Engine::pick_npu_asset_dir()`**

1. `<model_dir>/npu_matmul_f32` — model-local override wins.
2. `<xclbin_prefix>/xclbins/Embedding-Gemma-300M-OpenNPU2/npu_matmul_f32`.
3. `<xclbin_prefix>/xclbins/embed-gemma/npu_matmul_f32` — family fallback.

`utils::find_xclbin_path()` **throws** when no xclbin tree is installed. The
engine wraps that call in `try/catch` and degrades to CPU-only, because the
open engine must never hard-require the xclbin tree. `find_xclbin_path()` is
forward-declared in `engine.cpp` rather than pulling in `utils/utils.hpp`,
which drags in the XRT-dependent buffer/typedef chain.

Verified paths (cosine against the bf16 reference array in
`src/test/gemma_embedding/test.cpp`):

| Scenario | Result |
|---|---|
| App-installed family kernels | NPU enabled, 6 shapes, cosine 0.99775 |
| Model-local kernels | NPU enabled, cosine 0.99775 |
| No xclbin tree | graceful CPU-only, cosine 0.997711 |
| `FLM_NPU_DISABLE=1` | CPU-only, cosine 0.997711 |

The standalone embedding test defines `FLM_USE_OPEN_EMBEDDING_NPU=1` and
compiles `open_embedding/npu_matmul.cpp` so the NPU path is actually exercised.
Set `FLM_XCLBIN_PATH` to exercise app-family discovery without installing.

Inventory classifiers treat `npu_matmul_f32` kernels as `open_npu_kernel`
distinct from `closed_npu_kernel`, so our own built kernels are never counted
as pending replacement work.

## Gemma3 Text Engine: Phase 0 Complete (2026-09-02)

Planning, decisions, and progress live in
`docs/plans/open_gemma3_text_plan.md`. This section records only what the next
agent must not have to rediscover.

**Phase 0 shipped** (built at `Models/Gemma-3-1B-OpenNPU2`, git-ignored):
340 BF16 tensors, 2.0 GB, byte-reproducible, tied embeddings, and 6 reference
fixtures (`[6, 10, 7, 7, 11, 2282]` tokens).

Builder: `q4nx-build --open-causal-lm -i <source> -o <dir>`
Oracle: `q4nx-build -i <dir> --make-reference <dir>/reference_v1.json`

**Gemma3 text facts confirmed by a working oracle:**

- Embeddings **are** scaled by `sqrt(hidden_size)` = 33.9411 for hidden 1152.
  An earlier analysis claimed otherwise; the oracle's coherent output settles
  it. Verify against the oracle, never assume.
- Global (full-attention) layers are `[5, 11, 17, 23]`, derived from
  `sliding_window_pattern: 6` as `(L+1) % 6 == 0`. `config.json` has **no**
  `layer_types` key — it must be derived.
- `attn_scale = 1/sqrt(query_pre_attn_scalar)` = 0.0625 (head_dim 256 gives the
  same number, so don't conflate the two sources).
- Tied embeddings: no `lm_head.weight` tensor; reuse
  `model.embed_tokens.weight` (262144 x 1152, ~302 M params, largest tensor).

**AMD environment constraints (this workspace):**

| Constraint | Consequence |
|---|---|
| `transformers` + ROCm torch segfaults in `from_pretrained`, even with the GPU hidden | Do not plan to use HF as the oracle generator here. |
| NumPy has no bfloat16; `safetensors.numpy` raises `data type 'bfloat16' not understood` | Decode bf16 via `uint16 << 16` viewed as `float32`. This is also what the C++ engine does, so oracle and engine agree bit for bit. |
| `/tmp` is a RAM-backed tmpfs with limited free space | Put large model artifacts on real disk; use the git-ignored `Models/` directory. |

**Decisions already made** (do not relitigate): ship bf16 (lossless, the weights
are natively bf16); embedding stays at the highest practical precision; own both
ends of the format rather than inheriting the closed Q4NX geometry; decode on CPU
first and only add a dedicated small-M kernel if simpler than padding the GEMM;
replace the closed path and its remnants entirely once understood.

## Download Verification Policy (2026-09-02)

**Hash checks are advisory, never fatal.** A pull must not be blocked by an oid
comparison.

Rationale: registry oids legitimately disagree with what a repo serves — LFS
files hash as sha256 while plain git blobs hash as sha1, and re-uploaded,
re-quantized, or mirrored files all change the expected value. A genuinely
broken download is self-evident when the model fails to load or emits garbage.

Changed behaviour:

- `src/pull/download_model.cpp` — a mismatch logs `[WARN] Hash mismatch ...;
  continuing` and the download still succeeds. It no longer consumes retries on
  an unfixable comparison.
- `src/pull/model_downloader.cpp` (`verify_and_clean_files`) — a mismatch logs a
  warning and keeps the file. It no longer deletes the file or reports an error,
  which previously forced endless re-downloads.
- A missing file is still a real error and still triggers a re-pull.

**Related: never abort on a missing xclbin tree for work that doesn't need
kernels.** `utils::find_xclbin_path()` throws when no tree is installed, which
broke both the open engine's CPU path and `flm pull` itself.

- `src/open_embedding/engine.cpp` catches it around the app-family kernel lookup
  and falls back to CPU-only.
- `src/include/lm_config.hpp::_resolve_paths()` catches it and leaves
  `exec_path` empty, so reading `config.json` during `flm pull` / `flm list`
  works without any xclbins installed.

Verified end to end against the live repo: `flm pull embed-gemma:300m` downloads
all 7 files, warns on advisory hash differences, reports success, and the pulled
model then loads and runs on NPU (cosine 0.99775).

## Validated: Full E-Suite Pass (2026-09-02)

`flm pull embed-gemma:300m` → `flm serve -e 1` → `flm-test --embed` gives a
clean sweep, **8/8 PASS, 0 FAIL**:

| Check | Result |
|---|---|
| E1 Response Structure | valid 768-dim embedding |
| E2 Repeatability | worst cosine 1.000000 |
| E3 Batch & Index Integrity | 3 embeddings returned in order |
| E4 Dimensionality | consistent 768-dim across batch |
| E5 Semantic Ordering | related 0.8434 > unrelated 0.5862 |
| E6 Cross-Path Consistency | cosine 1.000000 |
| E7 Batch Reference Consistency | 30 draws, worst cosine 1.000000 |
| E8 Reference Agreement | **worst cosine 0.999993** (threshold 0.999) |

E8 at 0.999993 reproduces the original proven value exactly, which pins the
whole path — task prefix, tokenizer, forward pass, pooling, dense projection,
and L2 normalisation — to the validated numpy oracle.

**Two cosine numbers, do not confuse them.**

- **0.999993** — vs the fp32 numpy oracle (E8). This is the real accuracy
  result.
- **0.99775 / 0.997711** — vs the bf16-quantized reference array hard-coded in
  `src/test/gemma_embedding/test.cpp`, inherited from the old closed engine.
  That array only has bf16 precision, so ~0.998 is expected and is *not* a
  regression. NPU (0.99775) and CPU (0.997711) agree with each other.

**Gotcha: run the repo's `flm-test`, not a system-installed one.** A stale
system copy predating E8 silently produced a CSV with only E1–E7 and no error,
which looks like E8 was skipped or crashed. Always invoke
`utilities/flm-test` explicitly (for example
`python -m flm_test --embed`) so the bundled
`test_files/embedding_reference.json` oracle and current checks are used.

Results are written next to the model as
`embedding_results_<version>.csv` (the working model directory, git-ignored).

### Ambiguous/Generalized Parts (User Questions)

> **Q1: Weight layout** — Are your projection weights stored as `[N,K]` (out-major, needs transpose) or `[K,N]` (row-major, direct use)?
> **Q2: Max sequence length** — What is the model's `max_position_embeddings`? Determines M=2048 vs larger pads.
> **Q3: Batch dimension** — Does the engine process batched inputs (M=B×S) or single sequence (M=S)?
> **Q4: Attention GEMMs** — Does the model use batched attention scores (S×K×S) or fused kernels? May need separate design.
> **Q5: Quantized weights** — Are any weights INT8/INT4? Requires INT8 kernel path (`dtype_in=i8`, `dtype_out=i32`).
> **Q6: Multi-head layout** — Are Q/K/V projections fused (single weight [3×D, D]) or separate?
> **Q7: Sliding window / sparse attention** — Any non-dense matmuls? May stay on CPU.
> **Q8: Contrastive/pooling head** — Final dense layers — include in NPU or keep CPU?
> **Q9: Target NPU** — NPU1 (Phoenix, 4 cols max) or NPU2 (Strix, 8 cols)? Affects `n_aie_cols` max.

---

## Asset Checklist for Handoff

- [ ] `matmul_whole_array.py` adapted with model-specific shapes
- [ ] All xclbin/insts in `model_dir/npu_matmul_f32/`
- [ ] `npu_matmul.cpp` compiled with `FLM_USE_OPEN_EMBEDDING_NPU`
- [ ] `is_npu_projection()` matches model's layer names
- [ ] Weight transpose logic verified (`[N,K]` → `[K,N]`)
- [ ] E-suite passes (E8 cosine ≥ 0.999)
- [ ] `FLM_NPU_DISABLE=1` tested for CPU fallback

---

## Repository-Wide Binary Inventory (2026-09-02)

Use the checked-in inventory utility instead of ad hoc shell pipelines:

```bash
python utilities/binary-inventory/inventory.py
python utilities/binary-inventory/inventory.py \
  --all --output docs/precompiled_artifacts_all.json
```

The shipping surface (`src/lib`, `src/xclbins`) contains 386 artifacts totaling
480,732,285 bytes:

- 222 XCLBINs in 38 model directories, but only 106 unique XCLBIN payloads.
- 139 closed engine/primitive artifacts across ELF, PE DLL, and COFF library
  forms.
- 23 third-party Windows runtime artifacts.
- 2 AIEBU runtime-support archives.

All 222 shipping XCLBINs lack `BUILD_METADATA`; they expose only the standard
AIE partition/connectivity sections. Do not assume exact original shapes or
compile flags can be recovered from them. Derive replacements from public
model configs and architecture, host API/symbol evidence, controlled shape
experiments, and CPU/binary-oracle validation.

Seven groups of complete XCLBIN directories are byte-identical. Fine-tunes and
derivatives must link to family/shape bundles rather than ship copied kernels.
See `docs/precompiled_replacement_map.md` for the exact groups and the 18
recommended shipping bundle boundaries. The raw evidence is in
`docs/precompiled_artifacts.json`.

Host-library findings:

- `q4_npu_eXpress`/`SafeTensors` is the highest-fan-out closed dependency.
- `mha` is the largest source-interface gap because no public header exists.
- Normal model engines implement `causal_lm`; Qwen Omni and Whisper require
  distinct contracts.
- XRT has 24 ELF files and HRX has 23; HRX lacks Gemma4-12B and includes an
  extra `libdequant_new.so`, so backend directories are not symmetric.
- HRX ELF files retain absolute build-machine RUNPATHs; never treat those paths
  as source or deployment requirements.
- An engine is not fully removed until installer, standalone test, and `src/lib`
  binaries are checked. The EmbeddingGemma cleanup is now complete: installer
  entries removed, the standalone test links only Boost/threads/tokenizers/XRT,
  and all six `gemma_embedding` binaries are gone.
- `src/test/CMakeLists.txt` unconditionally links `q4_npu_eXpress`, `lm_head`,
  `dequant`, `gemm`, and `mha`. Do not use that helper for open-engine tests;
  copy the standalone pattern from
  `src/test/gemma_embedding/CMakeLists.txt` instead.
- Shared typed buffers still depend on XRT types through `device_runtime.hpp`
  and `buffer.hpp`, so even CPU-only open-engine tests currently need XRT
  headers plus `xrt_coreutil` at link time. That is a header/runtime
  dependency, not a closed engine dependency.

New model builders extend `utilities/q4nx-build`. Reuse its architecture
detection, tensor mappings, and layout handling; add open output backends rather
than creating a separate converter.

---

## References

- mlir-aie v1.3.4 whole_array: `programming_examples/basic/matrix_multiplication/whole_array/whole_array.py`
- iron API: `aie.iron.kernels.mm()`, `Runtime()` context manager, `TensorTiler2D`
- XRT: `xrt::xclbin`, `register_xclbin`, `hw_context`, `xrt::kernel(context, name)`
- Upstream: `Xilinx/mlir-aie` commit ed23bba (v1.3.4 tag)
