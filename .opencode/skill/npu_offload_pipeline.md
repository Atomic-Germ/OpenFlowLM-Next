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
- ironvenv: Python 3.13, mlir-aie 1.3.4, aiecc LLVM 23, Peano toolchain
  - Wheel: `llvm_aie-21.0.0.2026080301+c9c5ecb7` from Xilinx/llvm-aie nightly
  - Install: `python3 -m zipfile -e wheel.whl ironvenv/lib/python3.13/site-packages/`
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

## References

- mlir-aie v1.3.4 whole_array: `programming_examples/basic/matrix_multiplication/whole_array/whole_array.py`
- iron API: `aie.iron.kernels.mm()`, `Runtime()` context manager, `TensorTiler2D`
- XRT: `xrt::xclbin`, `register_xclbin`, `hw_context`, `xrt::kernel(context, name)`
- Upstream: `Xilinx/mlir-aie` commit ed23bba (v1.3.4 tag)
