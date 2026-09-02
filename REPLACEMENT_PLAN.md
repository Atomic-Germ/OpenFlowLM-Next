# Open FastFlowLM Closed-Source Replacement Plan

## Overview
Map of all pre-compiled closed-source artifacts and replacement strategies for AMD NPU2/iron/mlirae backends.

---

## 1. Full Map of Pre-Compiled Closed-Source Artifacts

### `.xclbin` Kernel Binaries
- **39 model directories** in `src/xclbins/`, each with 4-10 `.xclbin` files
- Split by kernel component: `mm.xclbin`, `layer.xclbin`, `attn.xclbin`, `dequant.xclbin`, etc.
- Example: `Qwen3-8B-NPU2/mm.xclbin`, `Embedding-Gemma-300M-NPU2/attn_full_mask.xclbin`
- Contain NPU2-accelerated matrix multiplication and kernel binaries

### `.so` Shared Libraries (141 total)
- **72 in `src/lib/xrt/`**: `libgemma_npu.so`, `libqwen3_npu.so`, `libgemma_embedding.so`, `libgemm.so`, `libdequant.so`, `libmha.so`, `liblm_head.so`, `libphi4_npu.so`, `libnanbeige_npu.so`, `libgpt_oss_npu.so`, `liblfm2_npu.so`
- **69 in `src/lib/hrx/`**: HRX amdxdna runtime equivalent libraries
- Include runtime glue, NPU dispatch, and model-specific kernels

### `model.q4nx` Format
- Closed binary tensor format for NPU2 weights
- Present in all 80+ model entries in `model_list.json`
- Converted from GGUF/HF weights via `q4nx-build` tools
- Used by original `libgemma_embedding.so` stack

### Other Closed Components
- **`libgemma_embedding.so`**: Original closed embedding backend — already replaced by `open_embedding::Engine` in `docs/ExampleNPU/src/open_embedding/`
- **XRT runtime**: `libxrt_coreutil.so` + plugins (device management, xclbin loading)
- **HRX runtime**: `libhrx.so` (amdxdna alternative)

---

## 2. Replacement Priority Order

| Priority | Model Family | Key Replacement Need |
|----------|-------------|---------------------|
| **1** | Embedding-Gemma-300M | ✅ Already replaced |
| **2** | Gemma3-text (1B, 4B) | Same blocks as embedding; new xclbins for 4B shapes |
| **3** | Nanbeige (3B) | Standard transformer (llama-like) |
| **4** | Phi4 (4B) | Standard + long/short RoPE |
| **5** | Llama3 (1B, 3B, 8B) | Standard MHA transformer |
| **6** | Qwen3 (0.6B-8B) | MHA + Q/K norm |
| **7** | LFM2 (1.2B, 2.6B) | SSM hybrid (shortconv + delta net) |
| **8** | Qwen3.5 (0.8B-9B) | Linear attention + SSM + conv1d |
| **9** | Qwen3.6-MoE (35B) | MoE routing + expert dispatch |
| **10** | GPT-OSS (20B) | Attention sinks + expert FFN |
| **11** | VLMs (Qwen3-VL, Gemma4) | Vision encoder + cross-attention |
| **12** | Whisper (ASR) | Encoder-decoder, completely different |

---

## 3. Open Engine Pattern (reusable across models)

```
src/open_embedding/engine.hpp          - Engine class (load/safetensors/weights_manifest)
src/open_embedding/engine.cpp          - Transformer forward pass with NPU offload
src/open_embedding/npu_matmul.cpp      - NPU2 BF16 matmul backend (XRT)
src/open_embedding/tools/make_manifest.py - Generate weights_manifest.json from safetensors
src/open_embedding/tools/oracle/       - Reference validation scripts
```

**Key Design:**
- Weights loaded from `model.safetensors` via `weights_manifest.json` (no Q4NX)
- FP32 CPU forward pass: RMSNorm, RoPE, GQA, GELU-MLP
- Optional NPU2 offload for 5 large projections (q, o, gate, up, down) via bf16→f32 kernels
- Output dtype: bf16 inputs → FP32 output (bit-exact, cosine ≥ 0.999)
- `is_npu_projection_weight()` heuristic selects which proj weights go to NPU

---

## 4. CMake/build System Changes

### Phase 1 - Open Embedding Integration
- Remove closed embedding: `modeling_gemma_embedding.hpp`, `modeling_gemma_embedding.cpp`, `gemma_embedding/gemma_embedding.hpp`
- Add open_embedding sources: `src/open_embedding/engine.cpp`, `src/open_embedding/npu_matmul.cpp`
- Add compile definition: `FLM_USE_OPEN_EMBEDDING=1`
- Gate NPU offload: `FLM_USE_OPEN_EMBEDDING_NPU=1` when `FLM_USE_OPEN_EMBEDDING=ON` AND `FLM_USE_HRX=OFF`
- Update `model_list.json` files list: remove `model.q4nx`, add `weights_manifest.json` + safetensors weight paths (`weights/2_Dense.safetensors`, `weights/3_Dense.safetensors`)
- Remove dead xclbins: `xclbins/Embedding-Gemma-300M-NPU2/`

### Phase 2+ - Subsequent Models
- Add `option(FLM_USE_OPEN_GEMMA3 ...)` and similar flags
- Conditionally link/open-link appropriate `lib*_npu.so` based on enabled flags
- Each model gets `src/open_<family>/engine.cpp`

### CMake Highlights (from existing CMakeLists.txt)
- `target_compile_definitions(flm PUBLIC FLM_USE_OPEN_EMBEDDING=1)` — always on when enabled
- `if(NOT FLM_USE_HRX) target_compile_definitions(flm PUBLIC FLM_USE_OPEN_EMBEDDING_NPU=1)` — NPU only with XRT
- Source glob: add `"src/open_embedding/*.cpp"` 
- Link: close `gemma_embedding` when open path enabled

---

## 5. Tooling Needs in `utilities/`

The existing utilities can be extended rather than rebuilt from scratch:

| Tool | Current Scope | Extensions Needed |
|------|--------------|------------------|
| **`q4nx-build/`** | Converts GGUF/HF → Q4NX | Also generate `weights_manifest.json` + safetensors splitting (2_Dense, 3_Dense heads) |
| **`flm-add/`** | Installs FLM models, symlinks xclbins | Also copy/convert weights to safetensors + generate `weights_manifest.json`; stop symlinking closed xclbins |
| **`flm-test/`** | E-suite embedding tests (E1-E8) | Extend E-suite for LLM models (perplexity/cosine checks); add NPU offload validation |
| **`q4nx-build/` builder extension** | Reuse the existing converter architecture | Add a builder mode/subcommand that generates config.json, weights_manifest.json, tokenizer assets, safetensors splits, and model-family build metadata |

**Key insight**: Do not build a separate converter from scratch. `q4nx-build/model_converter.py` and its architecture detection, tensor mapping, and packing infrastructure are the base for the open-model builder. Add open-output backends alongside Q4NX output so both pipelines share model-family knowledge.

---

## 6. Validation Milestones

Each open engine must pass:
1. **E-suite adaptation**: Embedding=E8 cosine agreement; LLM=perplexity/cosine checks
2. **NPU offload validation**: Cosine ≥ 0.999 between NPU bf16→f32 path and CPU FP32 reference
3. **Performance benchmarking**: Tokens/sec comparison with closed engine
4. **Cross-backend testing**: XRT and HRX compatibility

---

## 7. Execution Roadmap

1. Port open_embedding source from ExampleNPU into `src/open_embedding/`
2. Port `open_gemma_embedding.hpp` adapter class
3. Remove closed embedding files (`modeling_gemma_embedding.*`, `gemma_embedding.*`)
4. Simplify `AutoEmbeddingModel` (remove Q4NX dependency in embedding path)
5. Update `all_embedding_model.hpp` (always instantiate `OpenGemma_Embedding`)
6. Update `CMakeLists.txt` (add sources, remove closed lib, add tokenizers link)
7. Update `model_list.json` (embedding files list)
8. Remove dead xclbins (`xclbins/Embedding-Gemma-300M-NPU2/`)
9. Add `flm-test` install rule to `CMakeLists.txt`
10. Build and run E-suite to verify
11. Update `npu_offload_pipeline.md` skill with Gemma3-text notes
12. Begin Gemma3-text open engine (Phase 2)

---

## Key Files Already Existing (reusable)

- `docs/ExampleNPU/src/open_embedding/engine.hpp` — Engine class definition
- `docs/ExampleNPU/src/open_embedding/engine.cpp` — Full transformer forward with NPU offload
- `docs/ExampleNPU/src/open_embedding/npu_matmul.cpp` — NPU2 BF16 matmul backend (XRT)
- `docs/ExampleNPU/src/open_embedding/tools/make_manifest.py` — Generate weights_manifest.json
- `docs/ExampleNPU/src/open_embedding/tools/gemma3_reference.py` — Numpy reference validation
- `docs/plans/open_xclbin_plan.md` — Full transition plan from closed to open
- `docs/plans/open_embedding.md` — E8 validation summary (cosine 0.999993)
- `utilities/q4nx-build/` — Model converter (extend for manifest generation)
