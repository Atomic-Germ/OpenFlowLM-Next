# GGUF Code Isolation Plan

## Overview

This plan outlines the extraction/isolation of GGUF-related code from the OpenFlowLM-Next repository. The GGUF code spans multiple subsystems: file parsing, pool packing, model conversion, and installer utilities. The goal is to isolate these pieces one at a time while maintaining functionality.

## GGUF Code Components

### 1. GGUF File Parsing (`src/open_qwen36/gguf_file.cpp/hpp`)
- **Purpose**: Parse GGUF model files, extract tensor metadata and weights
- **Key functions**:
  - `GgufFile::GgufFile()` - Constructor that maps the GGUF file
  - `GgufFile::gguf_name()` - Tensor name mapping (GGUF -> internal)
  - `GgufFile::tensor()` - Get tensor info
  - `GgufFile::raw()` - Get raw tensor bytes
  - `GgufFile::meta()` - Get tensor metadata
  - `GgufFile::bf16()`, `f32()` - Dequantize tensor to float
  - `GgufFile::embed_row()` - Read a single row of a tensor
  - `GgufFile::kv_str()`, `kv_u64()`, `kv_f64()` - Read metadata key values

- **Dependencies**: Uses standard C++ file I/O, no external dependencies beyond the GGUF format spec

### 2. Pool Packing/Unpacking (`src/open_qwen36/pools.cpp/hpp`, `open_kernels/gguf_pool.py`)
- **Purpose**: Pack/unpack quantized weights into the NPU pool format (f32-scale chunks)
- **Key functions in pools.cpp**:
  - `pack_pool()` - Apply a sequence of pack ops to produce a pool
  - `pack_lmhead()` - Specialized LM head packing
  - `pack_norm()` - Pack layernorm weights
  - Various apply ops: `std_perm`, `std_perm_gguf`, `bf16_gemm`, `q8_perm`, `expert_stripes`, `expert_down`, `put`, `lmhead_q8`

- **Key functions in open_kernels/gguf_pool.py**:
  - `random_q4_blocks()`, `random_q8_0_blocks()` - Generate test GGUF blocks
  - `unpack_q4()`, `unpack_q8_0()` - Dequantize GGUF blocks
  - `dequant_q4()`, `dequant_q8_0()` - Dequantize pool chunks
  - `chunk_geometry()` - Compute pool chunk geometry
  - `pack_q4_pool()`, `pack_q8_pool()`, `pack_q8_pool_lmhead()` - Pack GGUF blocks into pool format
  - `dequant_pool()`, `dequant_pool_lmhead()` - Dequantize pool back to f32
  - `q4_chunk_bytes()`, `q8_chunk_bytes()` - Create individual chunks

- **Dependencies**: numpy for array operations, exact fp16->f32 widening

### 3. Model Manifest & Conversion (`src/open_qwen36/manifest.cpp/hpp`, `utilities/q4nx-build/`)
- **Purpose**: Model manifest management and GGUF->q4nx conversion
- **Key files**:
  - `manifest.cpp/hpp` - Manifest structure and packing plan interpreter
  - `q4nx-build/` utilities - Convert GGUF models to q4nx container format
  - Model converters for: llama, qwen2, qwen3, qwen35, qwen36moe, granite, k2, hunyuan, lfm2, nanbeige, phi4, qwen2vl, uno

- **Key conversion logic**:
  - Read GGUF file with `GGUFReader`
  - Dequantize weights
  - Requantize to q4nx format (bf16 scales + int4/int8 codes)
  - Write q4nx container

### 4. oflm-add Installer Changes (`utilities/oflm-add/`)
- **Purpose**: Install GGUF models via `oflm add` command
- **Key changes**:
  - GGUF install gate (manifest-based, not family-name list)
  - Per-model user settings (Issue 188)
  - Model registry integration
  - `--family` and `--xclbin-from` parameters for `oflm add`

### 5. Open-Kernel Designs (`open_kernels/designs/`, `open_kernels/recipes/`)
- **Purpose**: Kernel designs that support GGUF layouts
- **Key designs**:
  - `lm_head_q4/lm_head_q8` - LM head quant designs
  - `gemv_q4` - GEMV with q4 layouts
  - Recipe-based family detection (qwen3, llama3, etc.)
  - `spec.py` - Specification system

### 6. Skills Documentation
- `.agents/skills/open-gguf-direct/SKILL.md` - Documentation of GGUF-direct integration

## Isolation Strategy

The isolation will proceed in **7 phases**, each focusing on a distinct subsystem:

### Phase 1: GGUF File Parsing Isolation ✅ COMPLETED
- **Goal**: Extract `src/open_qwen36/gguf_file.cpp/hpp` as a standalone module in `src/gguf/`
- **What was done**:
  - Created `src/gguf/gguf_file.hpp` and `src/gguf/gguf_file.cpp` with the full GGUF file parsing code
  - Updated includes in consumer files to reference the new location:
    - `src/open_embedding/engine.hpp` → `#include "gguf/gguf_file.hpp"`
    - `src/open_qwen36/core.hpp` → `#include "gguf/gguf_file.hpp"`
    - `src/open_qwen36/pools.cpp` → `#include "gguf/gguf_file.hpp"`
    - `src/open_qwen36/gguf_pack_test.cpp` → `#include "gguf/gguf_file.hpp"`
    - `src/open_qwen36/gguf_file.cpp` → `#include "gguf/gguf_file.hpp"` (self-reference)
  - Verified all include paths resolve correctly from `src/` directory
- **Risk**: Low - this is self-contained file parsing, now isolated in `src/gguf/`

### Phase 2: Pool Packing/Unpacking Isolation ✅ COMPLETED
- **Goal**: Isolate `open_kernels/gguf_pool.py` and `open_kernels/gguf_scale_width.py` into a dedicated package
- **What was done**:
  - Created `open_kernels/gguf_pool/` package directory with:
    - `gguf_pool.py` - Core pool operations (CH_Q4, CH_Q8, chunk_geometry, random_q4_blocks, random_q8_0_blocks, unpack/dequant functions, pack functions)
    - `gguf_scale_width.py` - F32 scale width validation (Q8_CH_F32, kernel_rne, test functions)
    - `__init__.py` - Package re-exports for convenient import
  - Created shim at `open_kernels/gguf_pool.py` that re-exports all symbols from the package, maintaining backward compatibility with `from gguf_pool import ...`
  - Updated `__init__.py` to properly export `Q8_CH_F32` and other symbols
  - Verified all import patterns work:
    - `from gguf_pool import CH_Q4, CH_Q8` ✅
    - `from gguf_pool import CH_Q4, chunk_geometry as gg_chunk_geometry` ✅
    - `from gguf_pool import CH_Q8 as CH_GGUF, dequant_pool_lmhead as gg_dequant` ✅
    - `import gguf_pool` ✅
- **Risk**: Low - the shim maintains full backward compatibility; all existing import patterns continue to work

### Phase 3: Manifest and Conversion Isolation ✅ COMPLETED
- **Goal**: Extract manifest code and conversion utilities for GGUF-direct twin support
- **What was done**:
  - Created `src/open_qwen36/manifest_gguf/` package with:
    - `manifest_gguf.hpp` - Interface declarations for GGUF twin manifest operations:
      - `load_gguf_twin()` - Load a GGUF twin manifest from JSON
      - `make_gguf_twin_spec()` - Create a GGUF twin ModelSpec with quant="q4_1_f32"
      - `emit_gguf_twin()` - Emit a GGUF twin manifest JSON fragment
      - `gguf_quant_allowed()` - Check if a quant type is supported by pool pack ops
      - `gguf_tensor_chunk_bytes()` - Get chunk bytes for a GGUF tensor type
    - `manifest_gguf.cpp` - Implementation of all above functions
  - Verified the `Manifest` struct already has `std::unique_ptr<Manifest> gguf;` field (line 256 of manifest.hpp)
  - Verified the engine code in `core.cpp` already uses `man_.gguf.get()` to swap in the GGUF twin (lines 218-221)
  - Verified existing manifests in `src/xclbins/*/open_kernels/manifest.json` already contain the `gguf` section generated by `export_qwen36_kernels.py`
  - Updated `src/CMakeLists.txt` to include `manifest_gguf.cpp` in sources
- **Risk**: High - conversion code is deeply integrated, but the isolated module provides a clean interface for the GGUF twin logic

### Phase 4: oflm-add Installer GGUF Gate ✅ (Code Already Present, Comment Updated)
- **Goal**: The oflg-add installer already had substantial GGUF support; the main update was to reference the isolated GGUF parser module
- **What was done**:
  - Verified the oflg-add `utilities/oflm-add/oflm_add/__init__.py` already has comprehensive GGUF support including:
    - `GGUF_FILES` and `GGUF_OPTIONAL` - file lists for GGUF install
    - `GGUF_QUANT_PREFERENCE` = `["Q4_1", "Q4_0", "Q4_K", "Q5_K", "Q6_K", "Q8_0", "IQ4"]` - quant preference order
    - `choose_gguf_file(names, want=None)` - picks the best compatible *.gguf from a repo
    - `manifest_supports_gguf(path)` - checks if manifest.json has the "gguf" section
    - `repo_has_q4nx(tree, cache=None)` - checks for q4nx container precedence
    - `gguf_kernel_set(name, roots)` - finds the kernel set for a model carrying GGUF-direct build
    - `EMBEDDING_FAMILIES = {"embed-gemma"}` - embedding GGUF uses bf16 matmul kernels, not f32-scale pools
    - `NPU_EMBED_FAMILIES` and `NPU_EMBED_FILES` - NPU-embedding models (NOT GGUF/q4nx)
    - `GGUF_QUANT_RE` regex and `gguf_quant_suffix()`, `gguf_quant_family()`, `gguf_quant_of()` - quant name parsing
    - `choose_gguf_file()` integration with `repo_has_q4nx()` for precedence logic
  - Updated comment in `utilities/oflm-add/oflm_add/__init__.py` line 2022:
    - Changed `# src/open_qwen36/gguf_file.hpp -- only metadata parsing needs these`
    - To `# src/gguf/gguf_file.hpp -- only metadata parsing needs these`
    - To reference the isolated GGUF parser module at `src/gguf/`
- **Risk**: Low - the oflg-add code was already well-integrated; the only change was updating the comment to reference the new isolated module location

### Phase 5: Open-Kernel GGUF Designs ✅ COMPLETED
- **Goal**: Isolate kernel designs that handle GGUF layouts and ensure family recipes properly route GGUF models
- **What was done**:
  - Created `open_kernels/designs/gguf/` directory with:
    - `lm_head_q4/lm_head_q4.py` - LM head design with `LMHEAD_SCALES_F32` env var for f32-scale (6144 B) chunks
    - `lm_head_q8/lm_head_q8.py` - LM head design with `LMHEAD_SCALES_F32` env var for f32-scale (9216 B) chunks
    - `gemv_q4/gemv_q4.py` - GEMV design with f32-scale kernel support and `gemv_q4s32_*` kernel entries
  - All designs already have built-in support for both bf16 (q4nx) and f32 (GGUF-direct) modes via environment variables:
    - `lm_head_q4`: `LMHEAD_SCALES_F32=1` → 6144 B chunks, `gemv_q4s32_gy` kernel
    - `lm_head_q8`: `LMHEAD_SCALES_F32=1` → 9216 B chunks, `lm_head_q8s32_group` kernel
    - `gemv_q4`: `LMHEAD_SCALES_F32=1` → 6144 B chunks, `gemv_q4s32_*` kernel entries
  - Verified the `dense.py` recipe already handles `spec.quant == "q4_1_f32"` correctly:
    - Uses `std_perm_gguf` op when quant is "q4_1_f32"
    - Emits f32-scale pool layout (6144 B or 9216 B depending on design)
    - Programs with `quant="q4_1_f32"` for the GGUF twin manifest
  - Verified existing `open_kernels/designs/dense/` build directories with `f32_f5c3cab5` suffix already contain the f32-scale (GGUF-direct) builds
- **Risk**: Low - the designs already supported both modes; the isolation organizes them clearly and ensures recipe integration is correct

### Phase 6: Skills Documentation
- **Goal**: Update/isolate the open-gguf-direct skill
- **What to do**:
  - Update `.agents/skills/open-gguf-direct/SKILL.md` to reflect isolated code
  - Ensure skill references correct paths
- **Risk**: Low - documentation only

### Phase 7: Integration Testing
- **Goal**: Verify all isolated pieces work together
- **What to do**:
  - Run existing tests to ensure no regressions
  - Test GGUF model loading and pool creation
  - Validate that the isolation doesn't break existing functionality
- **Risk**: High - need to ensure all paths still work

## Dependencies Between Pieces

```
gguf_file.cpp/hpp
    ↓
pools.cpp/hpp (uses GgufFile for std_perm_gguf op)
    ↓
manifest.cpp/hpp (interprets pack ops, some use GgufFile)
    ↓
q4nx-build converters (GGUF -> q4nx format)
    ↓
oflm-add (installs GGUF models per family:size)
    ↓
open_kernels/designs (GGUF pool layouts)
    ↓
open_kernels/gguf_pool.py (pool pack/unpack in Python)
```

## Recommended Starting Point

**Start with Phase 1 (GGUF File Parsing)** because:
1. It's the most self-contained component
2. Other components depend on it but it doesn't depend on them
3. It provides the foundation for understanding how GGUF models are loaded
4. The code has clear boundaries (file I/O -> tensor metadata -> tensor data)

After Phase 1, proceed with Phase 2 (pool packing) since it's the next most independent piece, and then Phase 3 (manifest/conversion) which bridges the two.

## Success Criteria

Each phase is complete when:
- [ ] The isolated code compiles/runs without errors
- [ ] All tests specific to that component pass
- [ ] No other code breaks due to the isolation
- [ ] Documentation is updated to reflect new paths
- [ ] The original code can be replaced with the isolated version

## Next Steps

1. Review this plan and confirm the phase order
2. Start with Phase 1: Create the GGUF parser module
3. After each phase, run the test suite to verify no regressions
4. Document any issues or unexpected dependencies discovered during isolation