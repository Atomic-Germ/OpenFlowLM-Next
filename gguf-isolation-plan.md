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

### Phase 2: Pool Packing/Unpacking Isolation
- **Goal**: Extract `open_kernels/gguf_pool.py` and related pool code
- **What to do**:
  - Create `open_kernels/gguf_pool/` directory with Python modules
  - Move `gguf_pool.py` and `gguf_scale_width.py` to the new directory
  - Update imports in dependent files (lm_head_q8, gemv_q4)
  - Create compatibility shims if needed
- **Risk**: Medium - pool code is used by multiple kernel designs

### Phase 3: Manifest and Conversion Isolation
- **Goal**: Extract manifest code and conversion utilities
- **What to do**:
  - Create `src/open_qwen36/manifest_gguf/` for manifest-specific code
  - Move/qurt the q4nx-build conversion utilities to a separate location
  - Update references in oflm-add and engine code
- **Risk**: High - conversion code is deeply integrated

### Phase 4: oflm-add Installer GGUF Gate
- **Goal**: Isolate the GGUF install gate logic
- **What to do**:
  - Extract the manifest-based install check from `utilities/oflm-add/`
  - Create a dedicated `oflm-add/gguf_install.py` module
  - Update `oflm add` command to use the new module
- **Risk**: Medium - installer logic touches many areas

### Phase 5: Open-Kernel GGUF Designs
- **Goal**: Isolate kernel designs that handle GGUF layouts
- **What to do**:
  - Create `open_kernels/designs/gguf/` for GGUF-specific designs
  - Move `lm_head_q4`, `lm_head_q8`, `gemv_q4` designs with their GGUF paths
  - Update recipe-based family detection to route GGUF models
- **Risk**: Medium - kernel designs affect core computation

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