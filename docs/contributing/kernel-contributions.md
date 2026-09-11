# Kernel Contributions

> "The code that matters." -- Keep it simple, clear, and focused.

This document covers contributing **kernel builds** (AIE xclbins) for OpenFlowLM's NPU engine.

---

## Overview

OpenFlowLM runs models on AMD's XDNA2 NPUs. The `open_kernels/` directory holds the **design sources** that are compiled into `.xclbin` files during build time.

**Two parts to understand:**
1. **Open engine kernels** -- Built with `ironvenv`/`mlir-aie` (Linux only)
2. **Open NPUE kernels** -- Built with `q4nx-build` (Linux only)

The executable builds with CMake, but the kernel xclbins need a separate toolchain.

---

## Build Requirements

### For Open Engine Kernels (`open_kernels/`)

- **XRT** installed on host (`/opt/xilinx/xrt`)
- **Kernel toolchain** (`ironvenv` with `mlir-aie` + Peano)
- **NPU present** on build host (for embedding models)

### For Open NPUE Kernels (`open_npue/`)

- **q4nx-build** tool (`utilities/q4nx-build`)
- **NPU present** (for BERT embedding export)
- **Python 3.11** (for `open_kernels/export_qwen36_kernels.py`)

### Prerequisites

See [docs/linux-getting-started.md](../docs/linux-getting-started.md) for full setup.

---

## Building Open Engine Kernels

The open engine dispatches to kernel sets derived from each model's `config.json`.

### Build All Families

```bash
# Activate kernel toolchain
source ironvenv/bin/activate

# Build all kernel sets
python open_kernels/export_qwen36_kernels.py
```

**Output:** `src/xclbins/<model>/open_kernels/`

### Build Specific Families

Configure with `-DOFLM_KERNEL_SPECS` when building with CMake, or specify `--spec` for the script.

```bash
python open_kernels/export_qwen36_kernels.py --spec qwen3-4b
python open_kernels/export_qwen36_kernels.py --only qwen3-4b:ax0
```

**Available specs:**
- `qwen3-4b` -- Qwen3 dense 4B (all sizes)
- `gemma3-4b` -- Gemma3 dense 4B
- `llama-8b` -- Llama 3.1 8B
- `hy-mt2-7b` -- Hy-MT2-7B
- `granite-3b` -- IBM Granite 4.2 3B
- `qwen35-4b` -- Qwen3.5 dense 4B
- `qwen36-moe` -- Qwen3.6-MoE

### Build and Verify

```bash
oflm list
```

The engine should report which kernel set it resolved when loading a model.

---

## Building Open NPUE Kernels

Open NPUE kernels are for embedding models (Gemma, MedGemma, etc.).

### Build All Families

```bash
cd <repo>
.\npu_offload\gemm_rtp\build.ps1
```

**Output:** `src/xclbins/`

### Build Specific Family

```bash
.\npu_offload\gemm_rtp\build.ps1 -Only Gemma-300M-OpenNPU2
```

**Build time:** 3–4 minutes per family (~20 minutes total).

### Build Flags

Build flags live in `families.json`, not the script:

- `-Force` -- Rebuild all families
- `-Only <name>` -- Build one family
- `--check DIR` -- Verify against reference

---

## Adding New Kernel Families

### Open Engine (XDNA2)

1. **Find the model's `config.json`**
   - Typically at `https://huggingface.co/<model>/<model>/resolve/main/config.json`
   - Or download the model and read `config.json`

2. **Check the recipe**
   - Open `src/open_qwen36/engine.cpp` to see how recipes work
   - Find or create a `ModelSpec` for your model

3. **Add the shape**
   - Edit `open_kernels/recipes/<family>.json` if adding new shape
   - Or add `ModelSpec` entries to `open_kernels/model_specs.cpp`

4. **Build the kernel set**
   ```bash
   python open_kernels/export_qwen36_kernels.py --model-dir DIR
   ```

5. **Verify it works**
   ```bash
   oflm list
   oflm run <model>
   ```

### Open NPUE (BERT)

1. **Add to families.json**
   - Edit `npu_offload/gemm_rtp/families.json`
   - Add your model's geometry and parameters

2. **Build**
   ```bash
   .\npu_offload\gemm_rtp\build.ps1
   ```

3. **Verify**
   ```bash
   oflm list
   oflm run <model>
   ```

---

## Debugging Kernel Builds

### Common Issues

**"No open kernels found"**
- Check that kernels were exported to `src/xclbins/`
- Verify `model_list.json` includes your model
- Run `oflm list` to see which kernel set was loaded

**"Kernels match but model fails"**
- Check `manifest.json` in the kernel set
- Verify `toolchain.json` matches your mlir-aie/Peano versions
- Run `oflm list` to confirm kernel set resolution

**Build fails on NPU**
- Ensure NPU driver is installed (32.0.203.311+)
- Check XRT is installed and accessible
- Run `oflm list` to verify kernel set was built

### Debug Commands

```bash
# List available kernels
oflm list

# Show kernel set resolution
oflm run <model> --log-level debug

# Check kernel manifest
cat src/xclbins/<model>/open_kernels/manifest.json
```

---

## Branch Naming for Kernels

Use the standard format:

- `feat/llama-8b` -- Add Llama 8B kernel support
- `fix/kernel-llama-8b` -- Fix Llama 8B kernel build
- `perf/lmhead-q4` -- Improve LM head quantization
- `docs/kernel-experiment` -- Kernel experiment documentation
- `chore/q4nx-build` -- q4nx-build tool changes

---

## Contributing to Kernel Recipes

### Recipe Structure

Each family has a recipe that defines how models are exported:

1. **Shape extraction** from `config.json`
2. **Kernel selection** (attention, gemv, lm_head, etc.)
3. **Quantization path** (Q4, Q8, bf16, etc.)

### Adding New Shapes

1. Read the existing recipe for your family
2. Identify the shape parameters needed
3. Add to `open_kernels/model_specs.cpp` or `recipes/<family>.json`
4. Test with `export_qwen36_kernels.py`

---

## Testing Kernel Contributions

**Before submitting:**
1. Build the kernel set
2. Run `oflm list` to verify
3. Run `oflm run <model>` to test locally
4. Check benchmarks if applicable

**For new models:**
1. Add benchmark results to `docs/benchmarks/`
2. Include `config.json` in your test model if helpful
3. Document known limitations

---

## Getting Help

- **Open an issue** before starting major kernel contributions
- **Join Discord** for real-time help
- **Ask questions** on PRs or in issues

**Kernel-specific resources:**
- `open_kernels/README.md` -- Open engine kernel structure
- `open_kernels/PROVENANCE.md` -- Build history and tracking
- `docs/linux-getting-started.md` -- Full build setup

---

## License

OpenFlowLM is released under the MIT License. See [LICENSE_RUNTIME.txt](./LICENSE_RUNTIME.txt) for details.

**Kernel families:**
- Open engine kernels (XDNA2) -- MIT licensed
- Open NPUE kernels -- MIT licensed  
- Upstream kernels (if retained) -- Original FastFlowLM license

---

## Quick Links

- [Branch naming convention](#branch-naming-convention)
- [Open engine kernels](#open-engine-kernels)
- [Open NPUE kernels](#open-npue-kernels)
- [Adding new families](#adding-new-family)
- [Debugging](#debugging)
- [Testing](#testing)
