# Tool Contributions

> "The code that matters." -- Keep it simple, clear, and focused.

This document covers contributing to OpenFlowLM toolchain components: `oflm-add`, `q4nx-build`, `flm-add`, and related utilities.

---

## Tool Overview

| Tool | Purpose | Location |
|---|---|---|
| `oflm-add` | Add models to registry | `utilities/oflm-add/` |
| `q4nx-build` | Build quantized GGUF kernels | `utilities/q4nx-build/` |
| `flm-add` | Legacy model addition | `utilities/flm-add/` |
| `oflm-test` | Run test suite | `utilities/oflm-test/` |
| `bench-configs` | Configure benchmarks | `utilities/bench-configs/` |
| `binary-inventory` | Inventory binaries | `utilities/binary-inventory/` |

---

## oflm-add

`oflm-add` adds models to the OpenFlowLM registry and links them to kernel sets.

### What It Does

1. Reads model `config.json`
2. Finds or creates matching kernel set
3. Updates `model_list.json` with new entry
4. Links model to its kernel family

### Build Requirements

- `python 3.11`
- `pyxrt` (XRT Python bindings)
- `third_party/tokenizers-cpp` submodule

### Building

```bash
cd utilities/oflm-add
python setup.py develop
```

### Contributing

1. **Read the code** -- Understand how models are registered
2. **Test locally** -- Add models and verify with `oflm list`
3. **Write tests** -- Add integration tests for new features
4. **Update README** -- Document new capabilities

### Common Tasks

**Adding a new model:**
```python
# In oflm-add, add to model registry
registry.add_model("qwen3-4b", "Qwen3 4B")
registry.link_to_kernel("qwen3-4b")
```

**Linking to existing kernels:**
```python
# Find existing kernel set
kernels = registry.find_kernels("qwen3-4b")

# Link model to kernels
model.link_to_kernels(kernels)
```

**Verifying:**
```bash
oflm list
```

---

## q4nx-build

`q4nx-build` builds quantized GGUF kernels for models that aren't covered by the open engine.

### What It Does

1. Reads GGUF model
2. Selects quantization format (Q4_K, Q4_0, etc.)
3. Builds kernels from source
4. Outputs `final.xclbin` and `insts.bin`

### Build Requirements

- **XRT** installed (`/opt/xilinx/xrt`)
- **NPU present**
- **Python 3.11**
- **third_party/q4nx** submodule

### Building

```bash
cd utilities/q4nx-build
source ironvenv/bin/activate
python build.py --model-dir <path> --format Q4_K
```

**Output:** `src/xclbins/<model>/open_kernels/`

### Contributing

1. **Add new quantizations** -- Add to `families.json`
2. **Fix bugs** -- Report and fix issues
3. **Add tests** -- Verify quantization correctness
4. **Update README** -- Document new formats

### Common Tasks

**Adding new quantization:**
```json
{
  "model": "gemma-300m",
  "format": "Q4_K",
  "layers": ["matmul", "gemv"],
  "precision": "q4"
}
```

**Building specific layers:**
```bash
python build.py --model-dir <path> --layers matmul,gemv
```

**Verifying:**
```bash
oflm list
oflm run <model>
```

---

## flm-add

`flm-add` is the legacy model addition tool, now deprecated in favor of `oflm-add`.

### Build Requirements

- `python 3.11`
- `third_party/tokenizers-cpp` submodule

### Building

```bash
cd utilities/flm-add
python setup.py develop
```

### Contributing

- Review PRs to ensure compatibility with `oflm-add`
- Add tests for legacy features
- Update README if needed

---

## oflm-test

`oflm-test` runs the full test suite for different model categories.

### What It Does

1. Builds the engine
2. Runs smoke tests
3. Tests kernel dispatch
4. Verifies model loading

### Running Tests

```bash
# Run full suite
oflm-test --llm
oflm-test --vision
oflm-test --embed
oflm-test --tools
```

### Contributing

1. **Add new tests** -- Follow existing test patterns
2. **Fix failing tests** -- Report bugs and fix
3. **Update benchmarks** -- Add results to `docs/benchmarks/`
4. **Improve coverage** -- Add tests for edge cases

---

## bench-configs

`bench-configs` manages benchmark configurations.

### What It Does

1. Defines benchmark parameters
2. Runs benchmark suites
3. Generates reports

### Running Benchmarks

```bash
# Run benchmark suite
bench-configs run --model qwen3.5:4b --ctx-len 131072
```

### Contributing

1. **Add new configs** -- Document in `bench-configs/configs/`
2. **Fix bugs** -- Report and fix issues
3. **Update results** -- Add to `docs/benchmarks/`

---

## binary-inventory

`binary-inventory` creates a manifest of all binaries and dependencies.

### What It Does

1. Scans repository for binaries
2. Creates `precompiled_artifacts.json`
3. Tracks dependencies and versions

### Running

```bash
cd utilities/binary-inventory
python binary-inventory.py
```

### Contributing

- Keep inventory up-to-date
- Track binary versions
- Document dependencies

---

## Contributing to Tools

### General Process

1. **Fork and branch**
   ```bash
   git checkout main
   git pull origin main
   git checkout -b feat/tool-feature-name
   ```

2. **Make changes**
   - Read existing code
   - Follow existing patterns
   - Test locally

3. **Test**
   ```bash
   # Build tool
   cd utilities/tool
   python setup.py develop
   
   # Test
   oflm test --tool <tool>
   ```

4. **Submit PR**
   - Push branch to fork
   - Create PR
   - Reference issue
   - Request review

### Common Tool Patterns

**Adding new functionality:**
```python
# In tool's main module
def new_command(args):
    """New command description"""
    # Implementation
    pass
```

**Fixing bugs:**
```python
# Find the bug
try:
    result = operation()
except Exception as e:
    log_error(e)  # Propagate error
```

**Testing:**
```python
# In tool's tests
def test_new_feature():
    # Setup
    env = Env::create()
    
    # Run
    result = cmd("new-command", "arg")
    
    # Assert
    EXPECT_EQ(result.code(), 0)
```

---

## Building Tool Chains

### Full Toolchain

```bash
# Activate toolchain
source ironvenv/bin/activate

# Build all tools
cd utilities/oflm-add
python setup.py develop

cd utilities/q4nx-build
python build.py --model-dir <path> --format Q4_K
```

### Individual Tools

```bash
# oflm-add only
cd utilities/oflm-add
python setup.py develop

# Verify
oflm list
```

---

## Testing Tools

### Integration Tests

Test tools with real models:

```bash
# Test oflm-add
oflm-add --model-dir <path>
oflm list

# Test q4nx-build
cd utilities/q4nx-build
python build.py --model-dir <path> --format Q4_K
```

### Regression Tests

Test that existing functionality still works:

```bash
# Run full test suite
oflm-test --llm
oflm-test --embed
oflm-test --vision
```

---

## Common Issues

### "Tool fails to find kernels"

- Check `model_list.json` includes your model
- Verify kernels were exported to `src/xclbins/`
- Run `oflm list` to see which kernel set was loaded

### "Build fails on Windows"

- Run in Visual Studio developer environment
- Use `vcvars64.bat` to set include paths
- Check CMakeLists.txt for platform-specific flags

### "Test passes locally but fails in CI"

- Check environment variables
- Verify kernel toolchain is active
- Check file paths are absolute

---

## Getting Help

- **Open an issue** before starting major tool changes
- **Join Discord** for real-time help
- **Ask questions** on PRs or in issues

**Tool-specific resources:**
- `utilities/oflm-add/README.md` -- oflm-add documentation
- `utilities/q4nx-build/README.md` -- q4nx-build documentation
- `utilities/oflm-test/README.md` -- Test suite documentation

---

## Quick Links

- [Code contributions](../contributing/code-contributions.md)
- [Kernel contributions](../kernel-contributions.md)
- [Documentation](../docs/contributing/doc-contributions.md)
- [Testing](../docs/contributing/test-contributions.md)
