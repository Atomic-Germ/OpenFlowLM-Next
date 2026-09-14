# Code Contributions

> "The code that matters." -- Keep it simple, clear, and focused.

This document covers contributing general code changes to OpenFlowLM's engine and CLI.

---

## Before You Start

- **Read the code** you're modifying to understand patterns and conventions
- **Check existing tests** to see what behavior to preserve
- **Look at similar changes** to follow the same style
- **Open an issue** for major changes to get feedback

---

## Code Style

### General Rules

- Use **existing patterns** -- read the code you're changing to understand conventions
- Prefer **readability** over cleverness
- Use **consistent naming** across the codebase
- Add **inline comments** for non-obvious logic
- Keep changes **focused** -- one feature per branch

### Naming Conventions

- **Functions/methods**: `snake_case` (e.g., `get_attention_output`)
- **Classes/types**: `PascalCase` (e.g., `AttentionLayer`)
- **Variables/constants**: `snake_case` (e.g., `model_size`, `MAX_TOKENS`)
- **Private members**: prefix with `_` (e.g., `_temp_buffer`)

### Error Handling

- **Propagate** errors rather than swallowing them
- Use `hrx` to propagate dispatch failures
- Log context when errors occur

### Performance

- Profile before optimizing
- Document performance gains with benchmarks
- Consider memory implications for large context

---

## Branching for Code Changes

Use the clean, clear branch naming convention:

```bash
git checkout main
git pull origin main
git checkout -b feat/your-feature-name
```

**Types:**
- `feat/` -- New features
- `fix/` -- Bug fixes
- `perf/` -- Performance improvements
- `docs/` -- Documentation only
- `build/` -- Build system changes
- `refactor/` -- Code restructuring
- `chore/` -- Maintenance, dependencies
- `test/` -- Test additions

**Examples:**
- `feat/llama-support` -- Add Llama model support
- `fix/attention-vision` -- Fix attention vision path
- `perf/decode-3b` -- Improve decode on 3B models
- `build/root-cmake` -- Root CMake changes
- `chore/dependencies` -- Dependency updates

---

## Making Changes

### 1. Fork and Branch

```bash
git checkout main
git pull origin main
git checkout -b feat/your-feature-name
```

### 2. Make Changes

- **Read** the code you're modifying
- **Test** frequently with `git diff`
- **Keep changes focused** -- one feature per branch
- **Follow patterns** -- don't reinvent existing patterns

### 3. Test Your Changes

```bash
# Build
cmake --build --preset linux-default

# Run tests
ctest --preset linux-default

# Test locally
oflm run <model>
oflm list
```

### 4. Write Tests (if adding new functionality)

- Add tests to `src/tests/`
- Follow existing test patterns
- Make tests fail clearly when broken
- Consider integration tests for complex flows

### 5. Submit a Pull Request

- Push your branch to your fork: `git push origin feat/your-feature-name`
- Create a PR on GitHub
- Reference relevant issues in the PR description
- Request a review from maintainers

---

## Testing Your Changes

### Running Tests

```bash
# Build with all presets
cmake --build --preset linux-default

# Run tests
ctest --preset linux-default

# Run specific test
ctest --preset linux-default --test-name-pattern <pattern>
```

### Testing Locally

```bash
# List available models
oflm list

# Run a model
oflm run <model>

# Serve a model
oflm serve <model>
```

### Debugging

```bash
# Enable verbose logging
oflm run <model> --log-level debug

# Check kernel resolution
oflm list --verbose
```

---

## Common Code Patterns

### Reading Models

```cpp
// Read model config
auto config = ModelConfig::from_file("config.json");

// Read GGUF weights
GgufFile gguf;
auto weights = gguf.read_weights();
```

### Dispatching to NPU

```cpp
// Create dispatch
auto dispatch = Dispatch::create();

// Run forward
dispatch.forward();

// Check result
if (dispatch.has_error()) {
    log_error(dispatch.error());
}
```

### Error Handling

```cpp
// Propagate errors
try {
    auto result = operation();
} catch (const hrx_error& e) {
    hrx.propagate(e);
}
```

---

## Contributing to the Engine

### Open Engine (`src/open_qwen36/`)

The open engine dispatches to kernel sets. Changes here affect which kernels are selected and how they're used.

**Common patterns:**
- **Model specs** -- Define how models are registered and loaded
- **Recipes** -- Define kernel compositions for each family
- **Adapters** -- Define how models connect to the engine

### Open NPUE (`src/open_npue/`)

Open NPUE provides embedding model support. Changes here affect embedding model dispatch.

**Common patterns:**
- **Adapter selection** -- How models select which kernel to use
- **Linking** -- How flm-add links models to kernels
- **Validation** -- How kernels are verified at runtime

---

## Contributing to CLI Tools

### oflm CLI

The CLI provides the main user interface. Changes here affect the command-line experience.

**Commands:**
- `oflm run` -- Run a model
- `oflm serve` -- Serve as OpenAI-compatible API
- `oflm list` -- List available models
- `oflm bench` -- Benchmark models
- `oflm add` -- Add models to registry

**Testing changes:**
```bash
# Build CLI
cmake --build --preset linux-default

# Test commands
oflm run llama3.2:1b
oflm list
oflm bench qwen3.5:4b
```

---

## Code Review

**When submitting a PR:**
- Write a clear description
- Reference relevant issues
- Include test results
- List any breaking changes
- Add links to relevant docs

**After review:**
- Address reviewer feedback
- Run tests again after changes
- Push updated branch
- Mark as ready for merge

---

## Branch Lifecycle

- Your branch lives in your fork until merged
- Keep it up-to-date: `git pull origin main; git push origin feat/your-feature-name`
- Clean up after merge: `git branch -D feat/your-feature-name`

**Tracking your branch:**
```bash
# Pull latest main and update branch
git checkout main
git pull origin main
git checkout -b feat/your-feature-name

# Push to your fork
git push origin feat/your-feature-name
```

---

## Common Issues

### "Build fails on Windows"

- Run in Visual Studio developer environment
- Use `vcvars64.bat` to set include paths
- Check CMakeLists.txt for platform-specific flags

### "Kernel not loaded"

- Check that kernels were exported to `src/xclbins/`
- Verify `model_list.json` includes your model
- Run `oflm list` to see which kernel set was loaded

### "Test passes but runtime fails"

- Test with different model sizes
- Check kernel manifest after build
- Verify `toolchain.json` matches your mlir-aie/Peano versions

---

## Getting Help

- **Open an issue** before starting major changes
- **Join Discord** for real-time help
- **Ask questions** on PRs or in issues
- **Review existing PRs** to understand the codebase

---

## Quick Links

- [Branch naming convention](#branch-naming-convention)
- [Testing](#testing-your-changes)
- [Kernel contributions](../kernel-contributions.md)
- [Documentation](../docs/contributing/doc-contributions.md)
- [Tools](../docs/contributing/tool-contributions.md)
