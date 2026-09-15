# Test Contributions

> "The code that matters." -- Keep it simple, clear, and focused.

This document covers contributing tests to OpenFlowLM.

---

## Testing Philosophy

Tests should be **fast, focused, and coverage-rich**.

**Guiding principles:**
- **Fast** -- Tests that run in seconds, not minutes
- **Focused** -- One thing per test
- **Coverage-rich** -- Test common paths, not just happy paths
- **Fail fast** -- Clear errors when broken

---

## Test Structure

Tests live in `src/tests/`.

**Common test types:**
- **Unit tests** -- Test individual functions
- **Integration tests** -- Test flows across components
- **Kernel tests** -- Test kernel dispatch and execution
- **CLI tests** -- Test command-line interface

### Unit Tests

```cpp
TEST(AttentionLayer, basic_forward) {
    // Setup
    const auto config = ModelConfig::from_file("config.json");
    
    // Test
    auto result = layer.forward(input);
    
    // Assert
    EXPECT_TRUE(result.has_value());
    EXPECT_NEAR(result->output.size(), expected_size, 0.01);
}
```

### Integration Tests

```cpp
TEST(ModelLoad, full_pipeline) {
    // Setup
    const auto model_dir = "tests/models/test-model";
    
    // Test
    auto registry = Registry::create();
    registry.add_model_dir(model_dir);
    
    // Assert
    EXPECT_TRUE(registry.has_model("test-model"));
}
```

---

## Writing Tests

### Best Practices

1. **Follow existing patterns** -- Read tests to understand conventions
2. **Use fixtures** -- Reuse setup across tests
3. **Clear assertions** -- Tests should fail clearly when broken
4. **Small scope** -- One test per concept
5. **Fast execution** -- Avoid expensive operations unless testing them

### Test Fixtures

```cpp
class ModelFixture {
public:
    ModelFixture() {
        auto config = ModelConfig::from_file("config.json");
        layer_ = std::make_unique<AttentionLayer>(config);
    }
    
    ~ModelFixture() = default;
    
    AttentionLayer& layer() { return *layer_; }
    const AttentionLayer& layer() const { return *layer_; }
    
private:
    std::unique_ptr<AttentionLayer> layer_;
};
```

### Test Organization

- **Unit tests** → `src/tests/unit/`
- **Integration tests** → `src/tests/integration/`
- **Kernel tests** → `src/tests/kernels/`
- **CLI tests** → `src/tests/cli/`

---

## Running Tests

### With CMake

```bash
# Build with test support
cmake --preset linux-default

# Run all tests
ctest --preset linux-default

# Run specific test
ctest --preset linux-default --test-name-pattern <pattern>

# Run with output
ctest --preset linux-default --output-on-failure
```

### Manual Testing

```bash
# Run tests manually
ctest --preset linux-default -V

# Run specific test file
ctest --preset linux-default -T <test_name>
```

---

## Common Test Patterns

### Testing Config Loading

```cpp
TEST(Config, from_file) {
    auto config = ModelConfig::from_file("tests/test-config.json");
    
    EXPECT_EQ(config.model_size, 4194304);  // 4B in bytes
    EXPECT_EQ(config.vocab_size, 151936);
}
```

### Testing Kernel Dispatch

```cpp
TEST(KernelDispatch, select_attention) {
    auto dispatch = Dispatch::create();
    
    auto config = ModelConfig::from_file("tests/test-config.json");
    auto model = Registry::create().add_model_dir(config.model_dir);
    
    EXPECT_TRUE(dispatch.select(model, "attention", config));
    
    auto result = dispatch.forward();
    EXPECT_TRUE(result.has_value());
}
```

### Testing CLI Commands

```cpp
TEST(CLI, run_command) {
    auto env = Env::create();
    env.set("FLM_OPEN_KERNELS_DIR", "tests/kernels");
    
    auto result = cmd("run", "test-model");
    
    EXPECT_EQ(result.code(), 0);
}
```

---

## Kernel Tests

### Building Test Kernels

```bash
# Activate toolchain
source ironvenv/bin/activate

# Build test kernel
python open_kernels/export_qwen36_kernels.py --only qwen3-4b:ax0
```

### Testing Kernel Execution

```cpp
TEST(KernelExecution, qwen34b_ax0) {
    auto kernels_dir = "src/xclbins/qwen3-4b/open_kernels";
    
    auto dispatch = Dispatch::create();
    dispatch.set_kernels_dir(kernels_dir);
    
    auto config = ModelConfig::from_file("tests/test-qwen3.json");
    auto result = dispatch.forward(config);
    
    EXPECT_TRUE(result.has_value());
    EXPECT_NEAR(result->output.size(), expected_size, 0.01);
}
```

### Verifying Kernel Build

```cpp
TEST(KernelVerification, manifest) {
    auto kernels_dir = "src/xclbins/qwen3-4b/open_kernels";
    
    auto manifest = Manifest::from_file(kernels_dir + "/manifest.json");
    
    EXPECT_TRUE(manifest.has_version());
    EXPECT_TRUE(manifest.has_kernel_set());
}
```

---

## Debugging Tests

### Common Issues

**Test passes locally but fails in CI**
- Check environment variables
- Verify kernel toolchain is active
- Check file paths are absolute

**Kernel test fails**
- Verify kernels were built and exported
- Check `model_list.json` includes your model
- Run `oflm list` to see which kernel set was loaded

**Test times out**
- Reduce model size for quick iteration
- Add `--timeout` to test runner
- Check for resource leaks

### Debug Commands

```bash
# Run with verbose output
ctest --preset linux-default -V

# Run specific test
ctest --preset linux-default -T <test_name>

# Run with output on failure
ctest --preset linux-default --output-on-failure
```

---

## Writing Better Tests

### Test Coverage

**Good:**
- Tests common paths
- Tests edge cases
- Tests error conditions
- Tests integration points

**Bad:**
- Only tests happy paths
- Skips error conditions
- Tests entire systems in one test

### Test Names

**Good:**
- `Config.from_file` -- Clear what's being tested
- `KernelDispatch.select_attention` -- Clear purpose
- `ModelLoad.full_pipeline` -- Clear scope

**Bad:**
- `test1` -- No meaning
- `test_config` -- Vague

---

## Contributing Test Changes

### When to Write Tests

- **Adding new features** -- Always write tests
- **Fixing bugs** -- Add tests that catch regressions
- **Changing APIs** -- Ensure backward compatibility
- **Performance changes** -- Add benchmarks

### Before Submitting Tests

- [ ] Tests pass locally
- [ ] Tests are fast (< 1 minute)
- [ ] Tests cover the change and related areas
- [ ] Test names are descriptive
- [ ] Tests follow existing patterns
- [ ] No unnecessary dependencies

### After Submitting

- **Run the full test suite** after changes
- **Monitor CI** -- Check that tests pass
- **Add benchmarks** for performance changes

---

## Benchmarking

### Adding Benchmarks

Benchmarks go in `docs/benchmarks/`.

**Structure:**
```
docs/benchmarks/
├── qwen3_results.md
├── llama3_results.md
└── index.md
```

**Content:**
- Model name and size
- Quantization format
- Context length
- Tokens per second
- Power consumption
- Known issues

### Running Benchmarks

```bash
# Run benchmark
oflm bench <model>
```

**Output:** JSON or markdown with metrics

**Example:**
```bash
oflm bench qwen3.5:4b --ctx-len 131072
```

---

## Common Test Pitfalls

### Memory Leaks

- Check for `unique_ptr` in fixtures
- Verify cleanup in destructors
- Use valgrind to check

### Race Conditions

- Use thread-safe operations
- Add synchronization where needed
- Test concurrent access

### File System Issues

- Use absolute paths
- Clean up test artifacts
- Check permissions

---

## Getting Help

- **Open an issue** before starting major test changes
- **Join Discord** for real-time help
- **Ask questions** on PRs or in issues

**Test-specific resources:**
- `src/tests/` -- Existing tests to follow
- `src/tests/unit/` -- Unit test patterns
- `src/tests/integration/` -- Integration test patterns
- `src/tests/kernels/` -- Kernel test patterns

---

## Quick Links

- [Code contributions](../contributing/code-contributions.md)
- [Kernel contributions](../kernel-contributions.md)
- [Documentation](../docs/contributing/doc-contributions.md)
- [Tools](../docs/contributing/tool-contributions.md)
