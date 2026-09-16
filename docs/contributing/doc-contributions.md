# Documentation Contributions

> "The code that matters." -- Keep it simple, clear, and focused.

This document covers contributing documentation to OpenFlowLM.

---

## Documentation Structure

OpenFlowLM uses a **modular Jekyll site** with Markdown files. The structure is:

```
docs/
├── README.md              # Main project README
├── BUILD.md              # CMake build documentation
├── linux-getting-started.md  # User-facing install guide
├── models.md             # Model documentation index
├── benchmarks.md         # Benchmark results
├── docs.md                # Docs hub/landing page
├── index.md              # Main landing page
├── docs/                 # Technical documentation
│   ├── index.md          # Docs landing page
│   ├── models/           # Model-specific docs
│   ├── instructions/     # Usage guides
│   └── benchmarks/       # Benchmark details
└── ...more docs
```

**Key principles:**
- **Simple structure** -- Markdown files, not nested directories
- **Consistent front matter** -- `layout: page` or `layout: docs`
- **Shared layouts** -- Use existing layouts for consistency

---

## Before You Write

- **Find existing docs** to reference or update
- **Update related code** before writing docs (if applicable)
- **Check for duplicates** -- similar content may already exist
- **Start with a draft** -- propose changes before committing

---

## Document Types

### User-Facing Guides (`docs/` root)

**Who reads these:** End users installing and using OpenFlowLM

**Examples:**
- `README.md` -- Project overview and quick start
- `linux-getting-started.md` -- Installation guide
- `docs/linux-getting-started.md` -- Linux-specific setup
- `docs/docs/install_lin.md` -- Linux installation details

**Format:**
```yaml
---
layout: page
title: "Linux Installation"
permalink: /docs/install/linux/
description: "Install OpenFlowLM on Linux"
---
```

### Technical Docs (`docs/docs/`)

**Who reads these:** Developers, contributors, power users

**Examples:**
- `docs/docs/index.md` -- Docs hub
- `docs/docs/models/gemma.md` -- Gemma model documentation
- `docs/docs/instructions/cli.md` -- CLI usage guide

**Format:**
```yaml
---
layout: docs
title: "Gemma Model"
---
```

### Kernel Build Docs (`kernel-contributions.md`)

**Who reads these:** Kernel contributors and maintainers

**Examples:**
- `kernel-contributions.md` -- Kernel build guide
- `docs/plans/open_xclbin_plan.md` -- Kernel roadmap

**Format:** Markdown, no front matter required

---

## Writing Documentation

### Best Practices

1. **Be specific** -- Explain what, not just how
2. **Use examples** -- Show code, not just describe
3. **Link related content** -- Reference other docs when relevant
4. **Keep it concise** -- One concept per section
5. **Update with code** -- When code changes, update docs

### Document Structure

**User guides:**
```markdown
# Title

**What it is** -- Brief description
**Prerequisites** -- Requirements list
**Installation** -- Step-by-step guide
**Verification** -- How to confirm it worked
```

**Technical docs:**
```markdown
# Title

**Overview** -- What this does
**Usage** -- How to use it
**Examples** -- Code or CLI examples
**Notes** -- Important considerations
```

### Writing Style

- **Imperative mood** -- Use commands, not "use" or "remember to"
- **Simple language** -- Avoid jargon when possible
- **Clear headings** -- Use `#`, `##`, `###` for structure
- **Code blocks** -- Use ``` for code examples
- **Tables** -- Use Markdown tables for comparisons
- **Links** -- Link to related docs, not raw URLs

### Code Examples

```bash
# Install OpenFlowLM
oflm pull llama3.2:3b
oflm run llama3.2:3b --ctx-len 131072
```

```cpp
// Read model config
auto config = ModelConfig::from_file("config.json");
```

---

## Documenting Kernel Builds

### When to Write Docs

Write kernel build docs when:
- Adding a new kernel family
- Changing build procedures
- Documenting known issues
- Explaining kernel recipes

### Kernel Doc Structure

```markdown
# Kernel Family: <name>

**What it is** -- Brief description
**Requirements** -- Build requirements
**Build steps** -- How to build
**Verification** -- How to verify it works
**Known issues** -- Limitations and workarounds
```

### Linking Kernel Docs

```markdown
# Qwen3.6-MoE Kernels

The Qwen3.6-MoE model uses the dense recipe with MoE dispatch.

## Build Requirements

- XRT 32.0.203.311+
- ironvenv with mlir-aie
- NPU present

## Build Steps

```bash
python open_kernels/export_qwen36_kernels.py --spec qwen36-moe
```

## Verification

```bash
oflm list
```

The kernel set should report `open_qwen36: kernels ...`.
```

---

## Updating Existing Docs

When you change code:
1. **Find related docs** -- Search for references to the changed feature
2. **Update the docs** -- Ensure they match the new behavior
3. **Add notes** -- Document any new behavior or changes

**Examples:**
- Code adds a new command → Update CLI documentation
- API changes → Update developer docs
- Performance improvements → Update benchmarks section

---

## Testing Documentation

### Checklist Before Submitting

- [ ] Docs compile and render correctly
- [ ] Code examples are accurate
- [ ] Links work (internal and external)
- [ ] Formatting is consistent
- [ ] Front matter is correct
- [ ] Related code is up-to-date

### Testing Docs

```bash
# Build and preview docs
jekyll build

# Check for errors
jekyll build --verbose

# Render specific page
jekyll build --source docs --destination _site --config_file _config.yml
```

---

## Documenting Issues

When fixing bugs or adding features:
1. **Reference the issue** in your PR description
2. **Add docs if needed** -- document the fix or feature
3. **Update examples** -- if the change affects examples

**Example:**
```markdown
## Issue #123: Memory leak in attention path

This PR fixes the memory leak in the attention path.

## Documentation

Updated `docs/contributing/code-contributions.md` to explain
the memory leak and the fix.
```

---

## Contributing to Model Docs

### Model Documentation Structure

Each model should have:
1. **Overview** -- What the model is and what it does
2. **Performance** -- Benchmarks and expected performance
3. **Usage** -- How to run the model
4. **Limitations** -- Known issues and workarounds

### Adding New Model Docs

1. **Create the doc file** -- `docs/docs/models/<model>.md`
2. **Copy patterns from existing docs** -- Look at similar models
3. **Add benchmarks** -- Include in `docs/benchmarks/`
4. **Link from models index** -- Update `docs/models.md`

---

## Common Document Patterns

### Installation Guide

```markdown
# Linux Installation

**What it is** -- Install OpenFlowLM on Linux

**Prerequisites**
- Linux (Ubuntu 22.04+ recommended)
- XRT 32.0.203.311+
- Python 3.11

**Installation**
```bash
git clone ...
cmake --preset linux-default
cmake --build build
cmake --install build
```

**Verification**
```bash
oflm list
oflm run llama3.2:3b
```
```

### API Reference

```markdown
# CLI Reference

## Commands

### oflm run

Run a model.

**Usage**
```bash
oflm run <model> [options]
```

**Options**
- `--ctx-len <int>` -- Context length
- `--log-level <level>` -- Logging level
```

### oflm serve

Serve as OpenAI-compatible API.

**Usage**
```bash
oflm serve <model>
```
```

---

## Getting Help

- **Open an issue** before starting major documentation changes
- **Join Discord** for real-time help
- **Ask questions** on PRs or in issues

**Documentation resources:**
- Existing docs -- Review before writing
- Code -- Read the code you're documenting
- PRs -- See how others have documented features

---

## Quick Links

- [Code contributions](../contributing/code-contributions.md)
- [Kernel contributions](../kernel-contributions.md)
- [Testing](../docs/contributing/test-contributions.md)
- [Tools](../docs/contributing/tool-contributions.md)
