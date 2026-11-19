---
title: "Contributing"
description: "Branch conventions, build workflows, and contributor guides for the OpenFlowLM runtime and NPU kernels."
icon: "verified"
weight: 50
---

OpenFlowLM is an open, MIT-licensed project. We welcome pull requests across documentation, benchmarks, C++ runtime code, Python tools, and open MLIR-AIE / IRON NPU kernels.

## Contributor Guides

| Area | Guide | Scope |
|---|---|---|
| Build & Packaging | [Build System](/docs/contributing/build/) | Root vs. `src/` CMake presets, Docker, and kernel export |
| C++ & Rust Runtime | [Code Contributions](/docs/contributing/code-contributions/) | Engine modules, code style, and `check-settings-ci.sh` |
| Open NPU Kernels | [Kernel Contributions](/docs/contributing/kernel-contributions/) | Geometry specs, recipes, hardware verification, and `manifest.json` |
| Tests & Conformance | [Test Contributions](/docs/contributing/test-contributions/) | CTest targets and live `oflm-test` modality suites |
| Python Utilities | [Tool Contributions](/docs/contributing/tool-contributions/) | `oflm-add`, `oflm-test`, `env.sh`, and registry generators |
| Docs & Benchmarks | [Documentation Contributions](/docs/contributing/doc-contributions/) | Hugo Lotus Docs structure, alerts, and benchmark verifiers |
| Releases | [Semantic Versioning](/docs/contributing/semantic-versioning/) | SemVer rules and `OFLM_VERSION` bumps |

---

## Branch & Commit Conventions

<div class="oflm-two-col">
  <div class="oflm-card">

### Branch Names (`<type>/<slug>`)

Use lowercase hyphenated names under 50 characters:

- `feat/`, `fix/` (or `bugfix/`), `perf/`
- `docs/`, `build/`, `refactor/`, `chore/`, `test/`
- Examples: `feat/llama-support`, `perf/decode-3b`

  </div>
  <div class="oflm-card">

### Commit Messages

Use imperative mood with optional rationale in the body:

```text
feat: add Llama 3.1 8B model support

Add narrowing recipe and standard attention path.
```

  </div>
</div>

---

## Pull Request Workflow

1. **Branch from `main`** in your fork of [`Atomic-Germ/OpenFlowLM-Next`](https://github.com/Atomic-Germ/OpenFlowLM-Next):

    ```bash
    git checkout main && git pull origin main
    git checkout -b feat/your-feature-name
    ```

2. **Stage explicit paths only:** Run `git add <file>`, never `git add -A` or `git add .` (`src/models/` and `build/` hold gigabytes of local weights and build artifacts).
3. **Build, test, and verify settings** from the repository root:

    ```bash
    cmake --preset linux-default
    cmake --build --preset linux-default
    ctest --preset linux-default --output-on-failure
    utilities/check-settings-ci.sh
    ```

    > [!NOTE]
    > GitHub Actions CI configures `linux-debug` (`OFLM_BUILD_KERNELS=OFF`) without an NPU attached. Verify kernel changes locally on XDNA2 hardware with `linux-default`.

4. **Update docs & tests** alongside behavior changes ([Getting Started](/docs/install/), [User Guide](/docs/instructions/), `src/test/`, or `utilities/oflm-test/`), then open a pull request on [GitHub](https://github.com/Atomic-Germ/OpenFlowLM-Next/pulls).

## Community & License

- **Issues & Discord:** Open an issue on [GitHub Issues](https://github.com/Atomic-Germ/OpenFlowLM-Next/issues) before large architectural changes, or join [Discord](https://discord.gg/AcFAKvVV8c).
- **License:** OpenFlowLM and its open MLIR-AIE / IRON kernels are MIT licensed ([`LICENSE`](https://github.com/Atomic-Germ/OpenFlowLM-Next/blob/main/LICENSE), [`LICENSE_OPEN_RUNTIME.md`](https://github.com/Atomic-Germ/OpenFlowLM-Next/blob/main/LICENSE_OPEN_RUNTIME.md)). Legacy third-party artifacts keep their original redistribution terms.
