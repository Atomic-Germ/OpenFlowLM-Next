# Contributing to OpenFlowLM

> "The code that matters." -- Keep it simple, clear, and focused.

OpenFlowLM is a community-driven project. We welcome contributions from everyone, from first-time contributors to seasoned maintainers.

## Where to Start

| What you're interested in | Where to look |
|---|---|
| General code contributions | [docs/contributing/code-contributions.md](docs/contributing/code-contributions.md) |
| Building kernels (NPU xclbins) | [kernel-contributions.md](kernel-contributions.md) |
| Documentation | [docs/contributing/doc-contributions.md](docs/contributing/doc-contributions.md) |
| Tests | [docs/contributing/test-contributions.md](docs/contributing/test-contributions.md) |
| Tools (oflm-add, q4nx-build, etc.) | [docs/contributing/tool-contributions.md](docs/contributing/tool-contributions.md) |

---

## Branch Naming Convention

We use a **clean, clear** style that makes the purpose of each branch obvious at a glance:

```
<type>/<description>
```

**Types:**
- `feat/` -- New features
- `fix/` -- Bug fixes
- `perf/` -- Performance improvements
- `docs/` -- Documentation only
- `build/` -- Build system / CMake changes
- `refactor/` -- Code restructuring (no behavior change)
- `chore/` -- Maintenance, dependencies
- `test/` -- Test additions or modifications

**Rules:**
- Lowercase with hyphens between words
- Keep descriptions concise and clear
- Use lowercase hyphens (not underscores or camelCase)
- Keep under 50 characters when possible

**Examples:**
- `feat/llama-support` -- Add Llama model support
- `fix/attention-vision` -- Fix attention vision path
- `perf/decode-3b` -- Improve decode performance on 3B models
- `docs/install-linux` -- Linux installation guide
- `build/root-cmake` -- Root CMake configuration

---

## Commit Message Convention

Commit messages are a **suggestion**, not a strict requirement. However, we encourage the **imperative mood** with a clear subject and optional context:

```
<type>: <subject>

<optional context line>

<optional body explaining why, not how>
```

**Examples:**

```
feat: add Llama 3.1 8B model support

Add Llama 3.1 support to the open engine with its
narrowing recipe and standard attention path.
```

```
fix: attention vision path

Fix M-RoPE position handling in vision tower images
by propagating dispatch failures correctly.
```

```
perf: decode 3b

Remove per-position scalar float in attention to
improve decode time by 32.1x.
```

**Tip:** Use `feat/` in your branch name and `feat:` in your commit message for consistency.

---

## Code Contribution Process

1. **Fork the repo** and create a branch from main
   ```bash
   git checkout main
   git pull origin main
   git checkout -b feat/your-feature-name
   ```

2. **Make your changes** -- follow existing code style and patterns
   - Read the code you're modifying to understand the patterns
   - Use `git diff` frequently to see what you're changing
   - Keep changes focused -- one feature per branch

3. **Test your changes**
   - Run `cmake --build --preset linux-default` to build
   - Run `ctest --preset linux-default` to run tests
   - Test locally with `oflm run` or `oflm serve`

4. **Write tests** if you're adding new functionality
   - Add tests to `src/tests/`
   - Follow existing test patterns and naming

5. **Submit a pull request**
   - Push your branch to your fork
   - Create a PR on GitHub
   - Reference relevant issues in your PR description
   - Request a review from maintainers

**Branch lifecycle:**
- Your branch lives in your fork until merged
- After merging, keep your fork up-to-date: `git pull origin main; git push origin feat/your-feature-name`
- Don't forget to clean up: `git branch -D feat/your-feature-name`

---

## Documentation

Documentation should be **clear, concise, and accurate**. When you change code, update the docs too.

**Where to contribute:**
- User-facing guides → `docs/` root (e.g., `README.md`, `linux-getting-started.md`)
- Technical docs → `docs/docs/` (e.g., model docs, instructions)
- Kernel build docs → `kernel-contributions.md`

**Before writing docs:**
1. Find existing docs to reference
2. Update those docs if relevant
3. Link to related content from your docs

---

## Testing

Tests should be **fast, focused, and coverage-rich**.

**Writing tests:**
1. Add tests to `src/tests/`
2. Follow existing test patterns
3. Use fixtures for reusable setup
4. Write tests that fail clearly when broken

**Running tests:**
```bash
cmake --build --preset linux-default
ctest --preset linux-default
```

**Adding new tests:**
- Add to `src/tests/` for new features
- Fix existing tests to pass after your changes
- Consider adding integration tests for complex flows

---

## Tools

Contributions to tools (oflm-add, q4nx-build, etc.) follow the same process.

**Before contributing:**
1. Read the tool's README
2. Understand the existing implementation
3. Test locally before submitting

**After contributing:**
- Update the tool's README if needed
- Test the tool thoroughly
- Consider adding integration tests

---

## Getting Help

- **Open an issue** before starting major work
- **Join Discord** for real-time help
- **Ask questions** on PRs or in issues

---

## Community

OpenFlowLM thrives on contributions. We're happy to help you get started!

- Weekly office hours and demos
- Benchmarking nights
- Code reviews and feedback

**Want to help?**
- Review PRs
- Fix bugs you find
- Write tests
- Help new contributors
- Share your experience

---

## License

OpenFlowLM is released under the MIT License. See [LICENSE_RUNTIME.txt](./LICENSE_RUNTIME.txt) for details.

**For developers:**
- OpenFlowLM code is MIT licensed
- Upstream (FastFlowLM) kernels remain under their original license
- You can use OpenFlowLM for any purpose, including commercial use

---

## Quick Links

- [Branch naming convention](#branch-naming-convention)
- [Commit message convention](#commit-message-convention)
- [Code contribution process](#code-contribution-process)
- [Testing](#testing)
- [Tools](#tools)
- [Documentation](#documentation)
- [Getting help](#getting-help)
