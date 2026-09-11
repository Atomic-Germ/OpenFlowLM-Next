<p align="center">
  <img src="https://img.shields.io/badge/NPU-Optimized-red" />
</p>

## OpenFlowLM -- open NPU kernels for Ryzen™ AI

A community fork of [FastFlowLM](https://github.com/ROCm/FastFlowLM) that
replaces the closed NPU kernels with open ones, built from source in this
repository.

📦 **The only out-of-box, NPU-first runtime built exclusively for Ryzen™ AI.**  
🤝 **A familiar single-command CLI -- deeply optimized for NPUs.**  
✨ **From Idle Silicon to Instant Power -- OpenFlowLM Makes Ryzen™ AI Shine.**

> OpenFlowLM (OFLM) supports all Ryzen™ AI Series chips with XDNA2 NPUs (Strix, Strix Halo, Kraken, and Gorgon Point).
Run LLMs, embedding models and MoE models on **AMD Ryzen™ AI NPUs** -- no GPU
required.

> Supports Ryzen™ AI chips with XDNA2 NPUs (Strix, Strix Halo, Kraken and
> Gorgon Point).

---

## What is different from upstream

- **Open kernels.** `open_kernels/` holds the AIE designs the engine
  dispatches -- source, not pre-compiled binaries. Seven model families run on a
  shared recipe that works each model's shape out of its own `config.json`.
- **A second embedding backend.** Six encoder models beyond the one upstream
  ships, through [`src/open_npue/`](src/open_npue/).
- **GGUF and Q4_K containers**, so models are not confined to one weight format.
- **Built from source.** Use CMake presets for building; see
  [docs/BUILD.md](docs/BUILD.md).

Upstream remains the place to go for a turnkey install and for the closed,
tuned kernels.

---

## Getting started

1. **The NPU driver** -- use **32.0.203.311 or above** (Task Manager →
    Performance → NPU, or Device Manager). Earlier versions are not supported.
    Windows Update or [AMD's driver download](https://www.amd.com/en/support) is
    the recommended route; the
    [official install doc](https://ryzenai.docs.amd.com/en/latest/inst.html#install-npu-drivers)
    has the details.

2. **Build it** -- [docs/BUILD.md](docs/BUILD.md). The build system handles
    both the executable and kernel exports via CMake presets.

3. **Run it:**

   ```powershell
   oflm run llama3.2:1b
   ```

   or serve an OpenAI-compatible API:

   ```powershell
   oflm serve
   ```

🐧 [Linux getting-started guide](./docs/linux-getting-started.md)

---

## Highlights

- **Runs on the NPU** -- not the GPU, and not as CPU fallback
- **Open kernel path** -- the designs are here, built via CMake presets
- **Long context** -- up to 256k tokens on models that support it
- **Familiar CLI** -- `run`, `serve`, `list`, `bench`

---

## License

- Orchestration code and CLI tools are open source under the
  [MIT License](./LICENSE_RUNTIME.txt).
- The open AIE kernels in `open_kernels/` are part of this repository and carry
  its licence.
- Any closed binary kernels retained from upstream remain FastFlowLM's, under
  the terms upstream sets, and are not redistributed by this repository.

- All orchestration code and CLI tools are open-source under the [MIT License](./LICENSE_RUNTIME.txt).  
- These NPU-accelerated binary kernels are completely free for any use, including commercial use.
- Please acknowledge the upstream FastFlowLM and OpenFlowLM in your README/project page.
  
---

💬 Have **feedback/issues** or want **early access** to our new releases? [Open an issue](https://github.com/Atomic-Germ/OpenFlowLM/issues/new) or [Join our Discord community](https://discord.gg/z24t23HsHF)

---

## Acknowledgements

- Forked from [FastFlowLM](https://github.com/ROCm/FastFlowLM)
- Powered by the **AMD Ryzen™ AI NPU** architecture
- Inspired by [llama.cpp](https://github.com/ggml-org/llama.cpp) and
  [Ollama](https://github.com/ollama/ollama)
- Tokenization via [MLC-ai/tokenizers-cpp](https://github.com/mlc-ai/tokenizers-cpp)
- Chat formatting via [Google/minja](https://github.com/google/minja)
- Kernels written with [IRON](https://github.com/amd/iron) +
  [MLIR-AIE](https://github.com/Xilinx/mlir-aie)

---

## 🛠️ Building from Source

OpenFlowLM uses a unified CMake build system. From a clean recursive clone, configure, build, test, install, and package with preset-based commands from the **repository root**:

```bash
cmake --preset linux-default
cmake --build --preset linux-default
cmake --test --preset linux-default
cmake --install --preset linux-default
cpack --preset linux-package-tgz
```

For detailed instructions, see [docs/BUILD.md](docs/BUILD.md).

### Prerequisites

- Git
- CMake (version 3.25 or higher)
- A C++20 compatible compiler (e.g., GCC, Clang, MSVC)
- Ninja (recommended)

The full Linux build also compiles the open NPU kernel xclbins -- the `open_kernels`
families (Qwen3.6-MoE, Qwen3.5/3 dense, Llama 3, Gemma 3, HunYuan, Granite) and the
`open_npue` BERT embedding design sets -- which needs:

- **XRT** installed on the host (the AMD NPU runtime; `/opt/xilinx/xrt`), including its
  Python binding `pyxrt`. The installed XRT ships `pyxrt` for Python 3.11, so the
  kernel toolchain venv is pinned to 3.11.
- The kernel toolchain (`ironvenv` with `mlir-aie` + Peano, Python 3.11) -- **created
  automatically** by the build if absent.
- `third_party/mlir-aie` -- **cloned automatically** by the build if absent (best-effort;
  only used for `toolchain.json` metadata).
- An **NPU present** on the build host: the BERT design sets allocate NPU tensors, so
  that part of the export runs on the device (the `open_kernels` families are
  compile-only).

On Windows the engine builds, but the NPU kernel export is Linux-only (it requires the XRT/Peano toolchain and the NPU).

### Build Instructions

See [docs/BUILD.md](docs/BUILD.md) for detailed build instructions.

**Presets:**
- **Linux full distribution:** `cmake --preset linux-default`
- **Linux debug (engine only):** `cmake --preset linux-debug`
- **Linux portable:** `cmake --preset linux-portable`
- **Windows:** `cmake --preset windows-default`

**Build specific kernels:**
```bash
cmake --preset linux-default -DOFLM_KERNEL_SPECS=qwen3-4b
```
