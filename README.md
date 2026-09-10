<p align="center">
  <img src="https://img.shields.io/badge/NPU-Optimized-red" />
</p>

## OpenFlowLM — open NPU kernels for Ryzen™ AI

A community fork of [FastFlowLM](https://github.com/ROCm/FastFlowLM) that
replaces the closed NPU kernels with open ones, built from source in this
repository.

Run LLMs, embedding models and MoE models on **AMD Ryzen™ AI NPUs** — no GPU
required.

> Supports Ryzen™ AI chips with XDNA2 NPUs (Strix, Strix Halo, Kraken and
> Gorgon Point).

---

## What is different from upstream

- **Open kernels.** `open_kernels/` holds the AIE designs the engine
  dispatches — source, not pre-compiled binaries. Seven model families run on a
  shared recipe that works each model's shape out of its own `config.json`.
- **A second embedding backend.** Six encoder models beyond the one upstream
  ships, through [`src/open_npue/`](src/open_npue/).
- **GGUF and Q4_K containers**, so models are not confined to one weight format.
- **Built from source.** There is no packaged installer here; see
  [docs/BUILD.md](docs/BUILD.md).

Upstream remains the place to go for a turnkey install and for the closed,
tuned kernels.

---

## Getting started

1. **The NPU driver** — use **32.0.203.311 or above** (Task Manager →
   Performance → NPU, or Device Manager). Earlier versions are not supported.
   Windows Update or [AMD's driver download](https://www.amd.com/en/support) is
   the recommended route; the
   [official install doc](https://ryzenai.docs.amd.com/en/latest/inst.html#install-npu-drivers)
   has the details.

2. **Build it** — [docs/BUILD.md](docs/BUILD.md). There are two things to
   build: the executable, and the AIE design sets the NPU actually runs. The
   design sets are not checked in, and without them the binary starts and then
   refuses to load a model.

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

- **Runs on the NPU** — not the GPU, and not as CPU fallback
- **Open kernel path** — the designs are here, and rebuilding them is a
  documented step rather than a vendor drop
- **Long context** — up to 256k tokens on models that support it
- **Familiar CLI** — `run`, `serve`, `list`, `bench`

---

## License

- Orchestration code and CLI tools are open source under the
  [MIT License](./LICENSE_RUNTIME.txt).
- The open AIE kernels in `open_kernels/` are part of this repository and carry
  its licence.
- Any closed binary kernels retained from upstream remain FastFlowLM's, under
  the terms upstream sets, and are not redistributed by this repository.

If you build on this work, an acknowledgement of both projects is appreciated:

```
Powered by OpenFlowLM, a fork of FastFlowLM
```

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

💬 [Open an issue](https://github.com/Atomic-Germ/OpenFlowLM-Next/issues/new)
