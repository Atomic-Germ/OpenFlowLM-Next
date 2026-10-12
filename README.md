<p align="center">
  <img width="64" height="64" alt="OFLM_small" src="https://github.com/user-attachments/assets/36d563a8-034e-49db-a0ce-af920420e40f" />
</p>
<p align="center">
  <img src="https://img.shields.io/badge/NPU-Optimized-red" />
</p>

## OpenFlowLM -- open NPU kernels for Ryzen™ AI

Open NPU kernels for Ryzen AI, built from source in this repository, out of the
[FastFlowLM](https://github.com/ROCm/FastFlowLM) codebase.

**The point: a fine-tune nobody has ever heard of runs with zero new code.**
Resolve it, verify it, bind its kernels. You do not write kernels for it, and you
do not add its name to a list.

```bash
oflm pack -i Someone/Their-Model-NPU2    # work out what it is; build the container
oflm add  Someone/Their-Model-NPU2       # verify it, bind kernels, register it
```

Both are commands on the `oflm` binary itself, each handing off to the builder or
installer bundled beside it. Three tools carry this project, and they do
considerably more than their names suggest:

- **`oflm pack`** (the builder) is not just a converter. It identifies a GGUF by
  fingerprinting its actual tensor names when the metadata is absent or wrong,
  follows the model card's `base_model` chain to find the right skeleton assets, and
  has a converter per architecture. It is what makes an arbitrary model buildable.
- **`oflm add`** is not just an installer — `oflm pull` is that. It handles models
  nobody has registered: it picks between the converted and GGUF paths, checks the
  weights against what their filename claims, derives the tag/family/size, and
  binds kernels **by shape rather than by name**.
- **[`oflm-test`](utilities/oflm-test)** is the acceptance gate: `--llm`,
  `--vision`, `--embed`, `--tools`.

Kernel sets are per-*family* and matched by *geometry*, so every fine-tune and
derivative of a supported architecture works today, including ones we have never
seen. Work is under way to collapse the two commands into one: a llama.cpp-style
GGUF can now be installed directly, with no conversion step.

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
  dispatches -- source, not pre-compiled binaries. Families run on a shared
  recipe that works each model's shape out of its own `config.json` or GGUF
  metadata: Qwen3.6-MoE, Qwen3.5/3.8, Qwen3, Qwen2.5, Llama 3, Gemma 3, Phi-3/4,
  LFM2, HunYuan, Granite and K2.
- **Fine-tunes are first-class.** A model of a supported shape needs no new code,
  no registry entry and no new kernels. `oflm pack` builds it; `oflm add`
  matches its shape to an installed family kernel set.
- **A second embedding backend.** Six encoder models beyond the one upstream
  ships, through [`src/open_npue/`](src/open_npue/).
- **GGUF and Q4_K containers**, so models are not confined to one weight format.
- **Built from source.** Use CMake presets; see [docs/BUILD.md](docs/BUILD.md).

Not every *shape* has open kernels yet. Vision, audio and the Gemma 4, GPT-OSS
and Whisper families have no open design and fall back to the closed engine
binaries, which are legacy and being replaced. If your fine-tune's architecture is
unsupported, that is a **kernel gap to close**, not a model to exclude -- and it is
still two commands once it exists. See
[the support-status matrix](docs/docs/models/index.md) for what is open today.

Adding a model? Read [`src/create_new_model.md`](src/create_new_model.md). Adding
a kernel? Read [`AGENTS.md`](AGENTS.md) first -- it is the shortest accurate
description of how this project works.

---

## Getting started

1. **The NPU driver** -- use **32.0.203.304 or above** (Task Manager →
    Performance → NPU, or Device Manager). Earlier versions are not supported.
    Windows Update or [AMD's driver download](https://www.amd.com/en/support) is
    the recommended route; the
    [official install doc](https://ryzenai.docs.amd.com/en/latest/inst.html#install-npu-drivers)
    has the details.

2. **Build it** -- [docs/BUILD.md](docs/BUILD.md). The build system handles
    both the executable and kernel exports via CMake presets.

3. **Run a model:**

   ```powershell
   oflm run llama3.2:1b
   ```

   or serve an OpenAI-compatible API:

   ```powershell
   oflm serve
   ```

4. **Run your own model.** Convert it, then install it:

   ```bash
   oflm pack -i Someone/Their-Model-NPU2
   oflm add  Someone/Their-Model-NPU2
   oflm run their-model:9b
   ```

   `--dry-run` on `oflm pack` prints the plan without converting anything.

🐧 [Linux getting-started guide](./docs/linux-getting-started.md)

---

## Highlights

- **Any fine-tune of a supported shape, no code** -- convert and install
- **Runs on the NPU** -- not the GPU, and not as CPU fallback
- **Open kernel path** -- the designs are here, built via CMake presets
- **Long context** -- up to 256k tokens on models that support it
- **Familiar CLI** -- `run`, `serve`, `list`, `bench`, `add`

---

## License

- Orchestration code and CLI tools are open source under the
  [MIT License](./LICENSE_OPEN_RUNTIME.md).
- The open AIE kernels in `open_kernels/` are part of this repository and carry
  its licence.
- Any closed binary kernels retained from upstream remain FastFlowLM's, under
  the terms upstream sets, and are not redistributed by this repository.
- Please acknowledge the upstream FastFlowLM and OpenFlowLM in your README/project page.
  
---

💬 Have **feedback/issues** or want **early access** to our new releases? [Open an issue](https://github.com/Atomic-Germ/OpenFlowLM/issues/new) or [Join our Discord community]([https://discord.gg/8RexzmW7](https://discord.gg/AcFAKvVV8c)))

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
```

Or do all of it, in the host-native and portable package formats, with one command:

```bash
cmake --workflow --preset linux-package   # configures, builds, tests, packages RPM+TGZ (+ kernels)
```

The install lands in `/opt/openflowlm` and adds `/usr/bin/oflm` plus
`/etc/profile.d/openflowlm.sh`, so `oflm` is on `PATH` with no shell-rc edits.

For detailed instructions, see [docs/BUILD.md](docs/BUILD.md).

### Prerequisites

- Git
- CMake (version 3.25 or higher)
- A C++20 compatible compiler (e.g., GCC, Clang, MSVC)
- Ninja (recommended)

The full Linux build also compiles the open NPU kernel xclbins -- the `open_kernels`
families (Qwen3.6-MoE, Qwen3.5/3 dense, Qwen2.5, LFM2 hybrid, Llama 3.1, Gemma 3
4B/12B, HunYuan, Granite 4.2, Phi-4-mini, MiniCPM5) and the
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

On Windows the engine builds, and so do the NPU kernels, natively -- see
[docs/BUILD.md](docs/BUILD.md#2-the-npu-kernels). Only the one-shot
`utilities/export-kernels.py` driver (and the CMake target that calls it) is
Linux-only. The language-model kernels are compile-only; the BERT embedding sets need
an NPU on the build host.

### Build Instructions

See [docs/BUILD.md](docs/BUILD.md) for detailed build instructions.

**Presets:**
- **Linux full distribution:** `cmake --preset linux-default`
- **Linux full distribution (Fedora):** `cmake --preset fedora-default`
- **Linux debug (engine only):** `cmake --preset linux-debug`
- **Linux debug (Fedora, engine only):** `cmake --preset fedora-debug`
- **Linux portable:** `cmake --preset linux-portable`
- **Windows:** `cmake --preset windows-default`

**Build specific kernels:**
```bash
cmake --preset linux-default -DOFLM_KERNEL_SPECS=qwen3-4b
```
#### Docker build

Also you can build fast and simple with Docker.

Simply run(linux-package-deb preset is used by default):

```bash
./build_in_docker.sh
```

Or specify a workflow preset(for tar.gz package in this example):

```bash
./build_in_docker.sh linux-package
```

The script builds `openflowlm-build:ubuntu26`, passes `/dev/accel/accel0` into the container, enables XRT memory locking, and persists the build and NPU cache directories.

The container toolchain lives in `/opt/ironvenv` (`OFLM_VENV_DIR`), outside
the repository mounted at `/code`. This prevents a host `ironvenv` from hiding
the image's tested Python, mlir-aie and Peano environment. Rebuild the builder
image after changing the Dockerfile; the build script does this automatically.

The host must have the AMD XDNA driver installed and expose:

```text
/dev/accel/accel0
```

#### Run with Docker Compose

On a Linux host with Docker Compose and the AMD XDNA driver, first build the
Ubuntu DEB (including the kernel exports), then start the API:

```bash
./build_in_docker.sh
docker compose up -d --build
curl --fail http://127.0.0.1:52625/api/version
```
