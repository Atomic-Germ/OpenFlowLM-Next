<p align="center">
  <img width="64" height="64" alt="OFLM_small" src="https://github.com/user-attachments/assets/36d563a8-034e-49db-a0ce-af920420e40f" />
</p>
<p align="center">
  <img src="https://img.shields.io/badge/NPU-Optimized-red" />
</p>

## OpenFlowLM -- open NPU kernels for Ryzen™ AI

The open, NPU-first runtime for **AMD Ryzen™ AI** (XDNA2: Strix, Strix Halo,
Kraken and Gorgon Point) -- the open replacement for FastFlowLM, built from
source in this repository.

📦 **The only out-of-box, NPU-first runtime built exclusively for Ryzen™ AI.**  
🤝 **A familiar single-command CLI -- deeply optimized for NPUs.**  
✨ **From Idle Silicon to Instant Power -- OpenFlowLM Makes Ryzen™ AI Shine.**

Run language models, embedding models, MoE models, speech-to-text and
**diffusion** on **AMD Ryzen™ AI NPUs** -- no GPU required. Any model from
Hugging Face in a compatible format can be installed with one command and run
with one command:

```
oflm add mradermacher/gemma-4-12B-it-i1-GGUF
oflm run gemma-4-it:12b
```

> Supports Ryzen™ AI chips with XDNA2 NPUs (Strix, Strix Halo, Kraken and
> Gorgon Point).

---

## Why this exists

[FastFlowLM](https://github.com/ROCm/FastFlowLM) began as open source. AMD then
acquired it, and everything AIE went closed -- the NPU kernels, the `.xclbin`
binaries, and the `model.q4nx` container were stripped from the repository.
What remained was a runtime whose own issues thread documents why that was a
problem:

- [#246 — Adding own models](https://github.com/ROCm/FastFlowLM/issues/246):
  a user asked for a way to install their own GGUF model, and AMD's answer was
  *kernels are model-specific; we offer a paid conversion service.* The thread
  and the model-request issues are full of people blocked on exactly that.
- [#828 — Status of the official gguf → q4nx conversion tool announced in
  v0.9.26](https://github.com/ROCm/FastFlowLM/issues/828): FastFlowLM announced
  the tool in January 2026, never shipped it, and pointed a paying user at a
  closed-source environment instead.
- [#219 — Fine-tuned model support?](https://github.com/ROCm/FastFlowLM/issues/219):
  a user asks for LoRA fine-tunes on the NPU; no supported answer existed.
- [#690 — Let the open source community support you](https://github.com/ROCm/FastFlowLM/issues/690):
  the direct plea to stop, and the history of how it broke. This repo is the
  answer to it (and [#712](https://github.com/ROCm/FastFlowLM/issues/712) shows
  a model running on the open kernels after the closed path hit a 42-layer
  dispatch wall).
- [#502 — Support for coder model](https://github.com/ROCm/FastFlowLM/issues/502)
  shows how the model-support gap played out: users were handed closed
  alternatives one request at a time, per model.

This project is the other path: kernels that are **not** model-specific, a
container anybody can inspect, and no paid conversion step. The interesting
findings along the way:

- **`model.q4nx` is a rearranged Q4_1**, not a new format -- worked out by
  reading the file, and that's what
  [`q4nx-build`](https://github.com/Atomic-Germ/q4nx-build) is built on.
- **Kernels are architecture-specific, not model-specific** -- a shared recipe
  derives a schedule from the model's own geometry, which is what
  [`utilities/oflm-add`](utilities/oflm-add) builds on (one tuned recipe per
  family, one installed kernel set per geometry).
- **Any compatible model from Hugging Face can run.** `oflm add <repo>` picks
  the right quant, derives the family and geometry from the file itself, links
  a kernel set, and writes the tag. No allow-list, no container conversion,
  no manual steps.

The goal is the boringly simple one: `oflm add user/repo` → `oflm run tag`,
the way llama.cpp works, except on the NPU -- with diffusion (FLUX.2 klein),
speech-to-text (Whisper), embeddings and chat all living in the same tool.

OpenFlowLM is built and maintained by the community — primarily
[Atomic-Germ](https://github.com/Atomic-Germ) (this fork and the `oflm add`
toolchain), with major contributions from
[vegah](https://github.com/vegah) (the openGranite and Gemma 4 kernels, and
the GGUF-direct work) and [Cyronius](https://github.com/Cyronius) (kernel
review and fixes across the board). If you are the FastFlowLM maintainer and
want to land this work back upstream, the invite stands — it is
[the outcome
this fork was made for](https://github.com/ROCm/FastFlowLM/issues/690).

---

## What is different from upstream

- **Open kernels.** `open_kernels/` holds the AIE designs the engine
  dispatches -- source, not pre-compiled binaries. Ten model families run on a
  shared recipe that works each model's shape out of its own geometry:
  Qwen3.6-MoE, Qwen3.5, Qwen3, Qwen2.5, Llama 3, Gemma 3, Phi-3/4, LFM2,
  HunYuan and Granite. Gemma 4's text decoder is here too: the 12B, and the
  E2B / E4B from Google's GGUF.
- **`oflm add` -- install any compatible model in one command.** The weights
  are the authority: the tool derives the family and geometry from the file
  itself (header, not filename), links a kernel set built for that geometry,
  writes the tag, and gets out of the way. No allow-list, no paid conversion
  step, no lockfile.
- **GGUF-direct.** A plain GGUF repository works out of the box -- quant
  picked for compatibility first (`Q4_1`), others converted on the way in.
- **More than chat.** Speech-to-text (Whisper), embeddings and reranking
  (six BERT-family encoders through [`src/open_npue/`](src/open_npue/)), and
  FLUX.2 klein diffusion -- all in the same runtime and CLI, not bolt-ons.
- **No `model_list.json` requirement.** The shipped list is a display offer;
  it is not a gate. `oflm add` installs per the geometry the model declares,
  and anything else the upstream tool can't describe stays refused by name.
- **Built from source.** Use CMake presets for building; see
  [docs/BUILD.md](docs/BUILD.md).

Not every model runs on open kernels -- see
[the support-status matrix](docs/docs/models/index.md) for exactly what is open
today.

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

3. **Install and run any compatible model:**

   ```powershell
   oflm add mradermacher/gemma-4-12B-it-i1-GGUF   # or any fine-tune on the hub
   oflm run gemma-it:12b
   ```

   `oflm add` points the tag at a kernel set for the model's own geometry —
   no conversion, no registration with anyone else's list. Or serve an
   OpenAI-compatible API:

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
- The submodules (`git clone --recursive`, or `git submodule update --init
  --recursive` in an existing checkout)
- On Linux, the development packages listed for Fedora and Ubuntu in the
  [Linux getting-started guide](docs/linux-getting-started.md#before-the-first-build)

The full Linux build also compiles the open NPU kernel xclbins -- the `open_kernels`
families (Qwen3.6-MoE, Qwen3.5/3 dense, Qwen2.5, LFM2 hybrid, Llama 3.1, Gemma 3
4B/12B, HunYuan, Granite 4.2, Phi-4-mini, MiniCPM5) and the
`open_npue` BERT embedding design sets -- which needs:

- **XRT** installed on the host (the AMD NPU runtime; `/opt/xilinx/xrt`, or a distro
  package under `/usr`), including its Python binding `pyxrt`. `pyxrt` is built for
  one Python version, so the kernel toolchain venv is made for the version the build
  runs under (3.11 to 3.14).
- The kernel toolchain (`ironvenv` with `mlir-aie` + Peano) -- **created
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
