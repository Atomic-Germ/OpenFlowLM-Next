---
layout: docs
title: System Command and CLI Mode
nav_order: 1
parent: Instructions
---

# ⚡ CLI Mode

OFLM CLI mode offers a familiar terminal-based interactive experience, fully offline and accelerated exclusively on AMD NPUs. Here are detailed descriptions of commands and setup for CLI mode usage. It includes:

- **[🔧 Pre-Run PowerShell Commands (System)](#-pre-run-powershell-commands)**
- **[💻 Commands Inside CLI Mode](#-commands-inside-cli-mode)**
- **[🗂️ Others](#️-others)**

---

## 🔧 Pre-Run PowerShell Commands (System)

### 🖥️ System Compatibility Check

Verify that your hardware meets the necessary requirements to run OpenFlowLM:

```shell
oflm validate
```

Output the validation results as a JSON object:

```shell
oflm validate --json
```

---

### 🆘 Show Help

```shell
oflm help
```

---

### 🚀 Run a Model

Run a model interactively from the terminal:

```shell
oflm run llama3.2:1b
```

> `oflm` is short for OpenFlowLM. If the model isn't available locally, it will be downloaded automatically. This launches OpenFlowLM in CLI mode.

> **Linux note:** `oflm validate` checks the kernel DRM device and then opens the NPU through XRT, the way `oflm run` does. If it reports that the device runtime cannot open the NPU (`runtime_ok: false` with `--json`), confirm XRT can see the NPU:
> ```shell
> xrt-smi examine
> ```
> On Arch Linux, install `xrt-plugin-amdxdna` in addition to `xrt` and the `amdxdna` driver. See the [Linux install guide](/docs/install_lin/) for the full driver and firmware checklist.

---

### ⬇️ Pull a Model (Download Only)

Download a model from HuggingFace without launching it:

```shell
oflm pull llama3.2:3b
```

This code forces a re-download of the model, overwriting the current version.

```shell
oflm pull llama3.2:3b --force
```

> ⚠️ Use `--force` **only if the model file is corrupted** (e.g., incomplete download). Proceed with caution.

#### 📁 Default Model Storage Location

| Platform | Default Path |
|----------|-------------|
| Windows  | `C:\Users\<USER>\.oflm\models` |
| Linux    | `~/.config/oflm/models` |

#### 🔧 Changing the Model Storage Path

You can override the default location by setting the `OFLM_MODEL_PATH` environment variable.

**Windows** -- Update the existing system environment variable:
1. Open **Start** and search for **"Edit the system environment variables"**.
2. Click **Environment Variables…**.
3. Under **System variables**, find `OFLM_MODEL_PATH`, select it, and click **Edit…**.
4. Update the value to your desired path (e.g., `D:\models\oflm`).
5. Click **OK** and restart any open terminals for the change to take effect.

**Linux** -- Set temporarily for the current shell session:
```shell
export OFLM_MODEL_PATH="/your/custom/path"
```

To make the change permanent, add the line above to your `~/.bashrc`, then reload it:
```shell
echo 'export OFLM_MODEL_PATH="/your/custom/path"' >> ~/.bashrc
source ~/.bashrc
```

---

### 🌍 Pull from ModelScope

Every download command accepts `--modelscope` to fetch from ModelScope instead
of HuggingFace:

```shell
oflm pull llama3.2:3b --modelscope
oflm check llama3.2:3b --modelscope
oflm run llama3.2:3b --modelscope
oflm bench-embed bge-base:en-v1.5 --modelscope
```

---

### 📦 List Supported and Downloaded Models

Display all available models and locally downloaded models:

```shell
oflm list
```

Output as JSON:

```shell
oflm list --json
```


Filters flag:

```shell
# Show everything
oflm list --filter all

# Only models already installed
oflm list --filter installed

# Only models not yet installed
oflm list --filter not-installed
```

Quiet mode:

```shell
# Default view (pretty, with icons)
oflm list

# Quiet view (no emoji / minimal)
oflm list --quiet

# Show everything
oflm list --filter all --quiet

# Only models already installed
oflm list --filter installed --quiet

# Only models not yet installed
oflm list --filter not-installed --quiet
```

---

### ❌ Remove a Downloaded Model

Delete a model from local storage:

```shell
oflm remove llama3.2:3b
```

### ✅ Check a Downloaded Model

Verify the file hashes for a downloaded model:

```shell
oflm check llama3.2:3b
```

---

### 🚀 Start Server Mode (Local)

Launch OpenFlowLM as a local REST API server (also supports the OpenAI API):

```shell
oflm serve llama3.2:1b
```

---

### 🔌 Show Server Port 

Show current OFLM port (default) in PowerShell:  
  
```shell
oflm port
```

---

### ⚡ NPU Power Mode

By default, **OFLM runs in `performance` NPU power mode**. You can switch to other NPU power modes (`default`, `powersaver`, `balanced`, `performance`, or `turbo`) using the `--pmode` flag:

For **CLI mode**:
```shell
oflm run gemma3:4b --pmode balanced
```

For **Server mode**:
```shell
oflm serve gemma3:4b --pmode balanced
```

---

### 📏 Set Context Length at Launch

The default context length for each model can be found [here](https://openflowlm.com/docs/models/).   

Set the context length with `--ctx-len` (or `-c`).  

In PowerShell, run:

For **CLI mode**:
```shell
oflm run llama3.2:1b --ctx-len 8192
```

For **Server mode**:
```shell
oflm serve llama3.2:1b --ctx-len 8192
```

> - Internally, OFLM enforces a minimum context length of 512. If you specify a smaller value, it will automatically be adjusted up to 512.
> - The value is otherwise used **as given**. It is *not* rounded to a power of 2, so `--ctx-len 8000` gives you 8000.
> - The same 512 minimum applies to `--prefill-chunk-len`.
> - A context longer than the model's `default_context_length` may not be
>   supported by every architecture; check the model's card in
>   [Models](/docs/models/) for its maximum.

---

### 🖧 Set Server Port at Launch

Set a custom port at launch:

  ```shell
  oflm serve llama3.2:1b --port 8000
  oflm serve llama3.2:1b -p 8000
  ```

> ⚠️ `--port` (`-p`) only affects the **current run**; it won’t change the default port.

---

### 🛠️ Set Host at Launch

Specify a custom host address when starting the server:

```powershell
oflm serve llama3.2:1b --host 127.0.0.1
```

⚠️ Note: --host applies only to the current session. It does not modify the default host configuration (default: `127.0.0.1`).

> ⚠️ **Changed:** `--host` is now refused by `run`, `pull`, `remove`, `check` and `bench-embed`, as `--port` and `--cors` already were. Those commands used to accept it and ignore it, so a script that passes `--host` to one of them now fails and must drop the flag. `bench`, `list`, `version`, `port` and `validate` refuse the serve-only options too; the one exception is `oflm port --port N`, which prints the port that value resolves to.

---

### 🌐 Cross-Origin Resource Sharing (CORS)

CORS lets browser apps hosted on a different origin call your OFLM server safely.

- Enable CORS

```shell
oflm serve --cors 1
```
- Disable CORS

```shell
oflm serve --cors 0
```

> ⚠️ **Default:** CORS is **enabled**.  
> 🔒 **Security tip:** Disable CORS (or restrict at your proxy) if your server is exposed beyond localhost (127.0.0.1).

---

### ⏸️ Preemption

Preemption allows high-priority tasks to interrupt ongoing NPU jobs, improving responsiveness for critical workloads. To enable preemption:

For **CLI mode**:
```shell
oflm run llama3.2:1b --preemption 1
```

For **Server mode**:
```shell
oflm serve llama3.2:1b --preemption 1
```

> ⚠️ Note: Preemption is for **engineering testing/optimization** only. It requires a special driver + toolkit and is **not for public use**.


---

### 🧩 Change Prefill Chunk Size at Launch

The `--prefill-chunk-len` flag controls how many tokens are processed per chunk during the prefill phase of inference (default: 4096).

For **CLI mode**:

```shell
oflm run llama3.2:1b --prefill-chunk-len 8192
```

For **Server mode**:

```shell
oflm serve llama3.2:1b --prefill-chunk-len 8192
```

---

### 🎙️ ASR (Automatic Speech Recognition)

**Requirement:** The ASR model (e.g., `whisper-v3:turbo`) must run **with an LLM loaded concurrently**. Enabling `--asr 1` starts Whisper in the background **while** your chosen LLM loads.

#### CLI mode
```shell
oflm run gemma3:4b --asr 1  # Load Whisper (whisper-v3:turbo) in the background and load the LLM (gemma3:4b) concurrently.
```

#### Server mode
```shell
oflm serve gemma3:4b --asr 1  # Background-load Whisper and initialize the LLM (gemma3:4b) concurrently.
```

Pick a different speech-to-text model with `--asrmodel` (default `whisper-v3:turbo`):

```shell
oflm serve gemma3:4b --asr 1 --asrmodel whisper-v3:turbo
```

> **Note:** ASR alone isn’t supported--an LLM must be present for end-to-end voice→text→LLM workflows.

---

### 🔢 Serving an Embedding Model

`--embed 1` starts an encoder alongside the LLM for `/v1/embeddings`. Pick the
encoder with `--embeddingmodel` (default `embed-gemma:300m`):

```shell
oflm serve llama3.2:1b --embed 1 --embeddingmodel bge-base:en-v1.5
```

> ⚠️ `--embed` (`embed-gemma:300m`) is **server-only**. `oflm run` it drops the
> embedding and advices using `oflm serve -e 1`.
>
> The shipped encoders are `embed-gemma:300m` plus the BERT-family set
> (`bge-base:en-v1.5`, `bge-small:en-v1.5`, `bge-large:en-v1.5`,
> `all-minilm:l6-v2`, `nomic-embed-text:v1.5`, `gte-multilingual:base`) -- see
> [EmbeddingGemma](/docs/models/embeddinggemma/).

See the ASR guide [here](https://openflowlm.com/docs/models/whisper/)

---

## 💻 Commands Inside CLI Mode

Once inside the CLI, use the following commands. System commands always start with `/` (e.g., `/help`).

---

### 🆘 Help

```text
/?
/help
```

> Displays all available interactive system commands. Highly recommended for first-time users.
> `/help` is an alias for `/?`.

---

### 🪪 Model Info

```text
/show
```

> View model architecture, size, **max context size (Adjustable – see bottom)** and more.

---

### 🔄 Change Model

```text
/load [model_name]
```

> Unload the current model and load a new one. KV cache will be cleared.

---

### ⬇️ Pull a Model

```text
/pull [model_name]
```

> Download a model without leaving the session. `model_name` is a full tag,
> e.g. `/pull llama3.2:3b`.

---

### 💾 Save Conversation

```text
/save
```

> Save the current conversation history to disk.

---

### 🧹 Clear Memory

```text
/clear
```

> Clear the KV cache (model memory) for a fresh start.

---

### 📊 Show Runtime Stats

```text
/status
```

> Display runtime statistics like token count, throughput, etc.

---

### 🕰️ Show History

```text
/history
```

> Review the current session's conversation history.

---

### 🔍 Toggle Verbose Mode

```text
/verbose
```

> Enable detailed performance metrics per turn. Run again to disable.

---

### 📦 List Models

Display all available models and locally downloaded models:

```text
/list
```

---

### 👋 Quit CLI Mode

```text
/bye
```

> Exit the CLI.

---

### 🧠 Think Mode Toggle

Type `/think` to toggle Think Mode on or off interactively in the CLI.

> 💡 **Note**: This feature is only supported on certain models, such as **Qwen3**.

---

### 📂 Load a Local Text File in CLI Mode

Use any file that can be opened in Notepad (like `.txt`, `.json`, `.csv`, etc.).

Format (in CLI mode):

```shell
/input "<file_path>" prompt
```

Example:

```shell
/input "C:\Users\Public\Desktop\alice_in_wonderland.txt" Summarize it into 200 words
```

> Notes:

* Use quotes **only around the file path**
* **No quotes** around the prompt
* File must be plain text (readable in Notepad)

👉 Any long plain-text file works -- `Alice in Wonderland` from Project Gutenberg is a convenient ~150k-character stress prompt. Note that a model's supported context length is limited by its own `default_context_length` **and** by available DRAM.

> ⚠️ **Caution:** a model's supported context length is limited by available DRAM capacity. For example, with **32 GB** of DRAM, **LLaMA 3.1:8B** cannot run beyond a **32K** context length. Note that `llama3.1:8b` ships with a `default_context_length` of 16384, so raising it is an explicit opt-in.

If DRAM is heavily used by other programs while running **OpenFlowLM**, you may encounter errors due to insufficient memory, such as:

```error
[XRT] ERROR: Failed to submit the command to the hw queue (0xc01e0200):
Even after the video memory manager split the DMA buffer, the video memory manager
could not page-in all of the required allocations into video memory at the same time.
The device is unable to continue.
```

> 🤔 Interested in checking the DRAM usage?

<!-- **Method 1 – Task Manager (Quick View)**   -->
1. Press **Ctrl + Shift + Esc** (or **Ctrl + Alt + Del** and select **Task Manager**).  
2. Go to the **Performance** tab.  
3. Click **Memory** to see total, used, and available DRAM, as well as usage percentage.  

<!-- **Method 2 – Resource Monitor (Detailed View)**  
1. Press **Windows + R**.  
2. Type:  ```resmon```
3. Press **Enter**.  
4. Go to the **Memory** tab to view detailed DRAM usage and a per-process breakdown. -->

---

### 🌄 Loading Images in CLI Mode (for VLMs only, e.g. gemma3:4b)

Supports **.png** and **.jpg** formats.  

```shell
/input "<image_path>" prompt
```

Example:

```shell
/input "C:\Users\Public\Desktop\cat.jpg" describe this image
```

> Notes:

* Make sure the model you are using is a **vision model (VLM)** (e.g., gemma3:4b) 
* Put quotes **only around the file path**  
* Do **not** use quotes around the prompt  
* Image must be in **.jpg** or **.png** format  

#### Pre-resize the input image

Vision models preprocess images, and time-to-first-token scales with the
preprocessed resolution. `--img-pre-resize` (`-r`) picks a fixed height instead
of using the original, trading detail for latency:

| value | effect |
|---|---|
| `0` | use the original resolution |
| `1` | height 480 |
| `2` | height 720 |
| `3` | height 1080 |
| `4` … `8` | progressively larger heights, up to 4320 |

```shell
oflm run qwen3vl-it:4b -r 1
```

> ⚠️ Image understanding is served by the **closed** engine DLLs -- the open
> kernels are text-only. Passing an image selects the closed path even for a
> model whose text path runs open kernels. See the
> [support-status matrix](/docs/models/).

---

### ⚙️ Set Variables

```text
/set
```

`/set` takes a key and a value. The keys are exactly these -- an unrecognised
key prints `Invalid context:` and the list below:

| key | sets |
|---|---|
| `topk` | top-k |
| `topp` | top-p |
| `minp` | min-p |
| `temp` | temperature |
| `rep-pen` | repetition penalty |
| `freq-pen` | frequency penalty |
| `pres-pen` | presence penalty |
| `sys-msg` | the system message |
| `ctx-len` | the max context length |
| `gen-lim` | the upper limit on tokens generated per response |
| `r-eff` | reasoning effort (`low`\|`medium`\|`high`, GPT-OSS only, default `medium`) |
| `prefill-chunk-len` | accepted but currently a no-op |

Note the spellings: `topk`, not `top_k`, and `temp`, not `temperature`. An
invalid value for `ctx-len` aborts the process rather than being ignored.

```text
/set gen-lim 128
/set r-eff high
```

> ⚠️ **Note:** Providing invalid or extreme hyperparameter values may cause inference errors.

---

## 📊 Benchmarking Tool

Use the OFLM benchmarking tool to measure a model's performance across different context lengths.

Each benchmark tests context lengths from `1k` to `32k`, running `2` iterations at each length.

```shell
oflm bench llama3.2:1b
```

Change the iteration times by `bench-iterations`:

```shell
oflm bench llama3.2:1b --bench-iterations 4
```

Drive the sweep from a JSON config instead of the defaults with `-i` (the
configs live in `utilities/bench-configs/`):

```shell
oflm bench llama3.2:3b -i utilities/bench-configs/bench-1k.json
```

> ⚠️ The sweep runs context lengths from `1k` up to the config's `max_length`.
> A stage longer than the model's `default_context_length` will not produce a
> meaningful decode number, so keep `max_length` within what the model
> supports.

OFLM prints the results in your terminal and also saves them as a CSV file in the current folder for later reference.

```text
[OFLM]  === Benchmark Results ===

 Context Length |              TTFT (s) |      Prefill Speed (tok/s) |     Decoding Speed (tok/s)
----------------------------------------------------------------------------------------------------
             1k |       0.815 ±   0.012 |        1233.57 ±     17.91 |       61.39 ±      0.46
             2k |       1.144 ±   0.035 |        1728.16 ±     55.68 |       59.04 ±      0.36
             4k |       1.955 ±   0.009 |        2001.55 ±      9.22 |       54.32 ±      0.47
             8k |       3.821 ±   0.010 |        2037.62 ±      5.40 |       46.89 ±      0.19
            16k |       9.144 ±   0.011 |        1698.20 ±      2.08 |       37.20 ±      0.14
            32k |      26.346 ±   0.037 |        1177.16 ±      1.53 |       26.38 ±      0.07
----------------------------------------------------------------------------------------------------
```

### Embedding models: `bench-embed`

`bench` is for chat models. Encoders have no first token and no
prefill/decode split, and their sequence length is fixed by the compiled
design rather than by the request -- so they get their own command, which
sweeps **batch size** instead of context length.

```shell
oflm bench-embed bge-base:en-v1.5
oflm bench-embed bge-base --max-batch 32 --bench-iterations 5
oflm bench-embed nomic-embed-text:v1.5 --prompt-name query
oflm bench-embed bge-base -i utilities/bench-configs/bench-embed-32.json
```

| flag | meaning |
|---|---|
| `--max-batch N` | largest batch swept; it doubles 1, 2, 4 ... N (default 128). Must be a power of two. |
| `--bench-iterations N` | timed iterations per stage (default 2), after one discarded warm-up. |
| `--prompt-name NAME` | the task prompt, by its REST name (`query`, `document`, `clustering`, ...). **Required** for a model that declares prompt names (nomic). **Refused** for one with no task-prompt concept at all (the bge sizes, MiniLM, gte). `embed-gemma:300m` is the exception: it declares no names and still honours tasks through a hardcoded per-task prefix, so it accepts the flag and needs none. |
| `-i FILE` | a JSON config: `max_batch`, `iterations`, `task`, `texts`. |

Every stage times **two paths over the same texts**: one batched call, and the
same texts one at a time -- which is what a caller doing one request per text
gets. `Speedup` is the ratio, and on the NPU-backed encoders it is 5-10x.

```text
[OFLM]  === Embedding Benchmark Results ===

  Batch |         Batched (s) | Looped (s) |  Speedup |             Texts/s |          Tokens/s
--------------------------------------------------------------------------------------------------
      1 |      0.0253 +- 0.0008 |     0.0250 |    0.99x |        39.5 +-  1.2 |        1186 +-  36
      8 |      0.0314 +- 0.0008 |     0.2018 |    6.43x |       255.0 +-  6.9 |        5451 +- 148
     64 |      0.2056 +- 0.0020 |     1.5513 |    7.55x |       311.3 +-  3.0 |        6868 +-  65
    128 |      0.5059 +- 0.0018 |     3.1064 |    6.14x |       253.0 +-  0.9 |        5582 +-  20
--------------------------------------------------------------------------------------------------
```

Results also go to `bench_embed_<tag>_<date>[_<cpu>].csv` in the current
folder, with a `#` provenance header naming the model, the task, the corpus and
the identity-gate result.

Full methodology, what it refuses and why, and what it does **not** measure:
[`utilities/bench-configs/README.md`](https://github.com/Atomic-Germ/OpenFlowLM-Next/blob/main/utilities/bench-configs/README.md).
Measured results for every model: [Benchmarks -> Embeddings](/docs/benchmarks/embeddings_results/).

---

## 🗂️ Others

### 🛠 Change Default Context Length (max)

You can find more information about available models here:  

```
C:\Program Files\oflm\model_list.json
```

You can also change the `default_context_length` setting.

> ⚠️ **Note:** Be cautious! The system reserves DRAM space based on the context length you set.  
> Setting a longer default context length may cause errors on systems with smaller DRAM.
> Also, each model has its own default context length. These are the shipped
> values, not hard limits -- the model's own `config.json` is the real ceiling:
>
> | tag | `default_context_length` |
> |---|---|
> | `llama3.2:1b` | 131072 (128k) |
> | `gemma3:4b`, `llama3.2:3b` | 65536 (64k) |
> | `gemma3:1b`, `qwen3-tk:4b` | 32768 (32k) |
> | `llama3.1:8b` | 16384 (16k) |

`llama3.1:8b` ships with a deliberately small default; raise it with care.

#### Where `model_list.json` lives

| Platform | Path |
|----------|------|
| Windows (MSI install) | `C:\Program Files\oflm\model_list.json` |
| Linux (packaged install) | `/opt/openflowlm/share/oflm/model_list.json` |
| Relocatable bundle | `<dir of the oflm binary>/../share/oflm/model_list.json` |
| User override | `~/.config/oflm/model_list.json` |

The user override takes priority over the shipped copies. Set
`OFLM_CONFIG_PATH` to point at one explicitly.

#### Other environment variables

| Variable | Effect |
|---|---|
| `OFLM_MODEL_PATH` | Where models are stored (default `~/.oflm/models` on Windows, `~/.config/oflm/models` on Linux) |
| `OFLM_SERVE_PORT` | Default server port (default `52625`) |
| `OFLM_CONFIG_PATH` | Explicit `model_list.json` to load |
| `OFLM_MODELINFO_PATH` | Explicit `model_info.json` to load |
| `OFLM_HF_OWNER` | Hugging Face account to pull OpenFlowLM's own models from, instead of `model_list.json`'s `hf_owner` |
| `OFLM_XCLBIN_PATH` | Extra directory to search for xclbins |
| `OFLM_OPEN_KERNELS_DIR` | Where the open engine looks for exported kernels |

Every variable read through `getenv_oflm` also accepts its pre-rename `FLM_*`
spelling and prints a one-line notice naming the new one -- so an install from
before the `flm` -> `oflm` rename keeps working. `OFLM_OPEN_KERNELS_DIR` is the
exception: the open engine reads it with plain `getenv`, so
`FLM_OPEN_KERNELS_DIR` does nothing.

### 📦 Add a Converted Model (`oflm add`)

Register a pre-converted Q4NX model that is not in the shipped registry. This
is the supported way to install a model you packed yourself with `q4nx-build`,
or one published under a HuggingFace repo that is not yet in `model_list.json`:

```shell
oflm add Atomic-Germ/Model-3B-OpenNPU2 --family qwen3
```

| flag | meaning |
|---|---|
| `--tag NAME` | local tag to register under (e.g. `mymodel:3b`) |
| `--family NAME` | kernel family to link the xclbins from (required unless `--xclbin-dir` is given) |
| `--config FILE` | a `model_list.json` entry to use verbatim instead of generating one |
| `--models-root DIR` | install into this model store instead of the default |
| `--xclbin-dir DIR` | take the xclbins from here instead of resolving by family |
| `--xclbin-from REF` | source the family xclbins from another local model |
| `--system-list` | also register the tag in the system `model_list.json` |
| `--modelscope` | pull weights from ModelScope instead of HuggingFace |
| `--open-kernels DIR` | point the tag at a locally exported open-kernel set |
| `--no-xclbin` | register the weights without any xclbins |
| `--no-verify` | skip the post-install verification pass |
| `--force` | overwrite an existing installation |
| `--dry-run` | print what would happen, change nothing |

Every shape-identical model links to a **family** xclbin, so adding a new model
of an existing shape needs no kernel work:

```shell
oflm add Someone/Nanbeige4.1-3B-finetune-OpenNPU2 --family nanbeige --tag nanbeige-ft:3b
```

### 🏷️ Print the Version

```shell
oflm version
oflm version --json
```

`--json` prints `{ "version": "0.1.0" }`. The version is baked in at build time
from `OFLM_VERSION` in `CMakePresets.json`.

> ℹ️ Each registry entry carries an `oflm_min_version` -- the oldest `oflm`
> that can load that model. The downloader compares it against the running
> build and warns when the model is newer than the binary.

