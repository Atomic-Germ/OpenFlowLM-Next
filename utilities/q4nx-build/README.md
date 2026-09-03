# FLM Q4NX Converter

A utility for converting GGUF model files, usually fine-tunes, into the Q4NX format. This tool supports converting language, vision, and audio model weights.

## Supported Models
Based on the configuration, the converter supports several model architectures, including:
- Gemma 3
- GPT-OSS
- LFM 2
- LLaMA
- Phi-4
- Qwen 2 / Qwen 2.5
- Qwen 2 VL
- Qwen 3
- Qwen 3 VL
- Qwen 3.5
- Qwen 3.5 MoE
- Qwen 3.6 MoE

### Weight Type Support
- `language`: supported for model families in `configs/`
- `vision`: supported for vision-capable architectures (for example, Gemma 4 and Qwen3-VL)
- `audio`: currently supported for **Gemma 4** variants (`-t audio`)

## Setup

Create a virtual environment using python 3.13

1. Create and activate a virtual environment using pip or uv:
   ```bash
   uv venv venv
   venv/bin/activate
   ```
2. Install dependencies (one time)
   ```bash
   uv pip install -r requirements.txt
   ```

## Usage

The main entry point for conversion is `convert.py`.

### Help
```bash
python convert.py -h
```

### Basic Syntax
```bash
python convert.py [input_file] [output_folder] [-t TYPE]
```

### One-flag builds from an HF repo (installed `q4nx-build`)

When `-i` names an HF repo, `q4nx-build` fills in the rest from the repo's own model card:

```bash
q4nx-build -i numind/NuExtract3-GGUF
```

### Open (unquantized) embedding repos

Embedding models need no quantization, so `q4nx-build` also has a packaging
mode that produces a complete, uploadable HuggingFace repo directory in one
shot:

```bash
q4nx-build --open-embedding -i google/embeddinggemma-300m \
  -o ~/Embedding-Gemma-300M-OpenNPU2 \
  --npu-assets ~/.config/flm/models/.../npu_matmul_f32
```

`-i` accepts either an HF repo id or a local model directory. The output
contains:

- `config.json`, `tokenizer.json`, `tokenizer_config.json`
- `model.safetensors` (transformer body, copied verbatim)
- `2_Dense/model.safetensors`, `3_Dense/model.safetensors` (dense heads)
- `npu_matmul_f32/*.{xclbin,insts}` when `--npu-assets` is given
- `weights_manifest.json` with **paths relative to the model dir**, so the
  package carries no absolute builder paths
- `model_info_entry.json` — registry metadata (path/size/sha256 oid) to merge
  into `src/model_info.json` under the model tag

`--npu-assets` is the escape hatch for **new or prototype** models: it lets a
brand-new model ship its own compiled kernels inside the model directory so it
works end to end immediately. Established families should omit it and ship
kernels with the application from `src/xclbins/<Model-Dir>/npu_matmul_f32/`
instead, keeping model repos to weights and configuration only.

### Open (unquantized) causal-LM repos

The text-model counterpart, used for the open Gemma3 text engine
(`docs/plans/open_gemma3_text_plan.md`):

```bash
q4nx-build --open-causal-lm -i google/gemma-3-1b-it \
  -o ~/Models/Gemma-3-1B-OpenNPU2 \
  --make-reference ~/Models/Gemma-3-1B-OpenNPU2/reference_v1.json
```

It packs bf16 safetensors verbatim (lossless — the weights are natively bf16, so
fp32 would add size with no accuracy benefit) and records two things the runtime
cannot safely infer:

- **per-tensor dtype**, because the engine must not assume fp32;
- **tied embeddings** (`lm_head.weight` -> `model.embed_tokens.weight`), which
  Gemma3 uses instead of shipping an `lm_head` tensor.

### Reference oracle

`--make-reference` runs an independent **NumPy** implementation of the model and
writes prompt -> logit/token fixtures that the C++ engine must reproduce. It can
run standalone against any model directory:

```bash
q4nx-build -i ~/Models/Gemma-3-1B-OpenNPU2 \
  --make-reference ~/Models/Gemma-3-1B-OpenNPU2/reference_v1.json
```

Notes:

- NumPy only — no torch or transformers. This matters on AMD, where the ROCm
  torch build segfaults on model load and NumPy has no bfloat16. bf16 is decoded
  as `uint16 << 16` viewed as `float32`, matching the C++ engine bit for bit.
- fp32 compute, so implementation error is isolated from bf16 storage error.
- Prompt token ids are recorded explicitly, separating tokenizer correctness
  from forward-pass correctness.
- One prompt exceeds the sliding window (2282 tokens) to exercise hybrid
  sliding/global attention; greedy continuation is skipped there because
  re-forwarding is quadratic.
- The long prompt's greedy decode is intentionally omitted; everything else
  includes a 16-token greedy continuation.

The build is byte-reproducible: contents are copied verbatim, JSON is sorted,
and no timestamps or host paths are recorded. `model_info_entry.json` is a
build artifact; exclude it when uploading the repo directory.

1. **Base chain** — it follows the `base_model` frontmatter up the tree
   (`numind/NuExtract3-GGUF -> numind/NuExtract3 -> Qwen/Qwen3.5-4B -> ...`) and stops at the first
   ancestor with a `{org}/{base}-NPU2` mirror. Orgs are tried in order: **Atomic-Germ**, then **FastFlowLM**.
   That mirror becomes the `-s` skeleton source for tokenizer/config/vision assets.
2. **Weights type** — VLM pipeline tags (`image-text-to-text`, ...) or vision tags imply `-t vision`
   (language + vision weights); otherwise language.
3. **Output name** — `{model_name or repo name}-{size}-NPU2`, e.g. `NuExtract3-4B-NPU2`. The size token
   is parsed from the skeleton's name and skipped when the display name already carries it.

Explicit flags always win over derived values. Preview everything without converting:

```bash
q4nx-build -i numind/NuExtract3-GGUF --dry-run
```

### Arguments
- **`input_file`**  
  Also available as **`-i`** or **`--input`**.  
  Specifies the path to the input `.gguf` file that you want to convert.

- **`output_folder`**  
  Also available as **`-o`** or **`--output`**.  
   Specifies the destination folder for the converted output. The converter writes the generated tensors as **`model.q4nx`** into this folder.  
  If not provided, the converter uses the same directory as the input file.

- **`-t`, `--type`**  
  Specifies which weights to convert.  
  Available options are:
  - **`language`** — converts the language model weights
   - **`vision`** — converts the vision model weights
   - **`audio`** — converts audio model weights (currently Gemma 4)  
  If this option is not specified, the default is **`language`**.

- **`-f`, `--force`**  
  Forces the converter to use a specific model architecture instead of detecting it automatically from the GGUF metadata.  
  This can be useful when automatic detection is incorrect or when you want to explicitly select an architecture such as **`qwen2`**, **`llama`**, or **`gemma3`**.  
  Leave this option out if you want the converter to detect the architecture automatically.

- **`-s`, `--source-model`**  
  Source HF/ModelScope model for tokenizer/config assets (the NPU2 skeleton).  
  When omitted and `-i` is an HF repo, the base_model chain is walked automatically (see
  [One-flag builds](#one-flag-builds-from-an-hf-repo-installed-q4nx-build)).

- **`--dry-run`**  
  Print the resolved build plan (GGUF choice, base_model chain, skeleton source, output name,
  weights type) without converting or downloading anything.

### Examples

**1. Convert a language model (positional arguments):**
```bash
python convert.py model.gguf output_folder
```

**2. Convert a language model (flag arguments):**
```bash
python convert.py -i unsloth_gpt-oss-20b-Q4_0.gguf -o unsloth-gotoss20b-q40
```

**3. Convert a vision model:**
```bash
python convert.py -i qwen3vl-4b-mmproj-BF16.gguf -o unsloth-qwen3vl-vision -t vision
```

**4. Force a specific model architecture:**
```bash
python convert.py -i model.gguf -o output_folder -f qwen2
```
This is useful when the GGUF file metadata doesn't correctly identify the architecture or when you want to override the automatic detection.

**5. Convert an audio model (Gemma 4):**
```bash
python convert.py -i gemma4-2b-mmproj.gguf -o unsloth-gemma4-2b-audio -t audio -f gemma4
```

**6. Convert from a different quantization format (e.g. Q4_K_M):**
```bash
python convert.py -i model.gguf -o output_folder
```
The converter automatically dequantizes non-Q4_0/Q4_1 weights and re-quantizes them into Q4NX. See the [Converting from Other Quantization Formats](#converting-from-other-quantization-formats) section for details.

## Converting from Other Quantization Formats

The converter supports GGUF models that use quantization formats other than Q4_0 or Q4_1, such as **Q4_K_M**, **Q8_0**, **Q5_K_M**, and others. When the converter encounters weights in a non-native format, it automatically dequantizes them back to floating point and then re-quantizes them into the Q4NX target format.

This means you are not limited to sourcing GGUF files that are already in Q4_0 or Q4_1 — you can use a wider variety of community-quantized models as input.

The conversion process works as follows:

1. The converter reads the quantization type of each tensor from the GGUF file.
2. If the tensor is not already in the expected format (Q4_0 or Q4_1, depending on the model config), it is dequantized to FP32.
3. The dequantized weights are then re-quantized into the target format required by FLM.

This process is fully automatic and requires no additional flags beyond **`-f`** if the architecture cannot be auto-detected.

> **Note:** Dequantizing from a lossy format and re-quantizing introduces additional quantization error compared to converting directly from full-precision weights. For best quality, prefer starting from BF16, FP16, or Q8_0 sources when available.

When converting community-quantized models, the GGUF metadata may not always match the architecture names expected by the converter. Use the **`-f`** flag to explicitly specify the model architecture in those cases.

### Example

Convert a [Q4_K_M quantized model](https://huggingface.co/HauhauCS/Qwen3.5-9B-Uncensored-HauhauCS-Aggressive) from the community:

```bash
python convert.py -i Qwen3.5-9B-Uncensored-HauhauCS-Aggressive-Q4_K_M.gguf -o unsloth-qwen3_59b_uncensored -f qwen3.5-9B
```

Here, **`-f qwen3.5-9B`** tells the converter to use the Qwen 3.5 9B architecture, since the GGUF metadata from the community model may not be detected correctly.


## Step-by-Step Converting and Registering Custom FLM Models

This guide walks you through how to convert a supported GGUF model into FLM's Q4NX format, then either:

- **replace an existing installed model**, or
- **register your converted weights as a separate custom model**

The examples below use **`qwen3vl-it:4b`**, but the same overall workflow applies to other supported models.

---

### Before You Begin

Make sure you have:

- **FLM** installed on your system
- a **supported GGUF model file**
- the **FLM converter**
- the correct **conversion type** for your model family

#### Important compatibility note

Different model families expect different quantization formats. The expected format for each model family is defined in the configuration files under `configs/`.

For example:

- **`lfm2`** uses **`q4_0`**
- **`qwen3vl`** uses **`q4_1`**

If the source GGUF file uses a different quantization format (such as Q4_K_M or Q8_0), the converter will automatically dequantize and re-quantize the weights into the expected format. See [Converting from Other Quantization Formats](#converting-from-other-quantization-formats) for details.

---

### Option 1: Replace an Existing Installed Model

Choose this option if you want to **swap the original FLM model weights** with your own converted weights while keeping the same model name and launch command.

#### Step 1: Download a compatible GGUF model

Download a supported GGUF model from Hugging Face or another trusted source.

Before continuing, verify that:

- the model is compatible with the converter
- the quantization matches the expected format for the model family
- you know which FLM model you plan to replace

For the `qwen3vl` family, make sure the GGUF model matches **Q4_1** expectations.

---

#### Step 2: Convert the GGUF model to Q4NX

Run the converter using the instructions from the **Usage** section of the converter project.

Provide:

- the input GGUF file
- an output folder
- the correct conversion type

The converter will generate one or more `.q4nx` files, depending on the model architecture.

For vision-language models such as Qwen3-VL, this may include files such as:

- `model.q4nx`
- `vision_weights.q4nx`

> Keep the output folder handy. You will need these converted files in the next step.

---

#### Step 3: Locate the installed FLM model directory

Find the installed FLM model directory for the model you want to replace.

**Default model paths**

**Windows**
```text
C:\Users\<username>\Documents\flm\models\Qwen3-VL-4B-Instruct-NPU2
```

**Linux**
```text
/home/<username>/.config/flm/models/Qwen3-VL-4B-Instruct-NPU2
```

> Replace `<username>` with your actual system username.

---

#### Step 4: Replace the existing Q4NX file or files

Copy the newly converted files from your output folder into the installed FLM model directory.

Replace the corresponding existing file(s), such as:

- `model.q4nx`
- `vision_weights.q4nx`

Be careful to preserve the original filenames expected by FLM.

**Recommended best practice**

Before replacing anything, create a backup of the original model folder or at least the original `.q4nx` files. This makes it easy to restore the original model later if needed.

---

#### Step 5: Start the model

Once the replacement files are in place, start the model using either of the following commands:

```bash
flm run qwen3vl:4b
```

or

```bash
flm serve qwen3vl:4b
```

If the conversion and replacement were successful, FLM should now load your custom-converted weights under the original model name.

---

### Option 2: Add a Custom Model Configuration

Choose this option if you want to **keep the original FLM model intact** and register your converted model as a **separate custom model**.

This is the safer and more flexible option, especially if you want to compare the original model with your custom version.

Before starting, complete:

- **Step 1: Download a compatible GGUF model**
- **Step 2: Convert the GGUF model to Q4NX**

Then continue below.

---

#### Step 1: Open the main FLM models directory

Locate the main FLM models directory.

**Default model paths**

**Windows**
```text
C:\Users\<username>\Documents\flm\models\
```

**Linux**
```text
/home/<username>/.config/flm/models/
```

---

#### Step 2: Duplicate an existing model folder

Copy the existing model folder:

```text
Qwen3-VL-4B-Instruct-NPU2
```

Rename the copied folder to something new, for example:

```text
Qwen3-VL-4B-Custom
```

This duplicated folder becomes the base for your custom model.

---

#### Step 3: Replace the Q4NX files in the copied folder

Inside your newly copied model folder, replace the relevant `.q4nx` files with the files generated by the converter.

Typical files include:

- `model.q4nx`
- `vision_weights.q4nx`

Make sure the filenames match what FLM expects for that model.

---

#### Step 4: Edit `model_list.json`

Open your FLM installation directory and locate `model_list.json`.

**Default installation paths**

**Windows**
```text
C:\Program Files\flm
```

**Linux**
```text
/opt/fastflowlm/share/flm
```

You now have two ways to register your custom model:

- **Standalone model**  

- **Submodel under an existing family**  

Both approaches work. Choose the one that best fits your organization and naming preference.

**Option 1: Add a standalone custom model entry**

Add a new entry under `models`:

```json
"qwen3vl-it-custom": {
   "4b": {
      "name": "Qwen3-VL-4B-Custom",
      "url": "https://huggingface.co/FastFlowLM/Qwen3-VL-4B-Custom/resolve/v0.9.22-faster-q4-1",
      "file_url": "https://huggingface.co/api/models/FastFlowLM/Qwen3-VL-4B-Custom/tree/v0.9.22-faster-q4-1",
      "size": 4000000000,
      "flm_min_version": "0.9.22",
      "files": [
         "config.json",
         "model.q4nx",
         "tokenizer.json",
         "tokenizer_config.json",
         "vision_weights.q4nx"
      ],
      "vlm": true,
      "default_context_length": 32768,
      "details": {
         "format": "NPU2",
         "family": "qwen3vl",
         "think": false,
         "parameter_size": "4B",
         "quantization_level": "Q4_1"
      },
      "label": [
         "vision"
      ],
      "footprint": 3.9
   }
}
   ```

---

**Option B: Add a custom submodel under `qwen3vl-it`**

Add a new sub-entry under the existing `qwen3vl-it` model family.

```json
"qwen3vl-it": {
   "4b-custom": {
      "name": "Qwen3-VL-4B-Custom",
      "url": "https://huggingface.co/FastFlowLM/Qwen3-VL-4B-Custom/resolve/v0.9.22-faster-q4-1",
      "file_url": "https://huggingface.co/api/models/FastFlowLM/Qwen3-VL-4B-Custom/tree/v0.9.22-faster-q4-1",
      "size": 4000000000,
      "flm_min_version": "0.9.22",
      "files": [
         "config.json",
         "model.q4nx",
         "tokenizer.json",
         "tokenizer_config.json",
         "vision_weights.q4nx"
      ],
      "vlm": true,
      "default_context_length": 32768,
      "details": {
         "format": "NPU2",
         "family": "qwen3vl",
         "think": false,
         "parameter_size": "4B",
         "quantization_level": "Q4_1"
      },
      "label": [
         "vision"
      ],
      "footprint": 3.9
   }
}
```

**Note on `url` and `file_url`**

The `url` and `file_url` fields only matter when FLM needs to fetch the model from a remote source, for example when you run:

```bash
flm pull <custom-model-name>
```

If you want that workflow to work, make sure:

- the model files are already uploaded and reachable online
- the `files` list matches what is actually hosted

In this guide, you have already copied all required model files into the local model directory manually, FLM can load them directly from disk. In that case, `url` and `file_url` can be dummy placeholder values and do not need to point to real hosted files.


---

#### Step 5: Copy the matching `xclbins/` folder

In the FLM installation directory, open the `xclbins/` folder.

If you created a **standalone custom model**, copy:

```text
Qwen3-VL-4B-Instruct-NPU2
```

Then rename the copied folder to:

```text
Qwen3-VL-4B-Custom
```

This folder name should match the `"name"` value used in your custom model entry.

> If you are using the **submodel style**, you can usually skip this step.

---

#### Step 6: Confirm that FLM recognizes the custom model

Run:

```bash
flm list
```

You should see one of the following in the output:

- `qwen3vl-it-custom:4b` for a **standalone model**
- `qwen3vl-it:4b-custom` for a **submodel** of qwen3vl-it family

If the new model appears in the list, FLM has recognized your configuration successfully.

---

#### Step 7: Start the custom model

Use the command that matches the registration style you chose.

**Standalone model**

```bash
flm run qwen3vl-it-custom:4b
```

or

```bash
flm serve qwen3vl-it-custom:4b
```

**Submodel**

```bash
flm run qwen3vl-it:4b-custom
```

or

```bash
flm serve qwen3vl-it:4b-custom
```



## Project Structure
- `convert.py`: The main CLI script for running conversions.
- `setup_venv.sh`: Initializes the Python virtual environment and installs dependencies.
- `activate.sh`: Activates the virtual environment and sets up environment variables.
- `q4nx/`: Core package containing the conversion logic, gguf tensor parsing, and model-specific implementations.
- `configs/`: JSON configuration files for supported model architectures.


## Known Issues
- The converter outputs either Q4_0 or Q4_1 quantization format based on the setting in config files for each model. Input GGUF files can use other quantization formats — the converter will dequantize and re-quantize automatically, though this may introduce additional quantization error.
- For GPT-OSS:20B models, the converter currently uses the original `model.embed_tokens.weight` from the safetensors from OpenAI due to issues with Q4_1 quantization (from our experience, Q4_1 quantization messes up the quantization of the embedding layer for this model). 
  - **Workaround:** Place the [`model-00001-of-00001.safetensors`](https://huggingface.co/openai/gpt-oss-20b/tree/main) file in the root directory of this project before running the conversion for GPT-OSS. If the file is not found, the converter will print a warning and skip replacing the embedding weights. 
