---
layout: docs
title: Converting Custom Models
nav_order: 10
parent: Instructions
---

# 🔄 Converting Custom Models with `q4nx-build`

`q4nx-build` is the OpenFlowLM Q4NX converter. It takes a GGUF model (or an unquantized HuggingFace repo) and produces the `.q4nx` container that `oflm` loads at runtime.

It is included in this flake as a package, app, and dev shell.

---

## 🚀 Quick start (Nix)

### Enter the converter shell

```bash
nix develop .#q4nx-build
```

### Run a conversion without entering the shell

```bash
nix run .#q4nx-build -- -i peculiar-ragdoll/Cyber-Tiel-Coder-35B-A3B-GGUF-MTP --dry-run
```

---

## 🧩 What you need

- **OFLM installed** and on `PATH`.
- A **GGUF model file** or a HuggingFace repo that ships GGUFs.
- Enough disk space. Q4NX output is roughly the same size as the source GGUF, plus the source file itself and temporary dequant/requant workspace. A 22 GB Q4_K_M GGUF needs **~60–70 GB** free during conversion.
- A matching **official OFLM model** already installed or available online, used as the *skeleton source* for tokenizer, config, and xclbin metadata.

---

## 📋 Typical workflow

### 1. Preview the build plan

```bash
nix run .#q4nx-build -- \
  -i peculiar-ragdoll/Cyber-Tiel-Coder-35B-A3B-GGUF-MTP \
  --dry-run
```

This prints:
- the GGUF file `q4nx-build` will pick,
- the `base_model` chain,
- the skeleton source (e.g. `OpenFlowLM/Qwen3.6-35B-A3B-NPU2`),
- the target family it resolves to.

### 2. Force the family if auto-detection is wrong

Finetunes and re-quantizations sometimes have metadata that does not identify the underlying architecture. Use `-f` / `--force` to pin the OpenFlowLM family:

| Family | Typical use |
|---|---|
| `qwen3.6-moe` | Qwen3.6-MoE 35B-A3B, Ornith, Cyber-Tiel |
| `qwen3.5` | Qwen3.5 0.8B / 2B / 4B / 9B dense |
| `qwen3` | Qwen3 0.6B / 1.7B / 4B / 8B dense |
| `qwen3vl` | Qwen3-VL 4B / 7B vision |
| `qwen2` | Qwen2.5 3B, Qwen2 1.5B / 7B |
| `qwen2vl` | Qwen2.5-VL 3B / 7B |
| `llama3` | Llama 3.1 / 3.2 |
| `gemma3` | Gemma 3 1B / 4B |
| `gemma4e` | Gemma 4 E2B / E4B |
| `gemma4-12b` | Gemma 4 12B |
| `gpt-oss` | GPT-OSS 20B |
| `granite` | IBM Granite 4.2 3B |
| `nanbeige` | Nanbeige4.1 3B |
| `phi4` | Phi-4-mini 4B |
| `lfm2` | LiquidFM 2 1.2B / 2.6B |

```bash
nix run .#q4nx-build -- \
  -i peculiar-ragdoll/Cyber-Tiel-Coder-35B-A3B-GGUF-MTP \
  -f qwen3.6-moe \
  -o ~/Cyber-Tiel-35B-A3B-OFLM
```

### 3. Convert a local GGUF file

```bash
nix run .#q4nx-build -- \
  -i /path/to/model-Q4_K_M.gguf \
  -f qwen3.6-moe \
  -o ~/MyModel-OFLM
```

### 4. Install the converted model

```bash
nix run .#oflm-add -- \
  ~/MyModel-OFLM \
  --tag mymodel:35b \
  --family qwen3.6-moe
```

Then run it:

```bash
oflm run mymodel:35b
```

---

## 🏗️ What `q4nx-build` produces

The output directory contains the files `oflm` needs:

```text
~/MyModel-OFLM/
├── config.json
├── model.q4nx
├── tokenizer.json
├── tokenizer_config.json
└── chat_template.jinja        # optional
└── vision_weight.q4nx          # only for VL models
```

---

## ⚙️ Useful flags

| Flag | Meaning |
|---|---|
| `-i REPO_OR_FILE` | Input: HF repo id, URL, or local GGUF |
| `-o DIR` | Output directory |
| `-f FAMILY` | Force family (e.g. `qwen3.6-moe`, `qwen3.5`) |
| `-t TYPE` | Weights type: `language`, `vision`, `audio`, `embedding` |
| `--dry-run` | Resolve the plan without converting |
| `--deploy` | Not needed for local use; used for packaging |

---

## 🔥 Building open kernels for a converted model

Closed-source xclbins are linked from the matching official model automatically by `oflm-add`. If you want to run on the **open kernels** instead:

```bash
nix develop .#open-kernels
python open_kernels/export_qwen36_kernels.py \
  --model-dir ~/.config/oflm/models/<ModelName>
```

Then run with the open-kernel set linked:

```bash
OFLM_QWEN36_ENGINE=open oflm run mymodel:35b
```

---

## 🐛 Troubleshooting

### `unsupported model family`

The registry `details.family` does not match an engine in this build. Re-run `oflm-add` with the correct `--family`.

### Weight load assertion / buffer overrun

The converted Q4NX does not match the engine's expected layout. Common causes:

1. **Wrong family pin.** Re-convert with `-f` set to the real base model family.
2. **Outdated Q4NX format.** Some community repos were converted for older OpenFlowLM versions. Re-converting from the upstream GGUF with the current `q4nx-build` fixes this.
3. **Mismatched size.** A 9B model converted against a 4B skeleton will assert. Make sure the skeleton source matches the actual model dimensions.

### `No module named 'gguf'` / missing deps

Use the Nix shell/package. Do not install `q4nx-build` into the IRON `ironvenv`; that environment is only for kernel compilation.

---

## 📦 Full example: Cyber-Tiel 35B-A3B

```bash
# 1. plan
nix run .#q4nx-build -- \
  -i peculiar-ragdoll/Cyber-Tiel-Coder-35B-A3B-GGUF-MTP \
  -f qwen3.6-moe \
  --dry-run

# 2. convert (use /home with at least 70 GB free, or /tmp on a large filesystem)
nix run .#q4nx-build -- \
  -i peculiar-ragdoll/Cyber-Tiel-Coder-35B-A3B-GGUF-MTP \
  -f qwen3.6-moe \
  -o ~/Cyber-Tiel-35B-A3B-OFLM

# 3. register
nix run .#oflm-add -- \
  ~/Cyber-Tiel-35B-A3B-OFLM \
  --tag cybertiel:35b \
  --family qwen3.6-moe

# 4. run
oflm run cybertiel:35b
```

---

## 🧵 Non-Nix install

If you prefer `pip`:

```bash
pip install utilities/q4nx-build
q4nx-build -h
```

The Python package still requires `torch`, `transformers`, `gguf`, and `huggingface-hub`.
