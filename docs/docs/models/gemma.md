---
layout: docs
title: Gemma
nav_order: 4
parent: Models
---

## 🧩 Model Card: [gemma-3-1b-it](https://huggingface.co/google/gemma-3-1b-it)

- **Type:** Text-to-Text
- **Think:** No
- **Tool Calling Support:** No  
- **Base Model:** [google/gemma-3-1b-it](https://huggingface.co/google/gemma-3-1b-it)
- **Quantization:** Q4_1
- **Max Context Length:** 32k tokens  
- **Default Context Length:** 32k tokens ([change default](https://openflowlm.com/docs/instructions/cli/#-change-default-context-length-max))  
- **[Set Context Length at Launch](https://openflowlm.com/docs/instructions/cli/#-set-context-length-at-launch)**

▶️ Run with OpenFlowLM in PowerShell:  

```shell
oflm run gemma3:1b   # NOT RUNNABLE in this build -- see the note below
```

> ⚠️ **gemma3:1b is unavailable.** No open Gemma 3 kernel set was ever built for
> the 1B class, and the closed `gemma_text_npu` engine has been removed. There
> is no engine path for this tag in this build; it remains listed only because
> the registry has not been pruned yet. Use Gemma 3 4B via
> `OFLM_GEMMA_ENGINE=closed` (see below) or a Gemma 4 model instead.

---

## 🧩 Model Card: [gemma-3-4b-it](https://huggingface.co/google/gemma-3-4b-it)

- **Type:** Image-Text-to-Text
- **Think:** No
- **Tool Calling Support:** No  
- **Base Model:** [google/gemma-3-4b-it](https://huggingface.co/google/gemma-3-4b-it)
- **Quantization:** Q4_1
- **Max Context Length:** 128k tokens  
- **Default Context Length:** 64k tokens ([change default](https://openflowlm.com/docs/instructions/cli/#-change-default-context-length-max))  
- **[Set Context Length at Launch](https://openflowlm.com/docs/instructions/cli/#-set-context-length-at-launch)**

▶️ Run with OpenFlowLM in PowerShell:  

```shell
oflm run gemma3:4b
```

📝 **Note:** In CLI mode, attach an image with:

```shell
/input "file/to/image.jpg" describe this image.
```

✅ **Text runs on the open Gemma 3 kernels by default.** The open `gemma3` set
(4B and 12B geometries; no 1B) ships with the app, so `gemma3:4b`,
`medgemma:4b`, `medgemma1.5:4b` and `translategemma:4b` run open unless the
open engine refuses to serve them. Vision/audio still needs the closed
`gemma_npu` DLL, and there is no automatic fallback — set
`OFLM_GEMMA_ENGINE=closed` to run the legacy closed engine for text and images.
Gemma 4 (12B) runs open for text in this build; its image path needs the
closed `gemma4_12b_npu` engine, which only reads the upstream `.q4nx`
container -- a GGUF install of Gemma 4 has no working vision path today. See the
[support-status matrix](/docs/models/#open-vs-closed-support-status).

---

## 🧩 Model Card: [gemma-4-E2B-it](https://huggingface.co/google/gemma-4-E2B-it)

- **Type:** Any-to-Text
- **Think:** Yes
- **Tool Calling Support:** Yes  
- **Base Model:** [google/gemma-4-E2B-it](https://huggingface.co/google/gemma-4-E2B-it)
- **Quantization:** Q4_1
- **Max Context Length:** 128k tokens  
- **Default Context Length:** 32k tokens ([change default](https://openflowlm.com/docs/instructions/cli/#-change-default-context-length-max))  
- **[Set Context Length at Launch](https://openflowlm.com/docs/instructions/cli/#-set-context-length-at-launch)**

▶️ Run with OpenFlowLM in PowerShell:  

```shell
oflm run gemma4-it:e2b
```

> ⚠️ **Gemma 4 E2B and E4B have no open kernels.** Those tags dispatch
> entirely to the closed `gemma4e_npu` engine -- per-layer embeddings and
> cross-layer KV sharing are not built here. The **12B** text decoder runs on the
> open kernels (two dense geometries); see the
> [support-status matrix](/docs/models/#open-vs-closed-support-status).

🖼️ **Note:** In CLI mode, attach an image with:

```shell
/input "file/to/image.jpg" describe this image.
```

🗣️ **Note:** In CLI mode, attach an audio with:

```shell
/input "file/to/audio.mp3" summarize this audio.
```

📝 **Note:** 

- In server mode, Gemma 4 supports multimodal input with text, images, and audio. See the [OpenAI API multimodal example](https://openflowlm.com/docs/instructions/server/openapi/#%EF%B8%8F-example-multi-modal-input). 

- Change the visual token budget for images with the `image-max-tokens` parameter for different tasks. For more details, see the [Open WebUI custom parameters example](https://openflowlm.com/docs/instructions/server/webui/#%EF%B8%8F-example-add-oflm-custom-parameters).

---

## 🧩 Model Card: [gemma-4-E4B-it](https://huggingface.co/google/gemma-4-E4B-it)

- **Type:** Any-to-Text
- **Think:** Yes
- **Tool Calling Support:** Yes  
- **Base Model:** [google/gemma-4-E4B-it](https://huggingface.co/google/gemma-4-E4B-it)
- **Quantization:** Q4_1
- **Max Context Length:** 128k tokens  
- **Default Context Length:** 32k tokens ([change default](https://openflowlm.com/docs/instructions/cli/#-change-default-context-length-max))  
- **[Set Context Length at Launch](https://openflowlm.com/docs/instructions/cli/#-set-context-length-at-launch)**

▶️ Run with OpenFlowLM in PowerShell:  

```shell
oflm run gemma4-it:e4b
```

🖼️ **Note:** In CLI mode, attach an image with:

```shell
/input "file/to/image.jpg" describe this image.
```

🗣️ **Note:** In CLI mode, attach an audio with:

```shell
/input "file/to/audio.mp3" summarize this audio.
```

📝 **Note:** 

- In server mode, Gemma 4 supports multimodal input with text, images, and audio. See the [OpenAI API multimodal example](https://openflowlm.com/docs/instructions/server/openapi/#%EF%B8%8F-example-multi-modal-input). 

- Change the visual token budget for images with the `image-max-tokens` parameter for different tasks. For more details, see the [Open WebUI custom parameters example](https://openflowlm.com/docs/instructions/server/webui/#%EF%B8%8F-example-add-oflm-custom-parameters).

---

## 🧩 Model Card: [gemma-4-12B-it](https://huggingface.co/google/gemma-4-12B-it-qat-q4_0-unquantized)

- **Type:** Any-to-Text
- **Think:** Yes
- **Tool Calling Support:** Yes  
- **Base Model:** [google/gemma-4-12B-it](https://huggingface.co/google/gemma-4-12B-it-qat-q4_0-unquantized)
- **Quantization:** Q4_0
- **Max Context Length:** 128k tokens  
- **Default Context Length:** 32k tokens ([change default](https://openflowlm.com/docs/instructions/cli/#-change-default-context-length-max))  
- **[Set Context Length at Launch](https://openflowlm.com/docs/instructions/cli/#-set-context-length-at-launch)**

▶️ Run with OpenFlowLM in PowerShell:  

```shell
oflm run gemma4-it:12b
```

> ⚠️ **The 12B's text decoder runs on the open kernels** (source-built in this
> repository, two dense geometries). **Vision and audio do not** — an image or
> audio attach selects the closed `gemma4_12b_npu` engine, which is what
> upstream ships. Nothing here builds the vision tower from source.

🖼️ **Note:** In CLI mode, attach an image with:

```shell
/input "file/to/image.jpg" describe this image.
```

🗣️ **Note:** In CLI mode, attach an audio with:

```shell
/input "file/to/audio.mp3" summarize this audio.
```

📝 **Note:** 

- In server mode, Gemma 4 supports multimodal input with text, images, and audio. See the [OpenAI API multimodal example](https://openflowlm.com/docs/instructions/server/openapi/#%EF%B8%8F-example-multi-modal-input). 

- Change the visual token budget for images with the `image-max-tokens` parameter for different tasks. For more details, see the [Open WebUI custom parameters example](https://openflowlm.com/docs/instructions/server/webui/#%EF%B8%8F-example-add-oflm-custom-parameters).
