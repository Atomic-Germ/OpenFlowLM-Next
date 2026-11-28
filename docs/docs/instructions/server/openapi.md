---
title: "REST API Reference"
description: "Endpoints and Python/JavaScript SDK examples for chat, streaming, embeddings, vision, audio, and images."
icon: "api"
weight: 20
---

`oflm serve` exposes OpenAI-compatible (`/v1/*`) and Ollama-compatible (`/api/*`) HTTP endpoints on `http://127.0.0.1:52625`.

## Endpoints Reference

<div class="oflm-two-col">
  <div class="oflm-card">

### OpenAI Endpoints (`/v1/*`)

| Method | Path |
|---|---|
| `GET` | `/v1/models`, `/v1/version` |
| `POST` | `/v1/chat/completions`, `/v1/completions` |
| `POST` | `/v1/embeddings` |
| `POST` | `/v1/audio/transcriptions` |
| `POST` | `/v1/images/generations`, `/v1/images/edits` |

  </div>
  <div class="oflm-card">

### Ollama Endpoints (`/api/*`)

| Method | Path |
|---|---|
| `GET` | `/api/tags`, `/api/ps`, `/api/version`, `/api/npu/status` |
| `POST` | `/api/chat`, `/api/generate`, `/api/embeddings` |
| `POST` | `/api/show`, `/api/pull`, `/api/cancel` |

  </div>
</div>

> [!NOTE]
> `/v1/models` and `/api/tags` omit Whisper and EmbeddingGemma so chat clients do not select speech or embedding encoders as chat models. Co-load them via `--asr 1` and `--embed 1`.

### Request Body Extensions

`/v1/chat/completions` and `/v1/completions` accept standard `model`, `messages` / `prompt`, `stream`, `temperature`, `top_p`, `presence_penalty`, and `frequency_penalty`, plus these fields (pass via `extra_body` in typed SDKs):

| Field | Effect |
|---|---|
| `top_k` / `min_p` / `repetition_penalty` | Additional token sampling cutoffs and penalties |
| `think` | Toggle reasoning output (`true` or `false`) |
| `reasoning_effort` | Reasoning effort (`low`, `medium`, `high`) |
| `image-max-tokens` | Cap visual token budget on supported VLMs (such as Gemma 4) |

### Embeddings (`POST /v1/embeddings`)

`input` is required (string or array of strings); `prompt_name` and `task_type` select named task prompts where supported:

```json
{
  "model": "bge-base:en-v1.5",
  "input": ["first document", "second document"]
}
```

---

## Python & JavaScript SDK Quickstart

Start `oflm serve llama3.2:1b` in one terminal, then connect from Python or Node.js / TypeScript using the official OpenAI SDKs:

<div class="oflm-two-col">
  <div class="oflm-card">

### Python (`pip install openai`)

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:52625/v1", api_key="oflm")

resp = client.chat.completions.create(
    model="llama3.2:1b",
    messages=[
        {"role": "system", "content": "You are a concise assistant."},
        {"role": "user", "content": "Why is the sky blue?"},
    ],
)
print(resp.choices[0].message.content)
```

  </div>
  <div class="oflm-card">

### JavaScript / TypeScript (`npm i openai`)

```javascript
import OpenAI from "openai";

const client = new OpenAI({
  baseURL: "http://127.0.0.1:52625/v1",
  apiKey: "oflm",
});

const resp = await client.chat.completions.create({
  model: "llama3.2:1b",
  messages: [
    { role: "system", content: "You are a concise assistant." },
    { role: "user", content: "Why is the sky blue?" },
  ],
});
console.log(resp.choices[0].message.content);
```

  </div>
</div>

---

## Cookbook Examples

### Multi-Turn Chat & Streaming

The HTTP API is stateless: append assistant replies to `messages` across turns, and pass `stream=True` for real-time token deltas:

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:52625/v1", api_key="oflm")
messages = [
    {"role": "system", "content": "You are a concise assistant."},
    {"role": "user", "content": "Write a 2-sentence sci-fi hook."},
]

first = client.chat.completions.create(model="llama3.2:1b", messages=messages)
messages.append({"role": "assistant", "content": first.choices[0].message.content})
messages.append({"role": "user", "content": "Continue with a twist."})

stream = client.chat.completions.create(
    model="llama3.2:1b",
    messages=messages,
    temperature=0.8,
    top_p=0.95,
    extra_body={"top_k": 40, "repetition_penalty": 1.05},
    stream=True,
)
for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
print()
```

### Document Input

Read local UTF-8 files into user messages for long-context summarization (up to 256k tokens depending on model and RAM):

```python
from pathlib import Path
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:52625/v1", api_key="oflm")
doc = Path("long_prompt.txt").read_text(encoding="utf-8")

resp = client.chat.completions.create(
    model="llama3.2:1b",
    messages=[
        {"role": "system", "content": "Summarize key findings concisely."},
        {"role": "user", "content": doc},
    ],
)
print(resp.choices[0].message.content)
```

### Vision Input (`/v1/chat/completions`)

Encode local `.png` or `.jpg` images as Base64 data URLs for open-runtime VLMs such as `qwen3vl-it:4b` or `qwen3.5:4b` (including `oflm add` GGUFs with `mmproj` weights):

```python
import base64
from pathlib import Path
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:52625/v1", api_key="oflm")
img_b64 = base64.b64encode(Path("image.png").read_bytes()).decode("utf-8")

resp = client.chat.completions.create(
    model="qwen3vl-it:4b",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe this image concisely."},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
            ],
        }
    ],
)
print(resp.choices[0].message.content)
```

*(Pass multiple `image_url` entries in `content` for multi-image comparison.)*

### Audio Transcription (`/v1/audio/transcriptions`)

Start `oflm serve llama3.2:1b --asr 1` to enable speech-to-text with `whisper-v3:turbo`:

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:52625/v1", api_key="oflm")
with open("audio.wav", "rb") as audio_file:
    transcript = client.audio.transcriptions.create(
        model="whisper-v3:turbo",
        file=audio_file,
    )
print(transcript.text)
```

> [!NOTE]
> `whisper-v3:turbo` and inline `input_audio` / legacy Gemma vision (`gemma4-it:e2b`, `gemma4-it:e4b`, `gemma4-it:12b`) currently use prebuilt NPU `.xclbin` files and are unavailable on NixOS until open kernels ship ({{< b k="wip" issue="244" t="Issue #244" >}}, {{< b k="wip" issue="240" t="Issue #240" >}}, {{< b k="wip" issue="73" t="Issue #73" >}}; see [Nix & NixOS](/docs/install/linux/nix/)).

### Image Generation & Editing (FLUX.2 [klein] 4B)

Start `oflm serve llama3.2:1b --imagegen 1` to keep chat and diffusion co-resident (see [Image Generation and Editing](/docs/instructions/server/images/)):

```python
import base64
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:52625/v1", api_key="oflm")

gen = client.images.generate(
    model="flux2-klein:4b",
    prompt="a red fox in fresh snow",
    size="1024x1024",
    extra_body={"seed": 1},
)
with open("fox.png", "wb") as f:
    f.write(base64.b64decode(gen.data[0].b64_json))

with open("fox.png", "rb") as ref:
    edit = client.images.edit(
        model="flux2-klein:4b",
        image=ref,
        prompt="Make it a watercolor painting at sunset",
        size="512x512",
        extra_body={"seed": 7},
    )
with open("fox_watercolor.png", "wb") as f:
    f.write(base64.b64decode(edit.data[0].b64_json))
```
