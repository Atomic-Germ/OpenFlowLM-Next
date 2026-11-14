---
layout: docs
title: whisper
nav_order: 7
parent: Models
---

## 🧩 Model Card: [whisper-large-v3-turbo](https://huggingface.co/openai/whisper-large-v3-turbo)

- **Type:** Speech-to-Text (ASR: Automatic Speech Recognition)
- **Think:** No
- **Tool Calling Support:** No
- **Base Model:** [openai/whisper-large-v3-turbo](https://huggingface.co/openai/whisper-large-v3-turbo)
- **Quantization:** Q4_1
- **Max Output Length:** 448 tokens (fixed)
- **Maximum Encoder Window:** 30 seconds of audio (1500 frames)

> ⚠️ **From this repo, the open engine is the default Whisper.**
> `oflm add <whisper GGUF repo>` builds the open container from the base
> checkpoint the README names and installs the tag **`whisper-v3:turbo`** on the
> open engine. The closed engine is still available (`OFLM_WHISPER_ENGINE=closed`
> or an upstream `model.q4nx` container), but it is the fallback, not the
> product.

## Installing the open Whisper build

```shell
oflm add oxide-lab/whisper-large-v3-turbo-GGUF
```

One command. The installer:

1. Reads the GGUF header to get the architecture (`whisper`) and from it the
   family.
2. Detects the llm.cpp tensor layout (`enc.conv.*`, `dec.*`) — used when the
   open engine can read the GGUF directly; otherwise it builds
   `model.open.safetensors` + `weights_manifest.json` from the base HF
   checkpoint the README names (e.g. `openai/whisper-large-v3-turbo`) by the
   bundled `q4nx-build --open-whisper` — the same builder a user can invoke
   by hand:
   ```shell
   q4nx-build --open-whisper -i openai/whisper-large-v3-turbo -o Whisper-V3-Turbo-OpenNPU2
   oflm add Whisper-V3-Turbo-OpenNPU2 --name whisper-open:turbo
   ```
3. Registers the tag on the open engine and links the shipped `whisper_gemm`
   kernel set beside the model.

Then run anything with `--asr 1`; the default ASR model is `whisper-v3:turbo`,
and it is the open engine:

```shell
oflm run gemma3:4b --asr 1
```

The closed engine (`whisper_npu` from the shipped container, upstream-
only) still exists. It is a fallback away, not a path we maintain — if a
whisper repo is faster from it than from the open path, prefer the open path
and move on.

▶️ Run with OpenFlowLM in PowerShell:  

> The ASR model must be used with an LLM (loaded concurrently) in CLI Mode.
> The ASR model can be used as an independent ASR model in Server Mode .

### CLI Mode   

Start with ASR enabled: 

Load the ASR model (whisper-v3:turbo) in the background, with concurrent LLM loading (gemma3:4b).
```shell
oflm run gemma3:4b --asr 1 
```
or
```shell
oflm run gemma3:4b -a 1 
```

Then, type (replace `filename.mp3` with your audio file path):
```shell
/input "path\to\audio_sample.mp3" summarize it
```

### Server Mode 

Start with ASR enabled: 

- Load the ASR model (whisper-v3:turbo) in the background, with concurrent LLM loading (gemma3:4b).
```shell
oflm serve gemma3:4b --asr 1 
```
or
```shell
oflm serve gemma3:4b -a 1 
```

- Load the ASR model (whisper-v3:turbo) as a standalone ASR model.
```shell
oflm serve --asr 1 
```
or
```shell
oflm serve -a 1 
```

Send audio to `POST /v1/audio/transcriptions` via any OpenAI Client or Open WebUI.

> see more API details here → [/v1/audio/](https://platform.openai.com/docs/api-reference/audio)

**Example 1**: OpenAI Client

```python
# Import the official OpenAI Python SDK (OpenFlowLM mirrors the OpenAI API schema)
from openai import OpenAI

# Initialize the client to point at your local OpenFlowLM server
# - base_url: OpenFlowLM's local OpenAI-compatible REST endpoint
# - api_key: Dummy token; OpenFlowLM typically doesn't enforce auth, but the client requires a string
client = OpenAI(
    base_url="http://127.0.0.1:52625/v1",  # OpenFlowLM local API endpoint
    api_key="oflm",                         # Placeholder key
)

# Open the audio file in binary mode and create a transcription request
# - model: name of the speech-to-text model exposed by OFLM (e.g., "whisper-v3:turbo")
# - file: file-like object pointing to your audio
with open("audio.mp3", "rb") as f:
    resp = client.audio.transcriptions.create(
        model="whisper-v3:turbo",
        file=f,
    )

# Print the transcribed text returned by the server
print(resp.text)
```

**Example 2**: Open WebUI

- Follow Open WebUI setup [guide](https://openflowlm.com/docs/instructions/server/webui/).
- In the bottom-left corner, click User icon, then select Settings.
- In the bottom panel, open Admin Settings.
- In the left sidebar, navigate to Audio.
- Set Speech-to-Text Engine to OpenAI.
- Enter:
> API Base URL: `http://127.0.0.1:52625/v1` (Open WebUI Desktop) or `http://host.docker.internal:52625/v1` (Open WebUI in Docker)   
> API KEY: oflm (any value works)    
> STT Model: whisper-v3:turbo (type the tag; can be a different ASR model)    
- Save the setting.
- You're ready to upload audio files! (Choose an LLM to load and use concurrently)

---
