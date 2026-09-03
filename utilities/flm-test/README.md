# flm-test

A comprehensive testing framework *intended* for  **[FastFlowLM (FLM)](https://fastflowlm.com)** that validates the functionality of various AI model categories including Language, Embedding, Audio, and Vision models.

## Overview

flm-test is designed to thoroughly test FastFlowLM's API compatibility and model functionality across multiple modalities:

- **LLM Tests**: Language model inference with both streaming and non-streaming modes
- **Embedding Tests**: Text embedding validation (structure, determinism, batching, dimensionality, semantic ordering) with automated check verdicts. Runs exclusively on a server loaded with only an embed model (`flm serve -e 1`).
- **Audio Tests**: Audio understanding via chat completions, with a bundled music clip
- **Vision Tests**: Vision-Language Model (VLM) tests with multi-image support and automated response checking
- **Tool Calling Tests**: Function/tool-calling across five escalating complexity levels, in streaming and non-streaming modes

All test media is **bundled inside the package**, so no extra downloads or local paths are needed once installed.

Each test suite automatically:
- Detects the FLM server version
- Fetches available models
- Runs standardized test prompts against the bundled media
- Applies pass/fail response checks where applicable
- Saves results to CSV with timestamps
- Handles errors gracefully with detailed logging

## Prerequisites

- **FastFlowLM server** running locally or remotely
- **`uv` or `pip`** (Python package manager)

## Quick Start

### 1. Install the package

```bash
uv pip install git+https://github.com/Atomic-Germ/flm-test.git
```
```bash
pip install git+https://github.com/Atomic-Germ/flm-test.git
```

Or as an isolated tool with `uv`:

```bash
uv tool install git+https://github.com/Atomic-Germ/flm-test.git
```

> Note: PyPI hosting is planned; until then, install directly from GitHub.

### 2. Start FLM Server

Ensure your FastFlowLM server is running before running tests. Start the server with appropriate flags based on the tests you plan to run:

**Basic local server:**
```bash
flm serve
```

**Load embedding models (required for embedding tests):**
```bash
flm serve -e 1
```

### 3. Run Tests

Run tests with:

```bash
# Run all tests (embedding suite excluded — see below)
flm-test --all

# Run specific tests
flm-test --llm                    # LLM tests only
flm-test --embedding              # Embedding tests only
flm-test --audio                  # Audio tests only
flm-test --vision                 # Vision tests only
flm-test --tools                  # Tool-calling tests only

# Target a specific model (instead of all available models)
flm-test --llm --model gemma3:4b
flm-test --vision --model gemma3:4b qwen3vl-it:4b   # space-separated list
flm-test --audio --model whisper-v3:turbo

# Configuration
flm-test --llm --port 56354       # Set a custom port for LFM
flm-test --llm --gen-lim 32       # Limit LLM output to 32 tokens
flm-test --vision --temp 0.7      # Set sampling temperature (all chat-based tests; defaults to 0.3, a common tool-calling setting)
flm-test --tools --reasoning high # Set reasoning effort for all chat-based tests
```

### Embedding Test Exclusivity

The embedding suite is **mutually exclusive** with every other suite. It assumes the FLM server was started with *only* an embed model loaded (`flm serve -e 1`), so it must never run alongside the chat-based suites — otherwise a full model would have to be loaded just to run embeddings.

`--all` runs `--llm --audio --vision --tools` and intentionally excludes the embedding suite. Passing `--embedding` together with any other suite (or with `--all`) runs only the embedding suite, with a warning:

```bash
flm-test --all                      # all suites EXCEPT embedding
flm-test --embedding                # embeddings only
flm-test --llm --embedding          # embeddings only (mutually exclusive warning)
```

### Reasoning Control

Some models are trained to reason ("think") before answering and perform noticeably better with it enabled — especially on tool-calling tasks. FLM's OpenAI-compatible API exposes this via `reasoning_effort`, which flm-test forwards with every chat request when requested:

```bash
flm-test --llm --reasoning high    # deep thinking enabled
flm-test --tools --reasoning low   # light thinking enabled
flm-test --tools --reasoning none  # thinking explicitly disabled
```

| Value | Effect |
|-------|--------|
| *(flag omitted)* | Nothing is sent; each model keeps its own default behaviour |
| `none` | Thinking disabled for models that support it |
| `low` / `medium` / `high` | Thinking enabled with increasing effort |

Note that reasoning consumes completion tokens from the same budget as `--gen-lim`, so very small limits may cut thinking short before any answer text is produced.

## Test Types

### LLM Tests
Tests language models with conversation capabilities.

**What it tests:**
- Non-streaming mode: Single API calls with standard responses
- Streaming mode: Continuous token-by-token responses
- Multi-turn conversations: Context preservation across exchanges
- Reasoning content extraction (if supported by model)

**Test Flow:**

**Non-stream** test:
  1. Initial prompt: "Teach me Maxwell's equations."
  2. Follow-up: "Summarize your answer."

**Stream** test:
  1. Initial prompt: "Teach me Maxwell's equations."
  2. Follow-up: "Explain why they are important."

**Output:** `llm_results_v{version}_{timestamp}.csv`

### Vision Tests
Tests Vision-Language Models (VLMs) with multi-image analysis and objective response validation.

**What it tests:**
- OCR/text extraction from an image
- Multi-image understanding and detailed description generation
- Creative story generation connecting multiple images
- Streaming responses for image-to-text

**Test Flow:**
1. Initial prompt: "Extract text from the first image, describe the second one, and imagine what the spectrogram might sound like."
2. Follow-up: "Make a story that connects the images together."
3. Follow-up: "What kind of sound does the spectrogram represent?"

**Bundled Test Media:**
- `test_files/image/paris.png` - image containing a known English sentence
- `test_files/image/seagull.jpeg` - photograph of a seagull on a lamp post
- `test_files/image/spectrogram.png` - spectrogram of a musical clip

**Automated Checks:**
| Check | Verdict | Description |
|-------|---------|-------------|
| Text Extraction Check | PASS / FAIL | The first-round response must contain the exact sentence shown in `paris.png`, matched case-insensitively with flexible whitespace |
| Seagull Mention Check | PASS / FAIL | A seagull ("seagull", "sea gull", or "gull") must be recognized in the description or the story |
| Spectrogram Music Check | PASS / SOFT-FAIL | Informational only: the response should reference music-related terms (melody, rhythm, instruments, etc.). Failure is noted but does not count as a hard failure |

**Output:** `vision_results_v{version}_{timestamp}.csv`

### Audio Tests
Tests audio-capable models through chat completions using the OpenAI-style `input_audio` content part.

**What it tests:**
- Sending base64-encoded MP3 audio inline in a chat request
- Audio comprehension and description quality
- Multi-turn context preservation after an audio exchange
- Reasoning content extraction (if supported by model)

**Test Flow:**
1. Initial prompt: "Describe what you hear in this audio clip."
2. Follow-up: "What kind of mood or genre would this clip fit into?"

**Bundled Test Media:**
- `test_files/audio/atomic-germ.mp3` - short instrumental music clip (~64 seconds)

**Automated Checks:**
| Check | Verdict | Description |
|-------|---------|-------------|
| Music Mention Check | PASS / SOFT-FAIL | The description should reference music-related terms (melody, rhythm, beat, instrument, etc.). Failure is noted but does not count as a hard failure |

By default, audio tests run against known audio models (currently `whisper-v3:turbo`). An explicit `--model` filter always wins, so any audio-capable model can be targeted directly.

**Output:** `audio_results_v{version}_{timestamp}.csv`

### Embedding Tests
Tests text embedding models through the OpenAI-compatible `embeddings.create` API, with the same automated PASS/SOFT-FAIL/FAIL verdicts used by the other suites.

**What it tests:**
- Well-formed OpenAI embedding responses (object types, non-empty numeric vectors)
- Repeatability: the same input is drawn repeatedly and the count of inconsistent draws is reported, so a single outlier draw cannot flip a build from clean to defective (or back)
- Batch requests: as many embeddings as inputs, returned in order with unique indexes
- Dimensionality: every vector in a batch shares the same, positive dimension
- Semantic quality: related text pairs land closer together in the embedding space than unrelated pairs
- Cross-path consistency: the same input through the single-input and batch delivery paths in the same run
- Batch-reference stability: a larger sample of draws compared per-draw against the batch reference, so intermittent outliers surface even when a small sample misses them; the raw draw vectors are dumped to the CSV for cross-build comparison
- Reference agreement: embeddings matched against bundled oracle vectors from the validated numpy implementation of the official model (E8)

**Automated Checks:**
| Check | Verdict | Description |
|-------|---------|-------------|
| E1 Response Structure | PASS / FAIL | Response is a valid `list` payload containing an `embedding` object with a non-empty numeric vector |
| E2 Repeatability | PASS / SOFT-FAIL / FAIL | `SAMPLE_TEXT` drawn 10×; PASS when every pairwise cosine ≥ 0.999, SOFT-FAIL on a single-device flicker (≤ 25% of draw pairs inconsistent), FAIL when the outlier rate is a property of the build |
| E3 Batch & Index Integrity | PASS / FAIL | N inputs return exactly N embeddings in order with unique indexes |
| E4 Dimensionality | PASS / FAIL | All batch embeddings share the same consistent dimension |
| E5 Semantic Ordering | PASS / FAIL | Mean similarity of related pairs (`cat`/`kitten`, `ocean`/`sea`) exceeds that of unrelated pairs (`cat`/`car`, `ocean`/`desert`) |
| E6 Cross-Path Consistency | PASS / FAIL | The same weights reached via a single-input request and a one-item batch request agree (cosine ≥ 0.999), distinguishing a bad number from a bad machine |
| E7 Batch Reference Consistency | PASS / SOFT-FAIL / FAIL | `SAMPLE_TEXT` drawn 30×, each draw compared to the batch-path reference in the same run; PASS when all agree, SOFT-FAIL on sparse flicker (≤ 25% deviating), FAIL when the outlier rate is a property of the build |
| E8 Reference Agreement | PASS / FAIL | A corpus of 8 texts is compared against bundled reference vectors from the validated numpy implementation of the official google/embeddinggemma-300m pipeline (worst cosine ≥ 0.999), pinning the API path to a known-good implementation |

For E7 the CSV also carries one row per draw (`E7 … (draw N/30)`) with the **full raw 768-dim vector** in the Vector Preview column and its cosine to the batch reference, so embeddings can be diffed directly across builds. Only E7's rows carry full vectors; other checks keep the compact preview. E8's reference vectors ship with the package in `flm_test/test_files/embedding_reference.json`.

Because this suite is exclusive, only an embedding model is loaded on the server (`flm serve -e 1`) — no full model is required.

**Output:** `embedding_results_v{version}_{timestamp}.csv`

### Tool Calling Tests
Tests OpenAI-compatible function/tool-calling across five escalating complexity levels. Each level runs in both **non-streaming** and **streaming** mode.

**What it tests:**
- Emitting well-formed tool calls with valid JSON arguments (streamed and non-streamed)
- Inferring argument values from indirect references and resisting decoy tools
- Restraint: not calling tools when none are needed (negative control)
- Parallel tool calls for multiple independent requests in one turn
- The full tool loop: call → locally executed result → final answer grounded in the result

| Level | Name | Scenario |
|-------|------|----------|
| L1 | Basic Tool Call | Current weather in Paris; arguments appear verbatim in the prompt |
| L2 | Argument Extraction | Weather for "the city where the Eiffel Tower stands", with a forecast decoy tool available |
| L3 | Tool Restraint | Capital-of-France question with tools bound; nothing should be called |
| L4 | Parallel Tool Calls | Compare current weather in Paris and Tokyo in one turn |
| L5 | Multi-Turn Tool Loop | Look up widget price via a tool, then compute 3 widgets at 10% discount |

Tool results in L5 are produced by built-in mock implementations (deterministic fake weather/price databases), so no external services are required. Widget price is $20.00, so a correct final answer contains **54** (3 × $20 − 10%).

**Automated Checks:**
| Check | Verdict | Description |
|-------|---------|-------------|
| L1/L2 Tool Call Check | PASS / FAIL | `get_current_weather` called with valid JSON arguments and a location containing the expected city |
| L3 Restraint Check | PASS / SOFT-FAIL / FAIL | No tool calls and a direct answer mentioning Paris; SOFT-FAIL if answered correctly without naming Paris; FAIL if any tool was called or the answer is empty |
| L4 Parallel Check | PASS / SOFT-FAIL / FAIL | At least two calls covering both cities; SOFT-FAIL if only one call was issued |
| L5 Lookup Check | PASS / FAIL | `lookup_item_price` called with item 'widget' |
| L5 Final Answer Check | PASS / FAIL | Final answer reflects the computed total of $54 |

**Output:** `tools_results_v{version}_{timestamp}.csv`

## Understanding Results

Test results are saved as CSV files under timestamped directories:
```
results/{timestamp}/{backend_os}/{test_type}_results_v{flm_version}.csv
```

**Example filenames:**
- `results/20260821_203124/linux/vision_results_v1.0.1.csv`

### CSV Columns

**LLM Results:**
| Column | Description |
|--------|-------------|
| Model | Model ID/name |
| Mode | "Stream" or "Non-Stream" |
| Input | The prompt sent to the model |
| Reasoning Content | Internal reasoning (if available) |
| Output Content | Model's response |

**Vision Results:**
| Column | Description |
|--------|-------------|
| Model | VLM model ID |
| Input | The prompt sent to the model |
| Reasoning Content | Internal reasoning (if available) |
| Output Content | Model's response |
| Text Extraction Check | PASS / FAIL / ERROR for the paris.png sentence |
| Seagull Mention Check | PASS / FAIL / ERROR per round (description and story) |
| Spectrogram Music Check | PASS / SOFT-FAIL / ERROR / SKIPPED |

**Audio Results:**
| Column | Description |
|--------|-------------|
| Model | Audio model ID |
| Input | The prompt sent to the model |
| Reasoning Content | Internal reasoning (if available) |
| Output Content | Model's response |
| Music Mention Check | PASS / SOFT-FAIL / ERROR |

**Embedding Results:**
| Column | Description |
|--------|-------------|
| Model | Embedding model ID |
| Check | E1–E7 check name |
| Input | The text (or batch/JSON of texts) embedded |
| Embedding Dim | Vector dimensionality (or N/A on error) |
| Vector Preview | First few values plus total length |
| Check Result | Verdict with detail, e.g. "PASS: related avg 0.8241 > unrelated avg 0.1103" |

**Tools Results:**
| Column | Description |
|--------|-------------|
| Model | Model ID/name |
| Complexity Level | L1–L5 scenario name |
| Mode | "Stream" or "Non-Stream" |
| Input | The prompt sent to the model |
| Reasoning Content | Internal reasoning (if available) |
| Output Content | Model's textual response |
| Tool Calls | JSON summary of the tool calls requested by the model (name + parsed arguments), or "None" |
| Check Result | Verdict with detail, e.g. "PASS", "FAIL: no tool call issued" |

### Interpreting Results

- **N/A**: Feature/check not applicable to that row
- **PASS**: Response satisfied the check
- **FAIL**: Hard requirement not met (text extraction, seagull recognition)
- **SOFT-FAIL**: Noted for review only; not counted as a hard failure
- **ERROR: {message}**: Test failed with specific error
- **SKIPPED**: Round skipped due to an earlier failure
- **Empty content**: Model timeout or connection issue
