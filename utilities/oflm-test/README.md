# oflm-test

A comprehensive testing framework *intended* for  **[OpenFlowLM (OFLM)](https://openflowlm.com)** that validates the functionality of various AI model categories including Language, Embedding, Audio, and Vision models.

## Overview

oflm-test is designed to thoroughly test OpenFlowLM's API compatibility and model functionality across multiple modalities:

- **LLM Tests**: Language model inference in both streaming and non-streaming modes, checked for the things a client depends on: non-empty content, a `finish_reason`, the right model name, sane `usage` counts and context carried across turns
- **Embedding Tests**: Text embedding validation (structure, determinism, batching, dimensionality, semantic ordering, model identity, task prompts) with automated check verdicts. Runs exclusively on a server loaded with only an embed model (`oflm serve -e 1`, or `oflm serve <llm> --embed 1 --embeddingmodel <tag>` for a tag served by the `open_npue` backend).
- **Audio Tests**: Audio understanding via chat completions, with a bundled music clip
- **Vision Tests**: Vision-Language Model (VLM) tests with multi-image support and automated response checking
- **Tool Calling Tests**: Function/tool-calling across seven escalating complexity levels, in streaming and non-streaming modes
- **Server Conformance Tests**: What the OpenAI-compatible API owes every client whatever the weights say — error status codes, model identity, `finish_reason`, per-request isolation and stream/non-stream parity

All test media is **bundled inside the package**, so no extra downloads or local paths are needed once installed.

Each test suite automatically:
- Detects the OFLM server version
- Fetches available models
- Runs standardized test prompts against the bundled media
- Applies pass/fail response checks where applicable
- Saves results to CSV with timestamps
- Handles errors gracefully with detailed logging
- Prints a summary of its verdicts, and fails the run if any check failed

## Prerequisites

- **OpenFlowLM server** running locally or remotely
- **`uv` or `pip`** (Python package manager)

## Quick Start

### 1. Install the package

```bash
uv pip install git+https://github.com/Atomic-Germ/oflm-test.git
```
```bash
pip install git+https://github.com/Atomic-Germ/oflm-test.git
```

Or as an isolated tool with `uv`:

```bash
uv tool install git+https://github.com/Atomic-Germ/oflm-test.git
```

> Note: PyPI hosting is planned; until then, install directly from GitHub.

### 2. Start OFLM Server

Ensure your OpenFlowLM server is running before running tests. Start the server with appropriate flags based on the tests you plan to run:

**Basic local server:**
```bash
oflm serve
```

**Load embedding models (required for embedding tests):**
```bash
oflm serve -e 1
```

### 3. Run Tests

Run tests with:

```bash
# Run all tests (embedding suite excluded — see below)
oflm-test --all

# Run specific tests
oflm-test --llm                    # LLM tests only
oflm-test --embedding              # Embedding tests only
oflm-test --audio                  # Audio tests only
oflm-test --vision                 # Vision tests only
oflm-test --tools                  # Tool-calling tests only
oflm-test --api                    # Server conformance tests only

# Target a specific model (instead of all available models)
oflm-test --llm --model gemma3:4b
oflm-test --vision --model gemma3:4b qwen3vl-it:4b   # space-separated list
oflm-test --audio --model whisper-v3:turbo

# Configuration
oflm-test --llm --port 56354       # Set a custom port for LFM
oflm-test --llm --gen-lim 32       # Limit LLM output to 32 tokens
oflm-test --vision --temp 0.7      # Set sampling temperature (all chat-based tests; defaults to 0.3, a common tool-calling setting)
oflm-test --tools --reasoning high # Set reasoning effort for all chat-based tests
oflm-test --all --exit-zero        # Always exit 0, whatever the checks decided
```

### The Run Can Fail

When the run ends, `oflm-test` prints a summary block — what each suite decided, then every hard failure by name — and exits **1** if any suite recorded a FAIL or an ERROR:

```
=== Summary ===
  llm        FAIL 1, PASS 5
  api        PASS 9, SOFT-FAIL 1
    llm: gemma3:4b / Non-Stream / Teach me Maxwell's equations.: FAIL (empty response)

1 hard failure(s). SOFT-FAIL is noted but does not fail the run.
```

This is what makes the tool a test rather than a report. Before it, a FAIL landed in a CSV cell and waited for a human to open the file, so nothing — a script, a CI job, a PR check — could gate on the result. SOFT-FAIL appears in the summary but does not fail the run, which is what SOFT-FAIL already means everywhere else in the suite.

`--exit-zero` restores the old always-0 behaviour for anything already scripted against it.

### Embedding Test Exclusivity

The embedding suite is **mutually exclusive** with every other suite. It assumes the OFLM server was started with *only* an embed model loaded (`oflm serve -e 1`), so it must never run alongside the chat-based suites — otherwise a full model would have to be loaded just to run embeddings.

`--all` runs `--llm --audio --vision --tools --api` and intentionally excludes the embedding suite. Passing `--embedding` together with any other suite (or with `--all`) runs only the embedding suite, with a warning:

```bash
oflm-test --all                      # all suites EXCEPT embedding
oflm-test --embedding                # embeddings only
oflm-test --llm --embedding          # embeddings only (mutually exclusive warning)
```

### Reasoning Control

Some models are trained to reason ("think") before answering and perform noticeably better with it enabled — especially on tool-calling tasks. OFLM's OpenAI-compatible API exposes this via `reasoning_effort`, which oflm-test forwards with every chat request when requested:

```bash
oflm-test --llm --reasoning high    # deep thinking enabled
oflm-test --tools --reasoning low   # light thinking enabled
oflm-test --tools --reasoning none  # thinking explicitly disabled
```

| Value | Effect |
|-------|--------|
| *(flag omitted)* | Nothing is sent; each model keeps its own default behaviour |
| `none` | Thinking disabled for models that support it |
| `low` / `medium` / `high` | Thinking enabled with increasing effort |

Note that reasoning consumes completion tokens from the same budget as `--gen-lim`, so very small limits may cut thinking short before any answer text is produced.

## Test Types

### LLM Tests
Tests language models over two-round conversations, streamed and not, with the same automated verdicts the other suites use. The checks are deliberately not about answer quality: they are what the server owes a client whatever the weights say.

**What it tests:**
- Non-streaming mode: Single API calls with standard responses
- Streaming mode: Continuous token-by-token responses
- Non-empty content, and a `finish_reason` so a client can tell a complete answer from a truncated one
- The response names the model that was asked for
- `usage` token counts that are positive and add up (prompt + completion = total)
- Multi-turn conversations: Context preservation across exchanges, probed with a reference code only the earlier turn can supply
- Reasoning content extraction (if supported by model)

**Test Flow:**

**Non-stream** test:
  1. Initial prompt: "Teach me Maxwell's equations."
  2. Follow-up: "Summarize your answer."

**Stream** test:
  1. Initial prompt: "Teach me Maxwell's equations."
  2. Follow-up: "Explain why they are important."

**Context retention** probe (non-streamed):
  1. Initial prompt: "Remember this reference code exactly: QX-7731-ZB. Reply with just the code and nothing else."
  2. Follow-up: "What was the reference code I gave you? Reply with just the code."

**Automated Checks:**
| Check | Verdict | Description |
|-------|---------|-------------|
| Each round | PASS / SOFT-FAIL / FAIL | Content is non-empty, a `finish_reason` is present, and `usage` counts are positive and self-consistent. SOFT-FAIL when `usage` is incomplete, or when the response names a different model than the one requested, or names none at all |
| Context retention | PASS / SOFT-FAIL / FAIL | Round two repeats the code from round one; SOFT-FAIL if the model never echoed the code in round one, which makes the probe inconclusive rather than red |

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
- Model identity: the server serves the model it was asked for, or refuses (E9)
- Task prompts: the prompt named in the request reaches the model instead of every text being embedded under whichever task the server picked (E10), and a prompt name the server cannot resolve is refused rather than silently replaced with the default (E11)

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
| E8 Reference Agreement | PASS / FAIL / SKIP | A corpus of 8 texts is compared against bundled reference vectors from the validated numpy implementation of the official google/embeddinggemma-300m pipeline (worst cosine ≥ 0.999), pinning the API path to a known-good implementation. **SKIP** for any model the bundled vectors are not for — see below |
| E9 Model Identity | PASS / SOFT-FAIL / FAIL | A request naming a model the server cannot have loaded must be refused, not answered; and an accepted request must report the model that was asked for |
| E10 Task Prompt Honoured | PASS / SOFT-FAIL / FAIL / SKIP | The same text under `search_query` and under `search_document` must not embed identically; SOFT-FAIL when two draws of one prompt differ as much as the two prompts do, so the run cannot tell the prompt apart from noise. **SKIP** for a model with no task concept — see below |
| E11 Unknown Task Prompt | PASS / SOFT-FAIL / FAIL | `prompt_name: not_a_task` is refused with an OpenAI-shaped error; SOFT-FAIL when the refusal arrives inside an HTTP 200 envelope, FAIL when a vector comes back under some other task |

#### E8 covers one model, and says so

The bundled reference vectors are for `google/embeddinggemma-300m`. Two models
embed the same text into *different spaces* by design, so a cosine between them
carries no information about either — comparing another model's output against
these vectors is not a weaker test, it is a meaningless one. E8 therefore
reports **SKIP** for any model outside `EmbeddingTask.REFERENCE_MODELS`, rather
than a failure it cannot substantiate. To enable it for another model, add its
oracle vectors and its tag to that set.

#### Why E9 exists

Every other check in this suite passes on an embedding for the **wrong model**.
A substituted vector is correctly shaped, correctly normed, deterministic,
batch-consistent and semantically sensible, so E1–E7 all go green on it. E8
makes it worse rather than better: if the model that was substituted *in* is the
one the bundled reference was made *from*, E8 passes too, and the entire suite
reports success on an answer for a model nobody asked for.

That is not hypothetical. A server in this tree ignored the `model` field and
answered every request from whichever model it had loaded, echoing the requested
tag back so the response looked correct. E9 is the check that catches it: it
asks for a tag no server can have loaded and requires a refusal.

#### Why E10 exists

The same shape of defect, one field along. A handler in this tree passed the
query task to the model no matter which task the request named, so every
document was embedded as a query. The vector that came back was correctly
shaped, correctly normed, deterministic and semantically sensible, so E1–E9 all
went green on it, and nothing downstream could tell either: a document vector
and a query vector are the same size and the same kind of thing. The only
visible difference is against the same text under the other prompt, which is all
E10 measures. It needs no oracle vectors, because the assertion is that the
field does *something*, not that it produces any particular number.

Models with no task concept — the BERT-family encoders served by `open_npue` —
refuse a task prompt outright. That refusal is correct behaviour, so E10 reports
**SKIP** for them rather than a failure. E11 is the quieter version of the same
defect: a caller who names a task the server cannot resolve must be told so,
because a default substituted in its place leaves nothing in the response to
say the request was not honoured.

For E7 the CSV also carries one row per draw (`E7 … (draw N/30)`) with the **full raw 768-dim vector** in the Vector Preview column and its cosine to the batch reference, so embeddings can be diffed directly across builds. Only E7's rows carry full vectors; other checks keep the compact preview. E8's reference vectors ship with the package in `oflm_test/test_files/embedding_reference.json`.

Because this suite is exclusive, only an embedding model is loaded on the server (`oflm serve -e 1`) — no full model is required.

**Output:** `embedding_results_v{version}_{timestamp}.csv`

### Tool Calling Tests
Tests OpenAI-compatible function/tool-calling across seven escalating complexity levels. Each level runs in both **non-streaming** and **streaming** mode.

**What it tests:**
- Emitting well-formed tool calls with valid JSON arguments (streamed and non-streamed)
- Inferring argument values from indirect references and resisting decoy tools
- Restraint: not calling tools when none are needed (negative control)
- Parallel tool calls for multiple independent requests in one turn
- The full tool loop: call → locally executed result → final answer grounded in the result
- Tool result fidelity: a value in the last few tokens of a tool result comes back verbatim (catches prompt trimming that eats the tail of the result)
- Schema robustness: a tool whose parameter uses a JSON-schema type array (`["string", "null"]`) still renders and gets called

| Level | Name | Scenario |
|-------|------|----------|
| L1 | Basic Tool Call | Current weather in Paris; arguments appear verbatim in the prompt |
| L2 | Argument Extraction | Weather for "the city where the Eiffel Tower stands", with a forecast decoy tool available |
| L3 | Tool Restraint | Capital-of-France question with tools bound; nothing should be called |
| L4 | Parallel Tool Calls | Compare current weather in Paris and Tokyo in one turn |
| L5 | Multi-Turn Tool Loop | Look up widget price via a tool, then compute 3 widgets at 10% discount |
| L6 | Tool Result Fidelity | A completed `get_ticket` call whose result ends in `zz_code: ZQX-7731`; the model must echo that code |
| L7 | Nullable Schema | Search notes with a `folder` parameter typed `["string", "null"]`; the request must succeed and call `search_notes` |

Tool results in L5 are produced by built-in mock implementations (deterministic fake weather/price databases), so no external services are required. Widget price is $20.00, so a correct final answer contains **54** (3 × $20 − 10%).

**Automated Checks:**
| Check | Verdict | Description |
|-------|---------|-------------|
| L1/L2 Tool Call Check | PASS / FAIL | `get_current_weather` called with valid JSON arguments and a location containing the expected city |
| L3 Restraint Check | PASS / SOFT-FAIL / FAIL | No tool calls and a direct answer mentioning Paris; SOFT-FAIL if answered correctly without naming Paris; FAIL if any tool was called or the answer is empty |
| L4 Parallel Check | PASS / SOFT-FAIL / FAIL | At least two calls covering both cities; SOFT-FAIL if only one call was issued |
| L5 Lookup Check | PASS / FAIL | `lookup_item_price` called with item 'widget' |
| L5 Final Answer Check | PASS / FAIL | Final answer reflects the computed total of $54 |
| L6 Fidelity Check | PASS / FAIL | Answer contains `ZQX-7731` and no further tool call was made |
| L7 Nullable Check | PASS / FAIL | `search_notes` called with a query mentioning 'budget'; an HTTP error from the server is a FAIL |

**Output:** `tools_results_v{version}_{timestamp}.csv`

### Server Conformance Tests
Tests what the OpenAI-compatible API owes every client, whatever the weights say: that a refused request is refused with a status the caller can see, that the model asked for is the model that answers, that `finish_reason` reports what actually stopped generation, that one request's settings do not survive into the next, and that both response modes describe the same request the same way.

**What it tests:**
- Refusals: malformed JSON, a missing `messages` field and a `messages` value that is not a list
- A request for a tag no server can have loaded
- Truncated and complete answers, streamed and non-streamed
- A probe repeated either side of an unrelated request that raises the temperature and asks for reasoning
- The same request run both ways and compared

**Automated Checks:**
| Check | Verdict | Description |
|-------|---------|-------------|
| A1 Error Status | PASS / SOFT-FAIL / FAIL | A request that must be refused comes back non-2xx with an OpenAI-shaped error object (`message`, `type`, `code`). SOFT-FAIL when the refusal is right but the body carries no error a client can read; FAIL on a 2xx carrying an error, which never reaches the caller as an error at all |
| A2 Model Identity | PASS / SOFT-FAIL / FAIL / ERROR | `oflm-test-no-such-model:0b` must be refused with `model_not_found`, not answered by whichever model is loaded; and an accepted request must report the model that was asked for |
| A3 Finish Reason | PASS / SOFT-FAIL / FAIL | An answer cut at the token limit must report `length`, and a short complete answer must report `stop`. Both are checked in both modes; a cut answer reporting `stop` is a FAIL because the cut is then invisible to the caller |
| A4 Request Isolation | PASS / SOFT-FAIL / FAIL | The same probe before and after a request carrying `reasoning_effort: high` and a raised temperature must agree on whether reasoning content is present. Two probes run back to back first, so a text difference afterwards is only called a failure on a model that reproduces its own answer |
| A5 Stream Parity | PASS / SOFT-FAIL / FAIL | The same request reports the same `finish_reason` and the same model in both modes, and the streamed run ends with a chunk carrying a `finish_reason` |

#### Why this suite exists

Every defect these checks look for returned a well-formed HTTP 200 with a
plausible body. A model that was never loaded answered under the name of one
that was. An error message travelled inside a success envelope, so no client
read it as an error. An answer stopped at the token limit reported that it had
finished. One caller's reasoning setting stayed switched on for the next
caller's request, and both answers still read fine. None of that is visible to
a check that reads the answer, which is why the model-quality suites never saw
one of them.

So these checks do not read answers. They read the status code, the `model`
field, `finish_reason`, and what an identical request does after an unrelated
one — the parts of a response a client has to trust and cannot verify for
itself. A1 and A2 go through raw HTTP rather than the OpenAI SDK, because there
the status *is* the assertion and the SDK hides it: a 4xx becomes an exception
and a 200 carrying an error body becomes a success.

Without a `--model` filter the suite runs against the first chat model only.
The checks are about the server rather than the weights, so a second model would
make the server swap models on the NPU for no extra coverage.

**Output:** `api_results_v{version}_{timestamp}.csv`

## Understanding Results

Test results are saved as CSV files under timestamped directories:
```
results/{timestamp}/{backend_os}/{test_type}_results_v{oflm_version}.csv
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
| Finish Reason | What the server said stopped generation, or "N/A" |
| Check Result | Verdict with detail, e.g. "PASS: finish_reason 'stop', 812 chars" |

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
| Check | E1–E11 check name |
| Input | The text (or batch/JSON of texts) embedded |
| Embedding Dim | Vector dimensionality (or N/A on error) |
| Vector Preview | First few values plus total length |
| Check Result | Verdict with detail, e.g. "PASS: related avg 0.8241 > unrelated avg 0.1103" |

**Tools Results:**
| Column | Description |
|--------|-------------|
| Model | Model ID/name |
| Complexity Level | L1–L7 scenario name |
| Mode | "Stream" or "Non-Stream" |
| Input | The prompt sent to the model |
| Reasoning Content | Internal reasoning (if available) |
| Output Content | Model's textual response |
| Tool Calls | JSON summary of the tool calls requested by the model (name + parsed arguments), or "None" |
| Check Result | Verdict with detail, e.g. "PASS", "FAIL: no tool call issued" |

**API Results:**
| Column | Description |
|--------|-------------|
| Model | Model ID/name the probes were sent to |
| Check | A1–A5 check name |
| Probe | Which case within the check, e.g. "malformed JSON", "truncated / Stream" |
| HTTP Status | Status the server answered with, or "N/A" for probes made through the SDK |
| Detail | The raw body or the observed metadata the verdict was read from |
| Check Result | Verdict with detail, e.g. "PASS: refused with HTTP 400 model_not_found" |

### Interpreting Results

- **N/A**: Feature/check not applicable to that row
- **PASS**: Response satisfied the check
- **FAIL**: Hard requirement not met; fails the run
- **SOFT-FAIL**: Noted for review only; not counted as a hard failure and does not fail the run
- **ERROR: {message}**: Test failed with specific error; fails the run
- **SKIP / SKIPPED**: The check does not apply to this model, or the round was skipped after an earlier failure
- **Empty content**: Model timeout or connection issue
