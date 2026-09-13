# The OpenAI-compatible server

Everything a client can see on the wire: the HTTP status, the `model` field, `finish_reason`,
the shape of an error body, and the embeddings task prompt. The code is `src/server/` --
`server.cpp`'s responder lambda, `rest_handler.cpp`'s handlers, `openai_compat.hpp`'s wire
vocabulary and `streaming_ostream_openai.hpp`'s chunks. None of it is about the weights: every
requirement here holds for whichever model happens to be loaded, closed or open kernels alike.

Directory name gives the prefix: `SERVER`.

These landed as fixes in [#52](https://github.com/Cyronius/OpenFlowLM-Next/pull/52) and
[#59](https://github.com/Cyronius/OpenFlowLM-Next/pull/59) with no requirement IDs at all, which
is why this file exists. The background for each is in the code comment next to the fix, and the
defect-by-defect account is in `.claude/plans/oflm-test-server-conformance.md` (local, gitignored).

Every one of those defects has the same shape: a wrong answer that is **well formed**. HTTP 200
with an error body parses. A truncated answer reads like a complete one. A vector embedded under
the wrong task prompt is correctly shaped, correctly normed and deterministic. A smoke test that
asks "did the server answer?" passes through all of them, which is why the criteria below assert
the metadata rather than the content.

The decisions themselves are pure functions -- `status_for`, `finish_reason`, `preflight`,
`resolve_task`, `task_policy` -- and they already have unit tests in
`src/server/openai_compat_test.cpp`, which need no device, no weights and no network. The
requirements here are the other half: that the server actually puts those decisions on the wire,
in both response modes, on every endpoint.

The operator-facing version of these checks is `oflm-test --api` (checks A1 to A5) and
`oflm-test --embedding` (checks E10 and E11), which report a verdict per check and fail the run.
The tests under `tests/` assert the same behaviour a second time, deliberately: they import
nothing but the Python standard library, so a bare checkout can run them against a running server
without installing the `oflm-test` package first.

## Requirements

### SERVER-ERROR-STATUS: an error body never comes back with a 2xx status
**Applies to:** openflowlm-next (`src/server/server.cpp`, `src/server/openai_compat.hpp`)
**Test category:** integration (through `oflm serve`)
**Test:** `specs/server-api/tests/test_error_status.py`

The responder lambda used to read `error.code` as an int to pick the HTTP status. In the OpenAI
error shape that field is a string -- `"model_not_found"`, `"invalid_value"` -- so the read threw
`type must be number, but is string` out of the lambda, the outer handler turned the exception
into `{"error": "<exception text>"}`, and it went out with the 200 the lambda had already set.
Every OpenAI-shaped error the server built was invisible: clients saw a success carrying an
`error` key, and the SDKs, which look at the status first, saw nothing wrong at all.

The server shall give every response that carries a top-level `error` a status of 400 or above.
`openai_compat::status_for()` is the whole rule and it accepts both spellings of `code`: a numeric
one in the 400-599 range is taken as the status, and otherwise `type` decides
(`invalid_request_error` is 400, `authentication_error` 401, `permission_error` 403,
`not_found_error` 404, `rate_limit_error` 429). An error object it cannot classify is 500, and so
is the flat `{"error": "<text>"}` shape that the handlers' own catch blocks still emit, because
200 is the one answer that is certainly wrong.

**Acceptance criteria:**
- A chat request naming a model the server cannot load returns HTTP 400 and a body whose `error`
  is an object with `message`, `type`, `param` and `code`.
- That `code` is the JSON string `"model_not_found"`. A number there is the defect, not a variant
  spelling: the string is what the OpenAI shape says and what every client library expects.
- No response carrying a top-level `error` has a 2xx status, over at least these probes: an
  unknown model on `/v1/chat/completions`, the same on `/v1/completions`, an explicit
  `"model": ""`, and the `"model-faker"` sentinel.
- A request body that is not JSON is refused with a non-2xx status, and the next well-formed
  request is answered normally. Reporting the offending bytes back used to throw a second time
  inside the catch block, which left the NPU lock held and hung every later request, valid ones
  included; `safe_dump()` substitutes U+FFFD rather than throwing, so the error path cannot be
  broken by the input that made it run.

### SERVER-MODEL-IDENTITY: a model the server cannot load is refused, never substituted
**Applies to:** openflowlm-next (`src/server/rest_handler.cpp`, `src/server/openai_compat.hpp`)
**Test category:** integration (through `oflm serve`)
**Test:** `specs/server-api/tests/test_error_status.py`,
`specs/server-api/tests/test_embed_task_prompt.py` (the embeddings half)

`get_auto_model()` returned a Llama-3.2-1B engine for any tag it did not recognise, so a request
for a model this build has never heard of was answered fluently by a different model -- HTTP 200,
with the client's own requested tag echoed back in `model`, so a client comparing the response to
its request saw agreement. The embeddings endpoint had the same defect with the vectors: started
with `bge-base:en-v1.5` loaded, a request for `gte-multilingual:base` returned bge-base's vectors
byte for byte under the name it had been asked for.

The server shall refuse a request whose `model` it cannot serve, with 400 and `model_not_found`,
and shall resolve the tag before unloading anything, so a typo in a client's model field does not
evict the model that is serving everyone else. An accepted request reports the model it was asked
for. The three client mistakes -- unknown tag, not a chat model, no model loaded at all -- all
answer 400; a model that is known but fails to load is ours and answers 500.

**Acceptance criteria:**
- `"model": "oflm-test-no-such-model:0b"` on `/v1/chat/completions` returns 400 with
  `error.code == "model_not_found"`, and the body carries no `choices`.
- The same request with `stream: true` is refused the same way: a JSON error body with a 400
  status, not an SSE stream.
- An explicit `"model": ""` and the `"model-faker"` sentinel are refused with `model_not_found`
  rather than served by whatever is loaded; an *omitted* `model` field is served by the loaded
  model, which is how every client that names no model works.
- An accepted request's response `model` equals the tag the request asked for.
- After a refused request, a request for the served model is answered normally -- the refusal
  unloaded nothing.
- On `/v1/embeddings`, a request naming any model other than the loaded one returns 400
  `model_not_found` and a message naming what *is* loaded, rather than the loaded model's vectors
  under the requested name.

### SERVER-FINISH-REASON: finish_reason reports what actually stopped generation
**Applies to:** openflowlm-next (`src/server/rest_handler.cpp`, `src/server/openai_compat.hpp`, `src/server/streaming_ostream_openai.hpp`)
**Test category:** integration (through `oflm serve`)
**Test:** `specs/server-api/tests/test_finish_reason.py`

The engine computes `meta_info.stop_reason` and the handler dropped it, hardcoding `"stop"` in
the response. An answer cut off at `max_tokens` was therefore indistinguishable from one that had
finished -- the failure mode is an agent that reads a half-written tool call, or a summariser that
silently truncates, with nothing in the response to say so.

The server shall map the engine's stop reason into OpenAI's `finish_reason` vocabulary and report
it in both response modes: `"length"` when generation hit the token limit, `"tool_calls"` when it
stopped on a tool call, `"stop"` otherwise. The mapping is one function, `openai_compat::finish_reason()`,
because the engine's own vocabulary is wider than OpenAI's -- it also yields `"cancel"`, `"error"`
and `"UNKNOWN"`, none of which are values the OpenAI schema has, and a streamed cancellation used
to go out as `"cancel"`.

**Acceptance criteria:**
- A request with `max_tokens: 8` on a prompt that cannot be answered in 8 tokens reports
  `finish_reason: "length"`, non-streaming and streaming alike.
- A request that finishes on its own inside a generous `max_tokens` reports `"stop"`.
- `finish_reason` is always one of OpenAI's five values (`stop`, `length`, `tool_calls`,
  `content_filter`, `function_call`); the engine's `cancel`, `error` and `UNKNOWN` never reach a
  client.

### SERVER-PARAM-ISOLATION: a request gets the model's defaults for the fields it omits
**Applies to:** openflowlm-next (`src/server/rest_handler.cpp`)
**Test category:** integration (through `oflm serve`)
**Test:** `specs/tool-calling/tests/test_request_params_reset.py`

The engine-side requirement is `TOOLS-REQUEST-PARAMS-RESET` in `specs/tool-calling/spec.md`,
which states the rule and owns the list of affected settings; this one exists only so the
server's API surface is covered under the `SERVER` prefix too, and states the client-observable
half. Read the other one for the mechanism. The behaviour is one server serving many clients:
one client sending `reasoning_effort: "high"` turned thinking on for everybody until the next
request that happened to set it back.

**Acceptance criteria:**
- Two identical requests that omit `reasoning_effort` get the same answer regardless of what a
  request between them set it to.
- The same holds for `temperature`: a request that omits it samples at the model's load-time
  default, not at the `0` a previous request asked for.

### SERVER-EMBED-TASK-PROMPT: the request's task prompt reaches the model, or the request is refused
**Applies to:** openflowlm-next (`src/server/rest_handler.cpp`, `src/server/openai_compat.hpp`)
**Test category:** integration (needs an embedding server: `oflm serve -e 1`, or
`oflm serve <chat model> --embed 1 --embeddingmodel <tag>`)
**Test:** `specs/server-api/tests/test_embed_task_prompt.py`

Models like nomic-embed-text prepend a short per-task prefix to the text before embedding it, and
which prefix is chosen changes the vector materially -- measured on this server, the same text
under `search_query` against `search_document` is cosine 0.914, not 1. The handler passed
`task_query` unconditionally and ignored the request, so every *document* in a RAG index was
embedded as a *query*. Nothing downstream could tell: the vector is correctly shaped, correctly
normed and deterministic either way. Retrieval just gets quietly worse.

The server shall read the request's `prompt_name` (or its alias `task_type`), map it onto the
model's own prompt table, and pass the result to the engine. Where it cannot do that it shall
refuse, for the same reason: a substituted task prompt is indistinguishable from the right one.
Which refusal depends on what the model offers, and there are three distinct cases, because an
empty prompt table means two different things -- a model with no task concept at all (the BERT
family: bge, all-minilm, gte-multilingual) and one whose prefixes are hardcoded rather than
declared (OpenGemma), which is why `supports_task_prompts()` is asked separately from
`prompt_names()`.

The names the REST API accepts are its own vocabulary, not the container's: a container declares
names like `Retrieval-query`, and `openai_compat::task_names()` maps the REST spellings onto
them. An error message must quote the REST names, because quoting the container's sent clients to
values the validator then rejected.

**Acceptance criteria:**
- A model that declares prompt names, sent a request with neither `prompt_name` nor `task_type`:
  400 with `error.code == "missing_required_parameter"`, `param` `prompt_name`, and a message
  listing the accepted names. It does not pick one.
- A model with no task concept, sent a `prompt_name`: 400 with `error.code == "invalid_value"`.
  It does not embed the text unprefixed and answer 200.
- An unknown name such as `"not_a_task"`: 400 `invalid_value`, and on a model that has prompts
  the message quotes the offending value and lists the accepted ones (`search_query`,
  `search_document` among them). A non-string value is refused the same way.
- A model that honours prompts: the same text under `search_query` and under `search_document`
  comes back as two materially different vectors -- never identical, and cosine well under
  0.9999 (0.914 on nomic-embed-text) -- while each is bit-for-bit reproducible under its own
  prompt.
- `task_type` is an accepted alias: it alone produces the same vector as the same value sent as
  `prompt_name`. Sending both is fine when they name the same task; when they disagree the
  request is refused with 400 `invalid_value` naming both fields, rather than resolved by
  precedence.
- A model that has prompts but none serving the requested task refuses with 400 `invalid_value`
  carrying the engine's own message, rather than reaching the handler's catch block and going out
  as a 200.

### SERVER-STREAM-PARITY: both response modes report the same metadata
**Applies to:** openflowlm-next (`src/server/rest_handler.cpp`, `src/server/streaming_ostream_openai.hpp`)
**Test category:** integration (through `oflm serve`)
**Test:** `specs/server-api/tests/test_finish_reason.py`

Streaming and non-streaming are two code paths that build their responses independently, and
every metadata fix so far has had to be made twice. `finish_reason` is the example: the
non-streaming responder mapped it and the streaming one did not. A check that exercises one mode
proves nothing about the other, so this requirement says they agree, and the tests run the same
request both ways.

**Acceptance criteria:**
- The same request answered in both modes reports the same `model` and the same `finish_reason`.
- The stream ends with a chunk that carries a non-null `finish_reason` in `choices[0]`, followed
  by `data: [DONE]`. Content chunks before it carry `finish_reason: null`.
- A request refused before generation starts (an unknown model, say) is refused identically in
  both modes: a JSON error body with a non-2xx status, not an SSE stream that opens and then
  stops.
