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

### SERVER-REQUEST-VALIDATION: a malformed request is refused with a 400, and the server keeps serving
**Applies to:** openflowlm-next (`src/server/server.cpp`, `src/server/rest_handler.cpp`, `src/server/openai_compat.hpp`)
**Test category:** integration (through `oflm serve`)
**Test:** `specs/server-api/tests/test_request_validation.py`

Handlers read required fields with `request["field"]` on a `const json&`. For a key that is not
there that is undefined behaviour, and on this build it segfaults: `POST {}` killed the server on
`/api/show`, `/api/generate`, `/v1/completions` and `/v1/embeddings` (#70). A `request_id` that
is not a string threw before the handler ran, while the request held the NPU lock -- the same
leaked lock SERVER-ERROR-STATUS describes for a body that is not JSON, reached by a different
input. The process stayed up and `/v1/models` kept answering, because it does not take the lock,
so checking that a server survives a bad request means sending it a request that needs the NPU.

This extends two existing checks rather than replacing them: SERVER-ERROR-STATUS already covers a
body that is not JSON, and `oflm-test --api` check A1 already sends `/v1/chat/completions` a
request with no `messages` and one with `"messages"` as a string. The tests below repeat those two
probes, as the tests in this directory do for every `oflm-test` check.

The server shall check a request's required fields, and the type of `request_id`, before reading
them, and refuse a missing or mistyped one with 400 and an OpenAI-shaped error naming the field.
The embeddings endpoint shall also refuse a malformed `input` or `model`, and a server with no
embedding model loaded shall refuse an embeddings request rather than answer 200 with no body.

**Acceptance criteria:**
- `POST {}` to `/api/show`, `/api/generate`, `/api/chat`, `/v1/completions`,
  `/v1/chat/completions` and `/v1/embeddings` returns 400 with `error.param` naming the required
  field (`model`, `prompt`, `messages` or `input`), and after each one a chat request is answered.
- A required field of the wrong type (`"prompt": 5`, `"messages": "hi"`) returns 400 with
  `error.code == "invalid_value"`.
- A non-string `request_id` returns 400 with `error.param == "request_id"`, and a chat request
  sent after it is answered rather than left queued. A string `request_id` is accepted.
- On `/v1/embeddings`: `input` that is `null`, a number or an object returns 400; an array with a
  non-string element returns 400 with `param` naming that element (`input[1]`); `"model": null`
  returns 400; `"model": ""` returns 400 `model_not_found`; an omitted `model` is served by the
  loaded model and the response names it. `"input": []` stays a 200 with an empty `data`.
- A server started without an embedding model answers `/v1/embeddings` with 400
  `model_not_found`, not 200.

## The Images API

`POST /v1/images/generations` and `POST /v1/images/edits` are OpenAI's Images API over the open
diffusion engine (`src/open_diffusion`, FLUX.2 [klein] 4B with every op on the NPU; its own
requirements are in `specs/open-diffusion/spec.md`). The handlers are `rest_handler.cpp`'s
`handle_openai_images_*`. The request rules are one pure function, `openai_compat::images_request()`,
unit-tested in `src/server/openai_compat_test.cpp`. The plan these came from is
`archive/images-api.md`.

There is no `/v1/images/variations` (OpenAI no longer documents it), no `url` response format,
no `partial_images` streaming (a preview costs a full VAE decode per step: +35% time), no
A1111 `/sdapi` shim and no ComfyUI `/prompt`.

The integration tests are `specs/server-api/tests/test_images_api.py`: `oflm serve <chat model>`
with `flux2-klein:4b` installed, and the NPU.

### SERVER-IMAGES-GENERATIONS: a request gets images of the size and format it asked for
**Applies to:** openflowlm-next (`src/server/rest_handler.cpp`, `src/server/server.cpp`, `src/include/model_list.hpp`)
**Verification:** test
**Test:** `specs/server-api/tests/test_images_api.py`

`/v1/images/generations` answers `{created, data: [{b64_json, seed}], model, output_format, size}`.
`data` holds `n` images (1-10), made one after another with seeds `seed`, `seed + 1`, ...; each
item's `seed` says which, so a random-seed image can be made again. `output_format` is `png`
(the default) or `jpeg` (quality from `output_compression`, default 90). `webp` is refused as not
implemented: it needs `libwebp`. The response's `model` is the resolved tag.

The `model` field follows SERVER-MODEL-IDENTITY: it is resolved before anything is unloaded, a tag
the build does not have, an explicit `""` and a tag that makes no images are each a 400
`model_not_found`, and the llama3.2:1b fallback never applies. An omitted `model` is `--imagemodel`'s
(default `flux2-klein:4b`). `/v1/models` lists the image model; `/api/tags`, which chat clients
read, does not.

**Acceptance criteria:**
- `{"model": "flux2-klein:4b", "prompt": ..., "size": "512x512", "seed": 1}` returns 200, one
  `data` item whose `b64_json` decodes to a 512x512 PNG, an integer `created`, and
  `model == "flux2-klein:4b"`.
- `output_format: "jpeg"` gives a 512x512 JPEG.
- `n: 2, seed: 41` gives two different images with `seed` 41 and 42, and the second is byte for
  byte what `seed: 42` alone gives.
- The same request twice gives the same `b64_json`.
- `output_format: "webp"` is a 400 with `param == "output_format"` and `code == "not_implemented"`.
- `"model": "oflm-test-no-such-model:0b"`, `"model": ""` and a chat model's tag are 400
  `model_not_found`.
- `/v1/models` lists `flux2-klein:4b` and `/api/tags` does not.

### SERVER-IMAGES-PARAMS: one field under two names means one thing, and a refused control names itself
**Applies to:** openflowlm-next (`src/server/openai_compat.hpp`, `src/server/rest_handler.cpp`)
**Verification:** test
**Test:** `src/server/openai_compat_test.cpp` (`test_images_request`, the `openai_compat` CTest);
`specs/server-api/tests/test_images_api.py` (on the wire)

Clients written for vLLM-Omni send diffusers' names and clients written for Lemonade or A1111 send
A1111's, for the same controls. Each pair is one field: `steps` = `num_inference_steps`,
`cfg_scale` = `guidance_scale`, `sampler` = `sampler_name`. Both spellings in one request are
accepted when they agree and refused (400, `param` naming the second) when they do not: neither
silently wins.

- `steps`: 1-50, default the model's own (4 for klein, which is distilled for it).
- `seed`: a non-negative integer up to 2^64-1; `-1` (A1111's) and omitted mean a random seed.
- `cfg_scale` / `guidance_scale` (a number) and `negative_prompt` (a string) are accepted and
  ignored, with one log line: klein is guidance-distilled and has no CFG, and A1111 clients always
  send `cfg_scale: 7, negative_prompt: ""`.
- `sampler` / `sampler_name`: `euler`, `Euler`, `Euler a`, `flowmatch_euler` and
  `FlowMatchEulerDiscreteScheduler` are the model's flow-match Euler; any other name is a 400 that
  lists those.
- `response_format` must be `b64_json` (`url` is a 400 naming it); `stream: true` and
  `partial_images` > 0 are 400s naming them.
- Types are checked: a wrong one is a 400 with `param` naming the field. A float is not an
  integer, even `2.0`. `null` is the same as leaving the field out.
- Fields this server does not know (`quality`, `style`, `background`, `user`, ...) are ignored.

**Acceptance criteria:**
- `steps: 8, num_inference_steps: 8` is accepted; `steps: 8, num_inference_steps: 4` is a 400
  with `param == "num_inference_steps"`. The same holds for `cfg_scale`/`guidance_scale` (4 and
  4.0 agree) and `sampler`/`sampler_name`.
- `cfg_scale: 7, negative_prompt: ""` is accepted, and on the wire an image made with
  `guidance_scale: 3.5, negative_prompt: "blurry", sampler: "euler"` is byte for byte the plain
  request's.
- `sampler_name: "DPM++ 2M Karras"` is a 400 whose message lists `Euler a`.
- `steps: 2` runs and gives a different image from the default's.
- `n` 0, 11 or 2.0; `steps` 0 or 51; `seed` -2 or 1.5; `stream: true`; `partial_images: 1`;
  `response_format: "url"`; `prompt: 5`: each is a 400 naming its field.
- `seed: 18446744073709551615` is accepted; `seed: -1` is a random seed.

### SERVER-IMAGES-SIZE: only the engine's resolutions run
**Applies to:** openflowlm-next (`src/server/openai_compat.hpp`)
**Verification:** test
**Test:** `src/server/openai_compat_test.cpp`, `specs/server-api/tests/test_images_api.py`

`size` is `"WxH"` or `"auto"`. The sizes that run are the model's `image_sizes` in
`model_list.json`, which for klein are 512x512 and 1024x1024 (OPEN-DIFFUSION-RESOLUTIONS); `auto`
and an omitted size are the largest, 1024x1024. Any other size is a 400 that names the supported
ones. Non-square sizes need their own stream sets and are a separate item.

**Acceptance criteria:**
- `"512x512"` gives a 512x512 image and `"auto"` a 1024x1024 one.
- `"768x768"` and `"1024x512"` are 400s with `param == "size"` whose message contains
  `512x512, 1024x1024`; `"big"`, `"1024"` and the number `1024` are 400s with `param == "size"`.

### SERVER-IMAGES-EDITS: edits are checked, then answered 501 until the NPU has a VAE encoder
**Applies to:** openflowlm-next (`src/server/server.cpp`, `src/server/multipart.cpp`, `src/server/rest_handler.cpp`)
**Verification:** test
**Test:** `specs/server-api/tests/test_images_api.py`

klein edits by appending the input images' VAE latents as reference tokens, which needs a VAE
*encoder* on the NPU and DiT streams per reference size; `mask` is the inpainting pipeline on top.
Until then `/v1/images/edits` parses its multipart form, checks it -- one or more non-empty
`image` / `image[]` files (at most 16), at most one `mask`, then the same model and control rules
as generations -- and answers 501 naming what is missing.

The multipart parser accepts a quoted boundary (`boundary="..."`, RFC 2046), fills each part's
`content_type`, finds headers case-insensitively, and keeps repeated parts (`image[]`), which used
to overwrite each other.

**Acceptance criteria:**
- Two `image[]` parts, a `model` and a `prompt`, sent with a quoted boundary, are answered 501
  with a message naming the VAE encoder.
- A form with no image is a 400 with `param == "image"`.
- A form with an image and `size: 768x768` is a 400 with `param == "size"`.

### SERVER-IMAGES-NPU: an image request holds the NPU like chat, and always lets it go
**Applies to:** openflowlm-next (`src/server/server.cpp`)
**Verification:** test
**Test:** `specs/server-api/tests/test_images_api.py`

Both image paths are in `requires_npu_access()`: one request at a time on the NPU, queued behind
chat, embeddings and transcription. Every path through the handlers calls `send_response` exactly
once, which is what releases the lock (SERVER-ERROR-STATUS describes what a missed release does).
A client that disconnects during an `n` > 1 request stops it after the current image.

**Acceptance criteria:**
- After each of a refused size, an unknown model, `n: 0` and a request with no prompt, an image
  request is answered 200.
- Chat, image, chat, image on one server are each answered 200.

### SERVER-IMAGES-RESIDENCY: swap by default, both resident with --image 1
**Applies to:** openflowlm-next (`src/server/rest_handler.cpp`, `src/include/utils/vm_args.hpp`)
**Verification:** manual

The image engine holds six hardware contexts, 7.5 GB of weights and 1.4 / 4.6 GiB of activations
per resolution. By default an image request swaps the chat model off the NPU and loads the image
engine (5.2 s warm, the weights in the OS file cache), and a chat request swaps back; the log says
so each time. The chat model's tag is kept, so a chat request that names it, or names nothing, is
served by it again.

`oflm serve <tag> --image 1 [--imagemodel <tag>]` loads the image engine at startup beside the
chat model, allocates every resolution, and never swaps. It still shares the NPU lock: resident
saves the load, not the queue. A startup that cannot load it exits naming why. `--imagemodel`
alone sets the model a request naming none gets; a tag that is not an image model stops the
server at startup. Both flags are refused by every other command.

The engine opens its kernel sets on the server's device handle (`npu_device_inst`), as the chat,
embedding and Whisper engines do.

**Verification (manual):**
1. `oflm serve llama3.2:1b`; send chat, image, chat, image (`test_chat_and_images_alternate`).
   All four succeed, and the log shows `swapping the chat model ... off the NPU`, then
   `swapping the image engine ... off the NPU` and `reloading 'llama3.2:1b'`, at each switch.
   (2026-09-28: 3 swaps each way over the whole test file; image engine load 5.2 s.)
2. `oflm serve llama3.2:1b --image 1`: the log shows one `Loading image model` line at startup
   and none after the same four requests. (2026-09-28: loaded in 8.7 s with both resolutions
   allocated; llama3.2:1b and the image engine fit the NPU together.)
3. `oflm serve llama3.2:1b --image 1 --imagemodel llama3.2:3b` exits with `--imagemodel: model
   'llama3.2:3b' is not an image model`; `oflm run llama3.2:1b --image 1` is refused.
4. Not measured: resident beside `--asr 1` and `--embed 1` as well.
