# Tool calling through the OpenAI-compatible server

The server layer that turns a model's native tool-call text into OpenAI `tool_calls`,
and turns a client's tool schemas and tool results back into the model's prompt. Shared
by every engine (closed and open kernels); the requirements here are about the
`AutoModel` / `RestHandler` code, not the weights.

Directory name gives the prefix: `TOOLS`.

Background and evidence for the Gemma 4 requirements:
`.claude/plans/gemma4-tool-calling.md` (local, gitignored) and
[ROCm/FastFlowLM#722](https://github.com/ROCm/FastFlowLM/issues/722).

## Gemma 4

Gemma 4 emits `<|tool_call>call:NAME{ARGS}<tool_call|>` where `ARGS` is relaxed JSON:
bare keys, strings wrapped in `<|"|>...<|"|>`. The shared parser lives in
`src/include/AutoModel/gemma4_tool_parser.hpp` and serves the 12B and the E2B/E4B models.

### TOOLS-GEMMA4-CONTINUATION: the whole tool result reaches the model
**Applies to:** openflowlm-next (`src/common/AutoModel/modeling_gemma4_12b.cpp`)
**Test category:** integration (needs the NPU and `gemma4-it:12b`)
**Test:** `specs/tool-calling/tests/test_gemma4_continuation.py`

With thinking off, the Gemma 4 template ends a fresh model turn with an empty thought
block `<|channel>thought\n<channel|>`; the engine trims those four tokens from the prefill
and feeds them back during decode so its checkpoint sits before them. After a tool
response the template ends the prompt with `<tool_response|>` and adds nothing. The
engine shall trim and re-feed only when the rendered prompt actually ends with the empty
thought block, so a tool result is never shortened and no thought block is injected
where the template did not write one.

**Acceptance criteria:**
- A request whose last message is a tool result, with thinking off, prefills exactly as
  many tokens as the template renders (the server log's `Prefill chunk ... with N tokens`
  equals the tokenizer's count of the rendered prompt).
- A tool result `{"a_status": "open", "zz_code": "ZQX-7731"}` (the key sorts last, so the
  value sits in the final tokens) asked back verbatim at temperature 0 returns
  `ZQX-7731`, with thinking off and with thinking on.
- A fresh user turn with thinking off still prefills four fewer tokens than rendered and
  the generation still begins with the re-fed empty thought block (the existing
  behaviour is kept where it applies).
- In the continuation the model writes its own empty thought block (`<|channel>thought\n<channel|>`);
  a thought block that holds only whitespace is not reported as `reasoning_content`, in
  either response mode.

### TOOLS-GEMMA4-ENVELOPE: a wrapped call resolves to the named tool
**Applies to:** openflowlm-next (`src/include/AutoModel/gemma4_tool_parser.hpp`)
**Test category:** unit
**Test:** `src/test/gemma4_tool_parser/test.cpp`

The model sometimes wraps the real call in a generic envelope whose outer name is
`tool_call` (or `call`, `function`, `tool`, `function_call`) and whose arguments carry the
real name under `id` or `name` and the real arguments under `args`, `arguments`,
`parameters`, `params` or `input`. The parser shall return the inner name and inner
arguments in that case, and leave every other call untouched. Those outer names are legal
tool names too, so a call only counts as an envelope when every one of its argument keys is
one of the nine listed above; any other key means the call is real and is left alone.

**Acceptance criteria:**
- `call:tool_call{args:{query:"x"},id:<|"|>memory_search<|"|>}` parses to name
  `memory_search`, arguments `{"query": "x"}` (the trace from FastFlowLM#722).
- `call:function{name:<|"|>lookup_item_price<|"|>,arguments:{item:<|"|>widget<|"|>}}`
  parses to `lookup_item_price` / `{"item": "widget"}`.
- `call:tool_call{query:<|"|>x<|"|>}` (no inner name) stays `tool_call` / `{"query": "x"}`.
- `call:function{name:<|"|>widget<|"|>,quantity:2}` stays `function` /
  `{"name": "widget", "quantity": 2}` — `quantity` is not an envelope key, so this is a
  real tool named `function` and neither its name nor its arguments may be rewritten.
- Direct calls, empty argument lists and nested object/array arguments parse as before.

### TOOLS-GEMMA4-SCHEMA-TYPES: a type array in a tool schema does not fail the request
**Applies to:** openflowlm-next (`src/include/AutoModel/gemma4_tool_parser.hpp`, both Gemma 4 `apply_chat_template`s)
**Test category:** unit
**Test:** `src/test/gemma4_tool_parser/test.cpp`

The Gemma 4 template applies `| upper` to every parameter `type`; minja throws on a
JSON-schema type array such as `["string", "null"]`, which fails the whole request. Before
templating, the server shall fold every `type` array anywhere in a tool's parameter
schema into its first non-null member, adding `nullable: true` when `null` was listed.

**Acceptance criteria:**
- `{"type": ["string", "null"]}` becomes `{"type": "string", "nullable": true}`; other keys
  on that property are kept.
- Arrays nested under `items`, `properties` and sub-objects are folded too.
- A schema without type arrays is returned unchanged.
- A request whose tools include such a field succeeds through `oflm serve` and the
  model can call that tool (manual: `edge_cases.py` "nullable type array").

## All models

### TOOLS-REQUEST-PARAMS-RESET: request parameters do not leak between requests
**Applies to:** openflowlm-next (`src/server/rest_handler.cpp`, `src/common/AutoModel/automodel.cpp`)
**Test category:** integration (through `oflm serve`)
**Test:** `specs/tool-calling/tests/test_request_params_reset.py`

`temperature`, `top_p`, `top_k`, `min_p`, the penalties, `think`, `reasoning_effort` and
`system_prompt` are applied to the engine only when a request carries them. The server
shall restore the model's load-time defaults for those settings at the start of every
chat request before applying the request's own fields, so a request that omits a field
gets the model default and not whatever the previous request set. This holds for every
model that accepts the setting, not only the Gemma 4 pair: `AutoModel` owns the thinking
flag, the system prompt and the template's extra context, so its snapshot and reset cover
all of them, and a model only overrides them for request state it keeps elsewhere.

**Acceptance criteria:**
- After a request with `reasoning_effort: "low"`, a request without the field on a
  model whose default is no-think (Gemma 4) generates no reasoning content.
- The same holds on the other models that accept the field — Qwen 3, Qwen 3 MoE,
  Qwen 3 VL and GPT-OSS — since none of them keeps the flag privately any more.
- After a request with `temperature: 0`, a request without the field samples with the
  model's default temperature (observable: the raw output no longer repeats bit for bit
  across two identical prompts on a model whose default temperature is above 0).
