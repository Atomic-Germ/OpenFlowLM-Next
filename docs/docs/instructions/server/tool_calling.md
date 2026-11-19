---
title: "Tool Calling"
description: "Invoke external functions from OpenFlowLM chat models in Python scripts and Open WebUI."
icon: "handyman"
weight: 30
---

Tool calling lets instruction-tuned models (Qwen3, Qwen3.5, Qwen3.6-MoE, Llama 3.1/3.2, Gemma 4, Granite 4, including [`oflm add`](/docs/instructions/cli/#add-a-model-oflm-add) GGUFs) request structured function calls and incorporate local outputs into a grounded answer.

## Request Flow

1. **Declare `tools`:** Pass JSON Schema function signatures in `/v1/chat/completions`.
2. **Receive `tool_calls`:** The model emits function names and JSON arguments without executing code itself.
3. **Run & return `role: "tool"`:** Your client executes the local function, appends the `tool` result message, and calls `/v1/chat/completions` again for the final prose reply.

> [!NOTE]
> See the [OpenAI Function Calling guide](https://platform.openai.com/docs/guides/function-calling) for schema details, and run `oflm-test --tools` against `oflm serve` to verify model conformance.

---

## Python SDK Examples

### Non-Streaming Tool Call

```python
import json
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:52625/v1", api_key="oflm")
tools = [{
    "type": "function",
    "function": {
        "name": "get_temperature",
        "description": "Get current temperature for a city",
        "parameters": {
            "type": "object",
            "required": ["city"],
            "properties": {"city": {"type": "string"}},
        },
    },
}]

def get_temperature(city: str) -> str:
    return {"New York": "22°C", "London": "15°C", "Tokyo": "18°C"}.get(city, "Unknown")

messages = [{"role": "user", "content": "What is the temperature in New York?"}]
resp1 = client.chat.completions.create(model="qwen3.5:4b", messages=messages, tools=tools)
msg = resp1.choices[0].message
messages.append({"role": "assistant", "content": msg.content, "tool_calls": msg.tool_calls})

for tc in msg.tool_calls or []:
    if tc.function.name == "get_temperature":
        city = json.loads(tc.function.arguments)["city"]
        messages.append({
            "role": "tool",
            "tool_call_id": tc.id,
            "content": json.dumps({"temperature": get_temperature(city)}),
        })

resp2 = client.chat.completions.create(model="qwen3.5:4b", messages=messages, tools=tools)
print(resp2.choices[0].message.content)
```

### Streaming Tool Call

Set `stream=True` to collect `delta.tool_calls` chunks and stream the final answer:

```python
tool_calls, content = [], ""
for chunk in client.chat.completions.create(
    model="qwen3.5:4b", messages=messages[:1], tools=tools, stream=True
):
    delta = chunk.choices[0].delta
    if delta.content:
        content += delta.content
    if delta.tool_calls:
        tool_calls.extend(delta.tool_calls)
# Append assistant + tool messages as above, then call create(..., stream=True)
```

---

## Tool Calling in Open WebUI

1. Connect Open WebUI to `oflm serve` ([setup steps](/docs/instructions/server/webui/#run-open-webui-with-openflowlm)).
2. Disable conflicting built-in tools: under **Admin Settings -> Models**, select your model (e.g., `qwen3.5:4b`), uncheck **Builtin Tools**, and click **Save & Update**.
3. Import a tool: open **Workspace -> Tools -> Discover a tool**, import `arXiv Search Tool` (no API key needed) or `Play YouTube Video` (configure a restricted YouTube Data API v3 key under the tool's gear icon).
4. In a new chat, enable the tool under **Integrations -> Tools**, open **Controls** (top-right), and switch **Function Calling** from `Default` to `Native`.

For live web search without custom tool scripts, see [Agentic Web Search](/docs/instructions/server/webui/#example-agentic-ai-web-search-with-openflowlm).
