# Traces: TOOLS-GEMMA4-CONTINUATION (canonical spec: specs/tool-calling/spec.md)
# Integration: needs `oflm serve gemma4-it:12b` (or flm serve) listening on localhost:52625
# and the model's tokenizer + template on disk. Skips, with the reason, when either is absent.
import json
import os
import urllib.error
import urllib.request

import pytest

BASE = os.environ.get("OFLM_TEST_BASE_URL", "http://localhost:52625")
MODEL = "gemma4-it:12b"
MODEL_DIR_CANDIDATES = [
    os.path.join(os.environ.get("OFLM_MODEL_PATH", ""), "models", "Gemma4-12B-IT-NPU2"),
    os.path.join(os.environ.get("FLM_MODEL_PATH", ""), "models", "Gemma4-12B-IT-NPU2"),
    os.path.expanduser("~/.oflm/models/Gemma4-12B-IT-NPU2"),
    os.path.expanduser("~/.flm/models/Gemma4-12B-IT-NPU2"),
]

TOOL = {"type": "function", "function": {"name": "get_ticket", "description": "Fetch a support ticket by id.",
        "parameters": {"type": "object", "properties": {"ticket_id": {"type": "string"}}, "required": ["ticket_id"]}}}
RESULT = {"a_status": "open", "zz_code": "ZQX-7731"}  # zz_ sorts last, so the value sits in the final tokens


def _server_up():
    try:
        urllib.request.urlopen(f"{BASE}/v1/models", timeout=3).read()
        return True
    except Exception:
        return False


def _model_dir():
    for d in MODEL_DIR_CANDIDATES:
        if os.path.isfile(os.path.join(d, "chat_template.jinja")) and os.path.isfile(os.path.join(d, "tokenizer.json")):
            return d
    return None


pytestmark = pytest.mark.skipif(not _server_up(), reason=f"no server at {BASE}; start `oflm serve {MODEL}`")


def _chat(messages, **extra):
    body = dict(model=MODEL, messages=messages, tools=[TOOL], temperature=0.0, stream=False, **extra)
    req = urllib.request.Request(f"{BASE}/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        d = json.loads(r.read().decode())
    return d["choices"][0]["message"], d.get("usage", {})


def _continuation_messages():
    return [
        {"role": "user", "content": "Fetch ticket 42 and reply with ONLY the exact value of its `zz_code` field, nothing else."},
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "c1", "type": "function", "function": {"name": "get_ticket", "arguments": json.dumps({"ticket_id": "42"})}}]},
        {"role": "tool", "tool_call_id": "c1", "content": json.dumps(RESULT)},
    ]


def _rendered_token_count(messages_for_template, enable_thinking):
    jinja2 = pytest.importorskip("jinja2")
    tokenizers = pytest.importorskip("tokenizers")
    d = _model_dir()
    if d is None:
        pytest.skip("Gemma4-12B-IT-NPU2 tokenizer/template not found on disk")
    env = jinja2.Environment()
    env.globals["raise_exception"] = lambda m: (_ for _ in ()).throw(Exception(m))
    tpl = env.from_string(open(os.path.join(d, "chat_template.jinja"), encoding="utf-8").read())
    out = tpl.render(messages=messages_for_template, tools=[TOOL], add_generation_prompt=True,
                     bos_token="<bos>", enable_thinking=enable_thinking)
    tok = tokenizers.Tokenizer.from_file(os.path.join(d, "tokenizer.json"))
    return len(tok.encode(out, add_special_tokens=False).ids)


@pytest.mark.parametrize("reasoning", ["none", "low"])
def test_tail_of_tool_result_is_seen(reasoning):
    msg, _ = _chat(_continuation_messages(), reasoning_effort=reasoning)
    assert not msg.get("tool_calls"), msg
    assert "ZQX-7731" in (msg.get("content") or ""), msg


@pytest.mark.parametrize("stream", [False, True])
def test_models_own_empty_thought_block_is_not_reported(stream):
    body = dict(model=MODEL, messages=_continuation_messages(), tools=[TOOL], temperature=0.0,
                stream=stream, reasoning_effort="none")
    req = urllib.request.Request(f"{BASE}/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    raw = urllib.request.urlopen(req, timeout=600).read().decode("utf-8")
    reasoning = ""
    if stream:
        for line in raw.split("\n"):
            if line.startswith("data: ") and line != "data: [DONE]":
                delta = json.loads(line[6:])["choices"][0].get("delta", {})
                reasoning += delta.get("reasoning_content") or ""
    else:
        reasoning = json.loads(raw)["choices"][0]["message"].get("reasoning_content") or ""
    assert reasoning.strip() == "", repr(reasoning)


def test_prefill_count_matches_render_after_tool_result():
    # the server merges assistant tool_calls + tool messages into one assistant message
    # with tool_responses before templating (rest_handler convert_tool_responses_gemma4)
    merged = [
        _continuation_messages()[0],
        {"role": "assistant",
         "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "get_ticket", "arguments": {"ticket_id": "42"}}}],
         "tool_responses": [{"name": "get_ticket", "response": RESULT}]},
    ]
    expected = _rendered_token_count(merged, enable_thinking=False)
    _, usage = _chat(_continuation_messages(), reasoning_effort="none")
    assert usage.get("prompt_tokens") == expected, usage


def test_fresh_turn_still_trims_the_empty_thought_block():
    messages = [{"role": "user", "content": "Fetch ticket 42 using the tool."}]
    expected = _rendered_token_count(messages, enable_thinking=False)
    _, usage = _chat(messages, reasoning_effort="none")
    # four tokens (<|channel> thought \n <channel|>) are trimmed from the prefill and fed back in decode
    assert usage.get("prompt_tokens") == expected - 4, usage
