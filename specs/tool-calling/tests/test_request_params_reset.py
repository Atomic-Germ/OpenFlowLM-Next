# Traces: TOOLS-REQUEST-PARAMS-RESET (canonical spec: specs/tool-calling/spec.md)
# Integration: needs `oflm serve gemma4-it:12b` listening on localhost:52625 (Gemma 4 is the
# model whose default is no-think, which is what makes the leak observable). Skips with the
# reason when no server answers.
#
# The reset lives in AutoModel, so every model with a thinking flag is covered by the same
# two tests - point OFLM_TEST_MODEL at qwen3:8b or gpt-oss:20b (serving that model) to run
# them there.
import json
import os
import urllib.request

import pytest

BASE = os.environ.get("OFLM_TEST_BASE_URL", "http://localhost:52625")
MODEL = os.environ.get("OFLM_TEST_MODEL", "gemma4-it:12b")


def _server_up():
    try:
        urllib.request.urlopen(f"{BASE}/v1/models", timeout=3).read()
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _server_up(), reason=f"no server at {BASE}; start `oflm serve {MODEL}`")


def _chat(**extra):
    body = dict(model=MODEL, messages=[{"role": "user", "content": "In one short sentence, what is an NPU?"}],
                stream=False, **extra)
    req = urllib.request.Request(f"{BASE}/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(r.read().decode())["choices"][0]["message"]


def test_reasoning_effort_does_not_leak_into_the_next_request():
    with_thinking = _chat(reasoning_effort="low")
    assert with_thinking.get("reasoning_content"), with_thinking
    plain = _chat()
    assert not plain.get("reasoning_content"), plain


def test_temperature_does_not_leak_into_the_next_request():
    # Gemma 4's load-time temperature is 1.0; at 0 two identical prompts repeat bit for bit,
    # at the default they do not (same first sentence is fine, identical 60+ token text is not)
    a = _chat(temperature=0.0, max_tokens=60)["content"]
    b = _chat(temperature=0.0, max_tokens=60)["content"]
    assert a == b, "temperature 0 should be deterministic"
    c = _chat(max_tokens=60)["content"]
    d = _chat(max_tokens=60)["content"]
    assert not (c == d == a), "default temperature still pinned at 0 from the earlier request"
