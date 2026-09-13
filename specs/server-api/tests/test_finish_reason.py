# Traces: SERVER-FINISH-REASON, SERVER-STREAM-PARITY (canonical spec: specs/server-api/spec.md)
# Integration: needs `oflm serve llama3.2:1b` listening on localhost:52625. Any chat model does -
# the assertions are about the response metadata, not the text. On a model that thinks by default
# (qwen3, gpt-oss) raise OFLM_TEST_STOP_TOKENS, because the "stop" case has to leave room for the
# reasoning as well as the answer. Skips with the reason when no server answers.
import json
import os
import urllib.request

import pytest

BASE = os.environ.get("OFLM_TEST_BASE_URL", "http://localhost:52625")
MODEL = os.environ.get("OFLM_TEST_MODEL", "llama3.2:1b")
STOP_TOKENS = int(os.environ.get("OFLM_TEST_STOP_TOKENS", "256"))

# OpenAI's whole vocabulary. The engine's own is wider - it also says "cancel", "error" and
# "UNKNOWN" - and none of those are values a client is allowed to see.
OPENAI_FINISH_REASONS = {"stop", "length", "tool_calls", "content_filter", "function_call"}

# Forty primes cannot be written in 8 tokens, so this one is cut off every time.
LONG_PROMPT = "List the first forty prime numbers, separated by commas."
SHORT_PROMPT = "Reply with the single word ok and nothing else."


def _server_up():
    try:
        urllib.request.urlopen(f"{BASE}/v1/models", timeout=3).read()
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _server_up(), reason=f"no server at {BASE}; start `oflm serve {MODEL}`")


def _request(prompt, stream, max_tokens):
    payload = dict(model=MODEL, messages=[{"role": "user", "content": prompt}],
                   stream=stream, max_tokens=max_tokens)
    return urllib.request.Request(f"{BASE}/v1/chat/completions", data=json.dumps(payload).encode(),
                                  headers={"Content-Type": "application/json"})


def _chat(prompt, max_tokens):
    with urllib.request.urlopen(_request(prompt, False, max_tokens), timeout=600) as r:
        return json.loads(r.read().decode())


def _chunks(prompt, max_tokens):
    """Every SSE data chunk of a streamed answer, in order, [DONE] dropped."""
    out = []
    with urllib.request.urlopen(_request(prompt, True, max_tokens), timeout=600) as r:
        for line in r:
            line = line.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            payload = line[len("data:"):].strip()
            if payload == "[DONE]":
                break
            out.append(json.loads(payload))
    return out


def _streamed_finish_reason(chunks):
    reasons = [c["choices"][0].get("finish_reason") for c in chunks if c.get("choices")]
    final = [r for r in reasons if r is not None]
    assert final, f"no chunk carried a finish_reason: {reasons}"
    return final[-1]


def test_an_answer_cut_at_max_tokens_reports_length():
    body = _chat(LONG_PROMPT, 8)
    assert body["choices"][0]["finish_reason"] == "length", body["choices"][0]


def test_an_answer_that_ends_on_its_own_reports_stop():
    body = _chat(SHORT_PROMPT, STOP_TOKENS)
    assert body["choices"][0]["finish_reason"] == "stop", body["choices"][0]


def test_a_streamed_answer_cut_at_max_tokens_reports_length():
    assert _streamed_finish_reason(_chunks(LONG_PROMPT, 8)) == "length"


def test_a_streamed_answer_that_ends_on_its_own_reports_stop():
    assert _streamed_finish_reason(_chunks(SHORT_PROMPT, STOP_TOKENS)) == "stop"


def test_finish_reason_is_from_the_openai_vocabulary():
    body = _chat(LONG_PROMPT, 8)
    assert body["choices"][0]["finish_reason"] in OPENAI_FINISH_REASONS, body["choices"][0]
    assert _streamed_finish_reason(_chunks(LONG_PROMPT, 8)) in OPENAI_FINISH_REASONS


def test_the_stream_ends_with_a_chunk_carrying_finish_reason():
    chunks = _chunks(LONG_PROMPT, 8)
    assert chunks, "the stream carried no data chunks at all"
    last = chunks[-1]["choices"][0]
    assert last.get("finish_reason") is not None, f"last chunk before [DONE] has no finish_reason: {chunks[-1]}"
    for c in chunks[:-1]:
        assert c["choices"][0].get("finish_reason") is None, f"a content chunk claimed to be final: {c}"


def test_the_two_response_modes_agree_on_model_and_finish_reason():
    body = _chat(LONG_PROMPT, 8)
    chunks = _chunks(LONG_PROMPT, 8)
    assert body["model"] == MODEL, body["model"]
    assert chunks[-1]["model"] == body["model"], (chunks[-1]["model"], body["model"])
    assert _streamed_finish_reason(chunks) == body["choices"][0]["finish_reason"]
