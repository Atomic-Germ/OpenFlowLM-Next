# Traces: SERVER-ERROR-STATUS, SERVER-MODEL-IDENTITY (canonical spec: specs/server-api/spec.md)
# Integration: needs `oflm serve llama3.2:1b` listening on localhost:52625. Any chat model does -
# nothing here reads the generated text, only the status line and the error body. Point
# OFLM_TEST_MODEL at whatever is serving. Skips with the reason when no server answers.
import json
import os
import urllib.error
import urllib.request

import pytest

BASE = os.environ.get("OFLM_TEST_BASE_URL", "http://localhost:52625")
MODEL = os.environ.get("OFLM_TEST_MODEL", "llama3.2:1b")

# Not a tag any build has, and it never will be - the point is a model the server cannot load.
NO_SUCH_MODEL = "oflm-test-no-such-model:0b"

MESSAGES = [{"role": "user", "content": "Say ok."}]


def _server_up():
    try:
        urllib.request.urlopen(f"{BASE}/v1/models", timeout=3).read()
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _server_up(), reason=f"no server at {BASE}; start `oflm serve {MODEL}`")


def _body(raw):
    try:
        return json.loads(raw)
    except ValueError:
        return {"_not_json": raw}


def _post(path, payload, timeout=600):
    # The status IS the assertion here, so urllib rather than an SDK: the OpenAI client hides it
    # behind an exception type, and a 200 carrying an error body is exactly what we are looking for.
    data = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
    req = urllib.request.Request(f"{BASE}{path}", data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, _body(r.read().decode("utf-8", "replace"))
    except urllib.error.HTTPError as e:
        return e.code, _body(e.read().decode("utf-8", "replace"))


def _chat(**extra):
    payload = dict(model=MODEL, messages=MESSAGES, stream=False, max_tokens=16)
    payload.update(extra)
    return _post("/v1/chat/completions", payload)


def test_unknown_model_is_refused_with_a_400_error_body():
    status, body = _chat(model=NO_SUCH_MODEL)
    assert status == 400, (status, body)
    err = body["error"]
    assert err["type"] == "invalid_request_error", err
    assert err["param"] == "model", err
    assert NO_SUCH_MODEL in err["message"], err


def test_error_code_is_a_string():
    # Reading this field as an int threw out of the responder lambda; the outer handler caught the
    # exception and sent it as {"error": "<text>"} with the 200 the lambda had already set.
    _, body = _chat(model=NO_SUCH_MODEL)
    code = body["error"]["code"]
    assert isinstance(code, str), f"error.code came back as {type(code).__name__}: {code!r}"
    assert code == "model_not_found", code


def test_no_error_body_comes_back_with_a_2xx_status():
    probes = [
        ("chat, unknown model", "/v1/chat/completions",
         {"model": NO_SUCH_MODEL, "messages": MESSAGES, "max_tokens": 16}),
        ("completions, unknown model", "/v1/completions",
         {"model": NO_SUCH_MODEL, "prompt": "Say ok.", "max_tokens": 16}),
        ("chat, empty model string", "/v1/chat/completions",
         {"model": "", "messages": MESSAGES, "max_tokens": 16}),
        ("chat, the model-faker sentinel", "/v1/chat/completions",
         {"model": "model-faker", "messages": MESSAGES, "max_tokens": 16}),
    ]
    for name, path, payload in probes:
        status, body = _post(path, payload)
        assert "error" in body, f"{name}: expected a refusal, got {body}"
        assert status >= 400, f"{name}: error body sent with status {status}: {body}"


def test_unknown_model_is_not_answered_by_another_model():
    status, body = _chat(model=NO_SUCH_MODEL)
    assert status == 400, (status, body)
    assert "choices" not in body, f"a model that cannot be loaded was answered anyway: {body}"


def test_unknown_model_is_refused_in_streaming_mode_too():
    status, body = _chat(model=NO_SUCH_MODEL, stream=True)
    assert status == 400, (status, body)
    assert body["error"]["code"] == "model_not_found", body
    assert "choices" not in body, body


def test_an_accepted_request_reports_the_model_it_was_asked_for():
    status, body = _chat()
    assert status == 200, (status, body)
    assert body["model"] == MODEL, f"asked for {MODEL}, answered as {body.get('model')!r}"
    assert body["choices"][0]["message"]["content"].strip(), body


def test_a_refused_request_leaves_the_served_model_loaded():
    # The old order reset the engine and only then resolved the tag, so a typo evicted the model
    # that was serving everyone else.
    refused_status, refused = _chat(model=NO_SUCH_MODEL)
    assert refused_status == 400, (refused_status, refused)
    status, body = _chat()
    assert status == 200, (status, body)
    assert body["model"] == MODEL, body


def test_a_body_that_is_not_json_is_refused_and_the_server_keeps_serving():
    # Last on purpose: a body that is not JSON at all reaches the handler as a null JSON value, and
    # the second half of this test is the assertion that the server survives it. The lone 0xE5 is
    # the byte that used to wedge it - nlohmann put the offending bytes into the exception text,
    # dumping that text threw again inside the catch block, and the NPU lock was never released.
    status, body = _post("/v1/chat/completions", b'{"messages": "\xe5"')
    assert status >= 400, f"malformed body answered with status {status}: {body}"

    status, body = _chat()
    assert status == 200, (status, body)
    assert body["choices"][0]["message"]["content"].strip(), body
