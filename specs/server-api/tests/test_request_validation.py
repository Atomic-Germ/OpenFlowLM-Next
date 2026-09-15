# Traces: SERVER-REQUEST-VALIDATION (canonical spec: specs/server-api/spec.md)
# Integration: needs `oflm serve llama3.2:1b` on localhost:52625; point OFLM_TEST_MODEL at the chat
# model that is serving. The embeddings tests also need an embedding model with no task prompts:
# `oflm serve llama3.2:1b --embed 1 --embeddingmodel bge-base:en-v1.5`, with OFLM_TEST_EMBED_MODEL
# set to that tag. The no-embedding-model test runs only against a server started WITHOUT --embed.
# Each test says what it needs in its skip reason.
import json
import os
import urllib.error
import urllib.request

import pytest

BASE = os.environ.get("OFLM_TEST_BASE_URL", "http://localhost:52625")
MODEL = os.environ.get("OFLM_TEST_MODEL", "llama3.2:1b")
EMBED_MODEL = os.environ.get("OFLM_TEST_EMBED_MODEL", "bge-base:en-v1.5")

MESSAGES = [{"role": "user", "content": "Say ok."}]


def _body(raw):
    try:
        parsed = json.loads(raw)
    except ValueError:
        parsed = None
    return parsed if isinstance(parsed, dict) else {"_not_an_object": raw}


def _post(path, payload, timeout=120):
    data = json.dumps(payload).encode()
    req = urllib.request.Request(f"{BASE}{path}", data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, _body(r.read().decode("utf-8", "replace"))
    except urllib.error.HTTPError as e:
        return e.code, _body(e.read().decode("utf-8", "replace"))
    except Exception as e:  # a reset connection or a timeout: the server died or stopped serving
        return None, {"_transport": str(e)}


def _err(body):
    err = body.get("error")
    return err if isinstance(err, dict) else {}


def _server_up():
    try:
        urllib.request.urlopen(f"{BASE}/v1/models", timeout=3).read()
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _server_up(), reason=f"no server at {BASE}; start `oflm serve {MODEL}`")


def _assert_still_serving(after):
    # A request that needs the NPU. /v1/models does not take the NPU lock, so it kept answering
    # while a leaked lock left every NPU request queued forever -- it cannot show this failure.
    status, body = _post("/v1/chat/completions",
                         {"model": MODEL, "messages": MESSAGES, "max_tokens": 8, "stream": False})
    assert status == 200, f"after {after}: a chat request was not answered: {status} {body}"


def _assert_400(status, body, param, what):
    assert status == 400, f"{what}: expected 400, got {status} {body}"
    err = _err(body)
    assert err, f"{what}: expected an OpenAI-shaped error object, got {body}"
    assert err.get("param") == param, f"{what}: expected param {param!r}, got {err}"


# ---- every handler with a required field ----------------------------------------------------
# The /v1/chat/completions cases repeat oflm-test --api check A1 (no messages, messages a string).

REQUIRED = [
    ("/api/show", "model"),
    ("/api/generate", "prompt"),
    ("/api/chat", "messages"),
    ("/v1/completions", "prompt"),
    ("/v1/chat/completions", "messages"),
]


@pytest.mark.parametrize("path,field", REQUIRED)
def test_an_empty_object_is_refused_and_the_server_keeps_serving(path, field):
    status, body = _post(path, {})
    _assert_400(status, body, field, f"POST {path} {{}}")
    _assert_still_serving(f"POST {path} {{}}")


@pytest.mark.parametrize("path,payload,param", [
    ("/v1/completions", {"model": MODEL, "prompt": 5}, "prompt"),
    ("/api/generate", {"model": MODEL, "prompt": ["hi"]}, "prompt"),
    ("/v1/chat/completions", {"model": MODEL, "messages": "hi"}, "messages"),
    ("/api/chat", {"model": MODEL, "messages": {"role": "user"}}, "messages"),
])
def test_a_required_field_of_the_wrong_type_is_refused(path, payload, param):
    status, body = _post(path, payload)
    _assert_400(status, body, param, f"POST {path} {payload}")
    assert _err(body).get("code") == "invalid_value", body


# ---- request_id ---------------------------------------------------------------------------------

@pytest.mark.parametrize("bad", [5, None, {"a": 1}, ["x"]])
def test_a_non_string_request_id_is_refused_and_does_not_hold_the_npu(bad):
    status, body = _post("/v1/chat/completions",
                         {"model": MODEL, "messages": MESSAGES, "max_tokens": 8, "request_id": bad})
    _assert_400(status, body, "request_id", f"request_id {bad!r}")
    _assert_still_serving(f"request_id {bad!r}")


def test_a_string_request_id_is_accepted():
    status, body = _post("/v1/chat/completions",
                         {"model": MODEL, "messages": MESSAGES, "max_tokens": 8, "stream": False,
                          "request_id": "spec-test-1"})
    assert status == 200, (status, body)


# ---- /v1/embeddings -----------------------------------------------------------------------------

_EMBED_STATUS, _EMBED_BODY = _post("/v1/embeddings", {"input": "probe"}) if _server_up() else (None, {})
_NO_EMBED_MODEL = (_EMBED_STATUS == 400 and "no embedding model" in _err(_EMBED_BODY).get("message", ""))
_EMBED_READY = (_EMBED_STATUS == 200 and _EMBED_BODY.get("model") == EMBED_MODEL)

needs_embed = pytest.mark.skipif(
    not _EMBED_READY,
    reason=f"needs '{EMBED_MODEL}' loaded with no task prompts (probe answered {_EMBED_STATUS}); start "
           f"`oflm serve {MODEL} --embed 1 --embeddingmodel bge-base:en-v1.5`")


@needs_embed
def test_embeddings_empty_object_is_refused():
    status, body = _post("/v1/embeddings", {})
    _assert_400(status, body, "input", "POST /v1/embeddings {}")
    _assert_still_serving("POST /v1/embeddings {}")


@needs_embed
@pytest.mark.parametrize("bad", [None, 5, {"text": "x"}])
def test_embeddings_input_of_the_wrong_type_is_refused(bad):
    status, body = _post("/v1/embeddings", {"model": EMBED_MODEL, "input": bad})
    _assert_400(status, body, "input", f"input {bad!r}")


@needs_embed
def test_embeddings_a_non_string_element_is_named():
    status, body = _post("/v1/embeddings", {"model": EMBED_MODEL, "input": ["x", 7]})
    _assert_400(status, body, "input[1]", "input ['x', 7]")


@needs_embed
def test_embeddings_null_model_is_refused():
    status, body = _post("/v1/embeddings", {"model": None, "input": "x"})
    _assert_400(status, body, "model", "model null")


@needs_embed
def test_embeddings_empty_model_string_is_refused():
    status, body = _post("/v1/embeddings", {"model": "", "input": "x"})
    _assert_400(status, body, "model", 'model ""')
    assert _err(body).get("code") == "model_not_found", body


@needs_embed
def test_embeddings_omitted_model_is_served_and_named():
    status, body = _post("/v1/embeddings", {"input": ["one", "two"]})
    assert status == 200, (status, body)
    assert body.get("model") == EMBED_MODEL, body
    assert len(body.get("data", [])) == 2, body


@needs_embed
def test_embeddings_an_empty_array_is_a_valid_request():
    status, body = _post("/v1/embeddings", {"model": EMBED_MODEL, "input": []})
    assert status == 200, (status, body)
    assert body.get("data") == [], body


@pytest.mark.skipif(not _NO_EMBED_MODEL,
                    reason=f"needs a server started without --embed (probe answered {_EMBED_STATUS})")
def test_embeddings_without_an_embedding_model_is_refused_not_200():
    status, body = _post("/v1/embeddings", {"input": "x"})
    assert status == 400, (status, body)
    assert _err(body).get("code") == "model_not_found", body
