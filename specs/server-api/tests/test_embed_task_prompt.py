# Traces: SERVER-EMBED-TASK-PROMPT, SERVER-MODEL-IDENTITY (canonical spec: specs/server-api/spec.md)
# Integration: needs a server with an embedding model loaded - `oflm serve -e 1`, or
# `oflm serve <chat model> --embed 1 --embeddingmodel <tag>`. OFLM_TEST_EMBED_MODEL must name the
# tag that is actually loaded, because the server refuses a request naming any other one.
#
# Two kinds of model are needed to cover the whole requirement, and no server has both: a model
# that declares prompt names (nomic-embed-text:v1.5) for the "a prompt is required and it changes
# the vector" half, and one with no task concept (bge-base:en-v1.5) for the "a prompt is refused"
# half. Each test says which it needs in its skip reason.
import json
import math
import os
import urllib.error
import urllib.request

import pytest

BASE = os.environ.get("OFLM_TEST_BASE_URL", "http://localhost:52625")
EMBED_MODEL = os.environ.get("OFLM_TEST_EMBED_MODEL", "nomic-embed-text:v1.5")

# The BERT-family tags this build serves. They prepend nothing, so a task prompt has no meaning
# for them and the server refuses one rather than embedding the text unprefixed.
NO_TASK_CONCEPT = {"bge-base", "bge-small", "bge-large", "all-minilm", "gte-multilingual"}
FAMILY = EMBED_MODEL.split(":")[0]

TEXT = "The NPU runs the embedding model on device."


def _body(raw):
    # A server started without --embed used to answer 200 with a bare `null` (a 400 now, see
    # SERVER-REQUEST-VALIDATION), so "parsed to an object" is not something this file can assume.
    try:
        parsed = json.loads(raw)
    except ValueError:
        parsed = None
    return parsed if isinstance(parsed, dict) else {"_not_an_object": raw}


def _post(payload, timeout=300):
    req = urllib.request.Request(f"{BASE}/v1/embeddings", data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, _body(r.read().decode("utf-8", "replace"))
    except urllib.error.HTTPError as e:
        return e.code, _body(e.read().decode("utf-8", "replace"))
    except Exception as e:
        return None, {"_transport": str(e)}


def _embed_request(**extra):
    payload = dict(model=EMBED_MODEL, input=TEXT)
    payload.update(extra)
    return _post(payload)


def _err(body):
    err = body.get("error")
    return err if isinstance(err, dict) else {}


# One probe at import time answers both questions the file needs: is there an embedding endpoint
# with this tag on it, and does this model declare prompt names (in which case a request without
# one is refused, which is requirement case (a) and also how we detect it).
_PROBE_STATUS, _PROBE_BODY = _post({"model": EMBED_MODEL, "input": "probe"})


def _unavailable():
    if _PROBE_STATUS is None:
        return f"no server at {BASE}; start `oflm serve -e 1`"
    if _PROBE_STATUS == 404:
        return f"{BASE} serves no /v1/embeddings; start `oflm serve -e 1`"
    if "no embedding model" in _err(_PROBE_BODY).get("message", ""):
        return f"{BASE} was started without an embedding model; start `oflm serve -e 1`"
    if _err(_PROBE_BODY).get("code") == "model_not_found":
        return (f"the server does not have '{EMBED_MODEL}' loaded ({_err(_PROBE_BODY).get('message')}); "
                f"set OFLM_TEST_EMBED_MODEL to the tag it was started with")
    if _PROBE_STATUS == 200 and "data" not in _PROBE_BODY:
        return f"{BASE} was started without an embedding model; start `oflm serve -e 1`"
    return None


_DECLARES_PROMPTS = _err(_PROBE_BODY).get("code") == "missing_required_parameter"

pytestmark = pytest.mark.skipif(_unavailable() is not None, reason=_unavailable() or "")

# The condition includes "the server is up" so that when it is not, the module-level skip is the
# one that reports, with the reason that actually explains the run.
needs_prompts = pytest.mark.skipif(
    _unavailable() is None and not _DECLARES_PROMPTS,
    reason=f"'{EMBED_MODEL}' declares no prompt names; start "
           f"`oflm serve <chat model> --embed 1 --embeddingmodel nomic-embed-text:v1.5`")

needs_no_task_concept = pytest.mark.skipif(
    _unavailable() is None and FAMILY not in NO_TASK_CONCEPT,
    reason=f"'{EMBED_MODEL}' has a task concept; start "
           f"`oflm serve <chat model> --embed 1 --embeddingmodel bge-base:en-v1.5`")


def _vector(**extra):
    status, body = _embed_request(**extra)
    assert status == 200, (status, body)
    return body["data"][0]["embedding"]


def _cosine(a, b):
    assert len(a) == len(b), (len(a), len(b))
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb)


@needs_prompts
def test_a_model_that_declares_prompts_refuses_a_request_without_one():
    status, body = _embed_request()
    assert status == 400, (status, body)
    err = _err(body)
    assert err.get("code") == "missing_required_parameter", err
    assert err.get("param") == "prompt_name", err
    # The message has to quote the REST names, not the container's: naming the container's sent
    # clients to values the validator then rejected.
    assert "search_query" in err.get("message", ""), err
    assert "data" not in body, body


@needs_no_task_concept
def test_a_model_with_no_task_concept_refuses_a_prompt_name():
    status, body = _embed_request(prompt_name="search_query")
    assert status == 400, (status, body)
    assert _err(body).get("code") == "invalid_value", body
    assert "data" not in body, f"the text was embedded unprefixed and answered 200: {body}"


def test_an_unknown_task_name_is_refused():
    status, body = _embed_request(prompt_name="not_a_task")
    assert status == 400, (status, body)
    assert _err(body).get("code") == "invalid_value", body
    assert "data" not in body, f"an unknown task prompt was substituted rather than refused: {body}"


@needs_prompts
def test_the_refusal_names_the_offending_value_and_the_accepted_ones():
    _, body = _embed_request(prompt_name="not_a_task")
    message = _err(body).get("message", "")
    assert "not_a_task" in message, message
    assert "search_query" in message and "search_document" in message, message


def test_a_non_string_prompt_name_is_refused():
    status, body = _embed_request(prompt_name=7)
    assert status == 400, (status, body)
    assert _err(body).get("code") == "invalid_value", body


@needs_prompts
def test_the_task_prompt_changes_the_vector():
    # Measured on this server: the same text under search_query against search_document is cosine
    # 0.914. The handler used to pass task_query whatever the request said, so every document in a
    # RAG index was embedded as a query - and the vector is correctly shaped and correctly normed
    # either way, so nothing downstream could tell.
    query = _vector(prompt_name="search_query")
    document = _vector(prompt_name="search_document")
    assert len(query) == len(document)
    assert query != document, "both task prompts produced the same vector; the request was ignored"
    assert _cosine(query, document) < 0.9999, _cosine(query, document)


@needs_prompts
def test_each_task_prompt_is_reproducible():
    assert _vector(prompt_name="search_query") == _vector(prompt_name="search_query")
    assert _vector(prompt_name="search_document") == _vector(prompt_name="search_document")


@needs_prompts
def test_task_type_is_an_accepted_alias_for_prompt_name():
    assert _vector(task_type="search_document") == _vector(prompt_name="search_document")


@needs_prompts
def test_the_two_spellings_may_both_be_sent_when_they_agree():
    # "query" and "search_query" are two REST names for one task, so this is not a conflict.
    assert _vector(prompt_name="query", task_type="search_query") == _vector(prompt_name="search_query")


@needs_prompts
def test_prompt_name_and_task_type_that_disagree_are_refused():
    status, body = _embed_request(prompt_name="search_query", task_type="search_document")
    assert status == 400, (status, body)
    message = _err(body).get("message", "")
    assert _err(body).get("code") == "invalid_value", body
    assert "prompt_name" in message and "task_type" in message, message
    assert "data" not in body, f"a disagreement was resolved by precedence rather than refused: {body}"


def test_an_embedding_request_naming_another_model_is_refused():
    # Traces: SERVER-MODEL-IDENTITY. Started with bge-base loaded, a request for
    # gte-multilingual:base used to return bge-base's vectors byte for byte, labelled with the tag
    # that had been asked for.
    status, body = _embed_request(model="oflm-test-no-such-embed:0b")
    assert status == 400, (status, body)
    assert _err(body).get("code") == "model_not_found", body
    assert EMBED_MODEL in _err(body).get("message", ""), "the refusal does not name what is loaded"
    assert "data" not in body, f"another model's vectors were returned: {body}"
