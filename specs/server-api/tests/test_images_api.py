# Traces: SERVER-IMAGES-GENERATIONS, SERVER-IMAGES-PARAMS, SERVER-IMAGES-SIZE, SERVER-IMAGES-EDITS,
#         SERVER-IMAGES-NPU (canonical spec: specs/server-api/spec.md)
# Integration: needs `oflm serve <chat model>` on localhost:52625 with flux2-klein:4b installed (it is
# pulled on first use otherwise) and the NPU. OFLM_TEST_MODEL names the chat model that is serving;
# OFLM_TEST_IMAGE_MODEL the image model (default flux2-klein:4b). Images are made at 512x512 to keep
# the run short (~6 s each; the first request also loads the engine). Standard library only.
import base64
import json
import os
import struct
import urllib.error
import urllib.request
import uuid

import pytest

BASE = os.environ.get("OFLM_TEST_BASE_URL", "http://localhost:52625")
MODEL = os.environ.get("OFLM_TEST_MODEL", "llama3.2:1b")
IMAGE_MODEL = os.environ.get("OFLM_TEST_IMAGE_MODEL", "flux2-klein:4b")
TIMEOUT = 900   # a swap loads the engine (and may pull it) before the first image

PROMPT = "a red fox in fresh snow"


def _body(raw):
    try:
        parsed = json.loads(raw)
    except ValueError:
        parsed = None
    return parsed if isinstance(parsed, dict) else {"_not_an_object": raw}


def _send(req):
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
            return r.status, _body(r.read().decode("utf-8", "replace"))
    except urllib.error.HTTPError as e:
        return e.code, _body(e.read().decode("utf-8", "replace"))
    except Exception as e:  # a reset connection or a timeout: the server died or stopped serving
        return None, {"_transport": str(e)}


def _post(path, payload):
    data = json.dumps(payload).encode()
    return _send(urllib.request.Request(f"{BASE}{path}", data=data,
                                        headers={"Content-Type": "application/json"}))


def _post_form(path, fields, files):
    boundary = "----oflmtest" + uuid.uuid4().hex
    body = bytearray()
    for name, value in fields.items():
        body += (f"--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"\r\n\r\n"
                 f"{value}\r\n").encode()
    for name, filename, content in files:
        body += (f"--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"; "
                 f"filename=\"{filename}\"\r\nContent-Type: image/png\r\n\r\n").encode()
        body += content + b"\r\n"
    body += f"--{boundary}--\r\n".encode()
    # quoted, as RFC 2046 allows: the parser used to keep the quotes as part of the boundary
    return _send(urllib.request.Request(f"{BASE}{path}", data=bytes(body),
                                        headers={"Content-Type": f"multipart/form-data; boundary=\"{boundary}\""}))


def _get(path):
    return _send(urllib.request.Request(f"{BASE}{path}"))


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


def _generate(**extra):
    payload = {"model": IMAGE_MODEL, "prompt": PROMPT, "size": "512x512", "seed": 1}
    payload.update(extra)
    return _post("/v1/images/generations", {k: v for k, v in payload.items() if v is not None})


def _png_size(b):
    assert b[:8] == b"\x89PNG\r\n\x1a\n", "not a PNG"
    return struct.unpack(">II", b[16:24])


def _jpeg_size(b):
    assert b[:2] == b"\xff\xd8", "not a JPEG"
    i = 2
    while i < len(b):
        marker, length = b[i + 1], struct.unpack(">H", b[i + 2:i + 4])[0]
        if marker in (0xC0, 0xC1, 0xC2):
            h, w = struct.unpack(">HH", b[i + 5:i + 9])
            return w, h
        i += 2 + length
    raise AssertionError("no SOF marker in the JPEG")


def _images(body):
    return [base64.b64decode(d["b64_json"]) for d in body["data"]]


def _assert_400(status, body, param, what):
    assert status == 400, f"{what}: expected 400, got {status} {body}"
    assert _err(body).get("param") == param, f"{what}: expected param {param!r}, got {body}"


def _assert_image_served(after):
    # an NPU request: a leaked lock leaves it queued forever, which /v1/models cannot show
    status, body = _generate()
    assert status == 200, f"after {after}: an image request was not answered: {status} {body}"


# ---- SERVER-IMAGES-GENERATIONS ------------------------------------------------------------------

def test_a_png_of_the_requested_size():
    status, body = _generate()
    assert status == 200, (status, body)
    assert isinstance(body.get("created"), int), body
    assert len(body["data"]) == 1, body
    assert _png_size(_images(body)[0]) == (512, 512)
    assert body.get("model") == IMAGE_MODEL, body


def test_a_jpeg_when_asked():
    status, body = _generate(output_format="jpeg")
    assert status == 200, (status, body)
    assert _jpeg_size(_images(body)[0]) == (512, 512)


def test_n_images_with_consecutive_seeds():
    status, body = _generate(n=2, seed=41)
    assert status == 200, (status, body)
    assert [d.get("seed") for d in body["data"]] == [41, 42], body
    a, b = _images(body)
    assert a != b, "two seeds gave the same image"
    # the second is exactly what seed 42 alone makes
    status, one = _generate(seed=42)
    assert status == 200, (status, one)
    assert _images(one)[0] == b


def test_the_same_seed_gives_the_same_bytes():
    s1, a = _generate(seed=7)
    s2, b = _generate(seed=7)
    assert s1 == s2 == 200, (a, b)
    assert a["data"][0]["b64_json"] == b["data"][0]["b64_json"]


def test_webp_is_refused_as_not_implemented():
    status, body = _generate(output_format="webp")
    _assert_400(status, body, "output_format", "webp")
    assert _err(body).get("code") == "not_implemented", body


def test_the_image_model_is_listed_on_v1_models_only():
    status, body = _get("/v1/models")
    assert status == 200 and IMAGE_MODEL in [m["id"] for m in body["data"]], body
    status, body = _get("/api/tags")
    assert status == 200 and IMAGE_MODEL not in [m["name"] for m in body["models"]], body


# ---- the model field (SERVER-MODEL-IDENTITY's rules) ----------------------------------------------

@pytest.mark.parametrize("model", ["oflm-test-no-such-model:0b", "", MODEL])
def test_a_model_that_makes_no_images_is_refused(model):
    status, body = _generate(model=model)
    _assert_400(status, body, "model", f"model {model!r}")
    assert _err(body).get("code") == "model_not_found", body


# ---- SERVER-IMAGES-PARAMS -----------------------------------------------------------------------

def test_both_spellings_of_steps_agreeing_are_one_field():
    status, body = _generate(steps=4, num_inference_steps=4)
    assert status == 200, (status, body)


def test_both_spellings_of_steps_disagreeing_are_refused():
    status, body = _generate(steps=4, num_inference_steps=8)
    _assert_400(status, body, "num_inference_steps", "steps 4, num_inference_steps 8")


def test_the_a1111_controls_are_accepted_and_ignored():
    s1, a = _generate(cfg_scale=7, negative_prompt="", sampler_name="Euler a", seed=-1)
    assert s1 == 200, a
    # ignored means ignored: the image is the plain request's
    s2, b = _generate(guidance_scale=3.5, negative_prompt="blurry", sampler="euler", seed=3)
    s3, c = _generate(seed=3)
    assert s2 == s3 == 200, (b, c)
    assert b["data"][0]["b64_json"] == c["data"][0]["b64_json"]


def test_a_sampler_other_than_euler_is_refused_listing_the_accepted():
    status, body = _generate(sampler_name="DPM++ 2M Karras")
    _assert_400(status, body, "sampler_name", "DPM++ 2M Karras")
    assert "Euler a" in _err(body).get("message", ""), body


def test_a_step_count_other_than_the_default_runs():
    status, body = _generate(steps=2)
    assert status == 200, (status, body)
    four = _generate()[1]
    assert body["data"][0]["b64_json"] != four["data"][0]["b64_json"], "2 steps gave the 4-step image"


@pytest.mark.parametrize("extra,param", [
    ({"n": 0}, "n"), ({"n": 11}, "n"), ({"steps": 51}, "steps"), ({"seed": -2}, "seed"),
    ({"stream": True}, "stream"), ({"partial_images": 1}, "partial_images"),
    ({"response_format": "url"}, "response_format"), ({"prompt": 5}, "prompt"),
])
def test_a_bad_control_is_refused_naming_it(extra, param):
    status, body = _generate(**extra)
    _assert_400(status, body, param, str(extra))


# ---- SERVER-IMAGES-SIZE -------------------------------------------------------------------------

@pytest.mark.parametrize("size", ["768x768", "1024x512", "big"])
def test_a_size_the_engine_lacks_is_refused_naming_the_supported(size):
    status, body = _generate(size=size)
    _assert_400(status, body, "size", size)
    if "x" in size:
        assert "512x512, 1024x1024" in _err(body).get("message", ""), body


def test_auto_is_1024():
    status, body = _generate(size="auto")
    assert status == 200, (status, body)
    assert _png_size(_images(body)[0]) == (1024, 1024)


# ---- SERVER-IMAGES-EDITS ------------------------------------------------------------------------

def test_edits_validate_then_answer_501():
    png = base64.b64decode(_generate()[1]["data"][0]["b64_json"])
    status, body = _post_form("/v1/images/edits", {"model": IMAGE_MODEL, "prompt": "make it night"},
                              [("image[]", "a.png", png), ("image[]", "b.png", png)])
    assert status == 501, (status, body)
    assert "VAE encoder" in _err(body).get("message", ""), body


def test_edits_without_an_image_are_refused():
    status, body = _post_form("/v1/images/edits", {"model": IMAGE_MODEL, "prompt": "x"}, [])
    _assert_400(status, body, "image", "no image")


def test_edits_check_the_same_controls():
    status, body = _post_form("/v1/images/edits", {"prompt": "x", "size": "768x768"},
                              [("image", "a.png", b"\x89PNG not really")])
    _assert_400(status, body, "size", "edits size 768x768")


# ---- SERVER-IMAGES-NPU --------------------------------------------------------------------------

def test_the_server_keeps_serving_after_image_errors():
    for extra in ({"size": "768x768"}, {"model": "oflm-test-no-such-model:0b"}, {"n": 0}):
        _generate(**extra)
        _assert_image_served(f"a refused {extra}")
    status, _ = _post("/v1/images/generations", {"model": IMAGE_MODEL})   # no prompt
    assert status == 400
    _assert_image_served("a request with no prompt")


def test_chat_and_images_alternate():
    # without --image 1 each switch swaps the engines; with it, neither reloads (the log shows it)
    chat = {"model": MODEL, "messages": [{"role": "user", "content": "Say ok."}], "max_tokens": 8}
    for step in ("chat", "image", "chat", "image"):
        status, body = _post("/v1/chat/completions", chat) if step == "chat" else _generate()
        assert status == 200, f"{step}: {status} {body}"
