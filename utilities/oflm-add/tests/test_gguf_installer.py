import json
import sys
from pathlib import Path


OFLM_ADD = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(OFLM_ADD))

import oflm_add  # noqa: E402


def test_choose_gguf_prefers_q4_1_and_refuses_multipart():
    chosen, refused = oflm_add.choose_gguf_file([
        "model-Q8_0.gguf",
        "model-Q4_0.gguf",
        "model-Q4_1-00001-of-00002.gguf",
        "model-Q4_1.gguf",
    ])

    assert chosen == "model-Q4_1.gguf"
    assert "multi-part GGUF" in refused["model-Q4_1-00001-of-00002.gguf"]


def test_local_q4nx_can_take_precedence_over_gguf(tmp_path):
    (tmp_path / "model.q4nx").touch()
    tree = [{"path": "weights-Q4_1.gguf"}]

    assert oflm_add.repo_has_q4nx(tree, tmp_path)


def test_modelscope_tree_dict_and_remote_q4nx_precedence(tmp_path):
    tree = {
        "model.q4nx": {"Path": "model.q4nx"},
        "weights-Q4_1.gguf": {"Path": "weights-Q4_1.gguf"},
        "nested/other.gguf": {"Path": "nested/other.gguf"},
    }

    assert oflm_add.root_file_names(tree) == ["model.q4nx", "weights-Q4_1.gguf"]
    assert oflm_add.repo_has_q4nx(tree, tmp_path)


def test_modelscope_selected_gguf_downloads_as_canonical_name(monkeypatch, tmp_path):
    selected = "weights-Q4_1.gguf"
    tree = {
        selected: {"Path": selected, "Size": 123, "Sha256": "ABCD"},
    }
    calls = []

    monkeypatch.setattr(oflm_add, "ms_file_tree", lambda repo: ("modelscope.test", tree))
    monkeypatch.setattr(
        oflm_add,
        "download_file",
        lambda url, dest, **kwargs: calls.append((url, dest, kwargs)),
    )

    obtained = oflm_add.fetch_assets(
        "Org/Model", tmp_path, modelscope=True, gguf_name=selected, quiet=True
    )

    assert obtained == ["model.gguf"]
    url, dest, kwargs = calls[0]
    assert url.endswith(f"/resolve/master/{selected}")
    assert dest == tmp_path / "model.gguf"
    assert kwargs["expected_size"] == 123
    assert kwargs["expected_sha"] == "abcd"


def test_explicit_kernel_manifest_can_enable_gguf(tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"gguf": {}}), encoding="utf-8")
    assert oflm_add.manifest_supports_gguf(manifest)

    manifest.write_text("{}", encoding="utf-8")
    assert not oflm_add.manifest_supports_gguf(manifest)


# ------------------------------------------------ stock-repo GGUF support
# (bartowski/Meta-Llama-3.1-8B-Instruct-GGUF and friends ship ONLY the GGUFs:
# no Q4_1, no tokenizer files, and a gated base repo.)


def test_quant_suffix_must_run_to_end_of_stem():
    # Q4_0_4_4 is an ik_llama matmul layout, NOT Q4_0: installing it as Q4_0
    # pools was silent garbage. The suffix must be the whole quant token.
    assert oflm_add.gguf_quant_of("Model-Q4_0_4_4.gguf") is None
    assert oflm_add.gguf_quant_of("Model-Q4_0_8_8.gguf") is None
    assert oflm_add.gguf_quant_of("Model-f32.gguf") is None
    assert oflm_add.gguf_quant_of("Model.gguf") is None
    # K-subtype letters stay with their family: the tensors are plain Q4_K.
    assert oflm_add.gguf_quant_of("Model-Q4_K_M.gguf") == "Q4_K"
    assert oflm_add.gguf_quant_of("Model-Q6_K.gguf") == "Q6_K"
    assert oflm_add.gguf_quant_of("Model-Q8_0.gguf") == "Q8_0"


def test_choose_gguf_prefers_q4_k_m_over_q8_0_and_names_impostors():
    chosen, refused = oflm_add.choose_gguf_file([
        "M-Q4_0_4_4.gguf",
        "M-Q8_0.gguf",
        "M-Q4_K_M.gguf",
        "M-IQ2_M.gguf",
    ])
    assert chosen == "M-Q4_K_M.gguf"
    # The impostor names its full suffix: it must never read as Q4_0.
    assert refused["M-Q4_0_4_4.gguf"].startswith("quant Q4_0_4_4 not supported")


def test_stock_repo_names_derive_known_families():
    assert oflm_add.derive_family({}, "Meta-Llama-3.1-8B-Instruct-GGUF") == "llama3"
    assert oflm_add.derive_family({}, "gemma-3-4b-it-GGUF") == "gemma3"
    assert oflm_add.derive_family({}, "Phi-4-mini-instruct-GGUF") == "phi4"


def _gguf_bytes(kv_items, tensors=()):
    """Minimal GGUF v3 prefix: header + KV + tensor infos (no payload)."""
    import struct

    out = bytearray(b"GGUF" + struct.pack("<I", 3))
    out += struct.pack("<QQ", len(tensors), len(kv_items))

    def blob(b):
        return struct.pack("<Q", len(b)) + b

    def value(typ, val):
        if typ == 4:
            return struct.pack("<I", val)
        if typ == 5:
            return struct.pack("<i", val)
        if typ == 7:
            return bytes([1 if val else 0])
        if typ == 8:
            return blob(val.encode("utf-8", "surrogateescape"))
        if typ == 9:
            etyp, items = val
            buf = struct.pack("<IQ", etyp, len(items))
            for it in items:
                buf += value(etyp, it)
            return buf
        raise AssertionError(typ)

    for key, (typ, val) in kv_items:
        out += blob(key.encode()) + struct.pack("<I", typ) + value(typ, val)
    for name, tnum, dims in tensors:
        out += blob(name.encode()) + struct.pack("<I", len(dims))
        for d in dims:
            out += struct.pack("<Q", d)
        out += struct.pack("<IQ", tnum, 0)
    return bytes(out)


def _llama_kv():
    return [
        ("general.architecture", (8, "llama")),
        ("tokenizer.ggml.model", (8, "llama")),
        ("tokenizer.ggml.tokens", (9, (8, ["hello", " world", "<|begin_of_text|>",
                                                   "<|end_of_text|>", "<0x00>"]))),
        ("tokenizer.ggml.merges", (9, (8, ["h e", "he l", "l l", "o <0x00>"]))),
        ("tokenizer.ggml.token_type", (9, (5, [1, 1, 3, 3, 6]))),
        ("tokenizer.ggml.bos_token_id", (4, 2)),
        ("tokenizer.ggml.eos_token_id", (4, 3)),
        ("tokenizer.ggml.add_bos_token", (7, True)),
        ("tokenizer.chat_template", (8, "hello {{ x }}")),
    ]


def test_inventory_parses_and_rejects_bad_magic():
    kv, tensors = oflm_add.parse_gguf_inventory(
        _gguf_bytes(_llama_kv(), [("blk.0.attn_q.weight", 12, [256, 128])]))
    assert kv["general.architecture"] == "llama"
    assert kv["tokenizer.ggml.bos_token_id"] == 2
    assert tensors == [("blk.0.attn_q.weight", 12, [256, 128])]
    try:
        oflm_add.parse_gguf_inventory(b"NOPE" + b"\x00" * 32)
    except ValueError:
        pass
    else:
        raise AssertionError("bad magic must raise")
    try:
        oflm_add.parse_gguf_inventory(_gguf_bytes(_llama_kv())[:40])
    except oflm_add._GgufTruncated:
        pass
    else:
        raise AssertionError("short prefix must raise _GgufTruncated")


def test_verify_choice_matches_tensors_against_claim(tmp_path):
    q4k = tmp_path / "m.gguf"
    q4k.write_bytes(_gguf_bytes(_llama_kv(), [("w", 12, [8, 8]), ("n", 0, [8])]))
    ok, note = oflm_add.verify_gguf_choice(str(q4k), "m.gguf", "Q4_K")
    assert ok, note
    # Same bytes, wrong claim: refuse rather than pack the wrong layout.
    ok, note = oflm_add.verify_gguf_choice(str(q4k), "m.gguf", "Q4_0")
    assert not ok and "claims Q4_0" in note
    # A Q4_0_4_4 tensor (type 31) is refused even when the name said Q4_0.
    trap = tmp_path / "t.gguf"
    trap.write_bytes(_gguf_bytes(_llama_kv(), [("w", 31, [8, 8])]))
    ok, note = oflm_add.verify_gguf_choice(str(trap), "t.gguf", "Q4_0")
    assert not ok and "Q4_0_4_4" in note


def test_extract_tokenizer_from_gguf_round_trips(tmp_path):
    gguf = tmp_path / "model.gguf"
    gguf.write_bytes(_gguf_bytes(_llama_kv()))
    out = tmp_path / "model"
    out.mkdir()
    wrote = oflm_add.extract_tokenizer_from_gguf(gguf, out)
    assert wrote == ["tokenizer.json", "tokenizer_config.json", "chat_template.jinja"]

    tok = json.loads((out / "tokenizer.json").read_text(encoding="utf-8"))
    assert tok["model"]["type"] == "BPE"
    # ASCII maps to itself; the space byte maps to U+0120 like HF files.
    assert tok["model"]["vocab"]["hello"] == 0
    assert tok["model"]["vocab"]["\u0120world"] == 1
    assert tok["model"]["merges"][0] == "h e"
    assert tok["model"]["merges"][3] == "o <0x00>"
    specials = {t["content"]: t["id"] for t in tok["added_tokens"]}
    assert specials["<|begin_of_text|>"] == 2
    assert specials["<|end_of_text|>"] == 3

    cfg = json.loads((out / "tokenizer_config.json").read_text(encoding="utf-8"))
    assert cfg["bos_token_id"] == 2
    assert cfg["eos_token_id"] == [3]
    assert cfg["bos_token"] == "<|begin_of_text|>"
    assert cfg["add_bos_token"] is True
    assert "hello" in cfg["chat_template"]
    assert (out / "chat_template.jinja").read_text(encoding="utf-8").startswith("hello")


def test_extract_tokenizer_refuses_mergeless_spm(tmp_path):
    kv = [kv for kv in _llama_kv() if kv[0] != "tokenizer.ggml.merges"]
    gguf = tmp_path / "model.gguf"
    gguf.write_bytes(_gguf_bytes(kv))
    try:
        oflm_add.extract_tokenizer_from_gguf(gguf, tmp_path)
    except RuntimeError as ex:
        assert "no merges" in str(ex)
    else:
        raise AssertionError("mergeless vocab must raise")
