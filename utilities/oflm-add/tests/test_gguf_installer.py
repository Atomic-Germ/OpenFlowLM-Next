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


def test_embedding_gemma_repo_names_derive_embed_family():
    # Hyphenated ("embedding-gemma-...") and fused ("embeddinggemma-...") hosts
    # alike must resolve without --family.
    assert oflm_add.derive_family({}, "embedding-gemma-300M-GGUF") == "embed-gemma"
    assert oflm_add.derive_family({}, "embeddinggemma-300M-GGUF") == "embed-gemma"
    assert oflm_add.derive_family({}, "embed-gemma-300m") == "embed-gemma"


def test_gguf_arch_fallback_resolves_embed_family(tmp_path):
    gguf = tmp_path / "quant-Q8_0.gguf"
    gguf.write_bytes(_gguf_bytes([("general.architecture", (8, "gemma-embedding"))]))
    assert oflm_add.family_from_gguf_arch(str(gguf)) == "embed-gemma"


def test_gguf_arch_map_covers_common_families(tmp_path):
    cases = {
        "qwen3": "qwen3",
        "qwen2": "qwen2",
        "llama": "llama3",
        "gemma3": "gemma3",
        "phi4": "phi4",
        "granite": "granite",
        "hunyuan": "hunyuan",
    }
    for arch, family in cases.items():
        gguf = tmp_path / f"{arch}-Q4_K_M.gguf"
        gguf.write_bytes(_gguf_bytes([("general.architecture", (8, arch))]))
        assert oflm_add.family_from_gguf_arch(str(gguf)) == family
    # Unmapped arches and unreadable files yield None, not an error.
    other = tmp_path / "other-Q4_K_M.gguf"
    other.write_bytes(_gguf_bytes([("general.architecture", (8, "qwen3moe"))]))
    assert oflm_add.family_from_gguf_arch(str(other)) is None
    assert oflm_add.family_from_gguf_arch(str(tmp_path / "missing.gguf")) is None


def test_config_model_type_resolves_family():
    assert oflm_add.family_from_config_dict({"model_type": "qwen3"}) == "qwen3"
    assert oflm_add.family_from_config_dict({"model_type": "llama"}) == "llama3"
    assert oflm_add.family_from_config_dict(
        {"architectures": ["Qwen3ForCausalLM"]}) == "qwen3"
    assert oflm_add.family_from_config_dict(
        {"architectures": ["GraniteForCausalLM"]}) == "granite"
    # Vision-language shapes never fold onto a text family.
    assert oflm_add.family_from_config_dict({"model_type": "qwen2_5_vl"}) is None
    # Unknown shapes and garbage yield None, not an error.
    assert oflm_add.family_from_config_dict({"model_type": "mixtral"}) is None
    assert oflm_add.family_from_config_dict({}) is None
    assert oflm_add.family_from_config_dict(None) is None


def test_derive_tag_accepts_size_guess():
    assert oflm_add.derive_tag("My-Finetune-Q4_K_M", None, "4.4b") == "my-finetune:4.4b"
    try:
        oflm_add.derive_tag("My-Finetune-Q4_K_M")
    except SystemExit:
        pass
    else:
        raise AssertionError("sizeless slug without a guess must raise")


def test_gguf_size_token_math():
    assert oflm_add.gguf_size_token(2_500_000_000, "Q4_K") == "4.4b"
    assert oflm_add.gguf_size_token(333_590_944, "Q8_0") == "0.3b"
    assert oflm_add.gguf_size_token(0, "Q4_K") is None
    assert oflm_add.gguf_size_token(2_500_000_000, "IQ2_M") is None
    assert oflm_add.size_token_from_bytes(4_400_000_000) == "4.4b"
    assert oflm_add.size_token_from_bytes(None) is None


def test_gguf_byte_size_prefers_local_stat(tmp_path):
    gguf = tmp_path / "m-Q4_K_M.gguf"
    gguf.write_bytes(b"x" * 1024)
    assert oflm_add.gguf_byte_size("Org/M", False, tmp_path, None,
                                   "m-Q4_K_M.gguf", []) == 1024
    # HF tree listing (lfs size wins, plain size otherwise).
    tree = [{"path": "m-Q4_K_M.gguf",
             "lfs": {"oid": "a" * 64, "size": 2048}, "size": 2048}]
    assert oflm_add.gguf_byte_size("Org/M", False, None, None,
                                   "m-Q4_K_M.gguf", tree) == 2048
    # ModelScope dict listing.
    ms = {"m-Q4_K_M.gguf": {"Size": 4096}}
    assert oflm_add.gguf_byte_size("Org/M", True, None, None,
                                   "m-Q4_K_M.gguf", ms) == 4096
    assert oflm_add.gguf_byte_size("Org/M", False, None, None,
                                   "absent.gguf", tree) is None


def test_base_chain_climbs_and_stops_cycles(monkeypatch):
    frontmatters = {
        "User/Quant": ["User/Finetune"],
        "User/Finetune": ["google/base", "User/Quant"],  # cycle back up
        "google/base": [],
    }
    monkeypatch.setattr(oflm_add, "readme_base_model",
                        lambda repo, ms=False: frontmatters.get(repo, []))
    assert oflm_add.readme_base_chain("User/Quant") == ["User/Finetune", "google/base"]


def test_peek_config_prefers_local_and_climbs(monkeypatch, tmp_path):
    cfg = {"model_type": "qwen3", "hidden_size": 2560,
           "num_hidden_layers": 28, "vocab_size": 152064}
    (tmp_path / "config.json").write_text(json.dumps(cfg))
    got = oflm_add.peek_config_dict("Org/M", False, bases=["Org/Base"],
                                    local_dirs=[tmp_path])
    assert got["model_type"] == "qwen3"
    # No local file: the repo itself, then its bases, first hit wins.
    monkeypatch.setattr(oflm_add, "fetch_from_repo",
                        lambda r, f, dest, **kw: None if r == "Org/M" else dest)
    monkeypatch.setattr(oflm_add, "load_json", lambda p: {"model_type": "llama"})
    got = oflm_add.peek_config_dict("Org/M", False, bases=["Org/Base"],
                                    local_dirs=[tmp_path / "empty"])
    assert got["model_type"] == "llama"


def _dry_run_argv(repo, tmp_path):
    from pathlib import Path as _Path
    system_list = _Path(__file__).resolve().parents[3] / "src" / "model_list.json"
    return (["oflm-add", str(repo), "--system-list", str(system_list),
             "--config", str(tmp_path / "user_list.json"),
             "--models-root", str(tmp_path / "models"),
             "--dry-run", "--quiet"], system_list)


def test_named_local_q4nx_dry_run_unchanged(tmp_path, capsys, monkeypatch):
    """A well-named local q4nx repo plans exactly as before the heuristics.

    Guards the legacy path: name alias family, name size tag, official kernel
    match -- no detection fallback may fire here."""
    from pathlib import Path as _Path
    argv, system_list = _dry_run_argv(tmp_path / "Qwen3-4B-Custom-NPU2", tmp_path)
    if not system_list.is_file():
        return
    repo = tmp_path / "Qwen3-4B-Custom-NPU2"
    repo.mkdir()
    (repo / "model.q4nx").write_bytes(b"q4nx")
    (repo / "config.json").write_text(json.dumps({"model_type": "qwen3"}))
    (repo / "tokenizer.json").write_text("{}")
    (repo / "tokenizer_config.json").write_text("{}")
    monkeypatch.setattr("sys.argv", argv)
    oflm_add.main()
    out = capsys.readouterr().out
    assert "details.family : qwen3" in out
    assert "tag            : qwen3-custom:4b" in out
    assert "official match : qwen3:4b" in out
    assert "GGUF-direct" not in out


def test_chaotic_local_q4nx_dry_run_peeks_config(tmp_path, capsys, monkeypatch):
    """A marker-less local q4nx repo resolves via its own config.json.

    Family from model_type, tag size from the geometry estimate -- fully
    offline, so the peek must prefer the local file and do no network."""
    argv, system_list = _dry_run_argv(tmp_path / "ChaoticFinetune", tmp_path)
    if not system_list.is_file():
        return
    repo = tmp_path / "ChaoticFinetune"
    repo.mkdir()
    (repo / "model.q4nx").write_bytes(b"q4nx")
    (repo / "config.json").write_text(json.dumps({
        "model_type": "qwen3", "hidden_size": 2560, "num_hidden_layers": 28,
        "intermediate_size": 9728, "vocab_size": 152064}))
    (repo / "tokenizer.json").write_text("{}")
    (repo / "tokenizer_config.json").write_text("{}")
    monkeypatch.setattr("sys.argv", argv)
    oflm_add.main()
    out = capsys.readouterr().out
    assert "details.family : qwen3" in out
    assert "chaoticfinetune:5b" in out
    assert "official match : qwen3:4b" in out


def test_named_local_q4nx_full_install_registers(tmp_path, capsys, monkeypatch):
    """A local q4nx repo installs end to end: files copied, registry written.

    Fully offline (a README without frontmatter keeps the base chain local).
    This is the legacy path the GGUF automation must not disturb."""
    from pathlib import Path as _Path
    system_list = _Path(__file__).resolve().parents[3] / "src" / "model_list.json"
    if not system_list.is_file():
        return
    repo = tmp_path / "Qwen3-4B-Custom-NPU2"
    repo.mkdir()
    (repo / "model.q4nx").write_bytes(b"q4nx")
    (repo / "config.json").write_text(json.dumps({"model_type": "qwen3"}))
    (repo / "tokenizer.json").write_text("{}")
    (repo / "tokenizer_config.json").write_text("{}")
    (repo / "README.md").write_text("# local test repo\n")
    monkeypatch.setattr(
        "sys.argv",
        ["oflm-add", str(repo), "--system-list", str(system_list),
         "--config", str(tmp_path / "user_list.json"),
         "--models-root", str(tmp_path / "models"),
         "--xclbin-dir", str(tmp_path / "xclbins"), "--quiet"])
    oflm_add.main()
    target = tmp_path / "models" / "Qwen3-4B-Custom-NPU2"
    for name in ("model.q4nx", "config.json", "tokenizer.json",
                 "tokenizer_config.json"):
        assert (target / name).is_file()
    registry = json.loads((tmp_path / "user_list.json").read_text())
    entry = registry["models"]["qwen3-custom"]["4b"]
    assert entry["name"] == "Qwen3-4B-Custom-NPU2"
    assert entry["details"]["family"] == "qwen3"
    assert entry["details"]["format"] == "NPU2"
    assert entry["size"] == 4_000_000_000
    assert sorted(entry["files"]) == ["config.json", "model.q4nx",
                                      "tokenizer.json", "tokenizer_config.json"]


def test_user_install_never_touches_curated_list_or_kernels(tmp_path, monkeypatch):
    """The oflm-add invariant: a user install writes only user paths.

    The curated list is read, never written; kernel blobs are symlinked, never
    copied or downloaded; reinstalling a curated model lands under a different
    tag instead of overwriting it. A fake system root (via OFLM_EXECUTABLE)
    stands in for the install tree so the whole link flow runs offline."""
    import os
    from pathlib import Path as _Path
    checkout_list = _Path(__file__).resolve().parents[3] / "src" / "model_list.json"
    if not checkout_list.is_file():
        return
    curated = tmp_path / "system_list.json"
    curated.write_bytes(checkout_list.read_bytes())
    before = curated.read_bytes()

    # Fake install tree: <root>/share/oflm/xclbins/<official>/...
    fakebin = tmp_path / "fakebin"
    fakebin.mkdir()
    shipped = tmp_path / "share" / "oflm" / "xclbins" / "Qwen3-4B-NPU2"
    shipped.mkdir(parents=True)
    (shipped / "kernel.xclbin").write_bytes(b"kernels")
    monkeypatch.setenv("OFLM_EXECUTABLE", str(fakebin / "oflm"))

    repo = tmp_path / "Qwen3-4B-Custom-NPU2"
    repo.mkdir()
    (repo / "model.q4nx").write_bytes(b"q4nx")
    (repo / "config.json").write_text(json.dumps({"model_type": "qwen3"}))
    (repo / "tokenizer.json").write_text("{}")
    (repo / "tokenizer_config.json").write_text("{}")
    (repo / "README.md").write_text("# local test repo\n")
    monkeypatch.setattr(
        "sys.argv",
        ["oflm-add", str(repo), "--system-list", str(curated),
         "--config", str(tmp_path / "user_list.json"),
         "--models-root", str(tmp_path / "models"),
         "--xclbin-dir", str(tmp_path / "xclbins"), "--quiet"])
    oflm_add.main()

    # Curated list byte-identical; shipped kernels still a real dir, intact.
    assert curated.read_bytes() == before
    assert shipped.is_dir() and not shipped.is_symlink()
    assert (shipped / "kernel.xclbin").read_bytes() == b"kernels"
    # Both user links are symlinks into the shipped tree -- no blob was
    # copied or downloaded anywhere under the user paths.
    for name in ("Qwen3-4B-Custom-NPU2", "Qwen3-4B-NPU2"):
        link = tmp_path / "xclbins" / name
        assert link.is_symlink(), name
        assert os.readlink(link) == str(shipped)
    blobs = [p for p in (tmp_path / "xclbins").rglob("*")
             if p.is_file() and not p.is_symlink()]
    assert blobs == []
    # ... and the model lives under its own tag: the user list is a full
    # overlay seeded from the curated one, so the curated qwen3:4b entry must
    # be untouched while qwen3-custom:4b carries the install.
    registry = json.loads((tmp_path / "user_list.json").read_text())
    system = json.loads(before.decode())
    assert registry["models"]["qwen3"]["4b"] == system["models"]["qwen3"]["4b"]
    assert registry["models"]["qwen3-custom"]["4b"]["name"] == "Qwen3-4B-Custom-NPU2"


def test_chaotic_local_gguf_dry_run_detects_family_and_size(tmp_path, capsys,
                                                            monkeypatch):
    """A local dir with no family/size markers still plans an install.

    Family comes from the GGUF architecture, the tag size from its bytes and
    quant -- the llama.cpp-style automation for chaotic slugs. Fully offline:
    --dry-run returns before any download, --system-list points at the repo
    checkout (skipped when absent)."""
    from pathlib import Path as _Path
    system_list = _Path(__file__).resolve().parents[3] / "src" / "model_list.json"
    if not system_list.is_file():
        return
    repo = tmp_path / "MyCoolFinetune"
    repo.mkdir()
    kv = [("general.architecture", (8, "qwen3"))]
    gguf = repo / "mycool-Q4_K_M.gguf"
    gguf.write_bytes(_gguf_bytes(kv, [("w", 12, [8, 8]), ("n", 0, [8])]))
    with open(gguf, "r+b") as f:  # sparse: stat sees ~2.5 GB, reads stay small
        f.truncate(2_500_000_000)
    monkeypatch.setattr(
        "sys.argv",
        ["oflm-add", str(repo), "--system-list", str(system_list),
         "--config", str(tmp_path / "user_list.json"),
         "--models-root", str(tmp_path / "models"), "--dry-run", "--quiet"])
    oflm_add.main()
    out = capsys.readouterr().out
    assert "details.family : qwen3" in out
    assert "mycoolfinetune:4.4b" in out
    assert "official match : qwen3:4b" in out


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
