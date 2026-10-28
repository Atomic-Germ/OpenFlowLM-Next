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
    # A VISION family is its own family, never folded onto the text one it borrows kernels
    # from (qwen2vl runs on qwen2's set, and the adapter that prepares images is chosen by
    # the former).
    assert oflm_add.family_from_config_dict({"model_type": "qwen2_5_vl"}) == "qwen2vl"
    assert oflm_add.family_from_config_dict({"model_type": "qwen3_vl"}) == "qwen3vl"
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
    # ... and the model lives under its own tag. The user file is an OVERLAY
    # now, not a copy of the curated one: it holds what the user added and
    # nothing else, so the curated qwen3:4b entry is not in it at all -- the
    # shipped registry is read from its own place and the overlay is merged
    # on top. The install carries the derived tag, so it cannot collide with
    # the shipped qwen3:4b.
    registry = json.loads((tmp_path / "user_list.json").read_text())
    system = json.loads(before.decode())
    assert "qwen3" not in registry["models"], \
        "the user file must not carry a copy of the shipped registry"
    assert registry["models"]["qwen3-custom"]["4b"]["name"] == "Qwen3-4B-Custom-NPU2"
    # The curated list is still what ships the model, untouched.
    assert json.loads(curated.read_text())["models"]["qwen3"]["4b"] == \
        system["models"]["qwen3"]["4b"]


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
    # Same bytes, wrong claim: refuse rather than pack the wrong layout. The
    # name is a HINT, not a contract (llama.cpp mixes types by importance), so
    # what must hold is that no tensor is of the claimed family.
    ok, note = oflm_add.verify_gguf_choice(str(q4k), "m.gguf", "Q4_0")
    assert not ok and "Q4_0" in note and "refusing" in note
    # A file that MIXES types (a Q4_1 body with a Q6_K output) is fine: the
    # claimed family is present.
    mixed = tmp_path / "mix.gguf"
    mixed.write_bytes(_gguf_bytes(_llama_kv(), [("w", 12, [8, 8]), ("o", 14, [8, 8])]))
    ok, note = oflm_add.verify_gguf_choice(str(mixed), "mix.gguf", "Q4_K")
    assert ok, note
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


def test_a_quant_can_be_named_at_the_repo():
    """`repo:QUANT` picks that file out of a repo that carries one per quant."""
    assert oflm_add.split_quant_request("mradermacher/Qwen3.5-9B-i1-GGUF:IQ4_NL") == (
        "mradermacher/Qwen3.5-9B-i1-GGUF", "IQ4_NL")
    # a URL's port is not a quant request
    assert oflm_add.split_quant_request("https://example.com:8080/x") == (
        "https://example.com:8080/x", None)
    assert oflm_add.split_quant_request("Org/Name") == ("Org/Name", None)


def test_the_named_quant_wins_over_the_preference_order():
    names = ["M-IQ4_XS.gguf", "M-IQ4_NL.gguf", "M-Q5_K_S.gguf", "M-Q4_1.gguf"]
    # without a request the preference order decides (Q4_1 is byte-exact for the pool)
    assert oflm_add.choose_gguf_file(names)[0] == "M-Q4_1.gguf"
    # with one, the exact suffix wins
    assert oflm_add.choose_gguf_file(names, "IQ4_NL")[0] == "M-IQ4_NL.gguf"
    assert oflm_add.choose_gguf_file(names, "IQ4_XS")[0] == "M-IQ4_XS.gguf"
    # a family name takes the first file of that family
    assert oflm_add.choose_gguf_file(names, "IQ4")[0] in ("M-IQ4_NL.gguf", "M-IQ4_XS.gguf")
    # a quant the repo does not carry is refused, with what it does carry
    chosen, refused = oflm_add.choose_gguf_file(names, "Q6_K")
    assert chosen is None
    assert "no Q6_K GGUF" in refused["M-Q4_1.gguf"]


def test_gguf_repo_without_kernels_fails_directly_instead_of_q4nx_fallback(tmp_path, monkeypatch):
    """A GGUF repo whose family size has no GGUF-direct open kernels raises a
    direct error naming build-new-spec.py rather than falling back to model.q4nx."""
    from pathlib import Path as _Path
    system_list = _Path(__file__).resolve().parents[3] / "src" / "model_list.json"
    if not system_list.is_file():
        return
    repo = tmp_path / "Llama-3.2-1B-Instruct-GGUF"
    repo.mkdir()
    kv = [("general.architecture", (8, "llama"))]
    gguf = repo / "Llama-3.2-1B-Instruct-Q4_K_M.gguf"
    gguf.write_bytes(_gguf_bytes(kv, [("w", 12, [8, 8]), ("n", 0, [8])]))
    monkeypatch.setattr(oflm_add, "tree_xclbins", lambda: tmp_path / "no_tree_xclbins")
    monkeypatch.setattr(oflm_add, "find_system_xclbin_root", lambda: tmp_path / "no_sys_xclbins")
    monkeypatch.setattr(
        "sys.argv",
        ["oflm-add", f"{repo}:Q4_K_M", "--system-list", str(system_list),
         "--config", str(tmp_path / "user_list.json"),
         "--models-root", str(tmp_path / "models"),
         "--xclbin-dir", str(tmp_path / "xclbins"), "--quiet"])
    try:
        oflm_add.main()
    except SystemExit as e:
        msg = str(e)
        assert "no GGUF-direct open kernel set found" in msg
        assert "build-new-spec.py" in msg
        assert "model.q4nx" not in msg
    else:
        raise AssertionError("missing GGUF kernels must raise SystemExit directly")


def test_quant_request_on_repo_without_matching_quant_lists_files(tmp_path, monkeypatch):
    """Asking for repo:QUANT when that quant is absent lists the available quants and files."""
    repo = tmp_path / "Some-GGUF"
    repo.mkdir()
    (repo / "model-Q4_K_M.gguf").write_bytes(b"x")
    (repo / "model-Q8_0.gguf").write_bytes(b"x")
    monkeypatch.setattr("sys.argv", ["oflm-add", f"{repo}:Q4_1", "--quiet"])
    try:
        oflm_add.main()
    except SystemExit as e:
        msg = str(e)
        assert "no compatible 'Q4_1' GGUF" in msg
        assert "Q4_K_M, Q8_0" in msg
        assert "model-Q4_K_M.gguf, model-Q8_0.gguf" in msg
        assert "model.q4nx" not in msg
    else:
        raise AssertionError("missing requested quant must raise SystemExit")



# ---- config.json construction: the multimodal wrapper and the missing keys ----


def _kernel_set(tmp_path, check, defaults=None):
    kdir = tmp_path / "Qwen3.5-4B-NPU2" / "open_kernels"
    kdir.mkdir(parents=True)
    (kdir / "manifest.json").write_text(json.dumps({
        "hf_config_check": check,
        "hf_config_defaults": defaults or {},
    }), encoding="utf-8")
    return kdir.parent


LAYER_TYPES = ["linear_attention"] * 3 + ["full_attention"]


def _tower_config():
    """The shape empero-ai/Qwen3.8-4B and Qwen/Qwen3.5-4B ship: a VLM wrapper."""
    return {
        "model_type": "qwen3_5",
        "text_config": {
            "model_type": "qwen3_5_text",
            "hidden_size": 2560,
            "num_hidden_layers": 4,
            "vocab_size": 248320,
            "intermediate_size": 9216,
            "head_dim": 256,
            "num_attention_heads": 16,
            "num_key_value_heads": 4,
            "linear_num_value_heads": 32,
            "layer_types": list(LAYER_TYPES),
        },
        "vision_config": {"depth": 27},
    }


def _check_for(cfg):
    tower = cfg.get("text_config", cfg)
    return {k: v for k, v in tower.items() if k != "model_type"}


def test_a_multimodal_config_is_flattened_for_the_engines_top_level_checks(tmp_path):
    """The install used to succeed and the LOAD fail: 'config.json lacks layer_types'."""
    cfg = _tower_config()
    kdir = _kernel_set(tmp_path, _check_for(cfg))
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text(json.dumps(cfg), encoding="utf-8")

    oflm_add.reconcile_config_json(model, kdir)

    out = json.loads((model / "config.json").read_text(encoding="utf-8"))
    assert out["layer_types"] == LAYER_TYPES          # lifted out of text_config
    assert out["hidden_size"] == 2560
    assert out["text_config"]["layer_types"] == LAYER_TYPES   # the wrapper is kept
    assert out["vision_config"] == {"depth": 27}


def test_a_key_the_config_omits_entirely_comes_from_the_kernel_set(tmp_path):
    cfg = _tower_config()
    del cfg["text_config"]["layer_types"]
    kdir = _kernel_set(tmp_path, _check_for(_tower_config()))
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text(json.dumps(cfg), encoding="utf-8")

    filled, conflicting = oflm_add.reconcile_config_json(model, kdir)

    assert filled == ["layer_types"] and conflicting == []
    out = json.loads((model / "config.json").read_text(encoding="utf-8"))
    assert out["layer_types"] == LAYER_TYPES


def test_a_contradicting_config_is_reported_not_overwritten(tmp_path):
    """A different model must still be refused at load; add only says so earlier."""
    cfg = _tower_config()
    cfg["text_config"]["layer_types"] = ["full_attention"] * 4
    kdir = _kernel_set(tmp_path, _check_for(_tower_config()))
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text(json.dumps(cfg), encoding="utf-8")

    filled, conflicting = oflm_add.reconcile_config_json(model, kdir)

    assert conflicting == ["layer_types"] and filled == []
    out = json.loads((model / "config.json").read_text(encoding="utf-8"))
    assert out["text_config"]["layer_types"] == ["full_attention"] * 4


def test_a_key_with_a_manifest_default_may_stay_absent(tmp_path):
    cfg = _tower_config()
    del cfg["text_config"]["vocab_size"]
    kdir = _kernel_set(tmp_path, _check_for(_tower_config()), defaults={"vocab_size": 248320})
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text(json.dumps(cfg), encoding="utf-8")

    filled, conflicting = oflm_add.reconcile_config_json(model, kdir)

    assert filled == [] and conflicting == []
    assert "vocab_size" not in json.loads((model / "config.json").read_text())


# ---- tokenizer_config: an eos id the engine can read, and a chat template ----


def test_eos_falls_back_when_generation_config_exists_but_has_no_id(tmp_path):
    """Qwen3-4B-Base ships a generation_config.json WITHOUT eos_token_id; the file
    being found must not suppress the config.json fallback."""
    (tmp_path / "tokenizer_config.json").write_text(
        json.dumps({"model_max_length": 4096, "bos_token": "<|endoftext|>"}))
    (tmp_path / "generation_config.json").write_text(json.dumps({"do_sample": True}))
    (tmp_path / "config.json").write_text(json.dumps({"eos_token_id": 151643, "bos_token_id": 151643}))

    oflm_add.normalize_tokenizer_config(tmp_path, str(tmp_path), [])

    tc = json.loads((tmp_path / "tokenizer_config.json").read_text())
    assert tc["eos_token_id"] == [151643]
    assert tc["bos_token_id"] == 151643


def test_chat_template_is_climbed_from_the_chain(tmp_path):
    """The first repo with a tokenizer_config may have no template; a later one does."""
    calls = []

    def fake_fetch(repo, fname, dest, modelscope=False, force=False):
        calls.append(repo)
        templates = {"first/model": None, "second/model": "{% for m in messages %}{{ m['content'] }}{% endfor %}"}
        dest.write_text(json.dumps({"chat_template": templates.get(repo)}))
        return dest

    (tmp_path / "tokenizer_config.json").write_text(json.dumps({"model_max_length": 4096}))
    orig = oflm_add.fetch_from_repo
    oflm_add.fetch_from_repo = fake_fetch
    try:
        src = oflm_add.fill_chat_template(tmp_path, "first/model", ["second/model"])
    finally:
        oflm_add.fetch_from_repo = orig

    assert src == "second/model"
    assert calls == ["first/model", "second/model"]
    assert "chat_template" in json.loads((tmp_path / "tokenizer_config.json").read_text())


def test_official_repo_id_fills_hf_owner():
    entry = {"name": "Nanbeige4.1-3B-NPU2",
             "url": "https://huggingface.co/{hf_owner}/Nanbeige4.1-3B-NPU2"}
    assert oflm_add.official_repo_id(entry, {"hf_owner": "Atomic-Germ"}) == \
        "Atomic-Germ/Nanbeige4.1-3B-NPU2"
    assert oflm_add.official_repo_id({"name": "x", "url": "https://huggingface.co/acme/x"}, {}) == "acme/x"


def test_a_set_the_registry_does_not_list_still_names_its_family(tmp_path):
    """Our own conversion of a model we ship no container for: the manifest's spec
    says which family the kernels were built for, and that is enough to link."""
    kdir = tmp_path / "Nanbeige4.1-3B-NPU2" / "open_kernels"
    kdir.mkdir(parents=True)
    spec = {"family": "llama3", "hidden": 2560, "intermediate": 10496, "num_layers": 32,
            "layer_types": ["dense"] * 32, "vocab": 166144, "num_heads": 20, "num_kv_heads": 4,
            "head_dim": 128, "rotary_dim": 128}
    (kdir / "manifest.json").write_text(json.dumps({"spec": spec}))

    class _Spec:                       # the model's own spec, same geometry
        def kernel_hash(self, ignore=()):
            return "sha256:same"

    model_spec = _Spec()
    orig = oflm_add._model_spec_class
    oflm_add._model_spec_class = lambda: type("M", (), {
        "from_dict": staticmethod(lambda d: _Spec()),
    })
    try:
        fam, name = oflm_add.family_from_kernels({"models": {}}, model_spec, [tmp_path])
    finally:
        oflm_add._model_spec_class = orig

    assert (fam, name) == ("llama3", "Nanbeige4.1-3B-NPU2")


def test_gguf_kernel_set_with_no_name_is_a_refusal_not_a_crash():
    """No official match and no geometry match: the caller must get None (and say so),
    not `Path / None`."""
    assert oflm_add.gguf_kernel_set(None, ["/nonexistent"]) is None


def test_a_token_named_without_an_id_is_resolved_from_the_gguf(tmp_path):
    """Hy-MT2's tokenizer_config names <|startoftext|> and gives bos_token_id: null,
    and the string is not in tokenizer.json's vocab to look up."""
    (tmp_path / "tokenizer_config.json").write_text(json.dumps({
        "bos_token": "<|startoftext|>", "bos_token_id": None,
        "eos_token_id": [127960],
    }))
    (tmp_path / "tokenizer.json").write_text(json.dumps({"model": {"vocab": {}}}))

    class _G:
        @staticmethod
        def read(name, limit_mb=32):
            return ({"tokenizer.ggml.bos_token_id": 127958}, [])

    orig_inv, orig_tok = oflm_add.read_gguf_inventory, oflm_add.token_id_of
    (tmp_path / "model.gguf").write_bytes(b"GGUF")
    oflm_add.read_gguf_inventory = lambda *a, **k: _G.read(a[0])
    oflm_add.token_id_of = lambda target, tok: None
    try:
        oflm_add.normalize_tokenizer_config(tmp_path, "local", [])
    finally:
        oflm_add.read_gguf_inventory, oflm_add.token_id_of = orig_inv, orig_tok

    tc = json.loads((tmp_path / "tokenizer_config.json").read_text())
    assert tc["bos_token_id"] == 127958


def test_a_baked_rope_table_makes_rope_scaling_irrelevant(tmp_path):
    """llama.cpp bakes Llama-3's scaling into rope_freqs.weight; the GGUF's metadata then
    has no rope_scaling, and the set that serves it is the scaled one."""
    src = tmp_path / "m.gguf"
    src.write_bytes(b"GGUF")
    orig = oflm_add.read_gguf_inventory
    oflm_add.read_gguf_inventory = lambda p, limit_mb=96: ({"general.architecture": "llama"},
                                                          [("rope_freqs.weight", 0, [64])])
    try:
        assert oflm_add.rope_ignore(str(src)) == ("rope_scaling",)
    finally:
        oflm_add.read_gguf_inventory = orig
    oflm_add.read_gguf_inventory = lambda p, limit_mb=96: ({"general.architecture": "llama"},
                                                          [("token_embd.weight", 0, [128])])
    try:
        assert oflm_add.rope_ignore(str(src)) == ()
    finally:
        oflm_add.read_gguf_inventory = orig


def test_a_stub_config_is_rebuilt_from_the_gguf(tmp_path):
    """A gated base leaves {"model_type": "mistral"}; the GGUF names the geometry, and the
    engine's sampler segfaults without it."""
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "mistral"}))
    (tmp_path / "model.gguf").write_bytes(b"GGUF")

    class _Spec:
        family = "llama3"; hidden = 3072; num_layers = 28; intermediate = 8192
        vocab = 128256; num_heads = 24; num_kv_heads = 8; head_dim = 128
        rope_theta = 10000.0; norm_eps = 1e-5

    orig_spec, orig_checkout = oflm_add.spec_from_gguf_source, oflm_add.open_kernels_checkout
    oflm_add.spec_from_gguf_source = lambda s: _Spec()
    oflm_add.open_kernels_checkout = lambda: None
    try:
        oflm_add.repair_config_from_gguf(tmp_path)
    finally:
        oflm_add.spec_from_gguf_source, oflm_add.open_kernels_checkout = orig_spec, orig_checkout

    cfg = json.loads((tmp_path / "config.json").read_text())
    assert cfg["hidden_size"] == 3072 and cfg["vocab_size"] == 128256
    assert cfg["model_type"] == "llama"


def test_the_two_family_maps_keep_their_own_spellings():
    """GGUF arch names and HF model_type names are different vocabularies (k2-horizon vs
    k2_horizon, hunyuan-dense vs hunyuan_v1_dense). Mixing them into one map is how a
    correct config stops deriving a family."""
    assert oflm_add.GGUF_ARCH_FAMILIES["k2-horizon"] == "k2"
    assert oflm_add.family_from_config_dict({"model_type": "k2_horizon"}) == "k2"
    assert oflm_add.GGUF_ARCH_FAMILIES["hunyuan-dense"] == "hunyuan"
    assert oflm_add.family_from_config_dict({"model_type": "hunyuan_v1_dense"}) == "hunyuan"
    # a GGUF arch must not resolve through the config table and vice versa
    assert "hunyuan_v1_dense" not in oflm_add.GGUF_ARCH_FAMILIES


def test_an_i1_repo_takes_its_projector_from_the_sibling():
    """A mradermacher -i1 repo has no mmproj; the non-i1 repo of the same model does, and
    the only difference in the id is that marker."""
    assert oflm_add.projector_sibling("mradermacher/Qwen3-VL-4B-Instruct-i1-GGUF") == \
        "mradermacher/Qwen3-VL-4B-Instruct-GGUF"
    assert oflm_add.projector_sibling("mradermacher/gemma-3-4b-it-i1-GGUF") == \
        "mradermacher/gemma-3-4b-it-GGUF"
    # nothing to strip, and no repo part
    assert oflm_add.projector_sibling("Qwen/Qwen3-VL-4B-Instruct-GGUF") == ""
    assert oflm_add.projector_sibling("Qwen3-VL-4B-Instruct") == ""
    # "-i1" inside a model name is not the marker
    assert oflm_add.projector_sibling("someone/foo-i10-GGUF") == ""


def test_the_projector_prefers_the_lossless_quant(tmp_path):
    """The tower is host bf16: BF16/F16 needs no dequant at all, an 8-bit projector is
    smaller AND slower. Picking by file size would choose the Q8_0."""
    (tmp_path / "model.gguf").write_bytes(b"GGUF")
    seen = {}

    def fake_fetch(repo, fname, dest, modelscope=False, force=False):
        seen["repo"], seen["fname"] = repo, fname
        Path(dest).write_bytes(b"GGUF")
        return Path(dest)

    orig_fetch, orig_list, orig_inv = (oflm_add.fetch_from_repo, oflm_add._projectors_in,
                                       oflm_add.read_gguf_inventory)
    oflm_add._projectors_in = lambda repo, ms: (
        ["mmproj-Q8_0.gguf", "mmproj-f16.gguf", "mmproj-BF16.gguf"] if "sibling" in repo else [])
    oflm_add.fetch_from_repo = fake_fetch
    oflm_add.read_gguf_inventory = lambda p, limit_mb=8: (
        {"general.architecture": "clip", "clip.has_vision_encoder": True,
         "clip.projector_type": "qwen3vl_merger", "clip.vision.embedding_length": 1024,
         "clip.vision.block_count": 24}, [])
    try:
        got = oflm_add.install_projector(tmp_path, "org/model-i1-sibling", quiet=True)
    finally:
        (oflm_add.fetch_from_repo, oflm_add._projectors_in, oflm_add.read_gguf_inventory) = (
            orig_fetch, orig_list, orig_inv)
    assert got == "mmproj.gguf"
    assert seen["repo"] == "org/model-i1-sibling"      # it has its own projector, no sibling climb
    assert seen["fname"] == "mmproj-BF16.gguf"


def test_a_file_that_is_not_a_projector_is_dropped(tmp_path):
    """A language-model GGUF saved as mmproj.gguf would load as a tower and fail far from
    the install; it is checked BEFORE it is kept."""
    (tmp_path / "model.gguf").write_bytes(b"GGUF")

    def fake_fetch(repo, fname, dest, modelscope=False, force=False):
        Path(dest).write_bytes(b"GGUF")
        return Path(dest)

    orig = (oflm_add.fetch_from_repo, oflm_add._projectors_in, oflm_add.read_gguf_inventory)
    oflm_add._projectors_in = lambda repo, ms: ["mmproj-f16.gguf"]
    oflm_add.fetch_from_repo = fake_fetch
    oflm_add.read_gguf_inventory = lambda p, limit_mb=8: ({"general.architecture": "qwen3vl"}, [])
    try:
        got = oflm_add.install_projector(tmp_path, "org/model-GGUF", quiet=True)
    finally:
        (oflm_add.fetch_from_repo, oflm_add._projectors_in, oflm_add.read_gguf_inventory) = orig
    assert got == ""
    assert not (tmp_path / "mmproj.gguf").exists()


def test_qwen3vl_is_a_vl_model_on_qwen3_kernels():
    """The GGUF arch names the model (qwen3vl: the vision adapter); the spec that matches
    the kernels hashes as qwen3, because the decoder IS qwen3's."""
    assert oflm_add.GGUF_ARCH_FAMILIES["qwen3vl"] == "qwen3vl"
    assert oflm_add.family_from_config_dict({"model_type": "qwen3_vl"}) == "qwen3vl"


def test_a_gguf_vl_install_describes_its_tower_from_the_projector(tmp_path):
    """A GGUF VL model has no vision_config anywhere but the mmproj, and the adapter reads
    patch size / merge / mean from config.json. Translating the projector back into that
    block is what makes images preparable at all (patch size 0 otherwise)."""
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "qwen3_vl",
                                                      "vocab_size": 151936, "hidden_size": 2560,
                                                      "num_hidden_layers": 36}))
    (tmp_path / "model.gguf").write_bytes(b"GGUF")
    (tmp_path / "mmproj.gguf").write_bytes(b"GGUF")
    (tmp_path / "tokenizer.json").write_text(json.dumps({"added_tokens": [
        {"id": 151655, "content": "<|image_pad|>"},
        {"id": 151652, "content": "<|vision_start|>"},
        {"id": 151653, "content": "<|vision_end|>"}]}))

    class _Spec:
        family = "qwen3"; hidden = 2560; num_layers = 36; intermediate = 9728; vocab = 151936
        num_heads = 32; num_kv_heads = 8; head_dim = 128; rope_theta = 5000000.0; norm_eps = 1e-6

    clip = ({"general.architecture": "clip", "clip.has_vision_encoder": True,
             "clip.projector_type": "qwen3vl_merger",
             "clip.vision.block_count": 24, "clip.vision.embedding_length": 1024,
             "clip.vision.attention.head_count": 16, "clip.vision.feed_forward_length": 4096,
             "clip.vision.projection_dim": 2560, "clip.vision.patch_size": 16,
             "clip.vision.spatial_merge_size": 2},
            [["v.patch_embd.weight", 1, [16, 16, 3, 1024]],
             ["v.patch_embd.weight.1", 1, [16, 16, 3, 1024]],
             ["v.position_embd.weight", 0, [1024, 2304]],
             ["v.deepstack.5.fc1.weight", 1, [4096, 4096]],
             ["v.deepstack.11.fc1.weight", 1, [4096, 4096]],
             ["v.deepstack.17.fc1.weight", 1, [4096, 4096]]])
    mrope = {"general.architecture": "qwen3vl",
             "qwen3vl.rope.dimension_sections": [24, 20, 20, 0]}

    orig = (oflm_add.read_gguf_inventory, oflm_add.spec_from_gguf_source, oflm_add.open_kernels_checkout)
    oflm_add.read_gguf_inventory = lambda p, limit_mb=8: (
        clip if str(p).endswith("mmproj.gguf") else (mrope, []))
    oflm_add.spec_from_gguf_source = lambda s: _Spec()
    oflm_add.open_kernels_checkout = lambda: None
    try:
        oflm_add.repair_config_from_gguf(tmp_path)
    finally:
        oflm_add.read_gguf_inventory, oflm_add.spec_from_gguf_source, oflm_add.open_kernels_checkout = orig

    cfg = json.loads((tmp_path / "config.json").read_text())
    vc = cfg["vision_config"]
    assert vc["depth"] == 24 and vc["hidden_size"] == 1024 and vc["num_heads"] == 16
    assert vc["patch_size"] == 16 and vc["spatial_merge_size"] == 2 and vc["in_channels"] == 3
    assert vc["temporal_patch_size"] == 2            # two patch_embd slices
    assert vc["num_position_embeddings"] == 2304     # the position table's own shape
    assert vc["deepstack_visual_indexes"] == [5, 11, 17]
    assert cfg["vision_model_weight"] == "mmproj.gguf"
    assert cfg["image_token_id"] == 151655 and cfg["vision_start_token_id"] == 151652
    assert cfg["rope_parameters"]["mrope_section"] == [24, 20, 20]
    assert cfg["rope_parameters"]["mrope_interleaved"] is True


def test_a_vision_checkpoint_is_named_by_its_arch_not_its_kernels():
    """Qwen2.5-VL runs on the plain Qwen2.5-3B set (same decoder; M-RoPE is config), so the
    KERNEL family is qwen2. The model is still qwen2vl -- that is what picks the adapter
    that prepares images -- so the arch has to win over the kernel set."""
    assert oflm_add.GGUF_ARCH_FAMILIES["qwen2vl"] == "qwen2vl"
    assert oflm_add.family_from_config_dict({"model_type": "qwen2_5_vl"}) == "qwen2vl"
    assert oflm_add.family_from_config_dict({"model_type": "qwen2_5_vl_text"}) == "qwen2vl"
    assert oflm_add.GGUF_ARCH_FAMILIES["qwen3vl"] == "qwen3vl"


def test_a_bundle_links_the_kernel_set_whose_layout_matches(tmp_path):
    """A diffusion bundle names its kernels by LAYOUT HASH (the stream specs), not by a
    model geometry, so the match is exact -- a set built for other resolutions or edits
    must not be linked."""
    import json

    good = tmp_path / "xclbins" / "FLUX.2-klein-4B-NPU2" / "open_kernels"
    bad = tmp_path / "xclbins" / "Other" / "open_kernels"
    for d, layout in ((good, "e667"), (bad, "c327")):
        d.mkdir(parents=True)
        (d / "diffusion_kernels.json").write_text(json.dumps({"layout": layout, "complete": True}))
    assert oflm_add.find_diffusion_kernels("e667", [tmp_path / "xclbins"]) == good
    assert oflm_add.find_diffusion_kernels("dead", [tmp_path / "xclbins"]) is None
    # an incomplete set is not a candidate even when the layout matches
    (bad / "diffusion_kernels.json").write_text(json.dumps({"layout": "e667", "complete": False}))
    assert oflm_add.find_diffusion_kernels("e667", [bad.parent, tmp_path / "xclbins"]) == good


def test_a_symlinked_bundle_target_is_replaced_not_written_through(tmp_path, monkeypatch):
    """register-bundle.py left models/<name> as a link to the build directory. Installing
    with --force THROUGH that link unlinked the source's files one at a time (it ate a
    LICENSE.md), so the link has to be replaced first."""
    import argparse
    import json

    build = tmp_path / "build"
    build.mkdir()
    (build / "bundle.json").write_text(json.dumps({"layout": "L", "resolutions": {}, "edits": {}}))
    (build / "LICENSE.md").write_text("apache")
    (build / "weights.bin").write_bytes(b"w")

    root = tmp_path / "models"
    root.mkdir()
    (root / "my-klein").symlink_to(build)

    monkeypatch.setattr(oflm_add, "models_root_dir", lambda arg: root)
    monkeypatch.setattr(oflm_add, "find_system_model_list", lambda: tmp_path / "list.json")
    (tmp_path / "list.json").write_text(json.dumps({"models": {"flux2-klein": {"4b": {
        "name": "FLUX.2-klein-4B-NPU2", "details": {"family": "flux2-klein"}}}}}))
    monkeypatch.setattr(oflm_add, "register", lambda entry, *a, **k: "my-klein:512")
    monkeypatch.setattr(oflm_add, "log", lambda *a, **k: None)

    args = argparse.Namespace(models_root=None, system_list=None, tag="my-klein:512", name=None,
                              force=True, dry_run=False, no_xclbin=True, config=None)
    oflm_add.install_bundle(args, build, "my-klein")

    assert not (root / "my-klein").is_symlink()          # a real directory now
    assert (root / "my-klein" / "LICENSE.md").read_text() == "apache"
    assert (build / "LICENSE.md").is_file()              # the SOURCE is untouched
