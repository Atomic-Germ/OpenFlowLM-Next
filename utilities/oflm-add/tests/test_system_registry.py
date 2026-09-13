# Traces: OPEN-ADD-SYSTEM-REGISTRY (canonical spec: specs/open-engine/spec.md)
#
# Where oflm-add looks for the official model_list.json. The released engine
# installs as `flm`, not `oflm`, and the flag the failure message points at has
# to actually be read or there is no way out of the failure.
import json
import sys
from pathlib import Path

import pytest

OFLM_ADD = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(OFLM_ADD))

import oflm_add  # noqa: E402


@pytest.fixture
def nothing_on_path(monkeypatch, tmp_path):
    monkeypatch.setattr(oflm_add.shutil, "which", lambda n: None)
    monkeypatch.setattr(oflm_add, "SYSTEM_LIST_CANDIDATES", [])
    monkeypatch.setattr(oflm_add, "SYSTEM_XCLBIN_PREFIXES", [])


def write_list(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"models": {}}), encoding="utf-8")
    return path


def test_an_explicit_list_is_used_without_searching(nothing_on_path, tmp_path):
    mine = write_list(tmp_path / "elsewhere" / "model_list.json")
    assert oflm_add.find_system_model_list(mine) == mine


def test_an_explicit_list_that_is_not_there_is_refused_by_path(nothing_on_path, tmp_path):
    missing = tmp_path / "nope" / "model_list.json"
    with pytest.raises(SystemExit) as e:
        oflm_add.find_system_model_list(missing)
    assert str(missing) in str(e.value)


def test_the_released_engine_is_called_flm(monkeypatch, tmp_path):
    """The install on disk is flm.exe; looking only for `oflm` misses it."""
    write_list(tmp_path / "model_list.json")
    monkeypatch.setattr(oflm_add, "SYSTEM_LIST_CANDIDATES", [])
    monkeypatch.setattr(oflm_add.shutil, "which",
                        lambda n: str(tmp_path / "flm.exe") if n == "flm" else None)
    assert oflm_add.find_system_model_list() == tmp_path / "model_list.json"


def test_a_checkout_oflm_wins_over_an_installed_flm(monkeypatch, tmp_path):
    build, inst = tmp_path / "build", tmp_path / "inst"
    write_list(build / "model_list.json")
    write_list(inst / "model_list.json")
    monkeypatch.setattr(oflm_add, "SYSTEM_LIST_CANDIDATES", [])
    monkeypatch.setattr(oflm_add.shutil, "which", lambda n: str(
        (build if n == "oflm" else inst) / f"{n}.exe"))
    assert oflm_add.find_system_model_list() == build / "model_list.json"


def test_the_xclbin_root_finds_an_installed_flm_too(monkeypatch, tmp_path):
    (tmp_path / "xclbins").mkdir()
    monkeypatch.setattr(oflm_add, "SYSTEM_XCLBIN_PREFIXES", [])
    monkeypatch.setattr(oflm_add.shutil, "which",
                        lambda n: str(tmp_path / "flm.exe") if n == "flm" else None)
    assert oflm_add.find_system_xclbin_root() == tmp_path / "xclbins"


def test_the_refusal_names_the_paths_it_tried(monkeypatch, tmp_path):
    """The old message named two directories that were never looked at."""
    monkeypatch.setattr(oflm_add, "SYSTEM_LIST_CANDIDATES", [str(tmp_path / "share" / "model_list.json")])
    monkeypatch.setattr(oflm_add.shutil, "which",
                        lambda n: str(tmp_path / n / f"{n}.exe") if n == "flm" else None)
    with pytest.raises(SystemExit) as e:
        oflm_add.find_system_model_list()
    msg = str(e.value)
    assert str(tmp_path / "share" / "model_list.json") in msg
    assert str(tmp_path / "flm" / "model_list.json") in msg
    assert "--system-list" in msg


def test_the_xclbin_link_falls_back_to_a_junction(monkeypatch, tmp_path):
    """Windows only grants the symlink privilege to admins and developer mode.
    link_open_kernels already handles that; the xclbins link did not."""
    system_root = tmp_path / "sys" / "xclbins"
    (system_root / "Off-NPU2").mkdir(parents=True)
    user_root = tmp_path / "user" / "xclbins"

    def no_privilege(*a, **k):
        raise OSError(1314, "A required privilege is not held by the client")

    monkeypatch.setattr(oflm_add.os, "symlink", no_privilege)
    oflm_add.link_xclbins(system_root, user_root, "Off-NPU2", "Off-NPU2", quiet=True)
    assert (user_root / "Off-NPU2").is_dir()
