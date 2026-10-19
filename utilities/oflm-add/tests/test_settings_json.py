"""Tests for oflm-add writing to settings.json (not model_list.json)."""
import json
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
from oflm_add import register, user_settings_path, save_json_atomic


@pytest.fixture
def temp_settings(tmp_path):
    return tmp_path / "settings.json"


@pytest.fixture
def system_registry():
    return {
        "model_path": "models",
        "models": {
            "qwen3.8": {
                "27b": {
                    "name": "qwen3.8-27b",
                    "url": "https://example.com/qwen3.8",
                    "default_context_length": 8192,
                }
            }
        },
    }


def make_entry(model_type="my-model", size="1b", **kwargs):
    entry = {
        "type": model_type,
        "size": size,
        "name": f"{model_type}-{size}",
        "url": f"https://example.com/{model_type}",
        "default_context_length": 4096,
    }
    entry.update(kwargs)
    return entry


class TestRegister:
    def test_writes_to_settings_json(self, temp_settings, system_registry):
        entry = make_entry()
        tag = register(entry, temp_settings, system_registry)
        assert tag == "my-model:1b"
        assert temp_settings.exists()
        with open(temp_settings) as f:
            s = json.load(f)
        assert s["models"]["my-model"]["1b"]["url"] == entry["url"]

    def test_does_not_touch_model_list_json(self, temp_settings, system_registry, tmp_path):
        """oflm add should NOT create or modify model_list.json."""
        model_list_path = tmp_path / "model_list.json"
        entry = make_entry()
        register(entry, temp_settings, system_registry)
        assert not model_list_path.exists()

    def test_creates_settings_file_if_missing(self, temp_settings, system_registry):
        assert not temp_settings.exists()
        register(make_entry(), temp_settings, system_registry)
        assert temp_settings.exists()
        with open(temp_settings) as f:
            s = json.load(f)
        assert s["version"] == 1
        assert "defaults" in s
        assert "settings" in s
        assert "models" in s

    def test_collision_with_shipped_model(self, temp_settings, system_registry):
        """Cannot install a model with the same tag as a shipped model."""
        entry = make_entry("qwen3.8", "27b")
        with pytest.raises(SystemExit, match="already exists in the shipped registry"):
            register(entry, temp_settings, system_registry)

    def test_collision_with_shipped_model_with_name_override(self, temp_settings, system_registry):
        """--name allows installing with a different tag even if shipped has the same."""
        entry = make_entry("qwen3.8", "27b")
        tag = register(entry, temp_settings, system_registry, name_override="my-qwen:27b")
        assert tag == "my-qwen:27b"
        with open(temp_settings) as f:
            s = json.load(f)
        assert "my-qwen" in s["models"]

    def test_collision_with_user_model(self, temp_settings, system_registry):
        """Cannot install over an existing user model without --force."""
        entry = make_entry()
        register(entry, temp_settings, system_registry)
        with pytest.raises(SystemExit, match="already installed"):
            register(entry, temp_settings, system_registry)

    def test_force_overwrites_user_model(self, temp_settings, system_registry):
        entry1 = make_entry(url="https://example.com/old")
        entry2 = make_entry(url="https://example.com/new")
        register(entry1, temp_settings, system_registry)
        register(entry2, temp_settings, system_registry, force=True)
        with open(temp_settings) as f:
            s = json.load(f)
        assert s["models"]["my-model"]["1b"]["url"] == "https://example.com/new"

    def test_preserves_existing_settings(self, temp_settings, system_registry):
        """oflm add should not clobber existing user settings."""
        # Pre-populate settings.json with a user preference
        with open(temp_settings, "w") as f:
            json.dump({
                "version": 1,
                "defaults": {"pmode": "balanced"},
                "settings": {"qwen3.8": {"27b": {"prefill_mode": "lean"}}},
                "models": {},
            }, f)

        register(make_entry(), temp_settings, system_registry)

        with open(temp_settings) as f:
            s = json.load(f)
        # User preferences preserved
        assert s["defaults"]["pmode"] == "balanced"
        assert s["settings"]["qwen3.8"]["27b"]["prefill_mode"] == "lean"
        # New model added
        assert "my-model" in s["models"]

    def test_name_override_changes_tag(self, temp_settings, system_registry):
        entry = make_entry("qwen3.8", "27b")
        tag = register(entry, temp_settings, system_registry, name_override="custom:27b")
        assert tag == "custom:27b"
        with open(temp_settings) as f:
            s = json.load(f)
        assert "custom" in s["models"]
        assert "27b" in s["models"]["custom"]


class TestUserSettingsPath:
    def test_default_path(self, tmp_path, monkeypatch):
        monkeypatch.setenv("OFLM_MODEL_PATH", str(tmp_path))
        monkeypatch.delenv("OFLM_SETTINGS_PATH", raising=False)
        path = user_settings_path()
        assert path.name == "settings.json"
        assert str(tmp_path) in str(path)

    def test_env_override(self, tmp_path, monkeypatch):
        custom = tmp_path / "custom_settings.json"
        monkeypatch.setenv("OFLM_SETTINGS_PATH", str(custom))
        assert user_settings_path() == custom

    def test_cli_override(self, tmp_path):
        custom = tmp_path / "cli_settings.json"
        assert user_settings_path(str(custom)) == custom


class TestSaveJsonAtomic:
    def test_creates_file(self, tmp_path):
        path = tmp_path / "test.json"
        save_json_atomic(path, {"key": "value"})
        with open(path) as f:
            assert json.load(f) == {"key": "value"}

    def test_no_temp_files_left(self, tmp_path):
        path = tmp_path / "test.json"
        save_json_atomic(path, {"key": "value"})
        files = list(tmp_path.iterdir())
        assert len(files) == 1
        assert files[0].name == "test.json"

    def test_overwrites_existing(self, tmp_path):
        path = tmp_path / "test.json"
        save_json_atomic(path, {"v": 1})
        save_json_atomic(path, {"v": 2})
        with open(path) as f:
            assert json.load(f) == {"v": 2}
