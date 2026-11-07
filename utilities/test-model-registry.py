#!/usr/bin/env python3
"""Exercise the real CLI's shipped/user registry merge without models or hardware.

Usage: python3 utilities/test-model-registry.py /path/to/oflm src/model_list.json
All configuration and migration writes are confined to a temporary directory.
"""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


EXECUTABLE = str(Path(sys.argv[1]).resolve())
REGISTRY = json.loads(Path(sys.argv[2]).read_text())
del sys.argv[1:]


class RegistryMergeTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="oflm-registry-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.models = self.root / "user"
        self.models.mkdir()
        self.shipped_path = self.root / "shipped.json"
        self.settings_path = self.models / "settings.json"
        self.shipped = copy.deepcopy(REGISTRY)
        self.entry = copy.deepcopy(REGISTRY["models"]["qwen3.8"]["27b"])
        self.shipped["models"] = {"qwen3.8": {"27b": self.entry}}
        self.write(self.shipped_path, self.shipped)
        self.env = {k: v for k, v in os.environ.items() if not k.startswith("OFLM_")}
        self.env.update(OFLM_CONFIG_PATH=str(self.shipped_path),
                        OFLM_SETTINGS_PATH=str(self.settings_path),
                        OFLM_MODEL_PATH=str(self.models))

    @staticmethod
    def write(path, data):
        path.write_text(json.dumps(data))

    def cli(self, *args):
        result = subprocess.run([EXECUTABLE, *args], env=self.env, cwd=self.root,
                                text=True, capture_output=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return json.loads(result.stdout)

    def listed(self):
        return {m["name"]: m for m in self.cli("list", "--json")["models"]}

    def test_overlay_keeps_shipped_and_user_models_across_registry_updates(self):
        settings = {"version": 1, "defaults_version": self.shipped["defaults_version"],
                    "models": {"local-test": {"27b": self.entry}},
                    "settings": {"qwen3.8": {"27b": {"ctx_len": 4096}}}}
        self.write(self.settings_path, settings)
        shipped_bytes = self.shipped_path.read_bytes()
        settings_bytes = self.settings_path.read_bytes()
        self.assertEqual(set(self.listed()), {"qwen3.8:27b", "local-test:27b"})
        resolved = self.cli("config", "show", "--json", "--config-model", "qwen3.8:27b")
        self.assertEqual(resolved["ctx_len"]["value"], "4096")
        self.assertEqual(self.shipped_path.read_bytes(), shipped_bytes)
        # A later shipped catalogue must remain visible beside existing user models.
        self.shipped["models"]["new-shipped"] = {"27b": self.entry}
        self.write(self.shipped_path, self.shipped)
        self.assertEqual(set(self.listed()),
                         {"qwen3.8:27b", "local-test:27b", "new-shipped:27b"})
        self.assertEqual(self.settings_path.read_bytes(), settings_bytes)

    def test_user_model_collision_is_one_complete_user_entry(self):
        custom = copy.deepcopy(self.entry)
        custom["default_context_length"] = 4096
        custom.pop("url", None)
        self.write(self.settings_path, {"version": 1,
                   "defaults_version": self.shipped["defaults_version"],
                   "models": {"qwen3.8": {"27b": custom}}})
        listed = self.listed()
        self.assertEqual(set(listed), {"qwen3.8:27b"})
        self.assertEqual(listed["qwen3.8:27b"]["default_context_length"], 4096)
        self.assertNotIn("url", listed["qwen3.8:27b"])
        resolved = self.cli("config", "show", "--json", "--config-model", "qwen3.8:27b")
        self.assertEqual(resolved["ctx_len"]["value"], "4096")

    def test_first_run_migrates_legacy_models_and_settings_once(self):
        legacy = copy.deepcopy(self.shipped)
        legacy["models"]["legacy-local"] = {"27b": copy.deepcopy(self.entry)}
        legacy["models"]["qwen3.8"]["27b"]["default_context_length"] = 4096
        legacy_path = self.models / "model_list.json"
        self.write(legacy_path, legacy)
        before = legacy_path.read_bytes()
        shipped_bytes = self.shipped_path.read_bytes()
        self.assertEqual(set(self.listed()), {"qwen3.8:27b", "legacy-local:27b"})
        self.assertFalse(legacy_path.exists())
        self.assertEqual(legacy_path.with_suffix(".json.bak").read_bytes(), before)
        migrated = self.settings_path.read_bytes()
        resolved = self.cli("config", "show", "--json", "--config-model", "qwen3.8:27b")
        self.assertEqual(resolved["ctx_len"]["value"], "4096")
        self.assertEqual(set(self.listed()), {"qwen3.8:27b", "legacy-local:27b"})
        self.assertEqual(self.settings_path.read_bytes(), migrated)
        self.assertEqual(self.shipped_path.read_bytes(), shipped_bytes)


if __name__ == "__main__":
    unittest.main()
