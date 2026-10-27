"""Traces: OPEN-CONVERT-FAMILY-TAG (canonical spec: specs/open-engine/spec.md)."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from q4nx.model_assets import oflm_family_of  # noqa: E402


class OflmFamilyTest(unittest.TestCase):
    def family(self, cfg):
        with tempfile.TemporaryDirectory() as out:
            (Path(out) / "config.json").write_text(json.dumps(cfg))
            return oflm_family_of(Path(out))

    def test_k2_horizon(self):
        self.assertEqual(self.family({"model_type": "k2_horizon"}), "k2")

    def test_model_type_case_insensitive(self):
        self.assertEqual(self.family({"model_type": "K2_Horizon"}), "k2")

    def test_unknown_model_type(self):
        self.assertIsNone(self.family({"model_type": "no_such_arch"}))


if __name__ == "__main__":
    unittest.main()
