"""Traces: OPEN-CONVERT-EOS-GENCONFIG (canonical spec: specs/open-engine/spec.md)."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from q4nx.model_assets import merge_generation_eos  # noqa: E402


class GenerationEosTest(unittest.TestCase):
    def run_merge(self, tok_eos, gen):
        with tempfile.TemporaryDirectory() as out, tempfile.TemporaryDirectory() as src:
            (Path(out) / "tokenizer_config.json").write_text(json.dumps({"eos_token": "<e>", "eos_token_id": tok_eos}))
            if gen is not None:
                (Path(src) / "generation_config.json").write_text(json.dumps(gen))
            merge_generation_eos(Path(out), [src])
            return json.loads((Path(out) / "tokenizer_config.json").read_text())["eos_token_id"]

    def test_k2_end_of_turn_added(self):
        self.assertEqual(self.run_merge([1], {"eos_token_id": [1, 250019]}), [1, 250019])

    def test_scalar_ids(self):
        self.assertEqual(self.run_merge(1, {"eos_token_id": 7}), [1, 7])

    def test_already_covered_untouched(self):
        self.assertEqual(self.run_merge([128009, 128001], {"eos_token_id": [128001, 128009]}), [128009, 128001])

    def test_no_generation_config(self):
        self.assertEqual(self.run_merge([2], None), [2])

    def test_no_eos_in_generation_config(self):
        self.assertEqual(self.run_merge([2], {"bos_token_id": 0}), [2])


if __name__ == "__main__":
    unittest.main()
