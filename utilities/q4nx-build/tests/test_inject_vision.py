"""Tests for inject_flm_keys vision_config precedence (skeleton vs arch config)."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from q4nx.model_assets import inject_flm_keys  # noqa: E402

# The arch config carries the 9B tower's numbers (the historical hardcode).
ARCH_VC = {
    "vision_mm_engine_xclbin_name": "vision_mm.xclbin",
    "QWEN3_5_PATCH_SIZE": 16,
    "QWEN3_5_VISION_EMBED_DIM": 1152,
    "QWEN3_5_VISION_NUM_LAYERS": 27,
    "QWEN3_5_VISION_OUT_HIDDEN_SIZE": 4096,
}
# A skeleton's config.json ships the numbers matching its own blob.
SKELETON_VC = {
    "vision_mm_engine_xclbin_name": "vision_mm.xclbin",
    "QWEN3_5_PATCH_SIZE": 16,
    "QWEN3_5_VISION_EMBED_DIM": 1024,
    "QWEN3_5_VISION_NUM_LAYERS": 24,
    "QWEN3_5_VISION_OUT_HIDDEN_SIZE": 2048,
}


class InjectVisionConfigTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.out = Path(self.tmp.name)
        (self.out / "vision_weight.q4nx").write_bytes(b"\x00")

    def tearDown(self):
        self.tmp.cleanup()

    def _q4nx_config(self):
        return {
            "addr_qk": 53248,
            "vision_config": {"vision_file": "vision_weight.q4nx", **ARCH_VC},
        }

    def test_skeleton_vision_config_wins_over_arch_config(self):
        config = {"model_type": "qwen3_5", "hidden_size": 2048, "vision_config": dict(SKELETON_VC)}
        result = inject_flm_keys(config, self._q4nx_config(), self.out, flm_version="1.0.0")
        vc = result["vision_config"]
        self.assertEqual(vc["QWEN3_5_VISION_EMBED_DIM"], 1024)
        self.assertEqual(vc["QWEN3_5_VISION_NUM_LAYERS"], 24)
        # projector width still follows THIS model's hidden size
        self.assertEqual(vc["QWEN3_5_VISION_OUT_HIDDEN_SIZE"], 2048)

    def test_arch_config_used_when_no_skeleton_vision_config(self):
        config = {"model_type": "qwen3_5", "hidden_size": 2560}  # pure-HF source
        result = inject_flm_keys(config, self._q4nx_config(), self.out, flm_version=None)
        vc = result["vision_config"]
        self.assertEqual(vc["QWEN3_5_VISION_EMBED_DIM"], 1152)
        self.assertEqual(vc["QWEN3_5_VISION_OUT_HIDDEN_SIZE"], 2560)

    def test_hf_style_nested_vision_config_is_rebuilt(self):
        # HF sources carry a nested vision blob without engine keys; that must
        # not count as a skeleton-provided block.
        hf_vc = {k: v for k, v in SKELETON_VC.items() if "xclbin" not in k}
        config = {"model_type": "qwen3_5", "hidden_size": 2048, "vision_config": hf_vc}
        result = inject_flm_keys(config, self._q4nx_config(), self.out, flm_version=None)
        self.assertEqual(
            result["vision_config"]["vision_mm_engine_xclbin_name"], "vision_mm.xclbin"
        )

    def test_no_vision_weight_leaves_config_text_only(self):
        (self.out / "vision_weight.q4nx").unlink()
        config = {"model_type": "qwen3_5", "hidden_size": 2048, "vision_config": dict(SKELETON_VC)}
        result = inject_flm_keys(config, self._q4nx_config(), self.out, flm_version=None)
        self.assertNotIn("vision_model_weight", result)


if __name__ == "__main__":
    unittest.main()
