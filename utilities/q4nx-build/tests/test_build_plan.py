"""Unit tests for q4nx.build_plan derivation logic (no network required)."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from q4nx.build_plan import (  # noqa: E402
    BuildPlan,
    derive_build_plan,
    derive_display_name,
    find_skeleton,
    format_chain,
    infer_weights_type,
    normalize_base_models,
    parse_size_token,
    walk_base_chain,
)


class NormalizeBaseModelsTest(unittest.TestCase):
    def test_string_form(self):
        self.assertEqual(normalize_base_models("numind/NuExtract3"), ["numind/NuExtract3"])

    def test_list_form(self):
        self.assertEqual(
            normalize_base_models(["Qwen/Qwen3.5-4B-Base"]), ["Qwen/Qwen3.5-4B-Base"]
        )

    def test_none_and_garbage(self):
        self.assertEqual(normalize_base_models(None), [])
        self.assertEqual(normalize_base_models(42), [])
        self.assertEqual(normalize_base_models(["", "  "]), [])

    def test_url_flattened_and_deduped(self):
        value = [
            "https://huggingface.co/Qwen/Qwen3.5-4B",
            "Qwen/Qwen3.5-4B",
        ]
        self.assertEqual(normalize_base_models(value), ["Qwen/Qwen3.5-4B"])


class WalkBaseChainTest(unittest.TestCase):
    CARDS = {
        "org/quant": {"base_model": "org/finetune"},
        "org/finetune": {"base_model": ["Qwen/Qwen3.5-4B"]},
        "Qwen/Qwen3.5-4B": {"base_model": ["Qwen/Qwen3.5-4B-Base"]},
        "Qwen/Qwen3.5-4B-Base": {},
    }

    def fetch(self, repo_id):
        return self.CARDS.get(repo_id, {})

    def test_walks_to_root(self):
        chain = walk_base_chain("org/quant", fetch=self.fetch)
        self.assertEqual(
            chain,
            ["org/quant", "org/finetune", "Qwen/Qwen3.5-4B", "Qwen/Qwen3.5-4B-Base"],
        )

    def test_cycle_guard(self):
        cards = {"a/b": {"base_model": "c/d"}, "c/d": {"base_model": "a/b"}}
        chain = walk_base_chain("a/b", fetch=lambda r: cards[r])
        self.assertEqual(chain, ["a/b", "c/d"])

    def test_depth_limit(self):
        cards = {f"o/m{i}": {"base_model": f"o/m{i+1}"} for i in range(20)}
        chain = walk_base_chain("o/m0", max_depth=5, fetch=lambda r: cards[r])
        self.assertEqual(len(chain), 5)

    def test_offline_empty_cards(self):
        chain = walk_base_chain("any/repo", fetch=lambda r: {})
        self.assertEqual(chain, ["any/repo"])

    def test_format_chain(self):
        self.assertEqual(format_chain(["a/b", "c/d"]), "a/b -> c/d")


class FindSkeletonTest(unittest.TestCase):
    def test_nearest_ancestor_with_mirror_wins(self):
        # Qwen/Qwen3.5-4B has a mirror even though -Base is further up.
        existing = {"Atomic-Germ/Qwen3.5-4B-NPU2"}
        skeleton = find_skeleton(
            ["x/quant", "x/ft", "Qwen/Qwen3.5-4B", "Qwen/Qwen3.5-4B-Base"],
            probe=existing.__contains__,
        )
        self.assertEqual(skeleton, "Atomic-Germ/Qwen3.5-4B-NPU2")

    def test_org_order(self):
        calls = []

        def probe(candidate):
            calls.append(candidate)
            return candidate.startswith("FastFlowLM/")

        skeleton = find_skeleton(["Qwen/Qwen3-4B"], probe=probe)
        self.assertEqual(skeleton, "FastFlowLM/Qwen3-4B-NPU2")
        self.assertEqual(calls[0], "Atomic-Germ/Qwen3-4B-NPU2")

    def test_no_mirror_anywhere(self):
        self.assertIsNone(find_skeleton(["x/y"], probe=lambda c: False))

    def test_input_that_is_a_skeleton_is_skipped(self):
        seen = []
        find_skeleton(
            ["Atomic-Germ/Foo-NPU2", "base/Base"],
            probe=lambda c: seen.append(c) or True,
        )
        self.assertNotIn("Atomic-Germ/Foo-NPU2-NPU2", seen)


class ParseSizeTokenTest(unittest.TestCase):
    CASES = [
        ("Qwen3.5-4B-NPU2", "4B"),
        ("Qwen3.5-0.8B-NPU2", "0.8B"),
        ("Ornith-1.5-35B-A3B-NPU2", "35B-A3B"),
        ("Gemma4-E4B-IT-NPU2", "E4B"),
        ("medgemma-1.5-4b-it-NPU2", "4b"),
        ("MiniCPM-V-4.6-NPU2", None),
        ("DynaGuard-4B-NPU2", "4B"),
        ("LFM2.5-1.2B-Thinking-NPU2", "1.2B"),
        ("Darwin-36B-Opus-NPU2", "36B"),
        ("Nanbeige4.1-3B-NPU2", "3B"),
    ]

    def test_cases(self):
        for name, expected in self.CASES:
            with self.subTest(name=name):
                self.assertEqual(parse_size_token(name), expected)


class DeriveDisplayNameTest(unittest.TestCase):
    def test_prefers_model_name_field(self):
        card = {"model_name": "NuExtract3"}
        self.assertEqual(derive_display_name(card, "numind/NuExtract3-GGUF"), "NuExtract3")

    def test_strips_gguf_suffix_from_repo_name(self):
        self.assertEqual(derive_display_name({}, "numind/NuExtract3-GGUF"), "NuExtract3")
        self.assertEqual(derive_display_name({}, "someone/model-gguf"), "model")

    def test_keeps_plain_names(self):
        self.assertEqual(derive_display_name({}, "Jackrong/Qwopus3.5-9B-v3.5"), "Qwopus3.5-9B-v3.5")

    def test_spaces_become_dashes(self):
        self.assertEqual(derive_display_name({"model_name": "My Model X"}, "a/b"), "My-Model-X")


class InferWeightsTypeTest(unittest.TestCase):
    def test_vlm_pipeline_tag(self):
        self.assertEqual(
            infer_weights_type({"pipeline_tag": "image-text-to-text"}), "vision"
        )
        self.assertEqual(infer_weights_type({"pipeline_tag": "image-to-text"}), "vision")

    def test_text_pipeline_tag(self):
        self.assertEqual(infer_weights_type({"pipeline_tag": "text-generation"}), "language")

    def test_vision_via_tags_when_pipeline_missing(self):
        self.assertEqual(
            infer_weights_type({"tags": ["qwen3_5", "vision-language", "vlm"]}), "vision"
        )

    def test_defaults_to_language(self):
        self.assertEqual(infer_weights_type({}), "language")


class DeriveBuildPlanTest(unittest.TestCase):
    def test_full_plan_for_nuextract_style_repo(self):
        cards = {
            "numind/NuExtract3-GGUF": {
                "base_model": "numind/NuExtract3",
                "pipeline_tag": "image-text-to-text",
            },
            "numind/NuExtract3": {"base_model": ["Qwen/Qwen3.5-4B"], "model_name": "NuExtract3"},
            "Qwen/Qwen3.5-4B": {"base_model": ["Qwen/Qwen3.5-4B-Base"]},
        }
        mirrors = {"Atomic-Germ/Qwen3.5-4B-NPU2"}
        plan = derive_build_plan(
            "numind/NuExtract3-GGUF",
            fetch=lambda r: cards.get(r, {}),
            probe=mirrors.__contains__,
        )

        self.assertIsInstance(plan, BuildPlan)
        self.assertEqual(
            plan.chain,
            ["numind/NuExtract3-GGUF", "numind/NuExtract3", "Qwen/Qwen3.5-4B",
             "Qwen/Qwen3.5-4B-Base"],
        )
        self.assertEqual(plan.skeleton, "Atomic-Germ/Qwen3.5-4B-NPU2")
        self.assertEqual(plan.display_name, "NuExtract3")
        self.assertEqual(plan.size_token, "4B")
        self.assertEqual(plan.output_name, "NuExtract3-4B-NPU2")
        self.assertEqual(plan.weights_type, "vision")
        self.assertEqual(plan.pipeline_tag, "image-text-to-text")

    def test_no_skeleton_still_yields_a_name(self):
        plan = derive_build_plan("some/Model-X", fetch=lambda r: {}, probe=lambda c: False)
        self.assertIsNone(plan.skeleton)
        self.assertEqual(plan.output_name, "Model-X-NPU2")
        self.assertEqual(plan.weights_type, "language")

    def test_size_token_not_duplicated_when_name_has_it(self):
        cards = {"x/Qwen3.5-9B-Uncensored": {}}
        mirrors = {"Atomic-Germ/Qwen3.5-9B-NPU2"}
        plan = derive_build_plan(
            "x/Qwen3.5-9B-Uncensored",
            fetch=lambda r: cards.get(r, {}),
            probe=mirrors.__contains__,
        )
        self.assertEqual(plan.output_name, "Qwen3.5-9B-Uncensored-NPU2")

    def test_lowercase_size_token_not_duplicated(self):
        cards = {"x/medgemma-1.5-4b-it": {}}
        mirrors = {"FastFlowLM/medgemma-1.5-4b-it-NPU2"}
        plan = derive_build_plan(
            "x/medgemma-1.5-4b-it",
            fetch=lambda r: cards.get(r, {}),
            probe=mirrors.__contains__,
        )
        self.assertEqual(plan.output_name, "medgemma-1.5-4b-it-NPU2")

    def test_weights_reason_reports_the_signal(self):
        cards = {
            "a/via-pipeline": {"pipeline_tag": "image-text-to-text"},
            "a/via-tags": {"tags": ["vlm"]},
            "a/text": {"pipeline_tag": "text-generation"},
        }
        for repo, expected in [
            ("a/via-pipeline", "pipeline_tag image-text-to-text"),
            ("a/via-tags", "card tags"),
            ("a/text", ""),
        ]:
            with self.subTest(repo=repo):
                plan = derive_build_plan(repo, fetch=lambda r: cards.get(r, {}), probe=lambda c: False)
                self.assertEqual(plan.weights_reason, expected)


if __name__ == "__main__":
    unittest.main()
