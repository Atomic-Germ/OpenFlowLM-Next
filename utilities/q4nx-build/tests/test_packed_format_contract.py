"""What a packed directory must contain for the engines and kernels to read it.

Each of these failed at least once as real drift: the file, key or tensor a
consumer requires was not the one the packer wrote, and nothing in either suite
noticed because the two halves were tested separately.

  - Granite's folded multipliers reached config.json only on a code path Granite
    cannot take (HF), so every Granite pack shipped an UNFOLDED config and
    `spec.py` / `dense.py` refused it at load.
  - `--quant Q4_K` writes 4736-byte chunks, which the spec deriver's chunk table
    did not know, so those containers never derived a spec and `oflm add` could
    never find their kernels.
  - The no-source GGUF fallback wrote llama.cpp's architecture string as
    `model_type`, which no recipe claims, and omitted `rope_theta` / `head_dim`.
  - An HF source that ships no tokenizer_config.json produced a directory
    `oflm add` refuses outright.
  - `--build-spec` and `-t` were missing from the recorded pack command, so
    re-running the card produced a directory with no spec.json in it.
"""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from q4nx import model_assets  # noqa: E402
from q4nx.constants import ModelArch  # noqa: E402

REPO = Path(__file__).resolve().parents[1]

GRANITE_HD = 64
GRANITE_MULT = 0.015625


class _Field:
    def __init__(self, value):
        self._value = value

    def contents(self):
        return self._value


class _GGUFReader:
    """A GGUFReader stand-in: `.fields` keyed by llama.cpp metadata name.

    Carries no provenance keys, so `resolve_repo_candidates` finds nothing and a
    pack reads only the source files it already has -- no network.
    """

    def __init__(self, fields):
        self.fields = {k: _Field(v) for k, v in fields.items()}


GRANITE_GGUF = _GGUFReader({
    "general.architecture": "granite",
    "granite.embedding_length": 2560,
    "granite.block_count": 40,
    "granite.vocab_size": 100352,
    "granite.attention.head_count": 40,
    "granite.attention.head_count_kv": 8,
    "granite.rope.dimension_count": GRANITE_HD,
    "granite.rope.freq_base": 10000000.0,
    "granite.attention.layer_norm_rms_epsilon": 1e-05,
    "granite.attention.scale": GRANITE_MULT,
})

LLAMA_GGUF = _GGUFReader({
    "general.architecture": "llama",
    "llama.embedding_length": 2048,
    "llama.attention.head_count": 16,
    "llama.rope.dimension_count": 128,
})


def _granite_q4nx_config():
    with open(REPO / "configs" / "granite.json", encoding="utf-8") as f:
        return json.load(f)["q4nx_config"]


class GranitePackTest(unittest.TestCase):
    """The fold that decides whether a Granite container can run at all.

    `models/granite.py` folds the four multipliers into the weights, and
    `open_kernels/recipes/spec.py` refuses a container whose config.json does not
    say so. That writing lived on a code path Granite cannot reach, so every
    Granite pack shipped an unfolded config and was refused at load.
    """

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def _out(self, config=None):
        out = self.dir / "Granite-4.2-3B-NPU2"
        out.mkdir(parents=True, exist_ok=True)
        cfg = {"model_type": "granite", "hidden_size": 2560, "num_attention_heads": 40,
               "head_dim": GRANITE_HD, "attention_multiplier": GRANITE_MULT,
               "vocab_size": 100352, "rope_theta": 1e7}
        if config is not None:
            cfg = config
        with open(out / "config.json", "w", encoding="utf-8") as f:
            json.dump(cfg, f)
        for name in ("tokenizer.json", "tokenizer_config.json"):
            (out / name).write_text("{}", encoding="utf-8")
        return out

    def _pack(self, arch, reader, out):
        model_assets.assemble_model_assets(
            reader, _granite_q4nx_config(), str(out), model_arch=arch)

    def _config(self, out):
        with open(out / "config.json", encoding="utf-8") as f:
            return json.load(f)

    def test_the_packed_config_states_the_folded_multiplier(self):
        out = self._out()
        self._pack(ModelArch.GRANITE, GRANITE_GGUF, out)
        cfg = self._config(out)
        # head_dim ** -0.5 is what the kernels' attn.h applies; 0.015625 is the
        # source's own multiplier, which is what shipped before the wiring fix.
        self.assertEqual(cfg["attention_multiplier"], GRANITE_HD ** -0.5)
        self.assertEqual(cfg["q4nx_folded_multipliers"]["attention_multiplier"],
                         GRANITE_MULT)
        for key in ("embedding_multiplier", "residual_multiplier", "logits_scaling"):
            self.assertEqual(cfg[key], 1.0)
        self.assertEqual(cfg["head_dim"], GRANITE_HD)

    def test_the_no_source_path_also_folds(self):
        """The degraded path builds config.json from GGUF metadata; the fold runs
        after the config is loaded either way, so it has to land there too."""
        out = self._out()
        (out / "config.json").unlink()
        self._pack(ModelArch.GRANITE, GRANITE_GGUF, out)
        cfg = self._config(out)
        self.assertEqual(cfg["model_type"], "granite")       # not llama.cpp's arch
        self.assertEqual(cfg["attention_multiplier"], GRANITE_HD ** -0.5)
        # and the keys the open recipe `_need`s, which the fallback used to drop
        self.assertEqual(cfg["head_dim"], GRANITE_HD)
        self.assertEqual(cfg["rope_theta"], 10000000.0)

    def test_a_non_granite_arch_is_left_alone(self):
        out = self._out({"model_type": "llama", "attention_multiplier": 0.03})
        self._pack(ModelArch.LLAMA, LLAMA_GGUF, out)
        cfg = self._config(out)
        self.assertNotIn("q4nx_folded_multipliers", cfg)
        self.assertEqual(cfg["attention_multiplier"], 0.03)   # untouched, not folded

    def test_the_hf_entry_point_has_no_fold_call(self):
        """The old call site named a `reader` that does not exist in that scope,
        which is a NameError rather than a fold -- and Granite never reaches it
        anyway, because its converter refuses an HF safetensors source."""
        import inspect

        self.assertNotIn("apply_granite_fold_to_config",
                         inspect.getsource(model_assets.assemble_model_assets_hf))


class GraniteArchRoutingTest(unittest.TestCase):
    """A Granite pack has to deploy at all: `ARCH_TO_FAMILY` had no entry for it,
    so deploy.py's minimal registry entry wrote details.family = "". """

    def test_granite_k2_and_hunyuan_have_a_family(self):
        from q4nx.arch_detect import ARCH_TO_FAMILY

        for arch in (ModelArch.GRANITE, ModelArch.K2, ModelArch.HUNYUAN_DENSE):
            self.assertIn(arch, ARCH_TO_FAMILY, arch.name)
            self.assertTrue(ARCH_TO_FAMILY[arch], arch.name)


class NoSourceConfigTest(unittest.TestCase):
    """The `generate_config_from_gguf` fallback feeds `ModelSpec.from_hf_config`,
    so its keys have to be HF's and it has to carry what the recipes `_need`."""

    def test_model_type_is_an_hf_name_not_a_llama_cpp_arch_string(self):
        for arch, model_type in [("llama", "llama"), ("qwen3", "qwen3"),
                                 ("qwen35", "qwen3_5"), ("qwen35moe", "qwen3_5_moe"),
                                 ("granite", "granite"), ("hunyuan-dense", "hunyuan_v1_dense"),
                                 ("k2", "k2_horizon")]:
            reader = _GGUFReader({"general.architecture": arch})
            self.assertEqual(
                model_assets.generate_config_from_gguf(reader)["model_type"], model_type, arch)

    def test_rope_theta_and_head_dim_are_written(self):
        cfg = model_assets.generate_config_from_gguf(_GGUFReader({
            "general.architecture": "qwen3",
            "qwen3.embedding_length": 2048,
            "qwen3.attention.head_count": 16,
            "qwen3.rope.dimension_count": 128,
            "qwen3.rope.freq_base": 1000000.0,
        }))
        # `_qwen3_hf` reads both outright; the old fallback popped head_dim when
        # heads * head_dim == hidden_size (16 * 128 == 2048) and never wrote
        # rope_theta at all.
        self.assertEqual(cfg["head_dim"], 128)
        self.assertEqual(cfg["rope_theta"], 1000000.0)

    def test_an_unknown_architecture_is_passed_through_verbatim(self):
        cfg = model_assets.generate_config_from_gguf(_GGUFReader({"general.architecture": "spleen"}))
        self.assertEqual(cfg["model_type"], "spleen")


class TokenizerConfigTest(unittest.TestCase):
    """An HF pack whose source ships no tokenizer_config.json is not installable."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def _write_source(self):
        (self.dir / "tokenizer.json").write_text(json.dumps({
            "model": {"type": "BPE", "vocab": {"<|endoftext|>": 0, "hello": 1, "Bye": 2}},
        }), encoding="utf-8")
        (self.dir / "config.json").write_text(json.dumps({
            "model_type": "llama", "bos_token": "<|endoftext|>", "eos_token": "Bye",
        }), encoding="utf-8")

    def test_a_missing_tokenizer_config_is_synthesized_from_the_source(self):
        self._write_source()
        path = model_assets.synthesize_hf_tokenizer_config(self.dir)
        self.assertIsNotNone(path)
        cfg = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(cfg["bos_token_id"], 0)
        self.assertEqual(cfg["eos_token_id"], 2)         # resolved by text, not guessed

    def test_a_shipped_one_is_left_alone(self):
        self._write_source()
        (self.dir / "tokenizer_config.json").write_text("{}", encoding="utf-8")
        model_assets.ensure_hf_tokenizer_ids(self.dir)
        self.assertEqual(json.loads((self.dir / "tokenizer_config.json").read_text()), {})

    def test_ensure_hf_tokenizer_ids_falls_through_to_the_synthesizer(self):
        self._write_source()
        model_assets.ensure_hf_tokenizer_ids(self.dir)
        cfg = json.loads((self.dir / "tokenizer_config.json").read_text())
        # eos_token_id is normalized to the list the sampler reads
        self.assertEqual(cfg["eos_token_id"], [2])

    def test_nothing_is_written_when_no_id_can_be_resolved(self):
        (self.dir / "tokenizer.json").write_text(json.dumps(
            {"model": {"type": "BPE", "vocab": {"a": 0}}}), encoding="utf-8")
        (self.dir / "config.json").write_text(json.dumps({"model_type": "llama"}),
                                              encoding="utf-8")
        self.assertIsNone(model_assets.synthesize_hf_tokenizer_config(self.dir))
        self.assertFalse((self.dir / "tokenizer_config.json").exists())


class EosTokenIdTest(unittest.TestCase):
    """248044 is Qwen3.5/3.6's end-of-text id, and it used to be written into
    every family's config.json that omitted one. The open engines read eos_token_id
    straight out of config.json (open_embedding/engine.cpp), so a defaulted Llama
    or Gemma carried a token id it has never heard of."""

    def test_only_qwen_gets_the_qwen_end_of_text_default(self):
        for mt in ("qwen3_5", "qwen3_5_moe", "qwen3_5_moe_text", "qwen3_6_moe",
                   "qwen3_6_moe_text"):
            cfg = {"model_type": mt}
            model_assets.inject_oflm_keys(cfg, {}, Path(self._dir), None)
            self.assertEqual(cfg.get("eos_token_id"), 248044, mt)

    def test_every_other_family_leaves_the_key_absent(self):
        for mt in ("llama", "gemma3", "granite", "qwen3", "qwen2", "lfm2", "phi3",
                   "gpt_oss", "hunyuan_v1_dense", "k2_horizon", "nanbeige"):
            cfg = {"model_type": mt}
            model_assets.inject_oflm_keys(cfg, {}, Path(self._dir), None)
            self.assertNotIn("eos_token_id", cfg, mt)

    def test_a_stated_eos_id_is_never_overwritten(self):
        cfg = {"model_type": "llama", "eos_token_id": 2}
        model_assets.inject_oflm_keys(cfg, {}, Path(self._dir), None)
        self.assertEqual(cfg["eos_token_id"], 2)

    @property
    def _dir(self):
        return tempfile.gettempdir()


class PackedCommandTest(unittest.TestCase):
    """The recorded `oflm pack` line has to reproduce the directory it shipped."""

    def _args(self, **kw):
        base = dict(force_model_type="", quant=None, pad_to_fit=False, prune_ffn=None,
                    deploy_tag=None, build_spec=False, weights_type=None, imatrix=None)
        base.update(kw)
        return SimpleNamespace(**base)

    def _recorded(self, args):
        from q4nx.cli import _packed_command

        return _packed_command(args, "in.gguf", "/tmp/out", None, {})

    def test_build_spec_is_recorded(self):
        self.assertIn("--build-spec", self._recorded(self._args(build_spec=True)))

    def test_the_weights_type_is_recorded(self):
        self.assertIn("-t vision", self._recorded(self._args(weights_type="vision")))

    def test_neither_is_recorded_when_not_asked_for(self):
        cmd = self._recorded(self._args())
        self.assertNotIn("--build-spec", cmd)
        self.assertNotIn("-t", cmd)


class DeployedFilesTest(unittest.TestCase):
    def test_spec_json_is_deployed_and_recognised(self):
        """`--build-spec` writes spec.json; deploy's copy set and oflm-add's asset
        list both had to know the name exists."""
        from q4nx.deploy import MODEL_FILES, deployed_files_in

        self.assertIn("spec.json", MODEL_FILES)
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            for name in MODEL_FILES:
                (out / name).write_text("{}", encoding="utf-8")
            self.assertIn("spec.json", deployed_files_in(out))

        add = REPOSITORY_OF_ADD
        if add.is_file():                       # the tool ships beside the packer
            self.assertIn('"spec.json"', add.read_text(encoding="utf-8"))
            # K2 and Hunyuan are open-kernel families with no closed engine and
            # no model_list.json bucket; `derive_family` exits before the
            # open-kernel link is attempted unless they are named.
            text = add.read_text(encoding="utf-8")
            for prefix in ('"k2"', '"hunyuan"', '"hy-mt2"'):
                self.assertIn(prefix, text)


REPOSITORY_OF_ADD = REPO.parents[0] / "oflm-add" / "oflm_add" / "__init__.py"


if __name__ == "__main__":
    unittest.main()
