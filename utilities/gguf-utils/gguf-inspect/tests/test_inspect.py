"""Tests for the GGUF metadata inspector (Phase 0 baseline).

Run with:
    python3 utilities/gguf-utils/gguf-inspect/tests/test_inspect.py
    pytest utilities/gguf-utils/gguf-inspect/tests/

These exercise real, multi-gigabyte model snapshots, which are not in the repo.
Their locations are therefore resolved from the local Hugging Face cache and can
be overridden per file with an environment variable:

    GGUF_TEST_QWEN     -- Qwen3.8-9B-Distill-GGUF Q6_K trunk model
    GGUF_TEST_MMPROJ   -- the same repo's mmproj-F16.gguf
    GGUF_TEST_LLAMA    -- Llama-3.2-1B-Instruct-GGUF Q8_0 model

Every test that needs a snapshot skips when it is absent, so the suite is green
on CI, where no model is cached. (It previously hard-coded one developer's home
directory, so on CI every functional case silently skipped while
test_provenance -- the one test with no guard -- failed on the missing path.)
"""

from __future__ import annotations

import importlib
import os
import re
import struct
import sys
import tempfile
import time
import unittest
from pathlib import Path

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from gguf_inspect import inspect, GGUFReport  # noqa: E402

# Module import for the internals under test below.
_inspect = importlib.import_module("gguf_inspect.inspect")


def _find_repo_file(*parts: str) -> Path | None:
    """Walk up from this file to the repository root and return `*parts`.

    tests/test_inspect.py lives at utilities/gguf-utils/gguf-inspect/tests/, so
    four parents up is the repo root. Searching upward rather than hard-coding
    the depth keeps the drift test working if the tool is ever moved. Returns
    None instead of raising: the caller decides, and raising here would run at
    import time and take the whole module's skip with it.
    """
    for parent in Path(__file__).resolve().parents:
        candidate = parent.joinpath(*parts)
        if candidate.is_file():
            return candidate
    return None

# GGUF value-type ids used by the fixture writer.
_GGUF_UINT32, _GGUF_STRING, _GGUF_ARRAY = 4, 8, 9
# GGML type id for Q8_0.
_GGML_TYPE_Q8_0 = 8


def _write_minimal_gguf(path: Path) -> Path:
    """Write the smallest GGUF v3 that still carries real metadata.

    Lets the pure-logic tests (provenance, dtype allowlist, tag derivation) run
    against an actual parsed header instead of a stand-in object, so they do
    not need a multi-gigabyte snapshot.
    """

    def _str(s: str) -> bytes:
        b = s.encode()
        return struct.pack("<Q", len(b)) + b

    kvs = [
        ("general.architecture", _GGUF_STRING, _str("llama")),
        ("general.alignment", _GGUF_UINT32, struct.pack("<I", 32)),
        ("llama.block_count", _GGUF_UINT32, struct.pack("<I", 4)),
        ("tokenizer.ggml.model", _GGUF_STRING, _str("llama-bpe")),
    ]
    out = b"GGUF" + struct.pack("<I", 3)
    out += struct.pack("<Q", 1)          # one tensor
    out += struct.pack("<Q", len(kvs))
    for key, typ, val in kvs:
        out += _str(key) + struct.pack("<I", typ) + val
    out += _str("blk.0.attn_q.weight") + struct.pack("<I", 1)   # n_dims
    out += struct.pack("<Q", 32)                                 # row length
    out += struct.pack("<I", _GGML_TYPE_Q8_0)
    out += struct.pack("<Q", 0)                                  # offset
    out += b"\0" * 32                                            # align to 32
    out += b"\0" * (32 + 2 * 32)                                 # one Q8_0 row
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(out)
    return path

# Revisions are pinned so the baseline cannot silently drift.
_QWEN_REPO = "Atomic-Germ/Qwen3.8-9B-Distill-GGUF"
_QWEN_REV = "c43a63ce052c5a4906a20335ebf519907ea01ec5"
_LLAMA_REPO = "Atomic-Germ/Llama-3.2-1B-Instruct-GGUF"
_LLAMA_REV = "2baaed77eb97af7ec61f344afa096b90643e922c"


def _hf_cache_roots() -> list[Path]:
    roots = []
    if e := os.environ.get("HF_HUB_CACHE"):
        roots.append(Path(e))
    if e := os.environ.get("HF_HOME"):
        roots.append(Path(e) / "hub")
    home = os.environ.get("HOME") or os.environ.get("USERPROFILE")
    if home:
        roots.append(Path(home) / ".cache" / "huggingface" / "hub")
    return roots


def _cached(repo: str, rev: str, filename: str) -> Path | None:
    dirname = "models--" + repo.replace("/", "--")
    for root in _hf_cache_roots():
        p = root / dirname / "snapshots" / rev / filename
        if p.is_file() and p.stat().st_size > 0:
            return p
    return None


def _snapshot(env_var: str, repo: str, rev: str, filename: str):
    """A pinned snapshot path: the env override, else the local HF cache."""
    if override := os.environ.get(env_var):
        p = Path(override)
        return str(p) if p.is_file() and p.stat().st_size > 0 else None
    found = _cached(repo, rev, filename)
    return str(found) if found else None


QWEN = _snapshot("GGUF_TEST_QWEN", _QWEN_REPO, _QWEN_REV, "Qwen3.8-9B-Q6_K.gguf")
LLAMA = _snapshot("GGUF_TEST_LLAMA", _LLAMA_REPO, _LLAMA_REV,
                  "Llama-3.2-1B-Instruct-Q8_0.gguf")
MMPROJ = _snapshot("GGUF_TEST_MMPROJ", _QWEN_REPO, _QWEN_REV, "mmproj-F16.gguf")


def _exists(p: str | None) -> bool:
    return bool(p) and os.path.isfile(p) and os.path.getsize(p) > 0


class GGUFInspectorTests(unittest.TestCase):
    def _require(self, path, what):
        # The single guard every snapshot test uses, so a missing model skips
        # rather than failing.
        if not _exists(path):
            self.skipTest(f"{what} snapshot not available")
        return path

    def test_qwen_trunk_and_mtp(self):
        rep = inspect(self._require(QWEN, "Qwen Q6_K"))
        self.assertEqual(rep.architecture, "qwen35")
        self.assertEqual(rep.blocks.block_count, 33)
        self.assertEqual(rep.blocks.trunk_layers, 32)
        self.assertTrue(rep.blocks.has_mtp)
        self.assertIn(1, (rep.blocks.nextn_predict_layers,))

    def test_qwen_compatible_projector(self):
        rep = inspect(self._require(MMPROJ, "Qwen mmproj"))
        self.assertTrue(rep.projector.is_clip)
        self.assertIn("Qwen3.5", rep.projector.base_model)

    def test_llama_tensor_rope(self):
        rep = inspect(self._require(LLAMA, "Llama Q8_0"))
        self.assertEqual(rep.architecture, "llama")
        self.assertTrue(rep.rope.carried)
        self.assertIn("rope_freqs.weight", rep.rope.tensor_names)
        self.assertFalse(rep.blocks.has_mtp)

    def test_tokenizer_template_present(self):
        rep = inspect(self._require(LLAMA, "Llama Q8_0"))
        self.assertTrue(rep.tokenizer.has_chat_template)
        self.assertEqual(rep.tokenizer.pre, "llama-bpe")

    def test_unsupported_requirements(self):
        rep = inspect(self._require(QWEN, "Qwen Q6_K"))
        joined = " ".join(rep.unsupported.requirements)
        self.assertIn("speculative", joined.lower())

    def test_provenance(self):
        # The one test that used to run unguarded against a hard-coded path.
        rep = inspect(self._require(QWEN, "Qwen Q6_K"))
        self.assertEqual(rep.snapshot, "c43a63ce052c5a4906a20335ebf519907ea01ec5")

    def test_no_npu_no_payload_decode(self):
        # Metadata-only: inspecting a large file must not decode any tensor
        # payload (which would take many seconds to minutes). Warm the reader up
        # first so cold-import variance does not make the assertion flaky.
        path = self._require(QWEN, "Qwen Q6_K")
        inspect(path)  # warm
        start = time.perf_counter()
        inspect(path)
        self.assertLess(time.perf_counter() - start, 15.0)


class PureLogicTests(unittest.TestCase):
    """Cases that need no model file, so they run everywhere -- CI included.

    The dtype allowlist and the tag derivation were both wrong in ways only a
    real model could reveal, which is why they had no coverage at all when
    every other test skipped.
    """

    def _report(self, **general):
        class _Blocks:
            block_count = 0
            trunk_layers = 0
            nextn_predict_layers = 0
            has_mtp = False
            mtp_note = ""
            architecture = "qwen35"

        class _Rep:
            path = "p.gguf"
            version = 3
            endian = "little"
            architecture = "qwen35"
            snapshot = ""
            source = ""
            blocks = _Blocks()

        _Rep.general = general
        return _Rep()

    def test_dtype_allowlist_matches_runtime(self):
        # pools.cpp's std_perm_gguf accepts exactly these; the rest are refused.
        # Q4_0/Q4_1 go in byte-exactly; the K- and i-quants are dequantized and
        # requantized into the pool's q4_1 law.
        for ok in ("Q4_0", "Q4_1", "Q8_0", "Q4_K", "Q5_K", "Q6_K", "IQ4_NL", "IQ4_XS"):
            self.assertIn(ok, _inspect._RUNTIME_QUANT_TYPES)
        for bad in ("Q2_K", "Q3_K", "Q8_K", "Q5_0", "Q5_1", "IQ2_XXS", "IQ3_S"):
            self.assertNotIn(bad, _inspect._RUNTIME_QUANT_TYPES)
            self.assertNotIn(bad, _inspect._RUNTIME_TYPES)

    def test_quant_allowlist_tracks_pools_cpp(self):
        # The allowlist is written down TWICE: pools.cpp decides what the NPU can
        # actually pack, and _RUNTIME_QUANT_TYPES is what gguf-inspect tells a
        # user before they try. Nothing tied them together, so widening the C++
        # gate (a new dtype support) would leave the planner calling perfectly
        # loadable files unloadable -- both files reading correctly in review,
        # nothing failing CI. This derives one from the other.
        # Fail rather than skip: a checkout without pools.cpp means this guard cannot
        # do its job, and a silent skip would leave the drift unguarded while
        # the suite still reported green.
        pools_cpp = _find_repo_file("src", "open_qwen36", "pools.cpp")
        self.assertIsNotNone(
            pools_cpp,
            "cannot find src/open_qwen36/pools.cpp above " + str(Path(__file__).parent) +
            "; this test exists to keep it in sync with inspect.py")
        pools = pools_cpp.read_text(encoding="utf-8")
        start = pools.find("const bool kq")
        self.assertNotEqual(start, -1, "pools.cpp no longer has 'const bool kq' -- "
                                        "update this test along with the allowlist")
        # The allowlist is exactly two constructs: the `kq` union, and the
        # `if (!kq && t.type != ...)` guard beside it. Bound the region by the
        # guard's statement terminator, not by a newline -- a guard that wraps
        # across a line (which is how it grows when someone adds a dtype) would
        # otherwise be cut in half and the new dtype would slip past unnoticed.
        # The region does reach into the fail() message that follows, but its
        # literals carry no `GgufFile::Type::` prefix and cannot be matched.
        # It must stop BEFORE `has_min`, which names Q4_1 for an unrelated
        # reason (block size) and would make this test pass for the wrong one.
        guard = pools.find("if (!kq", start)
        self.assertNotEqual(guard, -1, "pools.cpp's std_perm_gguf guard changed shape")
        terminator = pools.find(";", guard)
        self.assertNotEqual(terminator, -1, "pools.cpp's std_perm_gguf guard has no terminator")
        region = pools[start:terminator]
        cxx = frozenset(re.findall(r"GgufFile::Type::(\w+)", region))
        self.assertTrue(cxx, "no types parsed out of pools.cpp -- the allowlist moved")
        self.assertNotIn("has_min", region, "the parsed region drifted past the allowlist")
        self.assertEqual(_inspect._RUNTIME_QUANT_TYPES, cxx,
                         "inspect.py's _RUNTIME_QUANT_TYPES has drifted from pools.cpp; "
                         "if pools.cpp just gained a dtype, add it here too")

    def test_layernorm_float_types_are_not_blockers(self):
        # A GGUF's norm weights are f32 by default. Flagging them as unsupported
        # would mark every real model unusable.
        for ok in ("F32", "F16", "BF16"):
            self.assertIn(ok, _inspect._RUNTIME_TYPES)
            self.assertNotIn(ok, _inspect._RUNTIME_QUANT_TYPES)

    def test_candidate_tag_preserves_unit(self):
        cases = {"300M": "w:300M", "4B": "w:4B", "1.5B": "w:1.5B",
                 "270": "w:270B", "9B ": "w:9B"}
        for label, expected in cases.items():
            rep = self._report(size_label=label, basename="w")
            self.assertEqual(_inspect._derive_candidate_tag(rep)[0], expected, label)

    def test_candidate_tag_without_size_label(self):
        rep = self._report(basename="w")
        tag, basis = _inspect._derive_candidate_tag(rep)
        self.assertEqual(tag, "")
        self.assertIn("no size_label", basis)

    def test_source_is_not_aliased_to_snapshot(self):
        # --source names a repo; the snapshot is a commit. The JSON report has
        # to keep the two distinguishable, which it could not when `source`
        # defaulted to the snapshot hash.
        rev = "c43a63ce052c5a4906a20335ebf519907ea01ec5"
        # Lay the fixture out like a real HF cache entry -- the snapshot hash is
        # derived from the path, so a bare /tmp file would carry none.
        cache = Path(tempfile.mkdtemp()) / "models--Atomic-Germ--Widget" / "snapshots" / rev
        path = _write_minimal_gguf(cache / "widget-Q8_0.gguf")
        rep = inspect(str(path))
        self.assertEqual(rep.snapshot, rev)
        self.assertEqual(rep.source, "")
        # and an explicit repo id is kept as the repo id
        rep = inspect(str(path), source="Atomic-Germ/Qwen3.8-9B-Distill-GGUF")
        self.assertEqual(rep.source, "Atomic-Germ/Qwen3.8-9B-Distill-GGUF")
        self.assertEqual(rep.snapshot, rev)

    def test_inspect_reads_a_real_header(self):
        # End-to-end over a synthetic file: no model download, no snapshot.
        path = _write_minimal_gguf(Path(tempfile.gettempdir()) / "minimal.gguf")
        rep = inspect(str(path))
        self.assertEqual(rep.version, 3)
        self.assertEqual(rep.architecture, "llama")
        self.assertEqual(rep.blocks.block_count, 4)
        self.assertEqual(rep.tokenizer.model, "llama-bpe")
        # q8_0 is in the allowlist, so nothing is reported as unsupported
        self.assertEqual(rep.execution.unknown_dtypes, [])
        self.assertEqual(rep.unsupported.requirements, [])


if __name__ == "__main__":
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromModule(sys.modules[__name__])
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
