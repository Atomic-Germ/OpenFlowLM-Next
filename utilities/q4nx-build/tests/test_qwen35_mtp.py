"""Qwen3.5's MTP block is dropped from EVERY pack, and the depth follows the container.

A Qwen3.5 export ships its multi-token-prediction head as one EXTRA transformer
block: Qwen3.8-27B is GGUF `block_count` 65 with `num_hidden_layers` 64 and
`mtp_num_hidden_layers` 1. `blk.64` holds the four `.nextn.*` tensors and is the
only block that does.

That block existed only to serve speculative decoding, which this runtime has no
use for, so it is never converted. b3eccb5 dropped it for a PRUNED pack only and
left the unpruned case open ("a 65-layer container is a separate question"); it is
now dropped unconditionally. Converting it produced a container holding MORE
layers than config.json declared -- the kernel recipe builds a per-layer set and
the engine walks `num_hidden_layers`, so that is a link-time mismatch, and the
extra layer is not in the file to be found.

The block is identified by `.nextn.` being PRESENT, never by index arithmetic
against the layer count, so this stays right for a model that ships no MTP or
ships two.
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from q4nx.cli import _prune_meta  # noqa: E402
from q4nx.models.qwen35 import Qwen35  # noqa: E402

NEXTN = ("eh_proj", "enorm", "hnorm", "shared_head_norm")


def block(b: int, *tops: str) -> list[str]:
    return [f"blk.{b}.{t}.weight" for t in tops]


class _FakeGGUF:
    """Just enough reader for the block-count field the depth comes from."""

    def __init__(self, block_count: int) -> None:
        self.fields = {"qwen35.block_count": _Field(block_count)}


class _Field:
    def __init__(self, v: int) -> None:
        self._v = v

    def contents(self) -> int:
        return self._v


def converter(tensors: list[str], block_count: int) -> Qwen35:
    """A Qwen35 with only the two attributes _mtp_blocks / mtp_dropped read."""
    c = Qwen35.__new__(Qwen35)
    c.imatrix = None          # an UNPRUNED pack: the case b3eccb5 left open
    c._mtp_cache = None
    c.gguf_reader = _FakeGGUF(block_count)
    c.gguf_tensors = {t: _T(t) for t in tensors}
    c.q4nx_tensors = {}
    return c


class _T:
    def __init__(self, name: str) -> None:
        self.name = name


LAYER_TOPS = ("attn_qkv", "attn_gate", "ffn_up", "ffn_gate", "ffn_down", "ssm_out")


def tensors_27b() -> list[str]:
    """65 blocks: 64 layers, then the MTP block carrying `.nextn.*`."""
    out: list[str] = []
    for b in range(64):
        out += block(b, *LAYER_TOPS)
    out += block(64, *LAYER_TOPS)
    out += [f"blk.64.nextn.{t}.weight" for t in NEXTN]
    return out


def tensors_no_mtp() -> list[str]:
    return [n for b in range(64) for n in block(b, *LAYER_TOPS)]


class MTPBlockTest(unittest.TestCase):
    def test_the_nextn_block_is_the_mtp_one_and_only_that_one(self):
        mtp = converter(tensors_27b(), 65)._mtp_blocks()
        self.assertEqual(mtp, {64})

    def test_an_unpruned_pack_reports_the_drop(self):
        """The regression: this used to answer 0 without an imatrix, so an
        unpruned pack kept the speculative block and declared 65 layers."""
        c = converter(tensors_27b(), 65)
        self.assertIsNone(c.imatrix)
        self.assertEqual(c.mtp_dropped, 1)

    def test_a_model_without_mtp_reports_none(self):
        self.assertEqual(converter(tensors_no_mtp(), 64).mtp_dropped, 0)

    def test_two_mtp_blocks_are_both_dropped(self):
        """Identified by presence, not by 'the last one', so a model shipping
        two speculative blocks loses both."""
        ts = tensors_27b() + [f"blk.63.nextn.{t}.weight" for t in NEXTN]
        self.assertEqual(converter(ts, 65)._mtp_blocks(), {63, 64})

    def test_the_meta_carries_the_depth_without_a_prune(self):
        """_prune_meta returned {} for an unpruned pack, so the config's
        num_hidden_layers was never corrected to what the container holds."""
        c = converter(tensors_27b(), 65)
        # what was written: the converted names, in every layer but the MTP one
        c.q4nx_tensors = {f"model.layers.{b}.{t}.weight": _T(n)
                          for n in tensors_no_mtp()
                          for b in [int(n.split(".")[1])]
                          for t in [n.split(".")[2]]}
        meta = _prune_meta(c)
        self.assertEqual(meta["mtp_dropped"], 1)
        self.assertEqual(meta["layers_actual"], 64)
        self.assertIsNone(meta.get("kept"), "no prune ran, so no FFN declaration")


if __name__ == "__main__":
    unittest.main()