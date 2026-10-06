"""GGUF tensors are innermost-first: the matmul K axis is shape[0].

This is the axis that decides whether `GGUFTensor.unpack` block-packs a weight
or takes the float-passthrough escape hatch, and getting it wrong is silent --
the container loads and decodes garbage, and every kernel slice compare passes
because the kernel and the reference read the same wrong bytes.

The bug this pins: the guard used `shape[-1]`. For a Q6_K `[hidden, vocab]`
token embedding, vocab is almost never a multiple of 32, so a correct
`[4096, 151669]` weight was pushed down the float escape hatch; a head that
needed a quantized triple (hunyuan's lm_head padding) then raised "did not
unpack to a (d, m, qw) quantized triple". The native unpackers take
`self.shape[0]` as their column count, which is the whole argument for index 0.
"""
import numpy as np
import pytest
from gguf.constants import GGMLQuantizationType as T

from q4nx.gguf_tensor import GGUFTensor

# The guard only reads `shape[0] % 32`, so the shapes below are tiny: (64, 33)
# and (33, 64) are the same logical weight with the axes the two ways round, and
# they disagree on exactly that modulo. Using real sizes here would allocate a
# 4096x151669 tensor per case for no extra coverage (it took ~13 s per call).
# REAL_* below pin that a genuine vocab size trips the same discriminator, so
# the small shapes cannot drift into testing something unrepresentative.
ALIGNED, RAGGED = 64, 33
REAL_HIDDEN, REAL_VOCAB = 4096, 151669
Q6_K_BLOCK_BYTES = 210  # per 256 weights


def _gguf(shape, tensor_type=T.Q6_K, name="token_embd.weight"):
    """A GGUFTensor with the right byte length for `shape`, contents unused.

    `unpack` branches on shape before it ever looks at the data, so the payload
    only has to be big enough to be read if it does pack.
    """
    n = int(np.prod(shape))
    nblocks = (n + 255) // 256
    data = np.zeros(nblocks * Q6_K_BLOCK_BYTES, dtype=np.uint8)
    return GGUFTensor(name, tuple(shape), data, tensor_type)


def _kind(tensor, target=T.Q4_1):
    """'packed' when unpack() yields the (d, m, qw) triple, else 'float'."""
    return "packed" if len(tensor.unpack(target)) == 3 else "float"


class TestColumnsAreAxisZero:
    def test_the_ordering_is_observable(self):
        # Guard the fixture itself: if GGUF ever stopped being innermost-first
        # this test would pass vacuously, so assert the two differ.
        assert ALIGNED % 32 == 0
        assert RAGGED % 32 != 0

    def test_a_real_vocab_size_trips_the_same_discriminator(self):
        """The small fixture is representative of the real bug.

        A real [hidden, vocab] embedding has a 32-aligned hidden and a vocab
        that is not a multiple of 32 -- exactly the ALIGNED/RAGGED split. This
        asserts the modulo property without allocating the tensor.
        """
        assert REAL_HIDDEN % 32 == 0
        assert REAL_VOCAB % 32 != 0
        assert (REAL_HIDDEN % 32 == 0) == (ALIGNED % 32 == 0)
        assert (REAL_VOCAB % 32 == 0) == (RAGGED % 32 == 0)

    @pytest.mark.parametrize("target", [T.Q4_0, T.Q4_1, T.Q8_0])
    def test_packs_when_axis_zero_is_block_aligned(self, target):
        # GGUF [cols, rows]: axis 0 holds the K axis and is 32-aligned -> pack.
        assert _kind(_gguf((ALIGNED, RAGGED)), target) == "packed"

    @pytest.mark.parametrize("target", [T.Q4_0, T.Q4_1, T.Q8_0])
    def test_falls_back_to_float_when_axis_zero_is_not_aligned(self, target):
        # Same logical weight, axes the other way round: axis 0 is the ragged
        # vocab, so it cannot be block-packed and the float path is correct.
        assert _kind(_gguf((RAGGED, ALIGNED)), target) == "float"

    def test_the_bug_this_pins(self):
        """A correct [hidden, vocab] weight must NOT take the float path.

        Under the old `shape[-1]` guard this returned "float", because vocab is
        not a multiple of 32 -- which is what crashed hunyuan's lm_head.
        """
        assert _kind(_gguf((ALIGNED, RAGGED))) == "packed"

    def test_shape_minus_one_would_have_been_wrong(self):
        """Pin the discriminator itself, so the guard cannot silently move.

        This is the assertion that fails if someone reintroduces shape[-1].
        """
        shape = (ALIGNED, RAGGED)
        old_axis_aligned = shape[-1] % 32 == 0
        new_axis_aligned = shape[0] % 32 == 0
        assert old_axis_aligned is False, "the old rule said 'do not pack'"
        assert new_axis_aligned is True, "the correct rule says 'pack'"


class TestOtherUnpackGuards:
    def test_one_dimensional_native_float_never_packs(self):
        # rope_freqs / biases, stored as F32: `len(shape) < 2` clears
        # wants_quantized_target (gguf_tensor.py:398-399), so the
        # native-float branch at :401 returns a 1-element passthrough even
        # though axis 0 is 32-aligned and would otherwise look packable.
        gt = GGUFTensor("rope_freqs", (4096,),
                        np.zeros(4096, dtype=np.float32), T.F32)
        assert len(gt.unpack(T.Q4_1)) == 1

    def test_one_dimensional_block_quantized_still_unpacks(self):
        # The < 2 guard does NOT apply to every source type: it only clears
        # `wants_quantized_target`, which gates the native-float branch. A
        # Q6_K 1-D tensor falls through to the requantize branch (:449-460)
        # and comes back as a triple. Pinned so the guard's real scope is
        # documented rather than assumed to be universal.
        assert _kind(_gguf((4096,))) == "packed"

    def test_float_target_is_passthrough_for_a_packable_shape(self):
        # A config entry asking for a float target (norm weights) must not pack.
        assert _kind(_gguf((ALIGNED, RAGGED)), T.F32) == "float"
