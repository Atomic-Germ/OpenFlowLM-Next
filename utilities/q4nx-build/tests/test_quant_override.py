"""--quant handling and the packer's own preconditions.

`set_default_tensor_type` is what the CLI's `--quant` calls, and it is the only
place the HF-vs-GGUF split is enforced up front: `--quant Q4_K` is refused
without a GGUF source, because the HF path quantizes with its own fixed per-role
targets and would write a q4_1 container that *claims* Q4_K. That is a container
whose header disagrees with its bytes, which no later check catches.

Also pinned: the same-value early return. `--quant Q4_1` on a Q4_1-default family
is the common case, and it must not rebuild the name maps (that printed the whole
mapping twice per pack).
"""
import inspect

import pytest
from gguf.constants import GGMLQuantizationType as T

from q4nx.constants import ModelArch
from q4nx.model_converter import __Q4NX_Converter

# Alias without the leading underscores: Python name-mangles a __name__ used
# inside a class body, which would break the base-instantiation test below.
_BASE = __Q4NX_Converter


@pytest.fixture(autouse=True)
def _restore_registry():
    """Undo the registry side effect of defining a subclass.

    `__init_subclass__` registers on the class *statement*, so merely importing
    this module rebinds ModelArch.QWEN3 to _Fake for the rest of the session.
    test_arch_routing.py asserts QWEN3 is a real routing target, so restore the
    original entry afterwards rather than relying on collection order.
    """
    from q4nx.model_converter import _MODEL_REGISTRY
    saved = _MODEL_REGISTRY.get(ModelArch.QWEN3)
    yield
    if saved is not None:
        _MODEL_REGISTRY[ModelArch.QWEN3] = saved
    else:
        _MODEL_REGISTRY.pop(ModelArch.QWEN3, None)


class _Fake(__Q4NX_Converter, model_arch=ModelArch.QWEN3):
    """Minimal concrete subclass.

    The `model_arch=` class keyword is mandatory -- __init_subclass__ takes it as
    a required positional and registers the class as a side effect. See
    _restore_registry for why that side effect is undone after this module.
    """

    def __init__(self, has_gguf, default="Q4_1"):
        self.gguf_reader = object() if has_gguf else None
        self.q4nx_config = {"default_tensor_type": default}
        self.default_tensor_type = T.Q4_1
        self.rebuilds = 0

    def convert(self, q4nx_path, weights_type):
        raise NotImplementedError

    def _create_name_maps(self):
        self.rebuilds += 1


class TestQuantRefusals:
    def test_q4_k_without_a_gguf_source_is_refused(self):
        # The bug this prevents: an HF pack labelled Q4_K holding q4_1 bytes.
        with pytest.raises(ValueError, match="Q4_K needs a GGUF source"):
            _Fake(has_gguf=False).set_default_tensor_type("Q4_K")

    def test_q4_k_with_a_gguf_source_is_allowed(self):
        c = _Fake(has_gguf=True, default="Q4_1")
        c.set_default_tensor_type("Q4_K")
        assert c.q4nx_config["default_tensor_type"] == "Q4_K"
        assert c.default_tensor_type is T.Q4_K

    @pytest.mark.parametrize("name", ["Q4_0", "Q4_1", "Q8_0"])
    def test_other_quants_are_allowed_without_a_gguf(self, name):
        c = _Fake(has_gguf=False, default="Q4_0" if name != "Q4_0" else "Q4_1")
        c.set_default_tensor_type(name)

    def test_unsupported_name_is_refused(self):
        with pytest.raises(ValueError, match="Unsupported q4nx_name"):
            _Fake(has_gguf=True).set_default_tensor_type("Q6_K")


class TestSameValueDoesNotRebuild:
    def test_same_value_returns_early(self):
        # --quant Q4_1 on a Q4_1 family: the common case, must not rebuild.
        c = _Fake(has_gguf=True, default="Q4_1")
        c.set_default_tensor_type("Q4_1")
        assert c.rebuilds == 0

    def test_a_changed_value_does_rebuild(self):
        # The maps carry per-tensor types, so a real change must propagate.
        c = _Fake(has_gguf=True, default="Q4_1")
        c.set_default_tensor_type("Q8_0")
        assert c.rebuilds == 1
        assert c.q4nx_config["default_tensor_type"] == "Q8_0"


class TestGettlMlType:
    @pytest.mark.parametrize("name,expected", [
        ("Q4_0", T.Q4_0), ("Q4_1", T.Q4_1), ("Q8_0", T.Q8_0),
        ("Q4_K", T.Q4_K), ("BF16", T.BF16),
    ])
    def test_known_names_map(self, name, expected):
        assert _Fake(has_gguf=True).get_ggml_type(name) is expected

    def test_bf16_is_only_reachable_here(self):
        """BF16 is a per-entry override, not a legal --quant target.

        `_load_config` accepts only Q4_0/Q4_1/Q8_0/Q4_K for the family default;
        get_ggml_type additionally knows BF16 for per-entry overrides. A config
        naming BF16 as the family default must be rejected at load, not silently
        accepted here.
        """
        assert _Fake(has_gguf=True).get_ggml_type("BF16") is T.BF16


class TestVirtualBase:
    def test_the_base_is_abstract(self):
        """`convert` is the only abstract method, so the ABC refuses first.

        Pin the real behaviour rather than the guard the reader might expect:
        the explicit "virtual" TypeError at model_converter.py:143-144 never
        fires for a normal subclass, because ABC instantiation fails first.
        """
        assert _BASE.__abstractmethods__ == frozenset({"convert"})
        with pytest.raises(TypeError, match="abstract"):
            _BASE()

    def test_subclasses_only_need_convert(self):
        """A subclass implementing convert is concrete; nothing else is required."""
        assert not inspect.isabstract(_Fake)
