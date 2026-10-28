__all__ = ['create_converter', 'create_hf_converter', 'get_model_arch_from_gguf', 'get_registered_models', '__Q4NX_Converter']


def _converter_stack():
    """Import the GGUF -> Q4NX converter on demand, and return its module.

    It pulls `gguf`, `mpmath` and -- through the model modules, which register themselves
    by subclassing -- `torch`: the whole converter toolchain. `q4nx.open_diffusion` (the
    FLUX.2 bundle builder) shares nothing with it but the package, and a machine that only
    builds a diffusion repo should not need torch to do it. The import ORDER matters:
    `model_converter` defines the base class the model modules subclass to register.
    """
    from . import model_converter  # noqa: F401
    from . import models            # noqa: F401  (registers every converter)

    return model_converter


def __getattr__(name):
    """PEP 562: the converter names resolve on first use, not at package import."""
    if name in __all__:
        return getattr(_converter_stack(), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")