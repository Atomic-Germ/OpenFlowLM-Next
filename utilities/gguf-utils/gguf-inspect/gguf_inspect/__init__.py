"""GGUF metadata inspection for OpenFlowLM model installation.

Public API:
    from gguf_inspect import inspect, open_gguf, GGUFReport
"""

from __future__ import annotations

from .reader import GgufReaderError, GgufFile, open_gguf
from .inspect import (
    GGUFReport,
    BlockInfo,
    Projector,
    RopeFactors,
    Tokenizer,
    ExecutionOptions,
    Unsupported,
    inspect,
)

__all__ = [
    "inspect",
    "open_gguf",
    "GgufReaderError",
    "GgufFile",
    "GGUFReport",
    "BlockInfo",
    "Projector",
    "RopeFactors",
    "Tokenizer",
    "ExecutionOptions",
    "Unsupported",
]

__version__ = "1.0"
