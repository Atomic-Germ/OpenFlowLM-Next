"""Geometry of the shared 32-lane banked DeltaNet primitives.

Describes buildable input shapes, NOT catalogue validation or model support.
AB is a separate dispatch: adding xn to the fused conv glue would need three
input DMA channels on a core with only two.
"""
from dataclasses import dataclass
from typing import ClassVar


@dataclass(frozen=True)
class WideDeltaNet:
    hidden: int
    key_heads: int
    value_heads: int
    key_dim: int
    value_dim: int
    ab_bank_width: ClassVar[int] = 32
    ab_input_dma_channels: ClassVar[int] = 2

    def __post_init__(self):
        if self.hidden <= 0 or self.hidden % 64:
            raise ValueError("AB hidden width must be positive and divisible by 64")
        if self.key_heads <= 0 or self.value_heads <= 0 or self.value_heads % self.key_heads:
            raise ValueError("value heads must be a positive multiple of key heads")
        if self.key_dim != 128 or self.value_dim != 128:
            raise ValueError("not implemented: DeltaNet dimensions other than 128")

    @property
    def ab_banks(self):
        return (self.value_heads + self.ab_bank_width - 1) // self.ab_bank_width

    @property
    def banks(self):
        return tuple((base, min(self.ab_bank_width, self.value_heads - base))
                     for base in range(0, self.value_heads, self.ab_bank_width))

    @property
    def group_size(self):
        return self.value_heads // self.key_heads

    def key_head(self, value_head):
        if not 0 <= value_head < self.value_heads:
            raise ValueError(f"value head {value_head} outside [0, {self.value_heads})")
        return value_head // self.group_size

    @property
    def xn_chunks(self):
        return (self.hidden + 2047) // 2048

    @property
    def ab_tiles(self):
        return tuple(min(2048, self.hidden - h * 2048) // 64 for h in range(self.xn_chunks))

    @property
    def side_bytes(self):
        # Each bank: Wa[hidden,32], Wb[hidden,32], one small-constants element.
        return self.ab_banks * (2 * self.hidden * self.ab_bank_width * 2 + 4096)

    @property
    def result_bytes(self):
        # alpha, beta logits, decay, sigmoid(beta), each [value_heads] f32.
        return 4 * self.value_heads * 4
