from .llama import Llama
from ..constants import ModelArch


class K2(Llama, model_arch=ModelArch.K2):
    """K2-Horizon (IFM/K2-Horizon-3.7B, model_type `k2_horizon`).

    Llama's tensor mapping, tiling and Q4_1 target apply one to one -- K2's names
    (self_attn.{q,k,v,o}_proj, mlp.{up,gate,down}_proj, input/post_attention
    _layernorm, model.norm, an UNTIED lm_head) are the llama ones, so k2.json is
    llama.json. What differs is only where a GGUF keeps the RoPE pair count: a
    K2 GGUF rides its metadata under its own general.architecture prefix
    (k2_horizon.*), so the hook scans for the rope dimension under whichever
    prefix the file actually uses, llama's included (which is also what
    `-f llama` on a llama-arch K2 GGUF gives).
    """

    def _rope_dim_count(self) -> int:
        for name, field in self.gguf_reader.fields.items():
            if name.endswith(".rope.dimension_count"):
                return field.contents()
        raise KeyError("k2: the GGUF carries no '*.rope.dimension_count' metadata; "
                       "the q/k pair reorder needs it (K2: 128)")
