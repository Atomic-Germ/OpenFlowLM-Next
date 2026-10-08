from .llama import Llama
from ..constants import ModelArch


class K2(Llama, model_arch=ModelArch.K2):
    """K2-Horizon (IFM/K2-Horizon-3.7B and -7B, model_type `k2_horizon`).

    Llama's tensor mapping, tiling and Q4_1 target apply one to one -- K2's names
    (self_attn.{q,k,v,o}_proj, mlp.{up,gate,down}_proj, input/post_attention
    _layernorm, model.norm, an UNTIED lm_head) are the llama ones, so k2.json is
    llama.json. What differs is the q/k row order and where a GGUF keeps the RoPE
    pair count. A `k2-horizon` GGUF keeps HF's split-half q/k rows (llama.cpp ropes
    K2 NEOX-style and does not permute), so Llama's un-interleave would scramble
    them: on the 7B it raises a text's NLL from 2.44 to 2.96, still fluent, and an
    adapter trained on the HF order (K2-Horizon-7B-Uno) drafts nonsense. Only a
    llama-arch K2 GGUF is interleaved. The rope-dimension hook scans whichever
    prefix the file uses (k2_horizon.*, or llama.* under `-f llama`).
    """

    def _gguf_qk_interleaved(self) -> bool:
        arch = self.gguf_reader.fields.get("general.architecture")
        name = bytes(arch.parts[arch.data[0]]).decode() if arch is not None and arch.data else ""
        return name == "llama"

    def _rope_dim_count(self) -> int:
        for name, field in self.gguf_reader.fields.items():
            if name.endswith(".rope.dimension_count"):
                return field.contents()
        raise KeyError("k2: the GGUF carries no '*.rope.dimension_count' metadata; "
                       "the q/k pair reorder needs it (K2: 128)")
