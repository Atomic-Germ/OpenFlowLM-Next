from ..model_converter import __Q4NX_Converter
from ..constants import ModelArch
from gguf import GGUFReader, dequantize
import torch

# The open dense head (open_kernels/designs/lm_head_q4) splits the vocabulary into
# 64-row bands, so lm_head.weight needs a whole number of them. Every other family
# already publishes a padded vocab_size; HunYuan's is 128167, so the head (and only
# the head -- the embedding is indexed by token id) is zero-padded up to 128192 and
# the recipe's `real_vocab` keeps the extra rows out of the argmax.
BAND_ROWS = 64


class HunyuanDense(__Q4NX_Converter, model_arch=ModelArch.HUNYUAN_DENSE):
    """HunYuan V1 dense (Hy-MT2, Hunyuan-{1.8,4,7}B).

    Tensor-wise this is Qwen3 dense: the same GGUF names, q/k RMSNorm weights per
    head, and -- confirmed against llama.cpp's converter (conversion/hunyuan.py:
    HunYuanModel.modify_tensors defers to the base class) -- no q/k rotary
    permutation, so the weights arrive in the half-split order attn.h rotates in.
    What differs is the tied head and the unpadded vocabulary, both handled here.
    """

    def __init__(self, source, config_json_path=None):
        self.gguf_reader = None
        self.gguf_tensors = []
        self.hf_source = None
        self.hf_dir = None
        self.weight_map = {}
        self.hf_shards = {}
        if isinstance(source, GGUFReader):
            self.gguf_reader = source
            self.gguf_tensors = {t.name: t for t in source.tensors}
            self.initialize()
        else:
            self.hf_source = source
            self.hf_dir = self._resolve_source(source)
            self.initialize(config_json_path=config_json_path)

    def initialize(self, config_json_path=None):
        super().initialize()

    def convert(self, q4nx_path: str, weights_type: str = 'language'):
        self.q4nx_tensors = {}
        if self.gguf_reader is None:
            raise ValueError(
                "HunYuan dense conversion reads a GGUF: the HF safetensors path stores tensors "
                "unquantized, which the open kernels' q4_1 GEMV cannot read. Convert the model "
                "with llama.cpp's convert_hf_to_gguf.py (or use tencent/Hy-MT2-7B-GGUF) first."
            )
        self._convert_gguf(q4nx_path, weights_type)

    @staticmethod
    def _pad_head_rows(unpacked):
        """Zero-pad a q4_1 (d, m, qw) triple up to a whole number of 64-row bands."""
        if len(unpacked) != 3:
            raise ValueError(
                "lm_head did not unpack to a (d, m, qw) quantized triple -- the source tensor came "
                "back as a float passthrough, which the open kernels' q4_1 head cannot read"
            )
        d, m, qw = unpacked
        rows = d.shape[0]
        want = -(-rows // BAND_ROWS) * BAND_ROWS
        if want == rows:
            return unpacked
        print(f"[INFO] lm_head: padding {rows} rows to {want} (whole 64-row bands)")
        # d = m = qw = 0 dequantizes to 0, so the padded logits are exactly zero
        return tuple(torch.cat([t, t.new_zeros((want - rows, t.shape[1]))], 0) for t in (d, m, qw))

    def _convert_gguf(self, q4nx_path: str, weights_type: str):
        if not self._has_lm_head():
            print("[INFO] Model does not have a lm_head, use embedding weights as lm_head")
            unpacked = self.gguf_tensors["token_embd.weight"].unpack(self.default_tensor_type)
            self.q4nx_tensors["lm_head.weight"] = self._pack_q4nx(*self._pad_head_rows(unpacked))

        for key, gguf_tensor in self.gguf_tensors.items():
            name = self.forward_name_map[gguf_tensor.name]
            if "token_embd.weight" in gguf_tensor.name:
                w = dequantize(gguf_tensor.data, gguf_tensor.tensor_type)
                self.q4nx_tensors[name] = torch.from_numpy(w).contiguous().to(torch.bfloat16)
                continue

            unpacked = gguf_tensor.unpack(self.default_tensor_type)
            if name == "lm_head.weight":
                unpacked = self._pad_head_rows(unpacked)
            self.q4nx_tensors[name] = self._pack_q4nx(*unpacked)

        self._export_weights(q4nx_path, weights_type)
        self._extract_tokenizer_json(q4nx_path)
