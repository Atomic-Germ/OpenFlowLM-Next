from ..model_converter import __Q4NX_Converter
from ..constants import ModelArch
from gguf import GGUFReader, dequantize, quantize
from safetensors.torch import save_file
import torch
from gguf import dequantize
from einops import rearrange

class Qwen2(__Q4NX_Converter, model_arch=ModelArch.QWEN2):
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
        if self.gguf_reader is not None:
            self._convert_gguf(q4nx_path, weights_type)
        else:
            self._convert_hf(q4nx_path, weights_type)

    def _convert_gguf(self, q4nx_path: str, weights_type: str):
        if not self._has_lm_head():
            print("[INFO] Model does not have a lm_head, use embedding weights as lm_head")
            unpacked = self.gguf_tensors["token_embd.weight"].unpack(self.default_tensor_type)
            self.q4nx_tensors["lm_head.weight"] = self._pack_q4nx(*unpacked)

        for key, gguf_tensor in self.gguf_tensors.items():
            if "token_embd.weight" in gguf_tensor.name:
                w = dequantize(gguf_tensor.data, gguf_tensor.tensor_type)
                w = torch.from_numpy(w).contiguous().to(torch.bfloat16)
                self.q4nx_tensors[self.forward_name_map[gguf_tensor.name]] = w
                continue

            unpacked = gguf_tensor.unpack(self.default_tensor_type)

            if "ffn_down.weight" in gguf_tensor.name:
                d, m, q = unpacked
                din = q.shape[1]
                din_pad = (din + 511) // 512 * 512
                print(f"Padding ffn_down from {din} to {din_pad}")
                din_dm = din_pad // 32
                d_pad = torch.zeros((d.shape[0], din_dm), dtype=d.dtype)
                m_pad = torch.zeros((m.shape[0], din_dm), dtype=m.dtype)
                q_pad = torch.zeros((q.shape[0], din_pad), dtype=q.dtype)
                d_pad[:, :d.shape[1]] = d
                m_pad[:, :m.shape[1]] = m
                q_pad[:, :q.shape[1]] = q
                unpacked = (d_pad, m_pad, q_pad)
        
            if "ffn_up.weight" in gguf_tensor.name or "ffn_gate.weight" in gguf_tensor.name:
                d, m, q = unpacked
                dout = q.shape[0]
                dout_pad = (dout + 511) // 512 * 512
                print(f"Padding ffn_up/gate from {dout} to {dout_pad}")
                d_pad = torch.zeros((dout_pad, d.shape[1]), dtype=d.dtype)
                m_pad = torch.zeros((dout_pad, m.shape[1]), dtype=m.dtype)
                q_pad = torch.zeros((dout_pad, q.shape[1]), dtype=q.dtype)
                d_pad[:d.shape[0], :] = d
                m_pad[:m.shape[0], :] = m
                q_pad[:q.shape[0], :] = q
                unpacked = (d_pad, m_pad, q_pad)

            self.q4nx_tensors[self.forward_name_map[gguf_tensor.name]] = self._pack_q4nx(*unpacked)

        self._export_weights(q4nx_path, weights_type)
        self._extract_tokenizer_json(q4nx_path)

    def _convert_hf(self, q4nx_path: str, weights_type: str):
        if weights_type != "language":
            raise ValueError(f"Unsupported weights_type: {weights_type} for Qwen2 HF conversion")
        self.q4nx_tensors = {}
        hf_name_map = self._build_hf_name_map()

        for name in sorted(self.weight_map):
            key = name.replace("model.language_model.", "")
            if key not in hf_name_map:
                print(f"[WARN] Unmapped HF tensor: {name}")
                continue
            w = self._load_tensor(name)
            q4nx_name = hf_name_map[key]

            if key == "model.embed_tokens.weight":
                self.q4nx_tensors[q4nx_name] = w.to(torch.bfloat16)
                continue
            if key == "lm_head.weight":
                self.q4nx_tensors[q4nx_name] = w
                continue

            self.q4nx_tensors[q4nx_name] = w

        if "lm_head.weight" not in self.q4nx_tensors:
            print("[INFO] No lm_head in HF source, using embedding weights")
            if "model.embed_tokens.weight" in self.q4nx_tensors:
                self.q4nx_tensors["lm_head.weight"] = self.q4nx_tensors["model.embed_tokens.weight"]

        print(f"[INFO] Produced {len(self.q4nx_tensors)} Q4NX tensors")
        self._export_weights(q4nx_path, weights_type)
