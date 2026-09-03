from ..model_converter import __Q4NX_Converter
from ..constants import ModelArch
from gguf import GGUFReader, dequantize, quantize
from safetensors.torch import save_file
import torch
from gguf import dequantize
from einops import rearrange

class Phi4(__Q4NX_Converter, model_arch=ModelArch.PHI4):
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

            if "qkv" in gguf_tensor.name:
                attention_heads = self.gguf_reader.fields["phi3.attention.head_count"].contents()
                kv_heads = self.gguf_reader.fields["phi3.attention.head_count_kv"].contents()
                D_QKV = unpacked[2].shape[0]
                DH = D_QKV // (attention_heads + 2 * kv_heads)
                pp = DH // 2
                DQ = DH * attention_heads
                DK = DH * kv_heads
                DV = DH * kv_heads
                d, m, qw = unpacked
                d_q = d[:DQ, :].contiguous()
                d_k = d[DQ:DQ+DK, :].contiguous()
                d_v = d[DQ+DK:, :].contiguous()
                m_q = m[:DQ, :].contiguous()
                m_k = m[DQ:DQ+DK, :].contiguous()
                m_v = m[DQ+DK:, :].contiguous()
                qw_q = qw[:DQ, :].contiguous()
                qw_k = qw[DQ:DQ+DK, :].contiguous()
                qw_v = qw[DQ+DK:, :].contiguous()
                d_q = rearrange(d_q, '(g p q) c -> (g q p) c', p = pp, q = 2).contiguous()
                m_q = rearrange(m_q, '(g p q) c -> (g q p) c', p = pp, q = 2).contiguous()
                qw_q = rearrange(qw_q, '(g p q) c -> (g q p) c', p = pp, q = 2).contiguous()
                d_k = rearrange(d_k, '(g p q) c -> (g q p) c', p = pp, q = 2).contiguous()
                m_k = rearrange(m_k, '(g p q) c -> (g q p) c', p = pp, q = 2).contiguous()
                qw_k = rearrange(qw_k, '(g p q) c -> (g q p) c', p = pp, q = 2).contiguous()
                unpacked_k = (d_k, m_k, qw_k)
                unpacked_v = (d_v, m_v, qw_v)
                print(f"[INFO] Splitting qkv into q, k, v with shapes: q: {qw_q.shape}, k: {qw_k.shape}, v: {qw_v.shape}")
                self.q4nx_tensors[self.forward_name_map[gguf_tensor.name].replace("q_proj", "k_proj")] = self._pack_q4nx(*unpacked_k)
                self.q4nx_tensors[self.forward_name_map[gguf_tensor.name].replace("q_proj", "v_proj")] = self._pack_q4nx(*unpacked_v)
                unpacked = (d_q, m_q, qw_q)
            
            if "ffn_up" in gguf_tensor.name:
                d, m, qw = unpacked
                intermediate_size = qw.shape[0] // 2
                d_gate = d[:intermediate_size, :].contiguous()
                d_up = d[intermediate_size:, :].contiguous()
                m_gate = m[:intermediate_size, :].contiguous()
                m_up = m[intermediate_size:, :].contiguous()
                qw_gate = qw[:intermediate_size, :].contiguous()
                qw_up = qw[intermediate_size:, :].contiguous() 
                print(f"[INFO] Splitting up_gate into up and gate with shapes: up: {qw_up.shape}, gate: {qw_gate.shape}")
                unpacked_gate = (d_gate, m_gate, qw_gate)
                self.q4nx_tensors[self.forward_name_map[gguf_tensor.name].replace("up_proj", "gate_proj")] = self._pack_q4nx(*unpacked_gate)
                unpacked = (d_up, m_up, qw_up)

            self.q4nx_tensors[self.forward_name_map[gguf_tensor.name]] = self._pack_q4nx(*unpacked)

        self._export_weights(q4nx_path, weights_type)
        self._extract_tokenizer_json(q4nx_path)

    def _convert_hf(self, q4nx_path: str, weights_type: str):
        if weights_type != "language":
            raise ValueError(f"Unsupported weights_type: {weights_type} for Phi4 HF conversion")
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
