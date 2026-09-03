from ..model_converter import __Q4NX_Converter
from ..constants import ModelArch
from gguf import GGMLQuantizationType, GGUFReader, dequantize, quantize
from safetensors.torch import save_file
import torch
from gguf import dequantize
from einops import rearrange, unpack

class Gemma3(__Q4NX_Converter, model_arch=ModelArch.GEMMA3):
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
        if weights_type == "language":
            if not self._has_lm_head():
                print("[INFO] Model does not have a lm_head, use embedding weights as lm_head")
                unpacked = self.gguf_tensors["token_embd.weight"].unpack(self.default_tensor_type)
                self.q4nx_tensors["lm_head.weight"] = self._pack_q4nx(*unpacked)

            for key, gguf_tensor in self.gguf_tensors.items():
                if "token_embd.weight" in gguf_tensor.name:
                    w = dequantize(gguf_tensor.data, gguf_tensor.tensor_type)
                    w = w* float(self.hidden_size) **0.5
                    w = torch.from_numpy(w).contiguous().to(torch.bfloat16)
                    self.q4nx_tensors[self.forward_name_map[gguf_tensor.name]] = w
                    continue

                unpacked = gguf_tensor.unpack(self.default_tensor_type)
                torch.set_printoptions(threshold=16, edgeitems=5, linewidth=200)
                self.q4nx_tensors[self.forward_name_map[gguf_tensor.name]] = self._pack_q4nx(*unpacked)

            self._extract_tokenizer_json(q4nx_path)
        elif weights_type == "vision":
            for key, gguf_tensor in self.gguf_tensors.items():
                unpacked = gguf_tensor.unpack(GGMLQuantizationType.BF16)
                assert len(unpacked) == 1
                assert type(unpacked[0]) == torch.Tensor, "Vision model tensors"

                weights = unpacked[0]
                if weights.dtype != torch.bfloat16:
                    weights = weights.to(torch.bfloat16)
                
                new_name = self.forward_name_map[gguf_tensor.name]
                
                if new_name == "multi_modal_projector.mm_input_projection_weight":
                    weights = weights.t().contiguous()
                    weights = self.vision_mm_weight_rearrange(weights)                    
                elif new_name.endswith("fc2.weight") or new_name.endswith("fc1.weight") \
                    or new_name.endswith("k_proj.weight") or new_name.endswith("q_proj.weight")\
                    or new_name.endswith("v_proj.weight") or new_name.endswith("out_proj.weight"):
                    weights = self.vision_mm_weight_rearrange(weights)

                self.q4nx_tensors[self.forward_name_map[gguf_tensor.name]] = weights
        else:
            raise ValueError(f"Unsupported weights_type: {weights_type} for Gemma3 model")
        self._export_weights(q4nx_path, weights_type)

    def _convert_hf(self, q4nx_path: str, weights_type: str):
        if weights_type != "language":
            raise ValueError(
                f"HF safetensors conversion for weights_type='{weights_type}' "
                f"is not supported by Gemma3. Use GGUF source instead."
            )
        self.q4nx_tensors = {}
        hf_name_map = self._build_hf_name_map()

        config_path = self.hf_dir / "config.json"
        hidden_size = self.hidden_size
        if config_path.is_file():
            import json
            with open(config_path) as f:
                cfg = json.load(f)
            hidden_size = cfg.get("hidden_size", hidden_size)

        for name in sorted(self.weight_map):
            key = name.replace("model.language_model.", "")
            if key not in hf_name_map:
                print(f"[WARN] Unmapped HF tensor: {name}")
                continue
            w = self._load_tensor(name)
            q4nx_name = hf_name_map[key]

            if key == "model.embed_tokens.weight":
                w = w * float(hidden_size) ** 0.5
                self.q4nx_tensors[q4nx_name] = w.to(torch.bfloat16)
                continue
            self.q4nx_tensors[q4nx_name] = w

        if "lm_head.weight" not in self.q4nx_tensors:
            print("[INFO] No lm_head in HF source, using embedding weights")
            if "model.embed_tokens.weight" in self.q4nx_tensors:
                self.q4nx_tensors["lm_head.weight"] = self.q4nx_tensors["model.embed_tokens.weight"]

        print(f"[INFO] Produced {len(self.q4nx_tensors)} Q4NX tensors")
        self._export_weights(q4nx_path, weights_type)
