import os
from ..model_converter import __Q4NX_Converter
from ..constants import ModelArch
from gguf import GGUFReader, dequantize, quantize, GGMLQuantizationType
from safetensors.torch import save_file
import torch
from gguf import dequantize
from einops import rearrange
import torch.nn.functional as F
from safetensors.torch import save_file, load_file
class GPTOSS(__Q4NX_Converter, model_arch=ModelArch.GPT_OSS):
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

    def post_gpt_oss_process(self,result_tensors_map:dict[str, torch.Tensor], n_layers:int):
        NUM_CT_PER_COLUMN = 4
        
        for layer_idx in range(n_layers):
            weight_name_list = [
            f"model.layers.{layer_idx}.ffn_down_exps.weight",
            f"model.layers.{layer_idx}.ffn_up_exps.weight",
            f"model.layers.{layer_idx}.ffn_gate_exps.weight"
            ]
            bias_name_list = [f"model.layers.{layer_idx}.mlp.experts.down_proj_bias", 
                        f"model.layers.{layer_idx}.mlp.experts.up_proj_bias",                       
                        f"model.layers.{layer_idx}.mlp.experts.gate_proj_bias",
                        ]
            for i in range(len(weight_name_list)):
                weight = result_tensors_map[weight_name_list[i]]
                bias = result_tensors_map[bias_name_list[i]]            

                if weight.shape[1] %NUM_CT_PER_COLUMN != 0:
                    pad_amount = (NUM_CT_PER_COLUMN - (weight.shape[1] % NUM_CT_PER_COLUMN)) % NUM_CT_PER_COLUMN
                    weight = F.pad(weight, (0, 0, 0, 0, 0, pad_amount))
                
                if bias.shape[1] != weight.shape[1]*self.row_block_size:
                    pad_amount = weight.shape[1]*self.row_block_size - bias.shape[1]
                    bias = F.pad(bias, (0, pad_amount))
                
                bias = rearrange(
                    bias,
                    "batch (block Q4NX_ROW_SIZE) -> batch block Q4NX_ROW_SIZE",
                    Q4NX_ROW_SIZE = self.row_block_size
                ).contiguous()
                NUM_OF_32_set = self.col_block_size//32
                assert bias.dtype == torch.bfloat16
                bias_byte = bias.view(torch.uint8)
                for exp_id in range(weight.shape[0]):
                    for row_block_idx in range(weight.shape[1]):
                        
                        offset = self.row_block_size* NUM_OF_32_set
                        weight[exp_id][row_block_idx][0][offset: offset+2*self.row_block_size] = bias_byte[exp_id][row_block_idx]
                    
                
                weight = rearrange(
                    weight, 
                    "batch (row_div_four four_row) (col_div one) data_block -> batch row_div_four col_div (four_row one) data_block",
                    one=1,
                    four_row=NUM_CT_PER_COLUMN

                ).contiguous()
                
                result_tensors_map[weight_name_list[i]] = weight.contiguous()    
                
        for layer_idx in range(n_layers):
            weight_name_list = [
            f"model.layers.{layer_idx}.self_attn.k_proj.weight",
            f"model.layers.{layer_idx}.self_attn.q_proj.weight",
            f"model.layers.{layer_idx}.self_attn.v_proj.weight",
            f"model.layers.{layer_idx}.self_attn.o_proj.weight"
            ]

            for i in range(len(weight_name_list)):
                weight = result_tensors_map[weight_name_list[i]]       

                if weight.shape[0] %16 != 0:
                    pad_amount = (16 - (weight.shape[0] % 16)) % 16
                    weight = F.pad(weight, (0, 0, 0, 0, 0, pad_amount))

                weight = rearrange(
                    weight, 
                    "(row_div_four four_row) (col_div one) data_block -> row_div_four col_div (four_row one) data_block",
                    one=1,
                    four_row=4

                ).contiguous()
                
                result_tensors_map[weight_name_list[i]] = weight.contiguous()    
        
        for layer_idx in range(n_layers):

            gate_proj_weight = result_tensors_map[f"model.layers.{layer_idx}.ffn_gate_exps.weight"]
            up_proj_weight = result_tensors_map[f"model.layers.{layer_idx}.ffn_up_exps.weight"]

            down_weight =result_tensors_map[f"model.layers.{layer_idx}.ffn_down_exps.weight"]
            
            num_expert = gate_proj_weight.shape[0]
            assert num_expert == gate_proj_weight.shape[0]
            weight_row_block_div_4 = gate_proj_weight.shape[1]
            weight_col_block = gate_proj_weight.shape[2]
            weight_block_4_row = gate_proj_weight.shape[3]
            weight_block_size = gate_proj_weight.shape[4]
            
            weights_concat = torch.stack([gate_proj_weight, up_proj_weight], dim=1)
            
            weights_concat = rearrange(weights_concat, "e s r c m b -> e (r s) c m b")

            weights_concat = torch.cat( [weights_concat, down_weight], dim=1)
            
            result_tensors_map[f"model.layers.{layer_idx}.ffn_gate_up_down_exps.weight"] = weights_concat.contiguous()    

            del result_tensors_map[f"model.layers.{layer_idx}.ffn_gate_exps.weight"]
            del result_tensors_map[f"model.layers.{layer_idx}.ffn_up_exps.weight"]
            del result_tensors_map[f"model.layers.{layer_idx}.ffn_down_exps.weight"]


    def process_gptoss_router_weights(self, weight:torch.Tensor, new_name:str, result_tensors_map:dict[str, torch.Tensor] ) :
        
        BLOCK_ROWS=32
        BLOCK_COLS=64
        BLOCK_TILE_ROWS  = 16 

        original_weight = weight.clone()
        assert weight.shape[0] % BLOCK_TILE_ROWS == 0, f"Expected num_expert to be multiple of {BLOCK_TILE_ROWS}, but got {weight.shape[0]}"

        weight = rearrange(
            weight,
            "(num_block_row BLOCK_ROWS) (num_block_col BLOCK_COLS) -> (num_block_row num_block_col) BLOCK_ROWS BLOCK_COLS",
            BLOCK_ROWS=BLOCK_ROWS,
            BLOCK_COLS=BLOCK_COLS,
        ).contiguous()
        
        weight = rearrange(
            weight,
            "num_blocks (num_tile BLOCK_TILE_ROWS) (BLOCK_COLS one) -> num_blocks (num_tile BLOCK_COLS) (BLOCK_TILE_ROWS one)",
            BLOCK_TILE_ROWS=BLOCK_TILE_ROWS,
            BLOCK_COLS=BLOCK_COLS,
        ).contiguous()
        
        result_tensors_map[new_name] = weight.to(torch.bfloat16)
        
        original_weight_col = original_weight.shape[1]
        if original_weight_col % self.col_block_size !=0:
            pad_amount = self.col_block_size - (original_weight_col % self.col_block_size)
            original_weight = F.pad(original_weight, (0, pad_amount))
        original_weight= original_weight.contiguous()   
        result_tensors_map[new_name + "_prefill"] = original_weight.to(torch.bfloat16)
        
 
    def convert(self, q4nx_path: str, weights_type: str = 'language'):
        self.q4nx_tensors = {}
        if self.gguf_reader is not None:
            self._convert_gguf(q4nx_path, weights_type)
        else:
            self._convert_hf(q4nx_path, weights_type)

    def _convert_gguf(self, q4nx_path: str, weights_type: str):
        print("Enter into GPTOSS convert function")

        for key, gguf_tensor in self.gguf_tensors.items():
            print(f"[INFO] Converting tensor {gguf_tensor.name} to {self.forward_name_map[gguf_tensor.name]}")
            if "token_embd.weight"  ==  gguf_tensor.name:
                w = dequantize(gguf_tensor.data, gguf_tensor.tensor_type)
                w = torch.from_numpy(w).contiguous().to(torch.bfloat16)
                self.q4nx_tensors[self.forward_name_map[gguf_tensor.name]] = w.contiguous()
                continue

            unpacked = gguf_tensor.unpack(self.default_tensor_type)
            if self.forward_name_map[gguf_tensor.name] == "lm_head.weight":
                qw = self._pack_q4nx(*unpacked)
                qw = rearrange(
                    tensor=qw,
                    pattern="(row_div_two two_row) (col_div one) data_block ->row_div_two col_div (two_row one) data_block",
                    one=1,
                    two_row=2
                ).contiguous()   
                self.q4nx_tensors[self.forward_name_map[gguf_tensor.name]] = qw
            elif gguf_tensor.tensor_type == GGMLQuantizationType.MXFP4:
                self.q4nx_tensors[self.forward_name_map[gguf_tensor.name]] = self._pack_MXFP4_q4nx(*unpacked)
            elif gguf_tensor.tensor_type == GGMLQuantizationType.F32:
                if gguf_tensor.name.endswith("ffn_gate_inp.weight"):
                    assert len(unpacked) ==1
                    new_name = self.forward_name_map[gguf_tensor.name]
                    self.process_gptoss_router_weights(weight=unpacked[0], new_name=new_name, result_tensors_map=self.q4nx_tensors)
                elif gguf_tensor.name.endswith(".bias") or gguf_tensor.name.endswith(".weight") :
                    assert len(unpacked) ==1
                    self.q4nx_tensors[self.forward_name_map[gguf_tensor.name]] = unpacked[0].to(torch.bfloat16)
                else:
                    raise ValueError(f"Unsupported F32 tensor {gguf_tensor.name} in GPTOSS model")
            else:
                self.q4nx_tensors[self.forward_name_map[gguf_tensor.name]] = self._pack_q4nx(*unpacked)

        self.post_gpt_oss_process(self.q4nx_tensors, self.num_layers)
        
        safetensors_with_embed_tokens_weights = "model-00001-of-00001.safetensors"
        if os.path.exists(safetensors_with_embed_tokens_weights):
            self.q4nx_tensors["model.embed_tokens.weight"] = load_file(safetensors_with_embed_tokens_weights)["model.embed_tokens.weight"]
        else:
            print(f"[WARNING] {safetensors_with_embed_tokens_weights} not found. Skipping embed_tokens.weight replacement.")

        self._export_weights(q4nx_path, weights_type)
        self._extract_tokenizer_json(q4nx_path)

    def _convert_hf(self, q4nx_path: str, weights_type: str):
        if weights_type != "language":
            raise ValueError(
                f"HF safetensors conversion for weights_type='{weights_type}' "
                f"is not supported by GPTOSS. Use GGUF source instead."
            )
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
            self.q4nx_tensors[q4nx_name] = w

        print(f"[INFO] Produced {len(self.q4nx_tensors)} Q4NX tensors")
        print("[WARN] GPTOSS HF conversion produces raw weights without MXFP4 quantization or expert merging.")
        print("[WARN] For production use, convert from GGUF source instead.")
        self._export_weights(q4nx_path, weights_type)
