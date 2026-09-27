"""Bounded-memory GGUF conversion for H5120, 16 key / 48 value heads.

llama.cpp stores V in tiled [slot,key] order; the wide kernels use grouped
[key,slot] order. Restore that order before production Q4NX packing.
"""
from pathlib import Path

import torch
from gguf import GGMLQuantizationType as QT
from ..gguf_tensor import GGUFTensor
from ..streaming import TensorWriter


def grouped_heads(tensor,axis,dim):
    if tensor.shape[axis]!=48*dim: raise ValueError('expected 48 value heads')
    index=torch.tensor([((slot*16+group)*dim+element)
                        for group in range(16) for slot in range(3) for element in range(dim)])
    return tensor.index_select(axis,index).contiguous()


def restore_order(name,tensor,block=1):
    if tensor is None: return None
    if 'qkv_proj' in name:
        if tensor.shape[0]!=10240: raise ValueError('expected QKV rows 4096+6144')
        return torch.cat([tensor[:4096],grouped_heads(tensor[4096:],0,128)])
    if 'self_attn.gate_proj' in name: return grouped_heads(tensor,0,128)
    if any(key in name for key in ('ssm_alpha_proj','ssm_beta_proj','ssm_a','ssm_dt')):
        return grouped_heads(tensor,0,1)
    if 'ssm_out_proj' in name: return grouped_heads(tensor,1,128//block)
    if 'ssm_conv1d' in name:
        if tensor.shape!=(10240,4): raise ValueError('expected conv [10240,4]')
        return torch.cat([tensor[:4096],grouped_heads(tensor[4096:],0,128)]).T.contiguous()
    if 'self_attn.q_proj' in name:
        return tensor.reshape(24,2,256,*tensor.shape[1:]).transpose(0,1).reshape(tensor.shape).contiguous()
    return tensor


def convert(converter,path):
    reader=converter.gguf_reader
    expected={'embedding_length':5120,'block_count':64,'feed_forward_length':17408,
              'ssm.group_count':16,'ssm.time_step_rank':48,'ssm.state_size':128,'ssm.inner_size':6144}
    for key,value in expected.items():
        if reader.fields['qwen35.'+key].contents()!=value: raise ValueError(f'unsupported wide geometry: {key}')
    path=Path(path)
    if path.suffix!='.q4nx': path=path/'model.q4nx'
    with TensorWriter(path) as writer:
        for source in converter.gguf_tensors.values():
            if '.nextn.' in source.name: continue
            name=converter.forward_name_map[source.name]
            target=source.get_used_quantization_type(converter.tensor_q4nx_type_map[source.name])
            if source.name in ('token_embd.weight','output.weight'):
                def parts():
                    for start in range(0,int(source.shape[1]),512):
                        raw=source.data[start:start+512]
                        chunk=GGUFTensor(source.name,(int(source.shape[0]),len(raw)),raw,source.tensor_type)
                        if source.name=='token_embd.weight': yield chunk.dequantize()
                        else: yield converter._pack(*chunk.unpack(QT.Q8_0),tensor_type=QT.Q8_0)
                writer.add(name,parts())
            elif 'ssm_alpha_proj' in name or 'ssm_beta_proj' in name:
                # The open wide recipe consumes the native 48-row BF16 companion,
                # never a padded quantized alpha/beta tensor.
                bf_name=name.replace('_proj.weight','_proj.bf16.weight')
                writer.add(bf_name,[restore_order(name,source.dequantize())])
            else:
                unpacked=source.unpack(target)
                if len(unpacked)==1:
                    tensor=restore_order(name,unpacked[0])
                    dtype=torch.float32 if name.endswith(('ssm_a','ssm_dt.bias')) else torch.bfloat16
                    writer.add(name,[tensor.to(dtype)])
                else:
                    unpacked=tuple(restore_order(name,x,32 if i<2 else 1) for i,x in enumerate(unpacked))
                    writer.add(name,[converter._pack(*unpacked,tensor_type=source.get_used_quantization_type(target))])
                del unpacked
            print('Packed',name,flush=True)
