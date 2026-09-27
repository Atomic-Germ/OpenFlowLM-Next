"""Header-only preflight for the full wide Q4 layer/Q8 head bring-up path.

This establishes packing prerequisites, never hardware/model acceptance.
"""
import json
import math
from pathlib import Path
import struct

from . import qwen35
from .wide_attention_layer import check_geometry
from .wide_slice import layer_types


def tensor_requirements(s):
    check_geometry(s)
    kinds = layer_types(s,layers=64)
    if s.vocab!=248320 or (s.lin_key_heads,s.lin_value_heads,s.lin_key_dim,s.lin_value_dim,s.conv_kernel)!=(16,48,128,128,4):
        raise ValueError('not implemented: full wide model outside the validated primitive geometry')
    plan = qwen35.pack_plan(s)
    result = {}
    def add(name,dtype,shape,projection=False):
        r = dict(dtype=dtype,shape=shape,bytes=math.prod(shape)*{'I8':1,'BF16':2,'F32':4}[dtype],projection=projection)
        if name in result and projection:
            r['shape'][0] = max(r['shape'][0],result[name]['shape'][0])
            r['bytes'] = math.prod(r['shape'])
        result[name] = r
    for i,kind in enumerate(kinds):
        for op in plan['layer_types'][kind]['pool']+plan['layer_types'][kind]['consts']:
            name=op['tensor'].format(l=i)
            if op['op']=='std_perm':
                add(name,'I8',[op.get('chunk0',0)+op['nch'],5120],True)
                columns=op['in_dim']//256
                result[name]['grid']=[result[name]['shape'][0]//columns,columns]
            elif op['op'] in ('transpose','transpose_banked'):
                add(name,'BF16',[op['rows'],op['cols']])
            elif op['op']=='conv_transpose':
                add(name,'BF16',[op['taps'],op['groups']*op['width']])
            elif op['op']=='put':
                dtype='F32' if name.endswith(('ssm_a','ssm_dt.bias')) else 'BF16'
                # Layer norm entries reserve ELN bytes, but the tensor is H elements.
                n = s.hidden if name.endswith(('input_layernorm.weight','post_attention_layernorm.weight')) else op['cap']//(4 if dtype=='F32' else 2)
                add(name,dtype,[n])
            else:
                raise ValueError(f"not implemented: wide model pack op {op['op']}")
    add(plan['embed']['tensor'],'BF16',[s.vocab,s.hidden])
    add(plan['norm']['tensor'],'BF16',[s.hidden])
    add('lm_head.weight','I8',[s.vocab*s.hidden//8192,8704])
    result['lm_head.weight']['grid']=[s.vocab//32,s.hidden//256]
    return result


def read_header(path):
    path=Path(path)
    with path.open('rb') as f:
        prefix=f.read(8)
        if len(prefix)!=8: raise ValueError('Q4NX: missing header length')
        length=struct.unpack('<Q',prefix)[0]
        if not 2<=length<=16*1024*1024 or length>path.stat().st_size-8:
            raise ValueError('Q4NX: invalid or truncated header length')
        try: header=json.loads(f.read(length))
        except (ValueError,UnicodeError) as e: raise ValueError('Q4NX: invalid JSON header') from e
        if not isinstance(header,dict): raise ValueError('Q4NX: expected tensor dictionary')
    return header,path.stat().st_size-8-length


def validate_header(s,header,data_bytes):
    requirements=tensor_requirements(s)
    converted=[]
    for name,r in requirements.items():
        if name not in header: raise ValueError(f'missing required tensor: {name}')
        item=header[name]
        if not isinstance(item,dict): raise ValueError(f'{name}: invalid tensor record')
        shape=item.get('shape')
        valid=shape==r['shape'] or ('grid' in r and shape==r['grid']+[r['shape'][-1]])
        if r['projection']:
            valid=(isinstance(shape,list) and len(shape) in (2,3)
                   and shape[:-1] in ([r['shape'][0]],r['grid']) and shape[-1] in (5120,8704,4736))
            if valid and shape[-1]!=5120: converted.append(name)
        if item.get('dtype')!=r['dtype'] or not valid:
            raise ValueError(f"not implemented: {name}: expected {r['dtype']} {r['shape']} (Q4 layer projections also accept Q8/Q4_K source chunks)")
        offsets=item.get('data_offsets')
        size=math.prod(shape)*{'I8':1,'BF16':2,'F32':4}[r['dtype']]
        if not isinstance(offsets,list) or len(offsets)!=2 or any(type(v) is not int for v in offsets) or offsets[1]-offsets[0]!=size:
            raise ValueError(f'{name}: data range does not match tensor size')
    ranges=[]
    for name,item in header.items():
        if name=='__metadata__': continue
        offsets=item.get('data_offsets') if isinstance(item,dict) else None
        if not isinstance(offsets,list) or len(offsets)!=2 or any(type(v) is not int for v in offsets):
            raise ValueError(f'{name}: invalid data offsets')
        a,b=offsets
        if not 0<=a<=b<=data_bytes: raise ValueError(f'{name}: data outside container')
        if b>a: ranges.append((a,b,name))
    ranges.sort()
    for previous,current in zip(ranges,ranges[1:]):
        if current[0]<previous[1]: raise ValueError(f'overlapping tensors: {previous[2]}, {current[2]}')
    return dict(ready_for_packing=True,model_validated=False,layers=64,
                required_tensors=len(requirements),requantized_tensors=converted,
                note='Header checks only; tensor values, packed model and 64-layer NPU execution remain unvalidated.')
