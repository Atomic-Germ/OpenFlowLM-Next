#!/usr/bin/env python3
"""Compare GGUF 48-head order with selected official HF tensors via HTTP ranges."""
import json
from pathlib import Path
import struct
import sys
import urllib.request

import numpy as np
from gguf import GGUFReader,dequantize

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'Models/qwen38-27b/source'
REV='1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0'


def read_range(shard,start,size):
    url=f'https://huggingface.co/Qwen/Qwen3.8-27B/resolve/{REV}/{shard}?range_start={start}&range_size={size}'
    req=urllib.request.Request(url,headers={'Range':f'bytes={start}-{start+size-1}'})
    with urllib.request.urlopen(req,timeout=60) as response:
        if response.status!=206 or not response.headers.get('Content-Range','').startswith(f'bytes {start}-'):
            raise ValueError('server did not honor bounded range request')
        data=response.read(size+1)
    if len(data)!=size: raise ValueError('incorrect range size')
    return data


def main():
    index=json.loads((BASE/'model.safetensors.index.json').read_text())['weight_map']
    prefix='model.language_model.layers.0.linear_attn.'
    pairs=[('in_proj_a.weight','blk.0.ssm_alpha.weight'),('conv1d.weight','blk.0.ssm_conv1d.weight')]
    reader=GGUFReader(str(BASE/'Qwen3.8-27B-Q8_0.gguf'))
    tensors={t.name:t for t in reader.tensors}
    headers={};results=[]
    for suffix,gguf in pairs:
        name=prefix+suffix;shard=index[name]
        if shard not in headers:
            length=struct.unpack('<Q',read_range(shard,0,8))[0]
            headers[shard]=(length,json.loads(read_range(shard,8,length)))
        length,header=headers[shard];record=header[name];a,b=record['data_offsets']
        raw=read_range(shard,8+length+a,b-a)
        (BASE/('hf-layer0-'+suffix+'.bin')).write_bytes(raw)
        if record['dtype']=='BF16': hf=(np.frombuffer(raw,np.uint16).astype(np.uint32)<<16).view(np.float32)
        elif record['dtype']=='F32': hf=np.frombuffer(raw,np.float32)
        else: raise ValueError(record['dtype'])
        hf=hf.reshape(record['shape']).squeeze()
        t=tensors[gguf];got=dequantize(t.data,t.tensor_type)
        if suffix=='conv1d.weight':
            restored=got.copy();restored[4096:]=got[4096:].reshape(3,16,128,4).transpose(1,0,2,3).reshape(6144,4)
        else: restored=got.reshape(3,16,5120).transpose(1,0,2).reshape(48,5120)
        cosine=lambda x:float(np.dot(x.ravel().astype(np.float64),hf.ravel())/(np.linalg.norm(x.astype(np.float64))*np.linalg.norm(hf.astype(np.float64))))
        result=dict(tensor=name,gguf=gguf,raw_cosine=cosine(got),restored_cosine=cosine(restored),exact=bool(np.array_equal(restored,hf)),max_absolute=float(np.max(np.abs(restored-hf))),passed=cosine(restored)>.9999)
        results.append(result);print(result,flush=True)
    (BASE/'head-order-check.json').write_text(json.dumps(results,indent=2)+'\n')
    return 0 if all(r['passed'] for r in results) else 1


if __name__=='__main__':sys.exit(main())
