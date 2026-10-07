"""Independent references and lane decoding for the isolated K4096 Q4 trace."""
import numpy as np
from ml_dtypes import bfloat16
from q4_1_pack import dequant_chunk


def decode(raw):
    if raw.size!=16*1120 or not np.isfinite(raw).all():
        raise ValueError('block trace size/nonfinite')
    r=raw.reshape(16,1120)
    order=np.arange(32)//2+16*(np.arange(32)%2)
    blocks=r[:,:1024].reshape(128,4,32)[:,:,order]
    high=r[:,1024:1056][:,order].copy()
    high[-1]=r[-1,1024:1056]
    return blocks,high,r[:,1056:1088][:,order]


def block_sums(raw):
    if raw.size!=16*1120 or not np.isfinite(raw).all():
        raise ValueError('block trace size/nonfinite')
    return raw.reshape(16,1120)[:,1088:1112].reshape(16,3,8).transpose(0,2,1).reshape(128,3)


def reference(tiles,x):
    if tiles.size!=16*5120 or x.shape!=(4096,):
        raise ValueError('weight/activation shape')
    x=x.astype(bfloat16).astype(np.float64)
    result=[]
    for i in range(16):
        w=dequant_chunk(tiles[i*5120:(i+1)*5120]).astype(np.float64)
        for kb in range(8):
            start=i*256+kb*32
            result.append((w[:,kb*32:(kb+1)*32]*x[start:start+32]).sum(axis=1))
    return np.asarray(result)
