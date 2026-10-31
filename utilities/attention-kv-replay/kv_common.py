"""Host ABI for an experimental [K hi, K lo, V hi, V lo] attention cache."""
from pathlib import Path
import sys
import numpy as np

HERE=Path(__file__).resolve().parent
BASE=HERE.parent/'attention-replay'
sys.path.insert(0,str(BASE))
from replay_common import ROOT,geometry as base_geometry
from q4nx import bf16_to_f32


def geometry(spec,tokens):
    g=base_geometry(spec,tokens)
    if (g['heads'],g['kv_heads'],g['dim'],g['local_heads'])!=(24,4,256,4):
        raise ValueError('requires Qwen 24/4 heads, HD256, four heads per core')
    return dict(g,kv_format='bf16-hi-lo',kv_row_bytes=8*g['kw'])


def read_kv(path,g,mode):
    if mode not in ('bf16','pair'):raise ValueError('invalid KV mode')
    data=path.read_bytes()
    if len(data)!=g['tokens']*8*g['kw']:raise ValueError('KV size mismatch')
    planes=bf16_to_f32(np.frombuffer(data,np.uint16)).astype(np.float64)
    planes=planes.reshape(g['tokens'],2,2,g['kv_heads'],g['dim'])
    if not np.isfinite(planes).all():raise ValueError('non-finite KV')
    if mode=='bf16' and np.any(planes[:,:,1]!=0):raise ValueError('nonzero control KV low plane')
    return planes[:,:,0]+planes[:,:,1]
