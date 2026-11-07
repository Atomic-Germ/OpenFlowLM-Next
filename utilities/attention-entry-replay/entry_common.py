"""Private FP32 entry experiment: keep production offsets, append norm scratch."""
import hashlib
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT/'open_kernels'),str(ROOT/'open_kernels/model')]
from recipes.cache import build_key
from recipes.qwen35 import recipe
from q4nx import bf16_to_f32


def digest(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def geometry(spec,mode):
    r=recipe(spec);l=r.layout
    if r.attn is not None and r.attn.XN_F32:
        raise ValueError("production already uses FP32 entry; replay requires the pre-integration checkout")
    if mode not in ('bf16','fp32') or not r.attn.KV_PAIR or spec.hidden!=5120 or not r.ln_split:
        raise ValueError('requires paired Qwen H5120 split norm and bf16/fp32 mode')
    fp=mode=='fp32'
    return dict(mode=mode,hidden=spec.hidden,production_act_bytes=l.AA_BYTES,
                act_bytes=l.AA_BYTES+(spec.hidden*4 if fp else 0),
                norm_offset=l.AA_BYTES if fp else l.AA_XN,
                norm_bytes=spec.hidden*(4 if fp else 2),norm_elements=5 if fp else 3)


def source_hash(spec,mode):
    geometry(spec,mode)
    h=hashlib.sha256((build_key(spec)+mode).encode())
    for name in ('entry_common.py','entry_worker.py','build.py'):
        h.update(name.encode()+b'\0'+(HERE/name).read_bytes())
    return h.hexdigest()


def render_design(spec,source=None):
    geometry(spec,'fp32')
    path=ROOT/'open_kernels/designs/layer_x/ax.py'
    s=path.read_text() if source is None else source
    replacements=[
        ('HERE = Path(__file__).parent',f'HERE = Path({str(path.parent)!r})'),
        ('OG_ELEMS = D.OG_ELEMS',
         'AA_XN = AA_BYTES\nAA_BYTES += HID * 4\nXN_ELEMS = HID * 4 // ELEM\n'
         f'sys.path.insert(0, {str(HERE)!r})\nfrom entry_worker import install_entry_worker\n'
         'install_entry_worker(X)\nOG_ELEMS = D.OG_ELEMS'),
        ('X.prep_bands(win, xin, yout, B, K, HID, XN_ELEMS, 2 * Q_PC + 2 * KV_PC, "attn")',
         'X.prep_bands(win, xin, yout, B, K, HID, XN_ELEMS, 2 * Q_PC + 2 * KV_PC, "attn", fp32=True)'),
        ('lno.drain(a_act, tap=bt(AA_BYTES, AA_XN, ELN), wait=True, group=tg_ln)',
         'lno.drain(a_act, tap=bt(AA_BYTES, AA_XN, HID * 4), wait=True, group=tg_ln)')]
    for old,new in replacements:
        if s.count(old)!=1:raise ValueError(f'experimental source anchor changed: {old}')
        s=s.replace(old,new)
    return s+f'\nENTRY_SOURCE_HASH = {source_hash(spec,"fp32")!r}\nSPECIALIZE["srchash"] = int(ENTRY_SOURCE_HASH[:8], 16)\n'


def read_candidate_norm(path,g):
    data=Path(path).read_bytes();off=g['norm_offset'];n=g['norm_bytes']
    if len(data)!=g['act_bytes'] or off<0 or off+n>len(data):raise ValueError('invalid norm capture size/region')
    dtype=np.float32 if g['mode']=='fp32' else np.uint16
    if off%np.dtype(dtype).itemsize or n!=g['hidden']*np.dtype(dtype).itemsize:raise ValueError('invalid norm geometry')
    a=np.frombuffer(data,dtype,g['hidden'],off)
    a=(bf16_to_f32(a) if dtype==np.uint16 else a).astype(np.float64)
    if not np.isfinite(a).all():raise ValueError('non-finite norm capture')
    return a
