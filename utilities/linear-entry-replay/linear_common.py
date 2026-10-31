"""Isolated full-lx FP32 entry experiment; production ABI remains unchanged."""
import hashlib
import importlib.util
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
    r=recipe(spec);l=r.layout;d=r.linear
    if (mode not in ('bf16','fp32') or spec.family!='qwen35' or spec.hidden!=5120 or
            spec.lin_value_heads!=48 or not r.ln_split or not r.ffn.XM_F32 or not r.ffn.DOWN_SPLIT or
            d is None or d.XN_F32 or d.CONV_F32 or d.AB_LANES!=64 or not d.GLUE_HALF_OUTER):
        raise ValueError('requires Qwen H5120/48-head split FP32 FFN with BF16 linear entry')
    fp=mode=='fp32'
    return dict(mode=mode,hidden=spec.hidden,production_act_bytes=l.A_BYTES,
                act_bytes=l.A_BYTES+(spec.hidden*4 if fp else 0),state_bytes=l.STATE_BYTES,
                norm_offset=l.A_BYTES if fp else l.A_XN,norm_bytes=spec.hidden*(4 if fp else 2),
                norm_elements=5 if fp else 3,ab_tiles=[32]*5 if fp else [64,64,32])


def source_hash(spec,mode):
    geometry(spec,mode)
    h=hashlib.sha256((build_key(spec)+mode).encode())
    for name in ('linear_common.py','linear_worker.py','build.py','ab_fp32.cc','copy_fp32.cc'):
        h.update(name.encode()+b'\0'+(HERE/name).read_bytes())
    return h.hexdigest()


def generated_sources(spec):
    path=ROOT/'open_kernels/designs/layer_x/gen_kernels.py'
    ms=importlib.util.spec_from_file_location('_linear_entry_gen',path)
    module=importlib.util.module_from_spec(ms);ms.loader.exec_module(module)
    return module.files(recipe(spec))


def verify_generated(directory,spec):
    sources=generated_sources(spec)
    if {p.name for p in directory.glob('*.cc')}!=set(sources):
        raise ValueError('generated source coverage mismatch')
    hashes={}
    for name,source in sources.items():
        if (directory/name).read_bytes()!=source.encode():raise ValueError(f'stale generated source: {name}')
        hashes[name]=digest(directory/name)
    return hashes


def render_design(spec,source=None):
    geometry(spec,'fp32');path=ROOT/'open_kernels/designs/layer_x/lx.py'
    s=path.read_text() if source is None else source
    pairs=[('HERE = Path(__file__).parent',f'HERE = Path({str(path.parent)!r})'),
           ('OG_ELEMS = D.OG_ELEMS','A_XN = A_BYTES\nA_BYTES += HID * 4\nXN_ELEMS = HID * 4 // ELEM\n'
            f'sys.path.insert(0, {str(HERE)!r})\nfrom linear_worker import install_glue, private_kernels\ninstall_glue(XL)\n'
            'AB_TILES = XL.AB_TILES\nOG_ELEMS = D.OG_ELEMS'),
           ('XL.ln_worker(of_lni, of_lno, tl, L)','XL.ln_worker(of_lni, of_lno, tl, L, entry_fp32=True)'),
           ('X.prep_bands(win, xin, yout, B, K, HID, XN_ELEMS, QKV_PC + Z_PC, "linear")',
            'X.prep_bands(win, xin, yout, B, K, HID, XN_ELEMS, QKV_PC + Z_PC, "linear", fp32=True)'),
           ('lno.drain(a_act, tap=bt(A_BYTES, A_XN, ELN), wait=True, group=tg_ln)',
            'lno.drain(a_act, tap=bt(A_BYTES, A_XN, HID * 4), wait=True, group=tg_ln)'),
           ('K = X.kernels(inc, t)','K = private_kernels(X, inc, t, Path(__file__).parent / "generated")')]
    for old,new in pairs:
        if s.count(old)!=1:raise ValueError(f'linear replay source anchor changed: {old}')
        s=s.replace(old,new)
    return s+f'\nLINEAR_ENTRY_HASH = {source_hash(spec,"fp32")!r}\nSPECIALIZE["srchash"] = int(LINEAR_ENTRY_HASH[:8],16)\n'


def read_norm(path,g):
    data=Path(path).read_bytes();off=g['norm_offset'];n=g['norm_bytes']
    size=4 if g['mode']=='fp32' else 2
    if len(data)!=g['act_bytes'] or off<0 or off%size or off+n>len(data) or n!=g['hidden']*size:
        raise ValueError('invalid linear norm capture size/region')
    a=np.frombuffer(data,'<f4' if size==4 else '<u2',g['hidden'],off)
    a=(a if size==4 else bf16_to_f32(a)).astype(np.float64)
    if not np.isfinite(a).all():raise ValueError('non-finite linear norm')
    return a
