"""Private full-lx recurrence injection; production ABI and scheduling stay intact."""
import hashlib
import importlib.util
from pathlib import Path
from recurrence_common import ROOT,HERE,digest,geometry as head_geometry,sources
from recipes.qwen35 import recipe
from recipes.cache import build_key

LAYER=ROOT/'open_kernels/designs/layer_x'


def geometry(spec):
    g=head_geometry(spec);r=recipe(spec);d=r.linear
    if not (r.ln_split and r.ffn.XM_F32 and r.ffn.DOWN_SPLIT and d.XN_F32 and d.CONV_F32):
        raise ValueError('requires current FP32 production linear ABI')
    return dict(g,act_bytes=r.layout.A_BYTES,state_bytes=r.layout.STATE_BYTES,hidden=spec.hidden)


def production_sources(spec):
    p=LAYER/'gen_kernels.py';sp=importlib.util.spec_from_file_location('_full_recurrence_gen',p)
    m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m)
    # Private experiments inject dnx.h, even after production selects a new header.
    # Keep all non-recurrence TUs from the current recipe.
    return {**m.files(recipe(spec)),**m.DNX}


def generated_sources(spec,shared=False):
    geometry(spec);head=sources('factored')
    result={**production_sources(spec),'dnx.h':head['dnx.h'],'precise.h':head['precise.h']}
    if shared:
        for n in ('dnx.h','precise.h'):
            result[n]=result[n].replace('static __attribute__((noinline))','inline __attribute__((noinline))')
    return result


def source_hash(spec,shared):
    h=hashlib.sha256((build_key(spec)+str(shared)).encode())
    for n,s in sorted(generated_sources(spec,shared).items()):h.update(n.encode()+b'\0'+s.encode())
    h.update(Path(__file__).read_bytes());h.update((LAYER/'lx.py').read_bytes())
    return h.hexdigest()


def render_design(spec,shared=False,source=None):
    geometry(spec);s=(LAYER/'lx.py').read_text() if source is None else source
    pairs=[('HERE = Path(__file__).parent',f'HERE = Path({str(LAYER)!r})'),
           ('import linear_fp32 as LF','import linear_fp32 as LF\n'+f'sys.path.insert(0, {str(HERE)!r})\nfrom full_layer_common import private_kernels'),
           ('K = X.kernels(inc, t)','K = private_kernels(X, inc, t, Path(__file__).parent / "generated")')]
    for old,new in pairs:
        if s.count(old)!=1:raise ValueError('full-layer source anchor changed: '+old)
        s=s.replace(old,new)
    return s+f'\nSPECIALIZE["srchash"] = {int(source_hash(spec,shared)[:8],16)}\n'


def private_kernels(X,inc,t,directory):
    original=X.HERE
    try:
        X.HERE=directory
        return X.kernels([str(directory),*inc,str(original)],t)
    finally:X.HERE=original


def verify_generated(directory,spec,shared):
    expected=generated_sources(spec,shared)
    if {p.name for p in directory.iterdir() if p.is_file()}!=set(expected):raise ValueError('generated source coverage mismatch')
    for n,s in expected.items():
        if (directory/n).read_text()!=s:raise ValueError('stale generated source: '+n)
    return {n:digest(directory/n) for n in expected}
