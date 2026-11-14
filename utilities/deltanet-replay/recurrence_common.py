"""Contracts for isolated production-layout recurrence, without model weights."""
import hashlib
import importlib.util
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT/'open_kernels'),str(ROOT/'open_kernels/model')]
from recipes.qwen35 import recipe
from deltanet_state import finite,metric


def digest(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def geometry(spec):
    r=recipe(spec);c=r.common;l=r.layout
    if spec.family!='qwen35' or spec.hidden!=5120 or spec.lin_value_heads!=48 or c.DN_ROWS!=10 or l.S_ROWS!=130:
        raise ValueError('requires validated 48-head / 130-row production geometry')
    return dict(rows=10,slices=13,state_bytes=66560,call_bytes=5120,heads=48,scratch_floats=1280)


def sources(mode):
    if mode not in ('control','triple','factored'):raise ValueError('unknown recurrence mode')
    p=ROOT/'open_kernels/designs/layer_x/gen_kernels.py'
    sp=importlib.util.spec_from_file_location('_recurrence_gen',p);m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m)
    header=ROOT/'open_kernels/designs/layer_x/dnx.h' if mode=='control' else HERE/('factored.h' if mode=='factored' else 'precise.h')
    extra={'precise.h':(HERE/'precise.h').read_text()} if mode=='factored' else {}
    return {**m.DNX,**extra,'dnx.h':header.read_text(),
            'design.py':"import sys\nsys.path.insert(0,"+repr(str(ROOT/'open_kernels'))+")\n"+(HERE/'design.py').read_text()}


def factored_reference(state,records):
    """FP64 algebra check only; primary acceptance stays step_reference."""
    state,records=finite(state),finite(records)
    if records.ndim!=2 or records.shape[1]!=512 or state.shape!=(len(records),128,128):
        raise ValueError('invalid factored reference shape')
    k,q,v=(records[:,i:i+128] for i in (0,128,256))
    decay,beta=records[:,384,None],records[:,385,None]
    delta=beta*(v-decay*np.einsum('hij,hi->hj',state,k))
    updated=decay[:,:,None]*state+k[:,:,None]*delta[:,None,:]
    output=(decay*np.einsum('hij,hi->hj',state,q)+np.einsum('hi,hi->h',k,q)[:,None]*delta)/np.sqrt(128)
    return finite(updated),finite(output)


def replay_cfg(build,inputs,out,layers,tokens,heads=48):
    build,inputs,out=map(lambda p:Path(p).resolve(),(build,inputs,out))
    lines=['device',f'xclbin step {build}/final.xclbin',f'kernelx step step {build}/insts.bin',
           'buf s 66560','buf v 5120','buf so 66560','buf o 512']
    for layer in layers:
        for head in range(heads):
            lines.append(f'load s {inputs}/zero.bin')
            for token in range(tokens):
                name=f'L{layer}_H{head}_T{token}'
                lines += [f'load v {inputs}/{name}.bin',f'load so {inputs}/poison-state.bin',
                          f'load o {inputs}/poison-output.bin','run step s v so o',
                          f'dump so {out}/{name}-state.bin 66560',f'dump o {out}/{name}-output.bin 512',
                          'copy s 0 so 0 66560']
    return '\n'.join(lines)+'\n'


def score(state,output,control_state,control_output,ref_state,ref_output,*,control,query=None):
    for a,shape in [(state,(130,128)),(output,(128,)),(control_state,(130,128)),
                    (control_output,(128,)),(ref_state,(128,128)),(ref_output,(128,))]:
        if np.shape(a)!=shape:raise ValueError('invalid recurrence capture shape')
        finite(a)
    sm,om=metric(state[:128],ref_state),metric(output,ref_output)
    exact=state.tobytes()==control_state.tobytes() and output.tobytes()==control_output.tobytes()
    padding=bool(np.all(state[128:]==0))
    result=dict(passed=padding and (exact if control else sm['passed'] and om['passed']),
                state=sm,output=om,padding_zero=padding,byte_exact=exact)
    if query is not None:
        query=finite(query)
        if query.shape!=(128,):raise ValueError('invalid query shape')
        exact_readout=np.einsum('ij,i->j',finite(state[:128]),query)/np.sqrt(128)
        result.update(captured_state_readout=metric(output,exact_readout),state_effect=metric(exact_readout,ref_output))
    return result
