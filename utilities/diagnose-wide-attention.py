#!/usr/bin/env python3
"""Replay captured wide attention without changing model acceptance references."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np
from ml_dtypes import bfloat16

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'open_kernels'))
from wide_attention_reference import decode, metric

GUARD=bytes([0xA5])*64


def sha(path):
    with Path(path).open('rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()


def capture(path,dtype,shape,finite=True):
    raw=Path(path).read_bytes();size=int(np.prod(shape))*np.dtype(dtype).itemsize
    if len(raw)!=size+64 or raw[-64:]!=GUARD: raise ValueError(f'{path}: size/canary')
    a=np.frombuffer(raw[:size],dtype).reshape(shape)
    if finite and not np.isfinite(a.astype(np.float32)).all(): raise ValueError(f'{path}: nonfinite')
    return a


def conditional(qg,kvn,norms,cs,cache,pos):
    new,og=decode(qg,kvn,norms,cs,cache,pos)
    return new,og.astype(np.float32).astype(bfloat16)


def differences(got,local,reference):
    bits=lambda a: a.astype(bfloat16).view(np.uint16)
    ids=np.flatnonzero(bits(got).ravel()!=bits(local).ravel())
    return dict(local=len(ids),propagated=int(np.count_nonzero(bits(local)!=bits(reference))),
                device_vs_reference=int(np.count_nonzero(bits(got)!=bits(reference))),first_local_indices=ids[:32].tolist())


def prepare(source,tag,kernel,out,projection=None):
    source,kernel=source.resolve(),kernel.resolve()
    meta=json.loads((source/'slice-fixture.json').read_text())
    case=next(c for c in meta['cases'] if c['tag']==tag)
    if case['kind']!='full_attention': raise ValueError('requires a full-attention frame')
    f=json.loads((kernel/'attention-fixture.json').read_text())
    if tuple(f[k] for k in ('nh','kvh','hd','rot','rows'))!=(24,4,256,64,meta['rows']):
        raise ValueError('attention geometry mismatch')
    if not json.loads((kernel/'attention-results.json').read_text())['passed']:
        raise ValueError('attention primitive gate failed')
    for name,digest in f['sha256'].items():
        if sha(kernel/name)!=digest: raise ValueError(f'primitive fixture changed: {name}')
    for name,digest in meta['kernels'].items():
        if sha(name)!=digest: raise ValueError(f'source kernel changed: {name}')
    def checked(name):
        if sha(source/name)!=meta['fixtures'][name]: raise ValueError(f'model fixture changed: {name}')
        return source/name
    ref=np.load(checked(f'{tag}-ref.npz'))
    const=np.fromfile(checked(f"layer{case['layer']}/const.bin"),np.uint8)
    position=checked(f"{case['token']}-position.bin").read_bytes()
    pos,rows=np.frombuffer(position,np.int32,count=2)
    if pos!=case['pos'] or rows!=meta['rows']: raise ValueError('position mismatch')
    module_spec=importlib.util.spec_from_file_location('attention_layer',ROOT/'utilities/test-wide-attention-layer.py')
    module=importlib.util.module_from_spec(module_spec);module_spec.loader.exec_module(module)
    _,layout=module.geometry(int(rows))
    record=const[layout.CA_META:layout.CA_META+2048].tobytes()+position
    if len(record)!=4096: raise ValueError('metadata size')
    norms=np.frombuffer(record,bfloat16,count=512).reshape(2,256)
    cs=np.frombuffer(position,np.float32,count=64,offset=512)
    qg=capture(source/f'{tag}-got-qg.bin',np.float32,(2,24,256))
    kvn=capture(source/f'{tag}-got-kvn.bin',np.float32,(2,4,256))
    cache=capture(source/f'{tag}-input-state.bin',bfloat16,(int(rows),2,4,256),finite=False)
    if not np.isfinite(cache[:pos].astype(np.float32)).all(): raise ValueError('nonfinite past cache')
    new,og=conditional(qg,kvn,norms,cs,cache,int(pos))
    source_new=capture(source/f'{tag}-got-new.bin',bfloat16,(2,4,256))
    source_og=capture(source/f'{tag}-got-og.bin',bfloat16,(24,256))
    if projection is not None:
        projection=projection.resolve()
        pf=json.loads((projection/'projection-fixture.json').read_text())
        if (pf['k'],pf['n'])!=(5120,14336): raise ValueError('projection geometry mismatch')
        if not json.loads((projection/'projection-results.json').read_text())['passed']:
            raise ValueError('projection primitive gate failed')
        for name,digest in pf['sha256'].items():
            if sha(projection/name)!=digest: raise ValueError(f'projection artifact changed: {name}')
        xn=capture(source/f'{tag}-got-xn.bin',bfloat16,(6144,),finite=False)
        if not np.array_equal(xn[:5120],ref['xn'].astype(bfloat16)):
            raise ValueError('isolated projection replay requires exact model xn')
        pool=checked(f"layer{case['layer']}/pool.bin")
    out.mkdir(parents=True,exist_ok=False)
    # NumPy serializes custom BF16 as opaque V2; store exact FP32 expansions.
    arrays=dict(new=new,og=og,model_new=ref['new'],model_og=ref['og'].reshape(24,256),
                source_new=source_new,source_og=source_og,norms=norms,cs=cs)
    if projection is not None: arrays.update(qg=ref['qg'],kvn=ref['kvn'])
    np.savez(out/'reference.npz',**{k:v.astype(np.float32) for k,v in arrays.items()})
    links={'qg.bin':source/f'{tag}-got-qg.bin','kvn.bin':source/f'{tag}-got-kvn.bin',
           'cache.bin':source/f'{tag}-input-state.bin','final.xclbin':kernel/'final.xclbin','insts.bin':kernel/'insts.bin'}
    if projection is not None:
        links.update({'projection.xclbin':projection/'final.xclbin','projection-insts.bin':projection/'insts.bin',
                      'xn.bin':source/f'{tag}-got-xn.bin'})
        # Original contiguous Q,K,V,gate pools; no repacking or new weight format.
        with pool.open('rb') as stream:
            stream.seek(layout.POOL_Q);data=stream.read(45875200)
        if len(data)!=45875200 or layout.POOL_O-layout.POOL_Q!=45875200:
            raise ValueError('Q/K/V/gate pool layout mismatch')
        (out/'qkvg-weights.bin').write_bytes(data)
        (out/'poison-projected.bin').write_bytes(np.full(14336,np.nan,np.float32).tobytes()+GUARD)
    for name,target in links.items(): (out/name).symlink_to(target)
    (out/'meta.bin').write_bytes(record)
    (out/'poison-new.bin').write_bytes(np.full((2,4,256),np.nan,bfloat16).tobytes()+GUARD)
    (out/'poison-og.bin').write_bytes(np.full((24,256),np.nan,bfloat16).tobytes()+GUARD)
    cfg=['device','xclbin a final.xclbin','kernelx a a insts.bin','buf meta 4096 meta.bin',
         'buf qg 49216 qg.bin','buf kvn 8256 kvn.bin',f'buf cache {cache.nbytes+64}',
         'buf new 4160','buf og 12352']
    if projection is not None:
        cfg+=['xclbin p projection.xclbin','kernelx p p projection-insts.bin',
              'buf pw 45875200 qkvg-weights.bin','buf xn 12352 xn.bin','buf projected 57408']
    for i in range(2):
        if projection is not None:
            cfg+=['load projected poison-projected.bin','run p pw xn projected',
                  'copy qg 0 projected 0 24576','copy qg 24576 projected 32768 24576',
                  'copy kvn 0 projected 24576 8192',f'dump projected projected{i}.bin 57408',
                  f'dump qg qg{i}.bin 49216',f'dump kvn kvn{i}.bin 8256']
        cfg+=['load cache cache.bin','load new poison-new.bin','load og poison-og.bin',
              'run a meta qg kvn cache new og',f'dump new new{i}.bin 4160',f'dump og og{i}.bin 12352']
    (out/'replay.cfg').write_text('\n'.join(cfg)+'\n')
    files=[*links,'reference.npz','meta.bin','poison-new.bin','poison-og.bin','replay.cfg']
    if projection is not None: files+=['qkvg-weights.bin','poison-projected.bin']
    fixture=dict(diagnostic_only=True,tag=tag,source=str(source),kernel=str(kernel),pos=int(pos),rows=int(rows),
                 projection=str(projection) if projection is not None else None,
                 sha256={name:sha(out/name) for name in files})
    (out/'replay-fixture.json').write_text(json.dumps(fixture,indent=2)+'\n')
    print(out/'replay.cfg',flush=True)


def compare(out,exact=False):
    f=json.loads((out/'replay-fixture.json').read_text())
    for name,digest in f['sha256'].items():
        if sha(out/name)!=digest: raise ValueError(f'replay fixture changed: {name}')
    ref=dict(np.load(out/'reference.npz'));result=dict(diagnostic_only=True,tag=f['tag'])
    repeat_names=['new','og']
    if f.get('projection'):
        qg=capture(out/'qg0.bin',np.float32,(2,24,256))
        kvn=capture(out/'kvn0.bin',np.float32,(2,4,256))
        capture(out/'projected0.bin',np.float32,(14336,))
        cache=capture(out/'cache.bin',bfloat16,(f['rows'],2,4,256),finite=False)
        for name,data in [('qg',qg),('kvn',kvn)]:
            result[name]=dict(different=int(np.count_nonzero(data!=ref[name])),
                              maxabs=float(np.max(np.abs(data.astype(np.float64)-ref[name]))))
        ref['new'],ref['og']=conditional(qg,kvn,ref['norms'],ref['cs'],cache,f['pos'])
        repeat_names+=['qg','kvn','projected']
    passed=True
    for name,shape,tol in [('new',(2,4,256),.01),('og',(24,256),.02)]:
        got=capture(out/f'{name}0.bin',bfloat16,shape)
        local=ref[name];model=ref[f'model_{name}']
        diff=differences(got,local,model)
        result[name]=dict(**diff,metric=metric(got,local,tol),matches_source=bool(np.array_equal(got,ref[f'source_{name}'])))
        passed &= result[name]['metric']['passed']
        if name=='og':
            result['heads']=[metric(g,r,tol) for g,r in zip(got,local)]
            passed &= all(h['passed'] for h in result['heads'])
    result['repeat_exact']=all((out/f'{n}0.bin').read_bytes()==(out/f'{n}1.bin').read_bytes() for n in repeat_names)
    result['passed']=bool(passed and result['repeat_exact'] and (not exact or result['og']['local']==0))
    result['require_exact_og']=exact
    (out/'replay-results.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='heads'},indent=2),flush=True)
    return 0 if result['passed'] else 1


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=('prepare','compare'));p.add_argument('--out',type=Path,required=True)
    p.add_argument('--source',type=Path);p.add_argument('--kernel',type=Path)
    p.add_argument('--projection',type=Path,help='also replay Q/K/V/gate from exact model xn')
    p.add_argument('--tag',default='cold-0-layer3')
    p.add_argument('--require-exact-og',action='store_true',help='require exact BF16 output against the conditional oracle for captured device inputs')
    a=p.parse_args()
    if a.stage=='prepare':
        if a.source is None or a.kernel is None: p.error('prepare requires --source and --kernel')
        if a.require_exact_og:p.error('--require-exact-og requires compare')
        prepare(a.source,a.tag,a.kernel,a.out,a.projection)
    else:sys.exit(compare(a.out,a.require_exact_og))
