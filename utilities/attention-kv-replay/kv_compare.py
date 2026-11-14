"""Compare the paired-KV experiment with its widened control and earlier FP32 replay."""
import argparse
import json
from pathlib import Path
import numpy as np
from kv_common import read_kv
from replay_common import digest,read_debug


def load_run(path,mode):
    score=json.loads((path/'score.json').read_text());audit=json.loads((path/'audit.json').read_text())
    if score['mode']!=mode or audit['mode']!=mode:raise ValueError('mode mismatch')
    if score['audit_sha256']!=digest(path/'audit.json'):raise ValueError('audit hash mismatch')
    for name,h in audit['sha256'].items():
        if digest(name)!=h:raise ValueError(f'input hash mismatch: {name}')
    layers=audit['layers']
    if not layers or [r['layer'] for r in score['results']]!=layers:raise ValueError('layer coverage mismatch')
    expected={f'{n}_L{l}.bin' for l in layers for n in ('output','kv','debug')}
    if set(score['capture_sha256'])!=expected:raise ValueError('capture coverage mismatch')
    for name,h in score['capture_sha256'].items():
        if digest(path/name)!=h:raise ValueError(f'capture hash mismatch: {name}')
    return score,audit


def summary(score):
    heads=[dict(layer=l['layer'],**r) for l in score['results'] for r in l['results']]
    return dict(passed=score['passed'],heads=len(heads),failures=sum(not r['passed'] for r in heads),
                worst=max(heads,key=lambda r:r['maxabs']),min_cosine=min(r['cosine'] for r in heads))


def compare(original,control,pair):
    runs=[load_run(p,m) for p,m in zip((original,control,pair),('fp32','bf16','pair'))]
    g=runs[1][1]['geometry'];layers=runs[1][1]['layers']
    if any(a['layers']!=layers for s,a in runs):raise ValueError('different layer coverage')
    if runs[2][1]['geometry']!=g:raise ValueError('different paired geometry')
    for k in runs[0][1]['geometry']:
        if runs[0][1]['geometry'][k]!=g[k]:raise ValueError('different original geometry')
    rows=[]
    for l in layers:
        inp=[(p/f'input_L{l}.bin').read_bytes() for p in (original,control,pair)]
        debug=[read_debug(p/f'debug_L{l}.bin',a['geometry']) for p,(s,a) in zip((original,control,pair),runs)]
        ctl=read_kv(control/f'kv_L{l}.bin',g,'bf16')
        from q4nx import bf16_to_f32
        orig=bf16_to_f32(np.fromfile(original/f'kv_L{l}.bin',np.uint16)).reshape(ctl.shape)
        raw=[np.fromfile(p/f'kv_L{l}.bin',np.uint16).reshape(g['tokens'],2,2,g['kw']) for p in (control,pair)]
        rows.append(dict(layer=l,inputs_identical=inp[0]==inp[1]==inp[2],
            control_output_identical=(original/f'output_L{l}.bin').read_bytes()==(control/f'output_L{l}.bin').read_bytes(),
            control_debug_identical=(original/f'debug_L{l}.bin').read_bytes()==(control/f'debug_L{l}.bin').read_bytes(),
            control_kv_identical=bool(np.array_equal(orig,ctl)),
            high_planes_identical=bool(np.array_equal(raw[0][:,:,0],raw[1][:,:,0])),
            q_identical=bool(np.array_equal(debug[0]['q_scaled'],debug[1]['q_scaled']) and
                             np.array_equal(debug[0]['q_scaled'],debug[2]['q_scaled']))))
    identity=all(all(v for k,v in r.items() if k!='layer') for r in rows)
    return dict(passed=identity and runs[2][0]['passed'],control_verified=identity,comparisons=rows,
                scores={m:summary(s) for m,(s,a) in zip(('original','control','pair'),runs)},
                score_sha256={str(p.resolve()/'score.json'):digest(p/'score.json') for p in (original,control,pair)})


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for n in ('original','control','pair','out'):ap.add_argument('--'+n,type=Path,required=True)
    a=ap.parse_args();a.out.unlink(missing_ok=True)
    r=compare(a.original,a.control,a.pair);a.out.write_text(json.dumps(r,indent=2)+'\n')
    print(json.dumps(r,indent=2));return 0 if r['passed'] else 1


if __name__=='__main__':raise SystemExit(main())
