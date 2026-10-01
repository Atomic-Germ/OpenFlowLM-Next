#!/usr/bin/env python3
"""Replay immutable full-model fixtures with validated projection/FFN kernels.

Only kernel bindings change. References, weights, tokens and acceptance bounds
are inherited; device captures go into a fresh directory.
"""
import argparse
import hashlib
import json
from pathlib import Path


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f,'sha256').hexdigest()


def replay(source, out, kernel, *, ffn=None, attention=None, ln=None, attention_projection=None):
    source, out, kernel = source.resolve(), out.resolve(), kernel.resolve()
    meta = json.loads((source/'slice-fixture.json').read_text())
    fixture = json.loads((kernel/'projection-fixture.json').read_text())
    results = json.loads((kernel/'projection-results.json').read_text())
    if (fixture['k'],fixture['n']) != (6144,5120):
        raise ValueError('output projection geometry must be K6144/N5120')
    if not results['passed']:
        raise ValueError('output projection primitive gate failed')
    replacements = dict(out=kernel)
    fixtures = dict(out=fixture)
    if attention_projection is not None:
        ap = attention_projection.resolve()
        pf = json.loads((ap/'projection-fixture.json').read_text())
        if (pf['k'],pf['n']) != (5120,14336):
            raise ValueError('attention projection geometry must be K5120/N14336')
        if not json.loads((ap/'projection-results.json').read_text())['passed']:
            raise ValueError('attention projection primitive gate failed')
        replacements['a_qkvg'], fixtures['a_qkvg'] = ap, pf
    if ffn is not None:
        ffn = ffn.resolve()
        ff = json.loads((ffn/'segmented-fixture.json').read_text())
        if (ff['k'],ff['n'],ff['full'],ff['trace']) != (17408,5120,True,False):
            raise ValueError('FFN geometry must be full H5120/FF17408 without trace')
        if not json.loads((ffn/'segmented-results.json').read_text())['passed']:
            raise ValueError('FFN primitive gate failed')
        replacements['ffn'], fixtures['ffn'] = ffn, ff
    if attention is not None:
        attention = attention.resolve()
        af = json.loads((attention/'attention-fixture.json').read_text())
        if tuple(af[n] for n in ('nh','kvh','hd','rot','rows')) != (24,4,256,64,meta.get('rows',257)):
            raise ValueError('attention geometry must match Q24/KV4/HD256/ROT64 and the static DMA rows')
        if not json.loads((attention/'attention-results.json').read_text())['passed']:
            raise ValueError('attention primitive gate failed')
        replacements['a_attn'], fixtures['a_attn'] = attention, af
    if ln is not None:
        ln = ln.resolve()
        lf = json.loads((ln/'ln-fixture.json').read_text())
        if (lf['n'],lf['eps']) != (5120,1e-6):
            raise ValueError('LN geometry must match N5120/epsilon1e-6')
        if not json.loads((ln/'ln-results.json').read_text())['passed']:
            raise ValueError('LN primitive gate failed')
        replacements['ln'], fixtures['ln'] = ln, lf
    for role, directory in replacements.items():
        for name, digest in fixtures[role]['sha256'].items():
            if sha(directory/name) != digest:
                raise ValueError(f'{role} artifact changed: {name}')
    for path, digest in meta['kernels'].items():
        if sha(path) != digest:
            raise ValueError(f'source kernel changed: {path}')
    for name, digest in meta['fixtures'].items():
        if sha(source/name) != digest:
            raise ValueError(f'source fixture changed: {name}')
    lines = (source/'decode.cfg').read_text().splitlines()
    changed = set()
    for i, line in enumerate(lines):
        words = line.split()
        if len(words) >= 3 and words[0] in ('xclbin','kernelx') and words[1] in replacements:
            role = words[1]
            if words[0] == 'kernelx' and words[2] != role:
                raise ValueError('unexpected shared kernel binding')
            file = 'final.xclbin' if words[0] == 'xclbin' else 'insts.bin'
            del meta['kernels'][words[-1]]
            words[-1] = str(replacements[role]/file)
            meta['kernels'][words[-1]] = sha(replacements[role]/file)
            lines[i] = ' '.join(words)
            changed.add((role,file))
    if changed != {(role,file) for role in replacements for file in ('final.xclbin','insts.bin')}:
        raise ValueError('source has no complete replacement kernel binding')
    out.mkdir(parents=True,exist_ok=False)
    for name in meta['fixtures']:
        if name == 'decode.cfg': continue
        target = out/name
        target.parent.mkdir(parents=True,exist_ok=True)
        target.symlink_to(source/name)
    (out/'decode.cfg').write_text('\n'.join(lines)+'\n')
    meta['fixtures']['decode.cfg'] = sha(out/'decode.cfg')
    meta['replay_source'] = str(source)
    meta['kernel_overrides'] = {role:str(directory) for role,directory in replacements.items()}
    (out/'slice-fixture.json').write_text(json.dumps(meta,indent=2)+'\n')
    print(out/'decode.cfg',flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--output-projection',type=Path,required=True)
    p.add_argument('--ffn',type=Path)
    p.add_argument('--attention',type=Path)
    p.add_argument('--ln',type=Path)
    p.add_argument('--attention-projection',type=Path)
    a = p.parse_args()
    replay(a.source,a.out,a.output_projection,ffn=a.ffn,attention=a.attention,ln=a.ln,attention_projection=a.attention_projection)
