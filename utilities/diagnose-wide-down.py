#!/usr/bin/env python3
"""Replay captured FFN h through down only and diagnose segment/reduction error."""
import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('ffn_diagnosis', HERE/'diagnose-wide-ffn.py')
ffn = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ffn)
from wide_down_reference import partial_reference
from wide_down_trace import decode_trace, analyze


def verify_files(out, meta):
    for name, digest in meta['sha256'].items():
        if ffn.sha(out/name) != digest:
            raise ValueError(f'fixture changed: {name}')


def prepare(source, kernel, out):
    source, kernel = source.resolve(), kernel.resolve()
    tool = json.loads((kernel/'probe-toolchain.json').read_text())
    fixture = json.loads((kernel/'segmented-fixture.json').read_text())
    if tool['scope']!='down' or not tool.get('down_trace') or not tool.get('down_rne'):
        raise ValueError('requires down-only high/low trace with final RNE')
    if not json.loads((kernel/'segmented-results.json').read_text())['passed']:
        raise ValueError('down primitive gate failed')
    verify_files(kernel, fixture)
    checked = ffn.compare_trace(source)
    if checked['h_bf16_differences'] or not checked['repeat_exact']:
        raise ValueError('source FFN activation/repeat gate failed')
    src = json.loads((source/'trace-fixture.json').read_text())
    h = ffn.capture(source/'got0.bin',src['act_bytes'],np.float32,logical=ffn.F,offset=src['h_offset'])
    shape = ffn.ModelSpec.from_dict(json.loads((kernel/'probe-spec.json').read_text()))
    if (shape.hidden,shape.intermediate)!=(ffn.H,ffn.F):
        raise ValueError('requires H5120/FF17408')
    layout = ffn.qwen35.layout(shape)
    pool = np.memmap(source/'pool.bin',mode='r',dtype=np.uint8)
    down = pool[layout.POOL_FFN_DOWN:layout.POOL_FFN_DOWN+ffn.H*ffn.F//8192*5120]
    parts = partial_reference(down,h,ffn.H,ffn.F,tool['segment_k'])
    original = ffn.capture(source/'got0.bin',src['act_bytes'],np.float32,logical=ffn.H,offset=src['out_offset'])
    out.mkdir(parents=True,exist_ok=False)
    for name, path in {'pool.bin':source/'pool.bin','final.xclbin':kernel/'final.xclbin',
                       'insts.bin':kernel/'insts.bin'}.items():
        (out/name).symlink_to(path.resolve())
    np.save(out/'parts.npy',parts)
    original.tofile(out/'source-fo.bin')
    raw = bytearray(np.full(fixture['act_bytes']//4,np.nan,np.float32).tobytes()+ffn.GUARD)
    raw[layout.A_H:layout.A_H+ffn.F*4] = h.tobytes()
    (out/'act.bin').write_bytes(raw)
    tb=fixture['trace_bytes']
    (out/'trace-poison.bin').write_bytes(np.full(tb//4,np.nan,np.float32).tobytes()+ffn.GUARD)
    cfg=['device','xclbin p final.xclbin','kernelx p p insts.bin',
         f'buf w {pool.size} pool.bin',f'buf a {len(raw)}',f'buf t {tb+64}']
    for i in range(2):
        cfg += ['load a act.bin','load t trace-poison.bin','run p w a t',
                f'dump a got{i}.bin {len(raw)}',f'dump t trace{i}.bin {tb+64}']
    (out/'down.cfg').write_text('\n'.join(cfg)+'\n')
    meta=dict(diagnostic_only=True,source=str(source),n=ffn.H,k=ffn.F,segments=len(parts),
              act_bytes=fixture['act_bytes'],trace_bytes=tb,h_offset=layout.A_H,
              out_offset=layout.A_OUT2,snapshots=fixture['snapshots'],
              sha256={p.name:ffn.sha(p) for p in out.iterdir()})
    (out/'down-fixture.json').write_text(json.dumps(meta,indent=2)+'\n')
    print(out/'down.cfg',flush=True)


def compare(out, channels):
    meta=json.loads((out/'down-fixture.json').read_text())
    verify_files(out,meta)
    parts=np.load(out/'parts.npy')
    trace=decode_trace(ffn.capture(out/'trace0.bin',meta['trace_bytes'],np.float32),meta['n'],meta['segments'])
    got=ffn.capture(out/'got0.bin',meta['act_bytes'],np.float32,
                    logical=meta['snapshots']*meta['n'],offset=meta['out_offset']).reshape(-1,meta['n'])
    h=ffn.capture(out/'got0.bin',meta['act_bytes'],np.float32,logical=meta['k'],offset=meta['h_offset'])
    before=ffn.capture(out/'act.bin',meta['act_bytes'],np.float32,logical=meta['k'],offset=meta['h_offset'])
    reconstructed=(trace[:,2].astype(np.float64)+trace[:,3]).astype(np.float32)
    if meta['snapshots']==1: reconstructed=reconstructed[-1:]
    repeat=all((out/f'{p}0.bin').read_bytes()==(out/f'{p}1.bin').read_bytes() for p in ('got','trace'))
    checks=dict(repeat_exact=repeat,activation_unchanged=bool(np.array_equal(h,before)),
                trace_matches_output=bool(np.array_equal(got,reconstructed)),
                matches_full_ffn=bool(np.array_equal(got[-1],np.fromfile(out/'source-fo.bin',np.float32))))
    result=dict(diagnostic_only=True,passed=all(checks.values()),checks=checks,
                fp32_differences=int(np.count_nonzero(got[-1]!=parts.sum(axis=0).astype(np.float32))),
                channels=[analyze(trace,parts,c) for c in channels])
    (out/'down-results.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True)
    return 0 if result['passed'] else 1


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=('prepare','compare'))
    p.add_argument('--source',type=Path,help='validated real full-FFN trace directory')
    p.add_argument('--kernel',type=Path)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--channel',type=int,action='append')
    a=p.parse_args()
    if a.stage=='prepare':
        if a.source is None or a.kernel is None: p.error('prepare requires --source and --kernel')
        prepare(a.source,a.kernel,a.out)
        return 0
    return compare(a.out,a.channel or [4872])


if __name__=='__main__':
    raise SystemExit(main())
