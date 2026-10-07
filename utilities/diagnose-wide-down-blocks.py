#!/usr/bin/env python3
"""Isolate one K4096 segment's Q4 blocks from a validated down trace."""
import argparse
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE=Path(__file__).resolve().parent
s=importlib.util.spec_from_file_location('down_diagnosis',HERE/'diagnose-wide-down.py')
down=importlib.util.module_from_spec(s)
s.loader.exec_module(down)
from wide_down_blocks import decode,reference,block_sums,product_trace,product_diagnosis
from ml_dtypes import bfloat16
from wide_down_trace import decode_trace
from recipes.segmented_dense import weight_slice

GUARD=down.ffn.GUARD
TRACE_BYTES=16*1120*4


def build_products(out,block):
    if not 0<=block<128: raise ValueError('product block must be 0..127')
    out.mkdir(parents=True,exist_ok=False)
    env=os.environ.copy()
    env['PROBE_PRODUCT_BLOCK']=str(block)
    env['OFLM_KEEP_FAILED']='1'
    env['PATH']='/opt/xilinx/xrt/bin:'+env.get('PATH','')
    env['LD_LIBRARY_PATH']='/opt/xilinx/xrt/lib:'+env.get('LD_LIBRARY_PATH','')
    subprocess.run([sys.executable,str(HERE.parent/'open_kernels/build_design.py'),
                    str(HERE.parent/'open_kernels/designs/layer_x/down_block_probe.py'),str(out)],
                   env=env,check=True)
    meta=dict(product_block=block,record_floats=2560,
              sha256={name:down.ffn.sha(out/name) for name in ('final.xclbin','insts.bin')})
    (out/'product-trace.json').write_text(json.dumps(meta,indent=2)+'\n')


def prepare(source,kernel,out,segment,channel):
    source,kernel=source.resolve(),kernel.resolve()
    meta=json.loads((source/'down-fixture.json').read_text())
    if (meta['n'],meta['k'])!=(5120,17408) or not 0<=segment<4 or not 0<=channel<5120:
        raise ValueError('requires H5120/FF17408, a full K4096 segment and valid channel')
    if down.compare(source,[channel]): raise ValueError('source trace reproduction failed')
    for name in ('final.xclbin','insts.bin'):
        if not (kernel/name).is_file(): raise ValueError(f'missing kernel {name}')
    product_meta=None
    if (kernel/'product-trace.json').is_file():
        product_meta=json.loads((kernel/'product-trace.json').read_text())
        down.verify_files(kernel,product_meta)
    trace_bytes=16*2560*4 if product_meta else TRACE_BYTES
    x=down.ffn.capture(source/'act.bin',meta['act_bytes'],np.float32,
                       logical=meta['k'],offset=meta['h_offset'])[segment*4096:(segment+1)*4096]
    t=decode_trace(down.ffn.capture(source/'trace0.bin',meta['trace_bytes'],np.float32),5120,5)
    cfg=json.loads((HERE.parent/'specs/open-engine/tests/fixtures/config_qwen38_27b.json').read_text())
    layout=down.ffn.qwen35.layout(down.ffn.ModelSpec.from_hf_config(cfg))
    offset,_=weight_slice(17408,channel//64,segment*4096,4096)
    pool=np.memmap(source/'pool.bin',mode='r',dtype=np.uint8)
    offset+=layout.POOL_FFN_DOWN+(channel%64//32)*5120
    tiles=np.concatenate([pool[offset+i*10240:offset+i*10240+5120] for i in range(16)])
    refs=reference(tiles,x)
    out.mkdir(parents=True,exist_ok=False)
    for name in ('final.xclbin','insts.bin'): (out/name).symlink_to(kernel/name)
    tiles.tofile(out/'weights.bin')
    x.tofile(out/'x.bin')
    # A zero frame between repeated real frames catches stale accumulator state.
    np.zeros_like(x).tofile(out/'zero.bin')
    np.save(out/'reference.npy',refs)
    base=channel//32*32
    np.save(out/'segment.npy',t[segment,:2,base:base+32])
    (out/'poison.bin').write_bytes(np.full(trace_bytes//4,np.nan,np.float32).tobytes()+GUARD)
    lines=['device','xclbin p final.xclbin','kernelx p p insts.bin',
           f'buf w {tiles.nbytes} weights.bin',f'buf x {x.nbytes}',f'buf y {trace_bytes+64}']
    for i,f in enumerate(('x.bin','zero.bin','x.bin')):
        lines += [f'load x {f}','load y poison.bin','run p w x y',f'dump y trace{i}.bin {trace_bytes+64}']
    (out/'blocks.cfg').write_text('\n'.join(lines)+'\n')
    record=dict(diagnostic_only=True,source=str(source),segment=segment,channel=channel,
                trace_bytes=trace_bytes,product_block=product_meta['product_block'] if product_meta else None,
                sha256={p.name:down.ffn.sha(p) for p in out.iterdir()})
    (out/'blocks-fixture.json').write_text(json.dumps(record,indent=2)+'\n')
    print(out/'blocks.cfg',flush=True)


def compare(out):
    meta=json.loads((out/'blocks-fixture.json').read_text())
    down.verify_files(out,meta)
    trace_bytes=meta.get('trace_bytes',TRACE_BYTES)
    raw=down.ffn.capture(out/'trace0.bin',trace_bytes,np.float32)
    products=None
    if meta.get('product_block') is not None:
        products=product_trace(raw,meta['product_block'])
        raw=raw.reshape(16,2560)[:,:1120].copy().ravel()
    blocks,high,low=decode(raw)
    xs=block_sums(raw).astype(np.float64).sum(axis=1)
    x=np.fromfile(out/'x.bin',np.float32).astype(bfloat16).astype(np.float64)
    sum_error=xs-x.reshape(128,32).sum(axis=1)
    tiles=np.fromfile(out/'weights.bin',np.uint8).reshape(16,5120)
    mins=tiles[:,512:1024].copy().view(bfloat16).astype(np.float64).reshape(128,32)
    z=down.ffn.capture(out/'trace1.bin',trace_bytes,np.float32)
    if products is not None:
        # Product operands contain weight factors on a zero activation; only
        # products, accumulated values and the base trace must clear to zero.
        z=z.reshape(16,2560)
        zp=product_trace(z.ravel(),meta['product_block'])
        z=np.concatenate([z[:,:1120].ravel(),zp[:,2:,:].ravel()])
    repeat=(out/'trace0.bin').read_bytes()==(out/'trace2.bin').read_bytes()
    expected=np.load(out/'segment.npy')
    ref=np.load(out/'reference.npy')
    local=blocks[:,0].astype(np.float64)+blocks[:,1]
    sums=blocks[:,2].astype(np.float64)+blocks[:,3]
    block_error=local-ref
    reduction_error=sums-local.cumsum(axis=0)
    lane=meta['channel']%32
    active=np.flatnonzero((block_error[:,lane]!=0)|(reduction_error[:,lane]!=0))
    rows=[dict(block=int(i),k_start=meta['segment']*4096+int(i)*32,
               local_error=float(block_error[i,lane]),reduction_error=float(reduction_error[i,lane]),
               block_high=float(blocks[i,0,lane]),block_low=float(blocks[i,1,lane]),
               accumulated_high=float(blocks[i,2,lane]),accumulated_low=float(blocks[i,3,lane]),
               activation_sum_error=float(sum_error[i]),
               min_times_sum_error=float(mins[i,lane]*sum_error[i]),
               reference=float(ref[i,lane])) for i in active]
    checks=dict(repeat_exact=repeat,zero_clears_state=bool(np.all(z==0)),
                matches_segment_high=bool(np.array_equal(high[-1],expected[0])),
                matches_segment_low=bool(np.array_equal(low[-1],expected[1])))
    if products is not None:
        block=meta['product_block']
        checks['product_trace_matches_block']=bool(np.array_equal(products[-1,3:],blocks[block,:2]))
    result=dict(diagnostic_only=True,passed=all(checks.values()),checks=checks,
                segment=meta['segment'],channel=meta['channel'],
                local_block_errors=int(np.count_nonzero(block_error)),
                reduction_errors=int(np.count_nonzero(reduction_error)),
                activation_sum_errors=int(np.count_nonzero(sum_error)),
                selected_channel=rows,
                interface_error=float(high[-1,lane].astype(np.float64)+low[-1,lane]-sums[-1,lane]))
    if products is not None:
        result['products']=product_diagnosis(products,ref[meta['product_block'],lane],lane)
    (out/'blocks-results.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True)
    return 0 if result['passed'] else 1


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=('build-products','prepare','compare'))
    p.add_argument('--source',type=Path)
    p.add_argument('--kernel',type=Path)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--segment',type=int,default=3)
    p.add_argument('--channel',type=int,default=4872)
    p.add_argument('--product-block',type=int,default=53)
    a=p.parse_args()
    if a.stage=='build-products':
        build_products(a.out,a.product_block)
        return 0
    if a.stage=='prepare':
        if a.source is None or a.kernel is None: p.error('prepare requires --source and --kernel')
        prepare(a.source,a.kernel,a.out,a.segment,a.channel)
        return 0
    return compare(a.out)


if __name__=='__main__': raise SystemExit(main())
