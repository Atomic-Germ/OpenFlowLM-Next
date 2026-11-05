"""Prepare or score the hardware table probe: exponent, rounding, saturation, sums."""
import argparse
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'open_kernels/model'))
from q4nx import bf16_to_f32

ap=argparse.ArgumentParser(description=__doc__)
ap.add_argument('--out',type=Path,required=True)
ap.add_argument('--build',type=Path,required=True)
ap.add_argument('--score',action='store_true')
a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=True)
if not a.score:
    x=np.random.default_rng(38).normal(size=(32,32)).astype(np.float32)
    x*=np.exp2(np.linspace(-100,100,32)).astype(np.float32)[:,None]
    x[0]=0;x[1]=np.nextafter(np.float32(2),np.float32(0));x[2]=-x[1]
    x[3]=np.tile(np.array([1.001,-1.,1e-8,-1e-8],np.float32),8)
    x[4]=np.float32(1e-38)
    x.tofile(a.out/'x.bin')
    b=a.build.resolve();o=a.out.resolve()
    (o/'run.cfg').write_text(f'device\nxclbin P {b}/final.xclbin\nkernelx prep P {b}/insts.bin\nbuf x 4096 {o}/x.bin\nbuf tab 2304\nrun prep x tab\ndump tab {o}/table.bin 2304\n')
else:
    x=np.fromfile(a.out/'x.bin',np.float32).reshape(32,32).astype(np.float64)
    data=(a.out/'table.bin').read_bytes()
    if len(data)!=2304:raise ValueError('wrong table size')
    xi=np.frombuffer(data,np.int16,count=1024).reshape(32,32)
    sh=np.frombuffer(data,np.int32,count=32,offset=2048)
    hi=bf16_to_f32(np.frombuffer(data,np.uint16,count=32,offset=2176)).astype(np.float64)
    lo=bf16_to_f32(np.frombuffer(data,np.uint16,count=32,offset=2240)).astype(np.float64)
    for i,block in enumerate(x):
        mx=np.max(abs(block))
        shift=min(126,14-int(np.floor(np.log2(mx)))) if mx else 126
        q=np.clip(np.rint(np.ldexp(block,shift)),-32768,32767).astype(np.int16)
        np.testing.assert_equal(sh[i],shift)
        np.testing.assert_array_equal(xi[i],q,err_msg=f'block {i}')
        # AIE conversion flushes a subnormal BF16 low part: the table ABI
        # cannot promise relative accuracy below the minimum normal value.
        bound=max(np.sum(abs(block))*2**-15, np.finfo(np.float32).tiny)
        if not abs(hi[i]+lo[i]-block.sum()) <= bound:
            raise ValueError(f'block {i}: sum exceeds BF16 hi/lo bound')
    print('PASS: 32 blocks, exact shifts/int16, FP32 block sums, endpoint saturation')
