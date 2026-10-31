"""Private glue ABI: FP32[1024] xn occupies the existing 4-KiB scratch slot."""
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent


def private_kernels(X,inc,t,directory):
    original=X.HERE
    try:
        X.HERE=directory
        return X.kernels([*inc,str(original)],t)
    finally:
        X.HERE=original


def ab_sequence(sin,xn,acc_a,acc_b,fab,fcopy,loop):
    for block in loop(5):
        e=sin.acquire(1);fcopy(e,xn,0);sin.release(1)
        for acc in (acc_a,acc_b):
            for tile in loop(32):
                w=sin.acquire(1);fab(w,xn,acc,tile,block);sin.release(1)


def install_glue(XL):
    from aie.iron.kernel import ExternalFunction
    if not XL.DENSE or not XL.AB_WIDE or not XL.HALF_OUTER or XL.HID!=5120:
        raise ValueError('unsupported linear-entry glue geometry')
    original_types=XL.glue_post_types;original_kernels=XL.glue_post_kernels
    XL.AB_TILES=[32]*5
    def types():
        g=original_types();g['fxn']=np.ndarray[(1024,),np.dtype[np.float32]];return g
    def kernels(inc,g):
        k=original_kernels(inc,g)
        k['ab']=ExternalFunction('linear_ab_fp32',source_file=str(HERE/'ab_fp32.cc'),
            arg_types=[g['u8_4k'],g['fxn'],g['facc'],np.int32,np.int32],include_dirs=inc,**XL.GLUE_FLAGS)
        k['copy']=ExternalFunction('linear_copy_fp32',source_file=str(HERE/'copy_fp32.cc'),
            arg_types=[g['u8_4k'],g['fxn'],np.int32],include_dirs=inc,**XL.GLUE_FLAGS)
        return k
    XL.glue_post_types=types;XL.glue_post_kernels=kernels
    def body(sin,ain,oout,acc_a,acc_b,decay,beta,qk,vt,xn,fab,fsmall,fconv,femit,fcopy):
        ab_sequence(sin,xn,acc_a,acc_b,fab,fcopy,XL.X.band_range)
        sm=sin.acquire(1);fsmall(sm,acc_a,acc_b,decay,beta);sin.release(1)
        # The production conv/record schedule and arithmetic are unchanged.
        for base,ntiles in ((0,XL.KEY_TILES),(XL.KEY_TILES,XL.VALUE_TILES)):
            for tt in XL.range_(ntiles):
                ww=sin.acquire(XL.CONVW_ELEMS);e=ain.acquire(2+XL.CONV_ROWS);o=oout.acquire(XL.CONV_ROWS)
                fconv(e[0],e[1],e[2],e[3],e[4],ww[0],ww[1],o[0],o[1],o[2],qk,vt,tt,base)
                oout.release(XL.CONV_ROWS);ain.release(2+XL.CONV_ROWS);sin.release(XL.CONVW_ELEMS)
                if base==XL.KEY_TILES:
                    for i in XL.range_(XL.D.HEADS_PER_TILE):
                        r=oout.acquire(1);femit(qk,vt,decay,beta,r,tt,i);oout.release(1)
    XL.glue_body=body
