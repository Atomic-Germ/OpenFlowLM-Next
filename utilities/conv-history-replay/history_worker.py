"""Two 512-channel calls retain the existing 2-KiB activation FIFOs."""
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent


def history_rows(channels,tile,half,state_bytes):
    from aie.helpers.taplib import TensorAccessPattern
    return TensorAccessPattern((1,state_bytes),(tile*1024+half*512)*4,
                               [1,1,3,2048],[0,0,channels*4,1])


def history_sequence(XL,*,pipe_factory,a_state,a_act,gact_p,gout_c,state_bytes,act_bytes):
    pipe=pipe_factory(3)
    for tile in range(XL.NT):
        for half in range(2):
            rows=history_rows(XL.NCH,tile,half,state_bytes)
            pipe.drain(gout_c,a_state,rows)
            pipe.fill(gact_p,a_act,XL.bt(act_bytes,XL.A_QKV+(tile*1024+half*512)*4,2048))
            pipe.fill(gact_p,a_state,rows)
        if tile>=XL.KEY_TILES:
            pipe.drain(gout_c,a_act,XL.bt(act_bytes,XL.A_VEC+(tile-XL.KEY_TILES)*XL.D.HEADS_PER_TILE*XL.D.RECORD_BYTES,
                                        XL.D.HEADS_PER_TILE*XL.D.RECORD_BYTES))
    pipe.finish()


def conv_sequence(XL,sin,ain,oout,qk,vt,decay,beta,fconv,femit):
    for base,ntiles in ((0,XL.KEY_TILES),(XL.KEY_TILES,XL.VALUE_TILES)):
        for tile in XL.range_(ntiles):
            w=sin.acquire(XL.CONVW_ELEMS)
            for half in XL.X.band_range(2):
                e=ain.acquire(4);o=oout.acquire(3)
                fconv(e[0],e[1],e[2],e[3],w[0],w[1],o[0],o[1],o[2],qk,vt,tile,base,half)
                oout.release(3);ain.release(4)
            sin.release(XL.CONVW_ELEMS)
            if base==XL.KEY_TILES:
                for i in XL.range_(XL.D.HEADS_PER_TILE):
                    r=oout.acquire(1);femit(qk,vt,decay,beta,r,tile,i);oout.release(1)


def install_history(XL):
    from aie.iron.kernel import ExternalFunction
    from linear_worker import ab_sequence
    original=XL.glue_post_kernels
    def kernels(inc,g):
        k=original(inc,g)
        k['conv']=ExternalFunction('history_conv',source_file=str(HERE/'history_conv.cc'),
            arg_types=[g['u8_2k']]*4+[g['u8_4k']]*2+[g['u8_2k']]*3+
                      [g['fqk'],g['fvt'],np.int32,np.int32,np.int32],include_dirs=inc,**XL.GLUE_FLAGS)
        return k
    XL.glue_post_kernels=kernels
    def body(sin,ain,oout,acc_a,acc_b,decay,beta,qk,vt,xn,fab,fsmall,fconv,femit,fcopy):
        ab_sequence(sin,xn,acc_a,acc_b,fab,fcopy,XL.X.band_range)
        sm=sin.acquire(1);fsmall(sm,acc_a,acc_b,decay,beta);sin.release(1)
        conv_sequence(XL,sin,ain,oout,qk,vt,decay,beta,fconv,femit)
    XL.glue_body=body
