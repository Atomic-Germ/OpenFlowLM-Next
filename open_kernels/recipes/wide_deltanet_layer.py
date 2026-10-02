"""Byte adapters for isolated wide-layer bring-up, preserving production packing.

No catalogue promotion or alternate model container format is defined here.
Offsets returned below are bytes: (destination, source, length).
"""


def projection_bands(n, cores):
    if cores<=0 or n<=0 or n%(cores*64):
        raise ValueError('projection output must tile 64 rows on every core')
    return n//(cores*64)


def post_groups(heads):
    if heads<=0 or heads%8:
        raise ValueError('post norm requires whole groups of eight 128-dimensional heads')
    return heads//8


def state_copies(layout, heads, dim, restore=False):
    valid=dim*dim*4
    if heads<=0 or dim<=0 or layout.S_HEAD_BYTES<valid or layout.STATE_S_OFF+heads*layout.S_HEAD_BYTES>layout.STATE_BYTES:
        raise ValueError('invalid padded recurrent state layout')
    copies=[(h*valid,layout.STATE_S_OFF+h*layout.S_HEAD_BYTES,valid) for h in range(heads)]
    return [(src,dst,size) for dst,src,size in copies] if restore else copies


def ab_copies(layout, hidden, heads):
    if hidden<=0 or heads<=0:
        raise ValueError('invalid AB geometry')
    bank=hidden*32*2
    copies=[]
    for b in range((heads+31)//32):
        dst=b*(2*bank+4096)
        copies.extend([(dst,layout.C_SIDE+layout.SIDE_ALPHA+b*bank,bank),
                       (dst+bank,layout.C_SIDE+layout.SIDE_BETA+b*bank,bank),
                       (dst+2*bank,layout.C_SIDE+layout.SIDE_SMALL,4096)])
    return copies


def setup_commands(s,l):
    """Extract existing packed regions into the standalone kernels' BOs."""
    hid,vw,nch=s.hidden,s.lin_value_width,s.lin_qkv_dim
    qbytes=nch*hid//8192*5120
    cmds=[f'copy qzw 0 pool {l.POOL_QKV} {qbytes}',
          f'copy qzw {qbytes} pool {l.POOL_Z} {vw*hid//8192*5120}',
          f'copy ow 0 const {l.C_WOUT} {hid*vw//8192*5120}',
          f'copy lnw 0 const {l.C_LNW} {hid*2}',
          f'copy postw 0 const {l.C_POSTLN} {hid*2}',
          f'copy nw 0 const {l.C_NW} 4096',
          f'copy side 4096 const {l.C_SIDE+l.SIDE_CONV} {s.conv_kernel*nch*2}']
    cmds += [f'copy abs {dst} const {src} {size}' for dst,src,size in ab_copies(l,hid,s.lin_value_heads)]
    return cmds


def token_commands(s,l):
    """One complete all-Q4 dense DeltaNet layer; neural math stays on the NPU."""
    if ((s.hidden,s.intermediate,s.lin_key_heads,s.lin_value_heads,s.lin_key_dim,
         s.lin_value_dim,s.conv_kernel,s.norm_eps,s.activation)
            != (5120,17408,16,48,128,128,4,1e-6,'silu') or s.q8_roles or s.num_experts):
        raise ValueError('not implemented: wide layer composition outside H5120/FF17408, 16/48-head all-Q4, conv4/eps1e-6/SiLU')
    hid,vw,nch=s.hidden,s.lin_value_width,s.lin_qkv_dim
    cmds=[f'copy cs 0 state 0 {l.STATE_S_OFF}']
    cmds += [f'copy si {dst} state {src} {size}' for dst,src,size in state_copies(l,48,128)]
    cmds += ['run ln x zero lnw res0 xn', 'run qz qzw xn qzout', 'run ab abs xn ab',
             f'copy qkv 0 qzout 0 {nch*4}', f'copy z 0 qzout {nch*4} {vw*4}',
             'copy side 0 ab 0 768', 'run glue side qkv cs conv vec',
             'run step si vec so o', 'run post o z nw og', 'run out ow og projout',
             'run ln projout x postw res1 xm', f'copy act {l.A_XM} xm 0 {hid*2}',
             'run ffn pool act trace', f'copy fo 0 act {l.A_OUT2} {hid*4}',
             # Reuse the residual add of LN; its normalization result is unused.
             'run ln fo res1 ones y discard', f'copy state 0 conv 0 {l.STATE_S_OFF}']
    cmds += [f'copy state {dst} so {src} {size}' for dst,src,size in state_copies(l,48,128,restore=True)]
    return cmds


def projection_table_bytes(k, base, correction=False):
    """Corrected standalone QKV table, within the measured main-core budget."""
    if k<=0 or k%32:
        raise ValueError('projection table requires whole 32-element blocks')
    result=max(base,4*k+k//4+k//16+512) if correction else base
    # Actual H5120/FF17408 probe uses40960 bytes beside its table.
    if 40960+result>65536:
        raise ValueError('corrected projection table exceeds main-core L1 budget')
    return result
