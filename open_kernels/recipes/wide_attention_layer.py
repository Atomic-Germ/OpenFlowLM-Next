"""Byte-only complete attention-layer composition for the B7 synthetic gate.

Use production pool/constant/activation offsets. Standalone contexts are reused
sequentially; this does not describe fused placement or export model support.
"""


def check_geometry(s):
    if ((s.hidden, s.intermediate, s.num_heads, s.num_kv_heads, s.head_dim,
         s.rotary_dim, s.attn_gate, s.norm_eps, s.activation)
            != (5120, 17408, 24, 4, 256, 64, True, 1e-6, 'silu')
            or s.q8_roles or s.num_experts or not s.has_full):
        raise ValueError('not implemented: attention layer outside all-Q4 H5120/FF17408 Q24/KV4/HD256/ROT64 gated eps1e-6/SiLU')


def projection_copies(s):
    """(BO, destination, source, bytes): q|k|v|gate -> q|gate and k|v."""
    check_geometry(s)
    q, k = s.attn_q_width*4, s.attn_kv_width*4
    return [('qg', 0, 0, q), ('qg', q, q+2*k, q), ('kvn', 0, q, 2*k)]


def setup_commands(s, l):
    check_geometry(s)
    h, q, k = s.hidden, s.attn_q_width, s.attn_kv_width
    commands, dst = [], 0
    for src, n in ((l.POOL_Q,q), (l.POOL_K,k), (l.POOL_V,k), (l.POOL_GATE,q)):
        size = n*h//8192*5120
        commands.append(f'copy qkvgw {dst} pool {src} {size}')
        dst += size
    return commands + [f'copy ow 0 pool {l.POOL_O} {h*q//8192*5120}',
                       f'copy lnw 0 const {l.CA_LNW} {h*2}',
                       f'copy postw 0 const {l.CA_POSTLN} {h*2}',
                       f'copy meta 0 const {l.CA_META} {l.E_A}']


def token_commands(s, l, pos, rows):
    check_geometry(s)
    if rows < 1 or rows > l.MAX_CTX or not 0 <= pos < rows:
        raise ValueError('cache write position must fit the streamed cache')
    h, q, k = s.hidden, s.attn_q_width, s.attn_kv_width
    commands = ['run ln x zero lnw res0 xn', 'run qkvg qkvgw xn projected']
    commands += [f'copy {bo} {dst} projected {src} {size}' for bo,dst,src,size in projection_copies(s)]
    commands += ['run attn meta qg kvn cache new og',
                 f'copy cache {pos*l.KV_ROW} new 0 {l.KV_ROW}',
                 'run out ow og projout', 'run ln projout x postw res1 xm']
    # Capture the real attention activation layout; the segmented FFN probe uses
    # the linear layout, whose XM/OUT2 offsets differ. Bridge explicitly.
    for name, offset, size in (('xn',l.AA_XN,h*2), ('qg',l.AA_QG,q*8),
            ('kvn',l.AA_KVN,k*8), ('og',l.AA_OG,q*2), ('projout',l.AA_OUT,h*4),
            ('res1',l.AA_RES,h*4), ('xm',l.AA_XM,h*2)):
        commands.append(f'copy act {offset} {name} 0 {size}')
    commands += [f'copy ffnact {l.A_XM} act {l.AA_XM} {h*2}',
                 'run ffn pool ffnact trace',
                 f'copy act {l.AA_OUT2} ffnact {l.A_OUT2} {h*4}',
                 f'copy fo 0 act {l.AA_OUT2} {h*4}',
                 'run ln fo res1 ones y discard']
    return commands
