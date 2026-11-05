"""Private FP32 conv-history experiment over the validated FP32 entry."""
import hashlib
from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'linear-entry-replay'))
import linear_common as entry
from linear_common import ROOT,digest,generated_sources,verify_generated,read_norm


def geometry(spec,mode):
    if mode not in ('bf16','fp32'):raise ValueError('invalid history mode')
    g=entry.geometry(spec,'fp32')
    offset=(spec.conv_kernel-1)*spec.lin_qkv_dim*2
    g.update(mode=mode,conv_dtype=mode,state_offset=offset*(2 if mode=='fp32' else 1))
    if mode=='fp32':g['state_bytes']+=offset
    return g


def source_hash(spec,mode):
    geometry(spec,mode);h=hashlib.sha256(entry.source_hash(spec,'fp32').encode()+mode.encode())
    for name in ('history_common.py','history_worker.py','history_conv.cc','build.py'):
        h.update(name.encode()+b'\0'+(HERE/name).read_bytes())
    return h.hexdigest()


def render_design(spec,source=None):
    s=entry.render_design(spec,source)
    begin='        pipe = Pipeline(3)\n        for tt in range(NT):'
    end='        pipe.finish()                                    # the records are in DDR'
    if s.count(begin)!=1 or s.count(end)!=1:raise ValueError('history sequence anchor changed')
    i=s.index(begin);j=s.index(end,i)+len(end)
    s=s[:i]+('        history_sequence(XL, pipe_factory=Pipeline, a_state=a_state, a_act=a_act,\n'
             '                         gact_p=gact_p, gout_c=gout_c, state_bytes=STATE_BYTES, act_bytes=A_BYTES)')+s[j:]
    anchor='install_glue(XL)'
    if s.count(anchor)!=1:raise ValueError('history glue anchor changed')
    s=s.replace(anchor,anchor+f'\nsys.path.insert(0, {str(HERE)!r})\n'
                'from history_worker import install_history, history_sequence\ninstall_history(XL)\n'
                'STATE_BYTES += STATE_S_OFF\nSTATE_S_OFF *= 2')
    return s+f'\nHISTORY_HASH = {source_hash(spec,"fp32")!r}\nSPECIALIZE["srchash"] = int(HISTORY_HASH[:8],16)\n'
