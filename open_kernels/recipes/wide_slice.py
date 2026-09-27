"""Compose eight validated standalone layers; weights stream, state stays per layer."""
from .spec import LINEAR, FULL
from . import wide_deltanet_layer as D, wide_attention_layer as A

SHARED = frozenset(('x','y','pool','const','ones','zero'))
SHARED_KERNELS = frozenset(('ln','out','ffn'))


def layer_types(s):
    kinds = tuple(s.layer_types[:8])
    if len(kinds)!=8 or kinds.count(LINEAR)!=6 or kinds.count(FULL)!=2:
        raise ValueError('eight-layer probe requires six linear and two full layers in the explicit prefix')
    return kinds


def buffer_name(name, prefix, index):
    if name in SHARED: return name
    if name in ('state','cache'): return f'{name}{index}'
    return f'{prefix}_{name}'


def scope(command, prefix, index):
    words = command.split()
    if words[0]=='copy':
        words[1] = buffer_name(words[1],prefix,index)
        words[3] = buffer_name(words[3],prefix,index)
    elif words[0]=='run':
        if words[1] not in SHARED_KERNELS: words[1] = f'{prefix}_{words[1]}'
        words[2:] = [buffer_name(w,prefix,index) for w in words[2:]]
    else:
        raise ValueError('only copy/run commands can be scoped')
    return ' '.join(words)


def layer_commands(s, l, index, pos, rows):
    kinds = layer_types(s)
    if not 0<=index<8: raise ValueError('layer index outside the eight-layer slice')
    if not 0<=pos<rows<=l.MAX_CTX: raise ValueError('cache position outside streamed rows')
    module,prefix = (D,'d') if kinds[index]==LINEAR else (A,'a')
    commands = [f'load pool layer{index}/pool.bin',f'load const layer{index}/const.bin']
    commands += [scope(c,prefix,index) for c in module.setup_commands(s,l)]
    if prefix=='a': commands.append(f'copy a_meta {l.E_A} position 0 {l.E_A}')
    body = D.token_commands(s,l) if prefix=='d' else A.token_commands(s,l,pos,rows)
    return commands + [scope(c,prefix,index) for c in body]
