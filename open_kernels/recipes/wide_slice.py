"""Compose standalone wide layers; weights stream, state stays per layer."""
from .spec import LINEAR, FULL
from . import wide_deltanet_layer as D, wide_attention_layer as A

SHARED = frozenset(('x','y','pool','const','ones','zero'))
SHARED_KERNELS = frozenset(('ln','out','ffn'))


def layer_types(s, layers=8):
    if layers not in (8,64):
        raise ValueError('wide composition supports an eight-layer prefix or the full 64 layers')
    if layers==64 and (s.num_layers!=64 or len(s.layer_types)!=64):
        raise ValueError('full-model composition requires exactly 64 explicit layer types')
    kinds = tuple(s.layer_types[:layers])
    if len(kinds)!=layers or kinds.count(LINEAR)!=layers*3//4 or kinds.count(FULL)!=layers//4:
        raise ValueError('eight-layer/full-model composition requires three linear layers per full layer')
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


def layer_commands(s, l, index, pos, rows, layers=8):
    kinds = layer_types(s,layers)
    if not 0<=index<layers: raise ValueError('layer index outside the requested wide composition')
    if not 0<=pos<rows<=l.MAX_CTX: raise ValueError('cache position outside streamed rows')
    module,prefix = (D,'d') if kinds[index]==LINEAR else (A,'a')
    commands = [f'load pool layer{index}/pool.bin',f'load const layer{index}/const.bin']
    commands += [scope(c,prefix,index) for c in module.setup_commands(s,l)]
    if prefix=='a': commands.append(f'copy a_meta {l.E_A} position 0 {l.E_A}')
    body = D.token_commands(s,l) if prefix=='d' else A.token_commands(s,l,pos,rows)
    return commands + [scope(c,prefix,index) for c in body]
