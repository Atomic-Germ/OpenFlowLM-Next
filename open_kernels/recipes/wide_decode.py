"""Final norm, full-vocabulary head and host token feedback for the B7 probe."""


def head_commands(hidden, vocab, tag):
    if hidden<=0 or vocab<=0:
        raise ValueError('decode requires positive hidden and vocabulary dimensions')
    return ['load finalres poison-finalres.bin','load finalxn poison-finalxn.bin',
            'load logits poison-logits.bin',
            'run ln y zero finalw finalres finalxn',
            'run lm lmw finalxn logits',
            f'dump finalres {tag}-finalres.bin {hidden*4+64}',
            f'dump finalxn {tag}-finalxn.bin {hidden*2+64}',
            f'dump logits {tag}-logits.bin {vocab*4+64}',
            f'greedy logits {vocab} token',f'dump token {tag}-next.bin 4']


def embedding_commands(hidden, vocab, tag):
    if hidden<=0 or vocab<=0:
        raise ValueError('decode requires positive hidden and vocabulary dimensions')
    return [f'dump token {tag}-token.bin 4',f'embed x embedding.bin token {vocab} {hidden}']
