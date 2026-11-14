"""Offline metadata normalization for an existing dense Qwen3.5 Q4NX directory."""
import json
import math
import re
import struct
from pathlib import Path

from .model_assets import inject_oflm_keys


def normalize_config(model_dir: Path) -> bool:
    """Validate container structure, back up config, then use converter normalization.

    This does not validate quantized tensor values or re-convert weights.
    The supported family is explicit because inject_oflm_keys has family-specific
    defaults; applying them to an arbitrary model would be unsafe.
    """
    model_dir = Path(model_dir)
    path = model_dir / 'config.json'
    original = path.read_bytes()
    config = json.loads(original)
    if not isinstance(config, dict):
        raise ValueError('config.json must contain an object')
    text = config.get('text_config', config)
    if not isinstance(text, dict):
        raise ValueError('text_config must contain an object')
    if text.get('model_type') not in ('qwen3_5', 'qwen3_5_text'):
        raise ValueError('config normalization supports dense qwen3_5 only')
    if text is not config:
        for key, value in text.items():
            if key != 'model_type' and config.get(key) is not None and config[key] != value:
                raise ValueError(f'conflicting top-level and text_config field: {key}')
    for key in ('hidden_size', 'num_hidden_layers', 'head_dim', 'num_attention_heads', 'num_key_value_heads'):
        if type(text.get(key)) is not int or text[key] <= 0:
            raise ValueError(f'config.json: invalid {key}')

    weights = model_dir / 'model.q4nx'
    with weights.open('rb') as stream:
        raw = stream.read(8)
        if len(raw) != 8:
            raise ValueError('model.q4nx: missing header length')
        length = struct.unpack('<Q', raw)[0]
        size = weights.stat().st_size
        if not 0 < length <= min(64 * 1024 * 1024, size - 8):
            raise ValueError('model.q4nx: invalid header length')
        header = json.loads(stream.read(length))
    if not isinstance(header, dict):
        raise ValueError('model.q4nx: header must be an object')
    end = size - 8 - length
    ranges, layers = [], set()
    sizes = dict(BF16=2, F16=2, F32=4, F64=8, I8=1, U8=1, I16=2, I32=4, I64=8)
    for name, entry in header.items():
        if name == '__metadata__':
            continue
        if not isinstance(entry, dict):
            raise ValueError(f'model.q4nx: invalid tensor {name}')
        shape, offsets, dtype = entry.get('shape'), entry.get('data_offsets'), entry.get('dtype')
        if (not isinstance(shape, list) or any(type(d) is not int or d <= 0 for d in shape)
                or not isinstance(offsets, list) or len(offsets) != 2
                or any(type(d) is not int for d in offsets)
                or not isinstance(dtype, str) or dtype not in sizes):
            raise ValueError(f'model.q4nx: invalid descriptor {name}')
        start, stop = offsets
        if not 0 <= start < stop <= end or stop - start != math.prod(shape) * sizes[dtype]:
            raise ValueError(f'model.q4nx: invalid byte range {name}')
        ranges.append((start, stop, name))
        match = re.match(r'model\.layers\.(\d+)\.', name)
        if match:
            layers.add(int(match[1]))
    ordered = sorted(ranges)
    if any(a[1] > b[0] for a, b in zip(ordered, ordered[1:])):
        raise ValueError('model.q4nx: overlapping tensor ranges')
    if len(layers) != text['num_hidden_layers'] or layers != set(range(text['num_hidden_layers'])):
        raise ValueError('model.q4nx: layer indices disagree with num_hidden_layers')
    for layer in layers:
        norm = header.get(f'model.layers.{layer}.input_layernorm.weight', {})
        if norm.get('shape') != [text['hidden_size']]:
            raise ValueError(f'model.q4nx: layer {layer} input norm disagrees with hidden_size')

    inject_oflm_keys(config, {}, model_dir, None)
    # Match assemble_model_assets' canonical runtime type for dense Qwen3.5.
    config['model_type'] = 'qwen3_5'
    if config == json.loads(original):
        return False
    backup = model_dir / 'config.json.before-normalize'
    # Never overwrite an earlier backup, even if the caller changed config again.
    with backup.open('xb') as stream:
        stream.write(original)
    temporary = model_dir / 'config.json.normalize.tmp'
    temporary.write_text(json.dumps(config, indent=2) + '\n', encoding='utf-8')
    temporary.replace(path)
    return True
