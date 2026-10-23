"""Repair existing dense Qwen3.5 metadata without rewriting model weights."""
import json
import struct
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def fixture(out, layers=2):
    cfg = dict(model_type='qwen3_5', text_config=dict(model_type='qwen3_5_text',
               hidden_size=4, num_hidden_layers=layers, head_dim=2,
               num_attention_heads=2, num_key_value_heads=1))
    (out / 'config.json').write_text(json.dumps(cfg))
    tensors = {f'model.layers.{l}.input_layernorm.weight': dict(dtype='BF16', shape=[4],
                data_offsets=[l*8, l*8+8]) for l in range(2)}
    h = json.dumps(tensors).encode()
    (out / 'model.q4nx').write_bytes(struct.pack('<Q', len(h)) + h + b'\0'*16)


def test_normalizes_via_converter_and_preserves_weights(tmp_path):
    from q4nx.config_normalize import normalize_config
    fixture(tmp_path)
    original = (tmp_path / 'config.json').read_bytes()
    weights = (tmp_path / 'model.q4nx').read_bytes()
    assert normalize_config(tmp_path)
    cfg = json.loads((tmp_path / 'config.json').read_text())
    assert cfg['head_dim'] == 2 and cfg['hidden_size'] == 4
    assert cfg['model_type'] == 'qwen3_5'
    assert 'text_config' not in cfg
    assert (tmp_path / 'config.json.before-normalize').read_bytes() == original
    assert (tmp_path / 'model.q4nx').read_bytes() == weights
    assert not normalize_config(tmp_path)
    assert (tmp_path / 'config.json.before-normalize').read_bytes() == original


@pytest.mark.parametrize('bad', ['layers', 'truncated', 'missing', 'conflict', 'unsupported'])
def test_invalid_container_or_geometry_leaves_config_untouched(tmp_path, bad):
    from q4nx.config_normalize import normalize_config
    fixture(tmp_path, layers=3 if bad == 'layers' else 2)
    if bad == 'truncated':
        p = tmp_path / 'model.q4nx'
        p.write_bytes(p.read_bytes()[:-1])
    elif bad == 'missing':
        (tmp_path / 'model.q4nx').unlink()
    elif bad in ('conflict', 'unsupported'):
        p = tmp_path / 'config.json'
        cfg = json.loads(p.read_text())
        if bad == 'conflict':
            cfg['hidden_size'] = 8
        else:
            cfg['text_config']['model_type'] = 'other'
        p.write_text(json.dumps(cfg))
    before = (tmp_path / 'config.json').read_bytes()
    with pytest.raises((ValueError, OSError)):
        normalize_config(tmp_path)
    assert (tmp_path / 'config.json').read_bytes() == before
    assert not (tmp_path / 'config.json.before-normalize').exists()


def test_cli_normalizes_without_starting_conversion(tmp_path, monkeypatch):
    from q4nx import cli
    fixture(tmp_path)
    monkeypatch.setattr(cli, 'create_converter', lambda *a: pytest.fail('conversion started'))
    assert cli.main(['--normalize-config', '-i', str(tmp_path)]) == 0
    assert 'text_config' not in json.loads((tmp_path / 'config.json').read_text())


@pytest.mark.parametrize('config', [[], {'text_config': None}])
def test_malformed_config_fails_cleanly(tmp_path, config):
    from q4nx.config_normalize import normalize_config
    fixture(tmp_path)
    (tmp_path / 'config.json').write_text(json.dumps(config))
    with pytest.raises(ValueError, match='config'):
        normalize_config(tmp_path)


def test_cli_rejects_conversion_options(tmp_path):
    from q4nx.cli import main
    fixture(tmp_path)
    before = (tmp_path / 'config.json').read_bytes()
    with pytest.raises(SystemExit, match='cannot be combined'):
        main(['--normalize-config', '-i', str(tmp_path), '-o', str(tmp_path / 'other')])
    assert (tmp_path / 'config.json').read_bytes() == before


@pytest.mark.parametrize('bad', ['overlap', 'dtype', 'width'])
def test_rejects_invalid_tensor_descriptors(tmp_path, bad):
    from q4nx.config_normalize import normalize_config
    fixture(tmp_path)
    weights = tmp_path / 'model.q4nx'
    data = weights.read_bytes()
    length = struct.unpack('<Q', data[:8])[0]
    header = json.loads(data[8:8+length])
    entry = header['model.layers.1.input_layernorm.weight']
    if bad == 'overlap':
        entry['data_offsets'] = [0, 8]
    elif bad == 'dtype':
        entry['dtype'] = []
    else:
        entry['shape'] = [2]
        entry['data_offsets'] = [8, 12]
    new = json.dumps(header).encode()
    weights.write_bytes(struct.pack('<Q', len(new)) + new + data[8+length:])
    before = (tmp_path / 'config.json').read_bytes()
    with pytest.raises(ValueError, match='model.q4nx'):
        normalize_config(tmp_path)
    assert (tmp_path / 'config.json').read_bytes() == before


def test_existing_backup_is_not_overwritten(tmp_path):
    from q4nx.config_normalize import normalize_config
    fixture(tmp_path)
    backup = tmp_path / 'config.json.before-normalize'
    backup.write_bytes(b'earlier backup')
    before = (tmp_path / 'config.json').read_bytes()
    with pytest.raises(FileExistsError):
        normalize_config(tmp_path)
    assert backup.read_bytes() == b'earlier backup'
    assert (tmp_path / 'config.json').read_bytes() == before
