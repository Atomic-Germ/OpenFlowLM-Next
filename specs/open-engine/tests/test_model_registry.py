"""The two registry files must agree, or `oflm pull` and `oflm check` disagree with it.

`model_list.json` says which files a model needs and what its tag is; `model_info.json` says
each file's size and oid. The downloader verifies against the SECOND and the UI reads the
FIRST, so a tag present in one and absent from the other is a model that lists but cannot be
downloaded -- or downloads through a path nothing knows about.

These checks are offline on purpose. Whether the recorded size still matches the bytes
HuggingFace serves is a question only the network can answer, and CI has none; that drift is
reported by `utilities/refresh-model-info.py --check`, run by hand. What CI CAN see is the
structural agreement, and that is what this file pins.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
LIST = ROOT / 'src' / 'model_list.json'
INFO = ROOT / 'src' / 'model_info.json'


def listing() -> dict:
    return json.loads(LIST.read_text())


def info() -> dict:
    return json.loads(INFO.read_text())


def tags() -> list[str]:
    return [f'{fam}:{size}' for fam, sizes in listing()['models'].items() for size in sizes]


def test_every_listed_model_has_a_file_record_set():
    """A tag in `model_list.json` with no `model_info.json` entry cannot be verified."""
    rec = info()
    missing = [t for t in tags() if t not in rec]
    assert not missing, f'listed but has no file records: {missing}'


def test_every_file_oflm_downloads_has_a_record():
    """The direction that breaks a pull: a required file with no size/oid to verify against.

    `registry_for` returns such a file as *unlisted*, and the pull path refuses the whole
    model with "model_info.json describes none of these files required by model_list.json".
    A recorded file that is not downloaded is NOT a defect -- `model_info.json` mirrors the
    whole repo, xclbins and documentation included, and only the tag's `files` list decides
    what `oflm` fetches.
    """
    rec, lst = info(), listing()
    problems = []
    for fam, sizes in lst['models'].items():
        for size, model in sizes.items():
            tag = f'{fam}:{size}'
            if tag not in rec:
                continue                      # reported by the test above
            have = {e['path'] for e in rec[tag]}
            absent = set(model['files']) - have
            if absent:
                problems.append(f'{tag}: downloaded but not recorded: {sorted(absent)}')
    assert not problems, problems


def test_a_size_is_recorded_for_every_download():
    """`status_of` compares `file_size` against this number; absent, the check is skipped."""
    rec = info()
    for tag, entries in rec.items():
        for e in entries:
            assert e.get('path'), (tag, e)
            assert isinstance(e.get('size'), int) and e['size'] >= 0, (tag, e['path'])
            lfs = e.get('lfs')
            if isinstance(lfs, dict):
                # The LFS object is what the downloader verifies; it must carry both.
                assert lfs.get('oid'), (tag, e['path'])
                assert isinstance(lfs.get('size'), int), (tag, e['path'])


def test_the_qwen35_family_still_points_at_the_same_repo_names():
    """The tags I refreshed must not have been re-pointed at a different model.

    A size correction is a correction; a changed `name` is a different model wearing the
    same tag, and that would make the refreshed records describe the wrong bytes.
    """
    lst = listing()['models']
    expected = {
        'qwen3.5': ('0.8b', 'Qwen3.5-0.8B-NPU2'),
        'qwen3.5': ('2b', 'Qwen3.5-2B-NPU2'),
        'qwen3.5': ('4b', 'Qwen3.5-4B-NPU2'),
        'qwen3.5': ('9b', 'Qwen3.5-9B-NPU2'),
        'qwen3.8': ('27b', 'Qwen3.8-27B-NPU2'),
    }
    for fam, (size, name) in expected.items():
        assert lst[fam][size]['name'] == name, f'{fam}:{size}'
        assert lst[fam][size]['details']['family'] in ('qwen3.5',), f'{fam}:{size}'


@pytest.mark.parametrize('tag', ['qwen3.5:0.8b', 'qwen3.5:4b'])
def test_the_refreshed_entries_record_the_current_upstream_sizes(tag):
    """The two models whose upstream files were re-uploaded.

    Pinning the sizes here is what makes the refresh reviewable: if a future pass changes one
    of these numbers, this fails and the change has to be looked at, which is the whole point
    of a registry that other layers trust.
    """
    rec = {e['path']: e for e in info()[tag]}
    if tag == 'qwen3.5:0.8b':
        assert rec['chat_template.jinja']['size'] == 27172
    else:
        assert rec['config.json']['size'] == 3091
        assert rec['tokenizer_config.json']['size'] == 15226
        # An LFS pointer's own size is not the file's; the payload size lives inside `lfs`.
        m = rec['model.q4nx']
        assert m['size'] == 4299177104
        assert m['lfs']['size'] == 4299177104
