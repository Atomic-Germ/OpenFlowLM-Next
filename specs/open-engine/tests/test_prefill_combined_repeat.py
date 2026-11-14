"""Resident repeats must capture both requests separately and reject missing evidence."""
import importlib.util
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[3]


def tool():
    spec = importlib.util.spec_from_file_location('combined_repeat',ROOT/'utilities/prefill-combined-cli/repeat.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def test_repeat_patch_preserves_request_reset_and_original_first_capture():
    m=tool()
    cpp=(ROOT/'src/open_qwen36/core.cpp').read_text()
    cli=(ROOT/'src/open_qwen36/cli.cpp').read_text()
    _,single=m.builder.parent.trace.patch(cpp,cli)
    repeated=m.patch(single)
    body=cli[cli.index('std::vector<int> request('):cli.index('\n}  // namespace',cli.index('std::vector<int> request('))]
    assert body in repeated and 'core.reset();' in body
    assert 'a.dump_prefix + "_repeat"' in repeated
    assert 'request(core, again_args)' in repeated
    assert 'writer.finish(' in repeated
    assert '!a.twice || a.repeat != 1' in repeated
    with pytest.raises(ValueError):m.patch(repeated)


def test_log_sections_require_two_complete_ordered_requests():
    m=tool()
    log='header\nTRACE_REQUEST_BEGIN 0\ntoken 1\nTRACE_REQUEST_END 0\nTRACE_REQUEST_BEGIN 1\ntoken 1\nTRACE_REQUEST_END 1\nrequest 2 REPRODUCED the first\nDONE\n'
    assert m.sections(log)==['token 1\n','token 1\n']
    for bad in (log.replace('TRACE_REQUEST_END 1\n',''),log.replace('TRACE_REQUEST_BEGIN 1','TRACE_REQUEST_BEGIN 0'),
                log.replace('REPRODUCED','DIFFERS FROM'),log+log):
        with pytest.raises(ValueError):m.sections(bad)
