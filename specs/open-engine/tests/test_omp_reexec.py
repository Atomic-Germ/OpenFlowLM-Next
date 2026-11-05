"""The OpenMP wait policy has to reach libgomp, and only a re-exec does that on Linux.

`src/open_qwen36/core.cpp` has carried a `setenv("OMP_WAIT_POLICY", "PASSIVE")` in a static
initialiser for a long time. On Linux it does nothing: libgomp reads the variable in its ELF
constructor, which runs when the library is mapped, before any constructor of the program.
The variable then reads back correctly, so the log line says PASSIVE and the workers still
spin -- measured as 34.8 s against 29.4 s for one 256-token prefill block, a 15% difference
between two runs of the same binary.

These tests pin the three things that make the fix work, without needing an NPU:

  * the guards are exactly `OFLM_OPEN_OMP_REEXEC` and `OFLM_OPEN_OMP_PASSIVE=0`, and in
    particular NOT "is OMP_WAIT_POLICY already set" -- the static initialiser sets it in this
    very process, which is the state the re-exec exists to escape;
  * the function is declared in the public header and defined outside the anonymous
    namespace, or the programs that call it cannot link;
  * the variable is classified, so the settings CI stays green.
"""
from __future__ import annotations

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
CORE_CPP = ROOT / 'src/open_qwen36/core.cpp'
CORE_HPP = ROOT / 'src/open_qwen36/core.hpp'
MAIN_CPP = ROOT / 'src/src/main.cpp'
CLI_CPP = ROOT / 'src/open_qwen36/cli.cpp'
REGISTRY = ROOT / 'utilities/generate-registry.py'


@pytest.fixture(scope='module')
def core() -> str:
    return CORE_CPP.read_text()


@pytest.fixture(scope='module')
def core_hpp() -> str:
    return CORE_HPP.read_text()


def _definition_body(core: str) -> str:
    """The body of reexec_with_passive, from its `#else` (the POSIX arm) to the next `#endif`."""
    at = core.index('bool reexec_with_passive(char** argv)')
    # the first definition is the _WIN32 arm; find the POSIX one after it
    posix = core.index('bool reexec_with_passive(char** argv)', at + 1)
    end = core.index('#endif', posix)
    return core[posix:end]


def _code_lines(body: str) -> list[str]:
    """`body` with comments stripped, so a comment that NAMES the wrong guard cannot satisfy
    (or trip) an assertion about the code."""
    return [ln.split('//')[0] for ln in body.splitlines()]


def test_guards_do_not_include_the_variable_itself(core: str):
    """The whole point: a process whose constructor already exported the variable must still
    re-exec, because libgomp has not read it. A getenv check here would return early and the
    fix would silently do nothing -- which is how the first version of it behaved."""
    code = '\n'.join(_code_lines(_definition_body(core)))
    assert 'OFLM_OPEN_OMP_REEXEC' in code
    assert 'getenv("OMP_WAIT_POLICY")' not in code


def test_opt_out_and_loop_guard_are_honoured(core: str):
    body = _definition_body(core)
    # OFLM_OPEN_OMP_PASSIVE=0 is the documented opt-out and must still turn the policy off
    assert 'OFLM_OPEN_OMP_PASSIVE' in body
    # the re-exec sets the guard, so a second hop cannot happen
    assert 'setenv("OFLM_OPEN_OMP_REEXEC"' in body
    assert 'execv(' in body


def test_declared_publicly_and_defined_outside_the_anonymous_namespace(core: str, core_hpp: str):
    """Both callers (the standalone CLI and the app's main) include the header; a definition
    inside `namespace { ... }` links as an internal symbol and the call fails at link time."""
    assert 'bool reexec_with_passive(char** argv);' in core_hpp
    anon_end = core.index('}  // namespace')
    posix = core.index('bool reexec_with_passive(char** argv)', core.index('#else\nbool reexec_with_passive'))
    assert posix > anon_end, 'the definition must sit outside the anonymous namespace'


@pytest.mark.parametrize('path', [MAIN_CPP, CLI_CPP], ids=['app-main', 'standalone-cli'])
def test_every_entry_point_calls_it(path: Path):
    text = path.read_text()
    assert 'reexec_with_passive(argv)' in text


def test_variable_is_classified_for_the_settings_ci():
    text = REGISTRY.read_text()
    assert '"OFLM_OPEN_OMP_REEXEC"' in text, 'a new OFLM_* variable must be classified'
