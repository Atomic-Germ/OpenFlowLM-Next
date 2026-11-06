"""Execute the real startup constructor and re-exec helper without XRT or weights."""
import os
from pathlib import Path
import shutil
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture(scope="module")
def startup(tmp_path_factory):
    compiler = shutil.which("g++")
    if not compiler:
        pytest.skip("g++ required")
    work = tmp_path_factory.mktemp("omp-startup")
    source = (ROOT / "src/open_qwen36/core.cpp").read_text()
    ctor = source[source.index("struct OmpWaitPolicy {"):source.index("/// Host threads for the block")]
    start = source.index("#ifdef _WIN32\nbool reexec_with_passive")
    helper = source[start:source.index("\n#endif", start) + len("\n#endif")]
    # Constructor, helper and actual execv are production code. Only the error case
    # replaces execv, so it is deterministic without requiring a restrictive sandbox.
    prefix = r'''#include <cstdlib>
#include <cstdio>
#include <string>
#include <cerrno>
#include <unistd.h>
#include <omp.h>
#ifdef FAIL_EXEC
static int failed_exec(const char*, char* const*) { errno = EACCES; return -1; }
#define execv failed_exec
#endif
'''
    main = r'''
int main(int argc, char** argv) {
    puts("ENTER"); fflush(stdout);
    reexec_with_passive(argv);
    const char* policy = getenv("OMP_WAIT_POLICY");
    const char* guard = getenv("OFLM_OPEN_OMP_REEXEC");
    printf("POLICY=%s\nGUARD=%s\n", policy ? policy : "unset", guard ? guard : "unset");
    for (int i = 1; i < argc; ++i) printf("ARG=%s\n", argv[i]);
    omp_display_env(0);
}
'''
    cpp = work / "startup.cpp"
    cpp.write_text(prefix + ctor + helper + main)
    for name, flags in [("normal", []), ("failure", ["-DFAIL_EXEC"])]:
        subprocess.run([compiler, "-std=c++17", "-fopenmp", *flags, str(cpp), "-o", str(work / name)], check=True)
    return work


def run(startup, env=None, binary="normal"):
    clean = {k: v for k, v in os.environ.items()
             if k not in ("OMP_WAIT_POLICY", "OFLM_OPEN_OMP_REEXEC", "OFLM_OPEN_OMP_PASSIVE")}
    return subprocess.run([str(startup / binary), "argument with spaces", "--literal=value"],
                          env=clean | (env or {}), capture_output=True, text=True, timeout=5, check=True)


def test_default_reexecs_once_and_preserves_arguments(startup):
    result = run(startup)
    assert result.stdout.count("ENTER\n") == 2
    assert "POLICY=PASSIVE\n" in result.stdout
    assert "OMP_WAIT_POLICY = 'PASSIVE'" in result.stderr
    assert "ARG=argument with spaces\nARG=--literal=value\n" in result.stdout


@pytest.mark.parametrize("policy", ["ACTIVE", "PASSIVE", "active", "passive"])
def test_explicit_policy_wins_without_reexec(startup, policy):
    result = run(startup, {"OMP_WAIT_POLICY": policy})
    assert result.stdout.count("ENTER\n") == 1
    assert f"POLICY={policy}\n" in result.stdout
    assert f"OMP_WAIT_POLICY = '{policy.upper()}'" in result.stderr


def test_opt_out_leaves_environment_untouched(startup):
    result = run(startup, {"OFLM_OPEN_OMP_PASSIVE": "0"})
    assert result.stdout.count("ENTER\n") == 1
    assert "POLICY=unset\nGUARD=unset\n" in result.stdout


def test_guard_disables_reexec_without_claiming_policy(startup):
    result = run(startup, {"OFLM_OPEN_OMP_REEXEC": "1"})
    assert result.stdout.count("ENTER\n") == 1
    assert "POLICY=unset\n" in result.stdout


def test_failed_exec_reports_error_and_restores_environment(startup):
    result = run(startup, binary="failure")
    assert result.stdout.count("ENTER\n") == 1
    assert "POLICY=unset\nGUARD=unset\n" in result.stdout
    assert "re-exec" in result.stderr and "Permission denied" in result.stderr


@pytest.mark.parametrize("path", ["src/src/main.cpp", "src/open_qwen36/cli.cpp"])
def test_entry_point_wiring(path):
    assert "reexec_with_passive(argv)" in (ROOT / path).read_text()


def test_windows_preamble_does_not_include_posix_header(tmp_path):
    """Run the preprocessor on real include guards with no POSIX SDK available.

    This checks platform selection; it does not claim a native MSVC build.
    """
    import re
    compiler = shutil.which('g++')
    if not compiler:
        pytest.skip('g++ required')
    source = (ROOT / 'src/open_qwen36/core.cpp').read_text().split('namespace open_qwen36 {')[0]
    for name in re.findall(r'^#include [<"]([^>"]+)', source, re.M):
        header = tmp_path / name
        header.parent.mkdir(parents=True, exist_ok=True)
        header.write_text('#error POSIX header unavailable on Windows\n' if name == 'unistd.h' else '')
    result = subprocess.run([compiler, '-E', '-P', '-x', 'c++', '-nostdinc', '-I', str(tmp_path),
                             '-D_WIN32', '-U__linux__', '-'], input=source, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
