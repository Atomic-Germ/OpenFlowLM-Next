"""Execute the production OpenMP startup helper without XRT or model weights."""
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture(scope="module")
def startup(tmp_path_factory):
    if sys.platform != "linux":
        pytest.skip("Linux libgomp startup behavior")
    compiler = shutil.which("g++")
    if not compiler:
        pytest.skip("g++ with libgomp required")
    work = tmp_path_factory.mktemp("omp-startup")
    binary = work / "startup"
    failure = work / "startup-failure"
    for output, flags in ((binary, []), (failure, ["-DOFLM_OMP_POLICY_FAIL_EXEC"])):
        result = subprocess.run([
            compiler, "-std=c++17", "-fopenmp", *flags,
            "-I", str(ROOT / "src"), "-I", str(ROOT / "src/include"),
            str(ROOT / "src/open_qwen36/omp_policy.cpp"),
            str(ROOT / "src/open_qwen36/omp_policy_test.cpp"),
            "-o", str(output),
        ], capture_output=True, text=True)
        if result.returncode:
            pytest.skip(f"g++ with libgomp required: {result.stderr.strip()}")
    return binary, failure


def run(startup, env=None, args=None, failure=False):
    clean = {k: v for k, v in os.environ.items()
             if k not in ("OMP_WAIT_POLICY", "OFLM_OPEN_OMP_REEXEC", "OFLM_OPEN_OMP_PASSIVE")}
    binary = startup[1] if failure else startup[0]
    return subprocess.run([str(binary), "startup", *(args or ["argument with spaces", "--literal=value"])],
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
    assert "POLICY=<unset>\nGUARD=<unset>\n" in result.stdout


def test_guard_disables_reexec_without_claiming_policy(startup):
    result = run(startup, {"OFLM_OPEN_OMP_REEXEC": "1"})
    assert result.stdout.count("ENTER\n") == 1
    assert "POLICY=<unset>\n" in result.stdout


def test_failed_exec_reports_error_and_restores_environment(startup):
    result = run(startup, failure=True)
    assert result.stdout.count("ENTER\n") == 1
    assert "POLICY=<unset>\nGUARD=<unset>\n" in result.stdout
    assert "re-exec" in result.stderr and "Permission denied" in result.stderr


def test_windows_helper_does_not_reexec(tmp_path):
    if sys.platform != "linux":
        pytest.skip("host simulation of the Windows helper branch")
    compiler = shutil.which("g++")
    if not compiler:
        pytest.skip("g++ required")
    binary = tmp_path / "startup-windows"
    subprocess.run([
        compiler, "-std=c++17", "-D_WIN32", "-I", str(ROOT / "src"),
        str(ROOT / "src/open_qwen36/omp_policy.cpp"),
        str(ROOT / "src/open_qwen36/omp_policy_test.cpp"), "-o", str(binary),
    ], check=True)
    result = subprocess.run([str(binary), "startup"], capture_output=True, text=True, check=True)
    assert result.stdout.count("ENTER\n") == 1
    assert "POLICY=<unset>\nGUARD=<unset>\n" in result.stdout
