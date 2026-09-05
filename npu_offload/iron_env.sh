#!/usr/bin/env bash
#
# iron_env.sh — shared IRON toolchain setup for the NPU kernel build scripts.
#
# Sourced (never executed) by npu_offload/gemm_rtp/build.sh and
# open_kernels/build.sh. Both builders need the identical environment, and the
# failure modes below each cost somebody a session, so they live here once
# rather than drifting across two copies.
#
# Caller contract: set REPO (tree root) and IRONVENV (venv dir, may be empty)
# before sourcing; then call iron_env_setup. On success PY names the python
# to run the export with. On any unsatisfied prerequisite it prints the fix
# and exits non-zero.
#
# What it does, in order:
#   1. Resolves PY: $IRONVENV/bin/python, else <repo>/ironvenv/bin/python,
#      else python3. Activates the venv in this shell (PATH for aiecc et al.).
#   2. Gates `import aie.iron` (shows the real import error, not a generic
#      message).
#   3. Loads the XRT userspace (/opt/xilinx/xrt/setup.sh) when pyxrt is not
#      yet importable, then gates on it. LOAD-BEARING: without pyxrt mlir_aie
#      degrades to CPUOnlyTensor and device="npu" allocations fail.
#      NPU_RUNTIME=hrx/hsa opts out (those backends probe for themselves).
#      A frequent cause of "installed but invisible" is diagnosed precisely:
#      XRT's bindings are per-Python-version .so files.
#   4. Gates on Peano (llvm-aie) discoverability.
#
# Step 1's manual equivalent is:
#   source ironvenv/bin/activate
#   source /opt/xilinx/xrt/setup.sh
#
# Peano installs into the SAME venv python via:
#   <venv>/bin/pip install -r third_party/mlir-aie/utils/peano-requirements.txt
# (or: uv pip install --python <venv>/bin/python -r ...).
# PEANO_INSTALL_DIR overrides discovery when it points at a valid install.

# shellcheck disable=SC2034
IRON_ENV_SH=1

iron_env_setup() {
    if [[ -n "${IRONVENV:-}" ]]; then
        IRON_VENV="$IRONVENV"
        PY="$IRON_VENV/bin/python"
        if [[ ! -x "$PY" ]]; then
            echo "ERROR: no python at $PY (--ironvenv $IRONVENV)." >&2
            return 1
        fi
    else
        if [[ -x "$REPO/ironvenv/bin/python" ]]; then
            IRON_VENV="$REPO/ironvenv"
            PY="$IRON_VENV/bin/python"
        else
            IRON_VENV=""
            PY="python3"
        fi
    fi
    if ! command -v "$PY" >/dev/null 2>&1; then
        echo "ERROR: python '$PY' not found." >&2
        return 1
    fi

    # Activate the venv in this shell so the export below inherits its PATH
    # (aiecc et al.) and Python context. Sourcing twice (user already
    # activated + this script) only duplicates PATH entries.
    if [[ -n "$IRON_VENV" && -f "$IRON_VENV/bin/activate" ]]; then
        # shellcheck disable=SC1091
        source "$IRON_VENV/bin/activate"
    fi

    # The IRON toolchain, checked before minutes of work rather than after.
    # Without it the failure is `ModuleNotFoundError: No module named 'aie'`,
    # which reads as a broken checkout rather than a shell that was never
    # set up. The real import error is shown, not swallowed.
    local iron_err
    if ! iron_err="$("$PY" -c "import aie.iron" 2>&1)"; then
        echo "ERROR: the IRON toolchain is not importable by '$PY'." >&2
        echo "$iron_err" | tail -n 4 | sed 's/^/  | /' >&2
        echo "" >&2
        echo "    source $REPO/ironvenv/bin/activate" >&2
        echo "    # only if the build then complains about missing tools:" >&2
        echo "    source $REPO/third_party/mlir-aie/utils/env_setup.sh" >&2
        echo "" >&2
        echo "Or pass the venv explicitly:  --ironvenv <dir>" >&2
        return 1
    fi

    # The XRT userspace, loaded the same way as by hand when pyxrt is not
    # yet importable.
    if [[ "${NPU_RUNTIME:-auto}" == auto || "${NPU_RUNTIME:-auto}" == xrt ]]; then
        if ! "$PY" -c "import pyxrt" 2>/dev/null; then
            if [[ -z "${XILINX_XRT:-}" && -f /opt/xilinx/xrt/setup.sh ]]; then
                export XILINX_XRT=/opt/xilinx/xrt
            fi
            if [[ -n "${XILINX_XRT:-}" && -f "$XILINX_XRT/setup.sh" ]]; then
                echo "(loading XRT userspace: source $XILINX_XRT/setup.sh)"
                source "$XILINX_XRT/setup.sh" >/dev/null 2>&1 || true
            fi
        fi
        if ! "$PY" -c "import pyxrt" 2>/dev/null; then
            # Tailor the message: XRT's bindings are per-Python-version .so
            # files, so "installed but invisible" usually means the running
            # python is not the version they were built for.
            local want have_dirs have
            want="$("$PY" -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))" 2>/dev/null)"
            have_dirs=()
            if [[ -n "${XILINX_XRT:-}" ]]; then have_dirs+=("$XILINX_XRT/python"); fi
            have_dirs+=(/usr/lib/python3/dist-packages /usr/lib/python3*/dist-packages)
            have="$(find "${have_dirs[@]}" -maxdepth 1 -name 'pyxrt*.so' 2>/dev/null | head -3 || true)"
            echo "ERROR: pyxrt is not importable by '$PY', so mlir_aie would fall" >&2
            echo "back to CPU-only tensors and the export cannot run." >&2
            if [[ -n "$have" ]]; then
                echo "" >&2
                echo "XRT bindings found on disk but not for this python:" >&2
                echo "$have" | sed 's/^/    /' >&2
                echo "this python wants extension suffix: ${want:-unknown}" >&2
                echo "Rebuild the iron venv on the matching Python (e.g. python3.11" >&2
                echo "if the bindings are cpython-311), or install XRT bindings for" >&2
                echo "$("$PY" -c "import sys; print(f'Python {sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null)." >&2
            else
                echo "Install the XRT userspace (it provides pyxrt) and make sure" >&2
                echo "$XILINX_XRT/setup.sh (or XILINX_XRT) points at it." >&2
            fi
            return 1
        fi
    fi

    # Peano (llvm-aie): without it the export dies inside the first compile
    # with "RuntimeError: Invalid Peano install directory: peano_not_found".
    if ! "$PY" -c "from aie.utils.config import peano_install_dir; peano_install_dir()" 2>/dev/null; then
        echo "ERROR: no Peano (llvm-aie) visible to '$PY'." >&2
        echo "Install the pinned nightly into that venv first:" >&2
        echo "    $PY -m pip install -r $REPO/third_party/mlir-aie/utils/peano-requirements.txt" >&2
        echo "(or: uv pip install --python $PY -r $REPO/third_party/mlir-aie/utils/peano-requirements.txt)" >&2
        echo "or point PEANO_INSTALL_DIR at an existing install." >&2
        return 1
    fi
}
