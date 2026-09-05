#!/usr/bin/env bash
#
# build.sh — Build every open_npue design set, in order (Linux).
#
# Linux counterpart of build.ps1. Usage:
#
#   ./build.sh [options]
#
# Options:
#   --only NAME     Build one family instead of all five (e.g. BERT-h1024-bfp16).
#   --dst DIR       Where the sets go (default: <repo>/src/xclbins).
#   --dev NAME      IRON device family to build for (default: npu2).
#                   Forwarded to export_gemm_rtp.py, which resolves it via
#                   aie.iron.device.from_name (npu1 = Phoenix/Hawk Point,
#                   npu2 = Strix/Krackan). Stated explicitly because the
#                   toolchain's own default is n_cols=1 (single column) and
#                   bare "npu" means NPU1, not NPU2.
#   --force         Rebuild even if the set is already there.
#   --ironvenv DIR  Python venv holding the IRON toolchain (must import
#                   aie.iron). Default: <repo>/ironvenv if it exists,
#                   otherwise whatever `python3` resolves to.
#   -h, --help      Show this help.
#
# ~3-4 minutes per family, five families, so budget ~20 minutes. Families that
# are already built are skipped; --force rebuilds, --only does one.
#
# THE FLAGS ARE IN families.json, NOT HERE. One machine-readable source that
# this script builds from and check_design_sets.py verifies against, so there
# is no second copy to drift. (--dev/--dst select the target and the
# destination, not the design, so they live here, like build.ps1's -Dst.)
#
# Environment: the IRON toolchain must be importable (mlir-aie + Peano), i.e.
# the same environment as npu_offload/matmul/. This script sets it up itself:
# it activates the iron venv and loads the XRT userspace
# (/opt/xilinx/xrt/setup.sh) when pyxrt is not yet importable -- the manual
# equivalent is:
#
#   source ironvenv/bin/activate
#   source /opt/xilinx/xrt/setup.sh
#
# The XRT step is load-bearing, not ceremonial: the export allocates
# device="npu" tensors, which need mlir_aie's XRTTensor backend. Without an
# importable pyxrt it degrades to CPUOnlyTensor and every family fails with
# "ValueError: Unsupported device: npu". (NPU_RUNTIME=hrx/hsa opts out of
# the XRT requirement; their own probes then decide.)
#
# Families must never be built concurrently: purge() deletes matching entries
# from the shared ~/.npu/cache on content markers, and the two hidden-768
# families own identical markers for 8 of their 16 entries. This script builds
# strictly sequentially (export_gemm_rtp.py also holds a lock and refuses).
#
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
DST="$REPO/src/xclbins"

ONLY=""
FORCE=0
IRONVENV=""
DEV="npu2"

needval() { [[ $# -ge 2 ]] || { echo "ERROR: $1 needs a value: $1=VALUE" >&2; exit 1; }; }

while [[ $# -gt 0 ]]; do
    case "$1" in
        --only=*) ONLY="${1#--only=}"; shift ;;
        --only) needval "$@"; ONLY="$2"; shift 2 ;;
        --dst=*) DST="${1#--dst=}"; shift ;;
        --dst) needval "$@"; DST="$2"; shift 2 ;;
        --dev=*) DEV="${1#--dev=}"; shift ;;
        --dev) needval "$@"; DEV="$2"; shift 2 ;;
        --ironvenv=*) IRONVENV="${1#--ironvenv=}"; shift ;;
        --ironvenv) needval "$@"; IRONVENV="$2"; shift 2 ;;
        --force) FORCE=1; shift ;;
        -h|--help)
            awk 'NR==1{next} /^#/{sub(/^# ?/,""); print; next} {exit}' "${BASH_SOURCE[0]}"
            exit 0 ;;
        *) echo "ERROR: unknown argument: $1 (try --help)" >&2; exit 1 ;;
    esac
done

die() { echo "ERROR: $*" >&2; exit 1; }

# ---------------------------------------------------------------- IRON python
if [[ -n "$IRONVENV" ]]; then
    VENV="$IRONVENV"
    PY="$VENV/bin/python"
    [[ -x "$PY" ]] || die "no python at $PY (--ironvenv $IRONVENV)."
else
    if [[ -x "$REPO/ironvenv/bin/python" ]]; then
        VENV="$REPO/ironvenv"
        PY="$VENV/bin/python"
    else
        VENV=""
        PY="python3"
    fi
fi
command -v "$PY" >/dev/null 2>&1 || die "python '$PY' not found."

# Activate the venv in this shell so the export below inherits its PATH
# (aiecc et al.) and Python context. Sourcing twice (user already
# activated + this script) only duplicates PATH entries.
if [[ -n "$VENV" && -f "$VENV/bin/activate" ]]; then
    # shellcheck disable=SC1091
    source "$VENV/bin/activate"
fi

# The IRON toolchain, checked before four minutes of work rather than after.
# Without it the failure is `ModuleNotFoundError: No module named 'aie'`,
# which reads as a broken checkout rather than a shell that was never set up.
if ! IRON_ERR="$("$PY" -c "import aie.iron" 2>&1)"; then
    echo "ERROR: the IRON toolchain is not importable by '$PY'." >&2
    echo "$IRON_ERR" | tail -n 4 | sed 's/^/  | /' >&2
    echo "" >&2
    echo "    source $REPO/ironvenv/bin/activate" >&2
    echo "    # only if the build then complains about missing tools:" >&2
    echo "    source $REPO/third_party/mlir-aie/utils/env_setup.sh" >&2
    echo "" >&2
    echo "Or pass the venv explicitly:  ./build.sh --ironvenv <dir>" >&2
    exit 1
fi

# The XRT userspace, loaded the same way as by hand
# (`source /opt/xilinx/xrt/setup.sh`) when pyxrt is not yet importable.
# LOAD-BEARING: the export allocates device="npu" tensors, which need
# mlir_aie's XRTTensor backend. Without pyxrt it degrades to CPUOnlyTensor
# and every family fails with "ValueError: Unsupported device: npu".
# NPU_RUNTIME=hrx/hsa opts out: those backends probe for themselves.
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
        WANT="$("$PY" -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))" 2>/dev/null)"
        HAVE_DIRS=()
        if [[ -n "${XILINX_XRT:-}" ]]; then HAVE_DIRS+=("$XILINX_XRT/python"); fi
        HAVE_DIRS+=(/usr/lib/python3/dist-packages /usr/lib/python3*/dist-packages)
        HAVE="$(find "${HAVE_DIRS[@]}" -maxdepth 1 -name 'pyxrt*.so' 2>/dev/null | head -3 || true)"
        echo "ERROR: pyxrt is not importable by '$PY', so mlir_aie would fall" >&2
        echo "back to CPU-only tensors and the export cannot run." >&2
        if [[ -n "$HAVE" ]]; then
            echo "" >&2
            echo "XRT bindings found on disk but not for this python:" >&2
            echo "$HAVE" | sed 's/^/    /' >&2
            echo "this python wants extension suffix: ${WANT:-unknown}" >&2
            echo "Rebuild the iron venv on the matching Python (e.g. python3.11" >&2
            echo "if the bindings are cpython-311), or install XRT bindings for" >&2
            echo "$("$PY" -c "import sys; print(f'Python {sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null)." >&2
        else
            echo "Install the XRT userspace (it provides pyxrt) and make sure" >&2
            echo "$XILINX_XRT/setup.sh (or XILINX_XRT) points at it." >&2
        fi
        exit 1
    fi
fi

# Peano (llvm-aie), checked the same way before any compile time is spent:
# without it the export dies inside the first build with "RuntimeError:
# Invalid Peano install directory: peano_not_found". It must be installed
# into the SAME venv python that runs the export:
#   <venv>/bin/pip install -r third_party/mlir-aie/utils/peano-requirements.txt
# (or: uv pip install --python <venv>/bin/python -r ...).
# PEANO_INSTALL_DIR overrides discovery when it points at a valid install.
if ! "$PY" -c "from aie.utils.config import peano_install_dir; peano_install_dir()" 2>/dev/null; then
    die "no Peano (llvm-aie) visible to '$PY'.
Install the pinned nightly into that venv first:
    $PY -m pip install -r third_party/mlir-aie/utils/peano-requirements.txt
(or: uv pip install --python $PY -r third_party/mlir-aie/utils/peano-requirements.txt)
or point PEANO_INSTALL_DIR at an existing install."
fi

# ---------------------------------------------------------------- spec
[[ -f "$HERE/families.json" ]] || die "no families.json beside $0."
if [[ -n "$ONLY" ]]; then
    KNOWN="$("$PY" -c "import json; print('\n'.join(f['name'] for f in json.load(open('$HERE/families.json'))['families']))")"
    MATCH=0
    while IFS= read -r name; do
        if [[ "$name" == "$ONLY" ]]; then MATCH=1; break; fi
    done <<< "$KNOWN"
    if [[ "$MATCH" -eq 0 ]]; then
        echo "ERROR: no family named '$ONLY'. Known:" >&2
        while IFS= read -r name; do echo "  $name" >&2; done <<< "$KNOWN"
        exit 1
    fi
fi

echo "Building into $DST (python: $PY)"
echo ""

# export_gemm_rtp.py's purge()/find_cache() call CACHE.iterdir() with no
# existence guard (that file is a synced upstream copy, so the guard lives
# here, not there): on a machine that has never run an IRON build,
# ~/.npu/cache does not exist yet and every family fails in 0s with
# FileNotFoundError before compiling anything. An empty dir purges nothing.
NPU_CACHE="$HOME/.npu/cache"
if [[ ! -d "$NPU_CACHE" ]]; then
    echo "(creating $NPU_CACHE)"
    mkdir -p "$NPU_CACHE"
fi

T_ALL=$SECONDS
BUILT=0; SKIPPED=0; FAILED=()

# One line per family: name<TAB>serves<TAB>args (args separated by \x1f).
SPEC="$("$PY" - "$HERE/families.json" <<'EOF'
import json, sys
spec = json.load(open(sys.argv[1]))
common = spec["common"]
for f in spec["families"]:
    args = "\x1f".join(f["args"] + common)
    print(f["name"] + "\t" + ", ".join(f["serves"]) + "\t" + args)
EOF
)"

while IFS=$'\t' read -r name serves argstr; do
    [[ -z "$name" ]] && continue
    if [[ -n "$ONLY" && "$name" != "$ONLY" ]]; then continue; fi
    out="$DST/$name"
    if [[ -f "$out/gemm_rtp/design.json" && "$FORCE" -eq 0 ]]; then
        printf '  %-24s already built (use --force to rebuild)\n' "$name"
        SKIPPED=$((SKIPPED + 1))
        continue
    fi
    printf '  %-24s %s\n' "$name" "$serves"
    if [[ -e "$out" ]]; then rm -rf "$out"; fi

    IFS=$'\x1f' read -r -a fargs <<< "$argstr"
    T0=$SECONDS
    LOG="$(mktemp)"
    if (cd "$HERE" && "$PY" export_gemm_rtp.py --dev "$DEV" "${fargs[@]}" --out "$out" >"$LOG" 2>&1); then
        CODE=0
    else
        CODE=$?
    fi
    SECS=$((SECONDS - T0))
    N=0
    if [[ -d "$out/gemm_rtp" ]]; then N="$(find "$out/gemm_rtp" -type f | wc -l)"; fi
    if [[ "$CODE" -ne 0 ]]; then
        echo "    FAILED (exit $CODE) after ${SECS}s"
        tail -n 15 "$LOG" | sed 's/^/      /'
        FAILED+=("$name")
    else
        echo "    ok  $N files, ${SECS}s"
        BUILT=$((BUILT + 1))
    fi
    rm -f "$LOG"
done <<< "$SPEC"

TOTAL=$((SECONDS - T_ALL))
echo ""
echo "built $BUILT, skipped $SKIPPED, failed ${#FAILED[@]}  (${TOTAL}s total)"

if [[ "${#FAILED[@]}" -gt 0 ]]; then
    echo "failed: ${FAILED[*]}"
    exit 1
fi

# The spec and the sets it produced must agree. This catches a flag edited in
# families.json without a rebuild, and a set built by some other route.
echo ""
echo "Checking the built sets against families.json:"
(cd "$HERE" && "$PY" check_design_sets.py --xclbins "$DST")
