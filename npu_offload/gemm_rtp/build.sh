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
# Environment: the IRON toolchain (mlir-aie + Peano) plus the XRT userspace.
# This script sets it up itself via npu_offload/iron_env.sh -- activating
# the iron venv and loading /opt/xilinx/xrt/setup.sh when pyxrt is not yet
# importable -- the manual equivalent being:
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

# -------------------------------- IRON toolchain
# Shared setup (venv resolution + activation, XRT userspace, preflights).
# See npu_offload/iron_env.sh. Sets PY on success, exits otherwise.
IRONVENV="${IRONVENV:-}"
# shellcheck disable=SC1091
source "$HERE/../iron_env.sh"
iron_env_setup || exit 1

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
