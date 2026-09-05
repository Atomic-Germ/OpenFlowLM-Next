#!/usr/bin/env bash
#
# build.sh — Build the open Qwen3.6-MoE kernel sets (Linux).
#
# Usage:
#
#   ./build.sh [options]
#
# Options:
#   --only LIST     Comma-separated subset of lx0,lx1,ax0,ax1,ln,lm_head_q8
#                   (default: all six).
#   --dst DIR       Where the sets go (default:
#                   <repo>/src/xclbins/Qwen3.6-35B-A3B-NPU2/open_kernels).
#   --force         Rebuild even if the set is already there.
#   --ironvenv DIR  Python venv holding the IRON toolchain (must import
#                   aie.iron). Default: <repo>/ironvenv if it exists,
#                   otherwise whatever `python3` resolves to.
#   -h, --help      Show this help.
#
# ~4-6 minutes for all six. Sets that are already built (final.xclbin +
# insts.bin present) are skipped; --force rebuilds. The one command behind
# this script is open_kernels/export_qwen36_kernels.py, which owns the SETS
# table (design source, build dir, compile-time knobs) -- see also
# src/open_qwen36/README.md. The device needs no flag: build_design.py pins
# npu2 itself (without it IRON silently targets NPU1).
#
# Environment: same shared setup as npu_offload/gemm_rtp/build.sh (iron venv
# activation, XRT userspace, preflights) via npu_offload/iron_env.sh.
#
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/.." && pwd)"
DST="$REPO/src/xclbins/Qwen3.6-35B-A3B-NPU2/open_kernels"

ONLY=""
FORCE=0
IRONVENV=""

needval() { [[ $# -ge 2 ]] || { echo "ERROR: $1 needs a value: $1=VALUE" >&2; exit 1; }; }

while [[ $# -gt 0 ]]; do
    case "$1" in
        --only=*) ONLY="${1#--only=}"; shift ;;
        --only) needval "$@"; ONLY="$2"; shift 2 ;;
        --dst=*) DST="${1#--dst=}"; shift ;;
        --dst) needval "$@"; DST="$2"; shift 2 ;;
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
source "$REPO/npu_offload/iron_env.sh"
iron_env_setup || exit 1

# ---------------------------------------------------------------- spec
# The set names are the exporter's SETS keys. Kept in one place (there),
# validated here before any build time is spent.
KNOWN="lx0 lx1 ax0 ax1 ln lm_head_q8"
if [[ -z "$ONLY" ]]; then
    WANT="$KNOWN"
else
    WANT=""
    for n in ${ONLY//,/ }; do
        MATCH=0
        for k in $KNOWN; do
            if [[ "$n" == "$k" ]]; then MATCH=1; break; fi
        done
        if [[ "$MATCH" -eq 0 ]]; then
            echo "ERROR: no set named '$n'. Known: $KNOWN" >&2
            exit 1
        fi
        WANT="$WANT $n"
    done
fi

echo "Building into $DST (python: $PY)"
echo ""

# Skip what is already built: the engine loads <name>/final.xclbin +
# <name>/insts.bin, so a set is complete when both exist. The exporter
# merges subset rebuilds into the existing toolchain.json itself.
TODO=""
SKIPPED=0
for n in $WANT; do
    if [[ -f "$DST/$n/final.xclbin" && -f "$DST/$n/insts.bin" && "$FORCE" -eq 0 ]]; then
        printf '  %-12s already built (use --force to rebuild)\n' "$n"
        SKIPPED=$((SKIPPED + 1))
    else
        TODO="$TODO${TODO:+,}$n"
    fi
done

T_ALL=$SECONDS
BUILT=0
if [[ -n "$TODO" ]]; then
    echo "  building: $TODO"
    if (cd "$HERE" && "$PY" export_qwen36_kernels.py --out "$DST" --only "$TODO"); then
        BUILT=$(echo "$TODO" | tr ',' '\n' | wc -l)
    else
        echo "FAILED: export_qwen36_kernels.py --only $TODO" >&2
        echo "The output above is the real diagnostic." >&2
        exit 1
    fi
fi

TOTAL=$((SECONDS - T_ALL))
echo ""
echo "built $BUILT, skipped $SKIPPED  (${TOTAL}s total)"

if [[ -f "$DST/toolchain.json" ]]; then
    "$PY" - "$DST/toolchain.json" <<'EOF'
import json, sys
t = json.loads(open(sys.argv[1]).read())
print(f"toolchain.json: mlir-aie {t.get('mlir_aie_version')}, "
      f"Peano {t.get('peano_version')}, source {t.get('source_git_head', '')[:12]}")
print(f"sets: {', '.join(sorted(t.get('sets', {})))}")
EOF
fi
