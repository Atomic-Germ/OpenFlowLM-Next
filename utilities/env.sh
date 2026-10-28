#!/usr/bin/env bash
# =============================================================================
# env.sh -- the environment this project needs, assembled at run time.
#
#   source utilities/env.sh        # from the repo root
#   source ./env.sh                # from anywhere
#   utilities/env.sh --check       # report what would be set, change nothing
#
# Nothing is hardcoded. The project's own files are found relative to this
# script (or to the current directory), and the system toolchains are looked
# up where they are actually installed, at the moment the script runs. A new
# ROCm, a new XRT, a new virtualenv is picked up without editing anything.
#
# What gets set:
#
#   XRT          XILINX_XRT, XRT_ROOT, PATH, LD_LIBRARY_PATH, PYTHONPATH
#                (xrt-smi, xclbinutil, aiebu-asm, pyxrt)
#   ROCm / HIP   ROCM_PATH, ROCM_HOME, ROCM_VERSION, HIP_PATH, HIP_HOME,
#                HIP_ROOT_DIR, PATH, LD_LIBRARY_PATH, CMAKE_PREFIX_PATH
#   MLIR-AIE     MLIR_AIE_INSTALL_DIR, PEANO_INSTALL_DIR, VIRTUAL_ENV, PATH,
#                LD_LIBRARY_PATH, PYTHONPATH (aiecc, aie-opt, aie-translate,
#                bootgen, the `aie` python package)
#   The project  OFLM_CONFIG_PATH, OFLM_MODELINFO_PATH, OFLM_XCLBIN_PATH,
#                OFLM_MODEL_PATH, OFLM_SETTINGS_PATH -- the local registry,
#                the local kernels, the local model store and the local
#                settings file of THIS checkout, so `oflm` run from the tree
#                reads the tree and not an installed copy.
#                OFLM_MODEL_PATH holds the models/ tree, not the tree
#                itself: the models are in $OFLM_MODEL_PATH/models.
#
# Overrides, all optional, all read before anything is set:
#
#   OFLM_ROOT        the project root, when this script does not live in it
#   OFLM_VENV_DIR    the virtualenv holding mlir-aie + Peano
#   XRT_SETUP        the XRT setup.sh to source
#   ROCM_PATH        the ROCm install to use
#   OFLM_ENV_QUIET=1 no summary at the end
# =============================================================================

# Only bash: this script uses indirect expansion and printf -v.
if [ -z "${BASH_VERSION:-}" ]; then
    echo "env.sh: bash is required (it uses bash-only features)." >&2
    return 1 2>/dev/null || exit 1
fi

# Sourced or executed? The summary behaves differently, and only a sourced
# run can change the caller's environment.
oflm_env_sourced=0
if [ "${BASH_SOURCE[0]}" != "${0}" ]; then
    oflm_env_sourced=1
fi

# --help, when executed. When sourced, the help goes to stdout and the
# script returns without touching the environment.
case "${1:-}" in
    -h|--help)
        sed -n '3,35p' "${BASH_SOURCE[0]}" | sed 's/^#\{1,\} \{0,1\}//'
        if [ "$oflm_env_sourced" = "1" ]; then return 0; else exit 0; fi
        ;;
esac

# -----------------------------------------------------------------------------
# small helpers
# -----------------------------------------------------------------------------

# oflm_env_prepend <dir> [VAR]
# Put <dir> at the front of <VAR> (PATH by default), once. A directory that
# does not exist is ignored, so a candidate that is not installed costs
# nothing.
oflm_env_prepend() {
    local dir="$1" var="${2:-PATH}"
    [ -n "$dir" ] || return 0
    [ -d "$dir" ] || return 0
    case ":${!var}:" in
        *":$dir:"*) return 0 ;;
    esac
    if [ -n "${!var}" ]; then
        printf -v "$var" '%s:%s' "$dir" "${!var}"
    else
        printf -v "$var" '%s' "$dir"
    fi
    export "$var"
}

# oflm_env_append <dir> [VAR]
# Same, at the back: the system copy keeps winning.
oflm_env_append() {
    local dir="$1" var="${2:-PATH}"
    [ -n "$dir" ] || return 0
    [ -d "$dir" ] || return 0
    case ":${!var}:" in
        *":$dir:"*) return 0 ;;
    esac
    if [ -n "${!var}" ]; then
        printf -v "$var" '%s:%s' "${!var}" "$dir"
    else
        printf -v "$var" '%s' "$dir"
    fi
    export "$var"
}

# oflm_env_root <start-dir>
# Walk up from <start-dir> until a directory that looks like this project
# turns up: it holds src/ and either src/model_list.json or src/CMakeLists.txt.
oflm_env_root() {
    local dir="${1:-.}"
    dir="$(cd -- "$dir" 2>/dev/null && pwd)" || return 1
    while [ -n "$dir" ] && [ "$dir" != "/" ]; do
        if [ -d "$dir/src" ]; then
            if [ -f "$dir/src/model_list.json" ] || [ -f "$dir/src/CMakeLists.txt" ]; then
                printf '%s\n' "$dir"
                return 0
            fi
        fi
        dir="$(dirname -- "$dir")"
    done
    return 1
}

# oflm_env_newest <dir-glob...>
# Print the highest-versioned match of a glob, e.g. core-7.13 vs core-7.14.
oflm_env_newest() {
    local d
    for d in "$@"; do
        [ -d "$d" ] && printf '%s\n' "$d"
    done | sort -V | tail -1
}

# oflm_env_dedup <VAR>
# Drop the repeated entries of a path variable, keeping the first one. The
# XRT setup.sh prepends its bin directory unconditionally, so sourcing this
# script twice would otherwise double it.
oflm_env_dedup() {
    local var="$1" out="" entry
    while IFS= read -r entry; do
        [ -n "$entry" ] || continue
        case ":$out:" in
            *":$entry:"*) continue ;;
        esac
        if [ -z "$out" ]; then out="$entry"; else out="$out:$entry"; fi
    done < <(printf '%s\n' "${!var}" | tr ':' '\n')
    printf -v "$var" '%s' "$out"
    export "$var"
}

# oflm_env_merge <VAR> <value>
# Fold a value produced elsewhere (the XRT setup.sh, run in a subshell) into
# a path variable: the entries that are new go in front, the ones the caller
# already had keep their place. Sourcing this script twice therefore lands on
# the same PATH, whatever the setup.sh does with its own prepend.
oflm_env_merge() {
    local var="$1" new="$2" cur="${!1}"
    local oldifs="${IFS-}" oldifs_set="${IFS+yes}" entry entry2 found out=""
    local -a newarr=() curarr=() added=()
    IFS=':'
    read -r -a newarr <<< "$new"
    read -r -a curarr <<< "$cur"
    if [ -n "$oldifs_set" ]; then IFS="$oldifs"; else unset IFS; fi
    for entry in "${newarr[@]}"; do
        [ -n "$entry" ] || continue
        found=0
        for entry2 in "${curarr[@]}"; do
            [ "$entry2" = "$entry" ] && { found=1; break; }
        done
        [ "$found" = "0" ] && added+=("$entry")
    done
    for entry in "${added[@]}"; do out="${out:+$out:}$entry"; done
    for entry in "${curarr[@]}"; do out="${out:+$out:}$entry"; done
    printf -v "$var" '%s' "$out"
    export "$var"
}

# -----------------------------------------------------------------------------
# 1. the project root
# -----------------------------------------------------------------------------

oflm_env_self="${BASH_SOURCE[0]}"
oflm_env_self_dir="$(cd -- "$(dirname -- "$oflm_env_self")" 2>/dev/null && pwd)"

OFLM_ROOT="${OFLM_ROOT:-}"
if [ -z "$OFLM_ROOT" ]; then
    # The script lives in <root>/utilities: try its own tree first, then the
    # current directory, so `source utilities/env.sh` and `source ./env.sh`
    # both land on the same root.
    OFLM_ROOT="$(oflm_env_root "$oflm_env_self_dir" || true)"
    [ -z "$OFLM_ROOT" ] && OFLM_ROOT="$(oflm_env_root "$PWD" || true)"
    # Last resort: the parent of the script's directory.
    [ -z "$OFLM_ROOT" ] && OFLM_ROOT="$(dirname -- "$oflm_env_self_dir")"
fi
OFLM_ROOT="$(cd -- "$OFLM_ROOT" 2>/dev/null && pwd)"
export OFLM_ROOT

# -----------------------------------------------------------------------------
# 2. XRT (xrt-smi, xclbinutil, aiebu-asm, pyxrt)
# -----------------------------------------------------------------------------

oflm_env_xrt_setup=""
for oflm_env_c in \
    "${XRT_SETUP:-}" \
    "${XRT_ROOT:-}/setup.sh" \
    "${XILINX_XRT:-}/setup.sh" \
    /opt/xilinx/xrt/setup.sh \
    /opt/xilinx/*/setup.sh \
    /opt/*/xrt/setup.sh \
    /usr/local/xilinx/xrt/setup.sh \
    /usr/local/*/xrt/setup.sh ; do
    if [ -n "$oflm_env_c" ] && [ -f "$oflm_env_c" ]; then
        oflm_env_xrt_setup="$oflm_env_c"
        break
    fi
done

if [ -n "$oflm_env_xrt_setup" ]; then
    # The XRT setup.sh ends by sourcing bash-completion and the xrt-smi
    # completion, which is not something to do to an interactive shell. It is
    # sourced in a subshell instead, and the four variables it sets are
    # copied back. (Same trick as utilities/fix-npu.sh.)
    oflm_env_xrt_env="$(bash -c '. "$0" >/dev/null 2>&1; env' "$oflm_env_xrt_setup" 2>/dev/null || true)"
    for oflm_env_var in XILINX_XRT XRT_ROOT; do
        oflm_env_val="$(printf '%s\n' "$oflm_env_xrt_env" | sed -n "s/^${oflm_env_var}=//p" | tail -n 1)"
        [ -n "$oflm_env_val" ] && export "$oflm_env_var=$oflm_env_val"
    done
    for oflm_env_var in LD_LIBRARY_PATH PATH PYTHONPATH; do
        oflm_env_val="$(printf '%s\n' "$oflm_env_xrt_env" | sed -n "s/^${oflm_env_var}=//p" | tail -n 1)"
        [ -n "$oflm_env_val" ] && oflm_env_merge "$oflm_env_var" "$oflm_env_val"
    done
    if [ -n "${XILINX_XRT:-}" ]; then
        export XILINX_XRT
        export XRT_ROOT="${XRT_ROOT:-$XILINX_XRT}"
    fi
fi

# pyxrt lives in $XILINX_XRT/python (added above), but the Ubuntu packages
# put it in the system dist-packages instead.
if ! python3 -c 'import pyxrt' >/dev/null 2>&1; then
    for oflm_env_c in /usr/lib/python3*/dist-packages /usr/lib/python3/dist-packages; do
        if ls "$oflm_env_c"/pyxrt*.so >/dev/null 2>&1; then
            oflm_env_prepend "$oflm_env_c" PYTHONPATH
            break
        fi
    done
fi

# -----------------------------------------------------------------------------
# 3. ROCm / HIP
# -----------------------------------------------------------------------------

oflm_env_rocm=""
for oflm_env_c in \
    "${ROCM_PATH:-}" \
    "${ROCM_HOME:-}" \
    "${HIP_PATH:-}" \
    /opt/rocm \
    /opt/rocm-* \
    /usr/local/rocm \
    /usr/local/rocm-* ; do
    if [ -n "$oflm_env_c" ] && [ -d "$oflm_env_c" ] && [ -d "$oflm_env_c/bin" ]; then
        oflm_env_rocm="$(cd -- "$oflm_env_c" && pwd)"
        break
    fi
done

if [ -n "$oflm_env_rocm" ]; then
    # The HIP toolchain lives in core-<version>/ inside the ROCm tree.
    oflm_env_rocm_core="$(oflm_env_newest "$oflm_env_rocm"/core-*)"
    # The version: the .info file when the install ships one, else the name of
    # the directory the alternatives symlink points at.
    oflm_env_rocm_version=""
    if [ -f "$oflm_env_rocm/.info/version" ]; then
        oflm_env_rocm_version="$(tr -d '[:space:]' < "$oflm_env_rocm/.info/version")"
    fi
    if [ -z "$oflm_env_rocm_version" ]; then
        oflm_env_rocm_version="$(basename -- "$(readlink -f -- "$oflm_env_rocm")")"
        case "$oflm_env_rocm_version" in
            rocm-*) oflm_env_rocm_version="${oflm_env_rocm_version#rocm-}" ;;
            rocm)   oflm_env_rocm_version="" ;;
        esac
    fi

    export ROCM_PATH="$oflm_env_rocm"
    export ROCM_HOME="$oflm_env_rocm"
    export ROCM_VERSION="$oflm_env_rocm_version"
    export HIP_PATH="$oflm_env_rocm"
    export HIP_HOME="$oflm_env_rocm"
    export HIP_ROOT_DIR="$oflm_env_rocm"
    export HIP_PLATFORM="${HIP_PLATFORM:-amd}"

    oflm_env_prepend "$oflm_env_rocm/bin" PATH
    oflm_env_append "$oflm_env_rocm/lib" LD_LIBRARY_PATH
    oflm_env_prepend "$oflm_env_rocm" CMAKE_PREFIX_PATH
    oflm_env_append "$oflm_env_rocm/include" CPATH

    if [ -n "$oflm_env_rocm_core" ]; then
        oflm_env_prepend "$oflm_env_rocm_core/bin" PATH
        oflm_env_append "$oflm_env_rocm_core/lib" LD_LIBRARY_PATH
        oflm_env_append "$oflm_env_rocm_core/include" CPATH
    fi

    # The ROCm python packages, when the install ships them.
    for oflm_env_c in "$oflm_env_rocm"/lib/python3*/site-packages; do
        oflm_env_append "$oflm_env_c" PYTHONPATH
    done
fi

# -----------------------------------------------------------------------------
# 4. MLIR-AIE + Peano (aiecc, aie-opt, aie-translate, bootgen)
# -----------------------------------------------------------------------------

oflm_env_venv=""
oflm_env_site=""
for oflm_env_c in \
    "${OFLM_VENV_DIR:-}" \
    "${VIRTUAL_ENV:-}" \
    "$OFLM_ROOT/ironvenv" \
    "$OFLM_ROOT/ironenv" \
    "$OFLM_ROOT/ironvenv142" \
    "$OFLM_ROOT/.venv" \
    "$OFLM_ROOT/third_party/mlir-aie/ironenv" ; do
    [ -n "$oflm_env_c" ] || continue
    [ -d "$oflm_env_c" ] || continue
    # Resolve it: the candidate may be relative to the caller's directory.
    oflm_env_c="$(cd -- "$oflm_env_c" 2>/dev/null && pwd)" || continue
    oflm_env_site="$(ls -d "$oflm_env_c"/lib/python*/site-packages 2>/dev/null | head -1)"
    [ -n "$oflm_env_site" ] || continue
    # Both halves are needed: mlir-aie for the drivers, llvm-aie (Peano) for
    # the compiler aiecc drives.
    if [ -d "$oflm_env_site/mlir_aie" ] && [ -d "$oflm_env_site/llvm-aie" ]; then
        oflm_env_venv="$oflm_env_c"
        break
    fi
    oflm_env_site=""
done

# Last resort: a virtualenv that is active but was not found above, or a
# pip-installed mlir_aie in the ambient python.
if [ -z "$oflm_env_venv" ]; then
    oflm_env_site="$(python3 -c 'import site,sys; p=site.getsitepackages(); print(p[0] if p else "")' 2>/dev/null || true)"
    if [ -n "$oflm_env_site" ] && [ -d "$oflm_env_site/mlir_aie" ]; then
        oflm_env_venv="$VIRTUAL_ENV"
    fi
fi

if [ -n "$oflm_env_site" ] && [ -d "$oflm_env_site/mlir_aie" ]; then
    export MLIR_AIE_INSTALL_DIR="$oflm_env_site/mlir_aie"
    export PEANO_INSTALL_DIR="$oflm_env_site/llvm-aie"

    # The real binaries, not the console scripts: the scripts carry a
    # hardcoded shebang to the python they were installed with, which breaks
    # the moment the tree moves.
    oflm_env_append "$MLIR_AIE_INSTALL_DIR/lib" LD_LIBRARY_PATH
    oflm_env_prepend "$MLIR_AIE_INSTALL_DIR/python" PYTHONPATH

    # Peano itself stays OUT of PATH: its clang would shadow the system one.
    # aiecc is handed the compiler through PEANO_INSTALL_DIR.
    export PEANO_BIN_DIR="$PEANO_INSTALL_DIR/bin"

    if [ -n "$oflm_env_venv" ]; then
        export VIRTUAL_ENV="$oflm_env_venv"
        oflm_env_prepend "$oflm_env_venv/bin" PATH
    fi

    # The venv's bin goes in first, the real drivers in front of it: aiecc,
    # aie-opt and aie-translate are the binaries, not the wrappers.
    oflm_env_prepend "$MLIR_AIE_INSTALL_DIR/bin" PATH
fi

# The mlir-aie checkout in the tree, when it is there: it carries the same
# env_setup.sh the docs mention, and the aie python package as a source
# tree rather than a wheel.
if [ -f "$OFLM_ROOT/third_party/mlir-aie/utils/env_setup.sh" ]; then
    export OFLM_MLIR_AIE_SRC="$OFLM_ROOT/third_party/mlir-aie"
fi

# -----------------------------------------------------------------------------
# 5. this project's own config, kernels, models and settings
# -----------------------------------------------------------------------------

# The registry of the tree, not of an installed copy.
if [ -f "$OFLM_ROOT/src/model_list.json" ]; then
    export OFLM_CONFIG_PATH="$OFLM_ROOT/src/model_list.json"
fi
if [ -f "$OFLM_ROOT/src/model_info.json" ]; then
    export OFLM_MODELINFO_PATH="$OFLM_ROOT/src/model_info.json"
fi
# The xclbin root: the directory that HOLDS the xclbins/ directory.
if [ -d "$OFLM_ROOT/src/xclbins" ]; then
    export OFLM_XCLBIN_PATH="$OFLM_ROOT/src"
fi
# The model store of the tree.
#
# OFLM_MODEL_PATH is the directory that HOLDS the models/ tree, not the models
# tree itself: the registry says where the models live ("model_path":
# "models"), and oflm joins the two -- <OFLM_MODEL_PATH>/models/<Model>. The
# oflm-add tool resolves it the same way (OFLM_MODEL_PATH + "/models"), so
# one value serves both. Pointing it at src/models would make oflm look for
# src/models/models and find nothing.
if [ -d "$OFLM_ROOT/src/models" ]; then
    export OFLM_MODEL_PATH="$OFLM_ROOT/src"
    # The user settings live NEXT to the models directory -- the file oflm
    # reads, and the file oflm-add writes. Point at the tree's copy so
    # `oflm config set` writes here and not into ~/.config/oflm.
    export OFLM_SETTINGS_PATH="$OFLM_ROOT/src/settings.json"
fi
# The open-kernel sets of this tree are found through OFLM_XCLBIN_PATH
# (src/xclbins/<Model>/open_kernels): OFLM_OPEN_KERNELS_DIR is deliberately
# left alone, it wants ONE kernel set -- a directory with a manifest.json --
# and a tree-wide value would only make the engine complain.

# The tools that live in the tree: oflm-test and q4nx-build run from their
# python package here, not from the installed one.
for oflm_env_c in "$OFLM_ROOT/utilities/oflm-test" "$OFLM_ROOT/utilities/q4nx-build"; do
    [ -d "$oflm_env_c" ] && oflm_env_append "$oflm_env_c" PYTHONPATH
done

# The binary this tree builds, when it has been built.
if [ -x "$OFLM_ROOT/build/src/oflm" ]; then
    export OFLM_BIN="$OFLM_ROOT/build/src/oflm"
    oflm_env_append "$OFLM_ROOT/build/src" PATH
fi

# The sentencepiece build needs the abseil headers, which are a nested
# checkout of the tokenizers-cpp submodule. The link that points at them is
# gitignored, so a tree that was copied from another machine keeps a link
# aimed at THAT machine -- a link to nowhere, and the build stops with
# "third_party/absl/log/check.h: No such file or directory". A relative one
# survives the tree being moved.
oflm_env_absl="$OFLM_ROOT/third_party/tokenizers-cpp/sentencepiece/third_party"
if [ -d "$oflm_env_absl/abseil-cpp/absl" ] && [ ! -e "$oflm_env_absl/absl" ]; then
    if ln -sfn abseil-cpp/absl "$oflm_env_absl/absl" 2>/dev/null; then
        :
    fi
fi

# -----------------------------------------------------------------------------
# 6. the NPU itself
# -----------------------------------------------------------------------------

for oflm_env_var in PATH LD_LIBRARY_PATH PYTHONPATH; do
    oflm_env_dedup "$oflm_env_var"
done

if command -v xrt-smi >/dev/null 2>&1; then
    oflm_env_npu="$(xrt-smi examine 2>/dev/null | tr -d '\r' | grep -E 'NPU Phoenix|NPU Strix|NPU Strix Halo|NPU Krackan|RyzenAI-npu[1456]' || true)"
    if printf '%s' "$oflm_env_npu" | grep -qiE 'NPU Strix|NPU Strix Halo|NPU Krackan|RyzenAI-npu[456]'; then
        export NPU2=1
    else
        export NPU2=0
    fi
fi

# -----------------------------------------------------------------------------
# 7. the summary
# -----------------------------------------------------------------------------

# The variables this script exports, for the fish front end (utilities/env.fish)
# and for anything else that needs to import them. The list is written out
# rather than tracked as it goes, so a variable that stops being set stops
# being advertised.
oflm_env_exported=(
    OFLM_ROOT
    XILINX_XRT XRT_ROOT
    PATH LD_LIBRARY_PATH PYTHONPATH
    ROCM_PATH ROCM_HOME ROCM_VERSION
    HIP_PATH HIP_HOME HIP_ROOT_DIR HIP_PLATFORM
    CMAKE_PREFIX_PATH CPATH
    MLIR_AIE_INSTALL_DIR PEANO_INSTALL_DIR PEANO_BIN_DIR VIRTUAL_ENV
    OFLM_MLIR_AIE_SRC
    OFLM_CONFIG_PATH OFLM_MODELINFO_PATH OFLM_XCLBIN_PATH
    OFLM_MODEL_PATH OFLM_SETTINGS_PATH OFLM_BIN
    NPU2
)

# OFLM_ENV_DUMP=1: print the exported variables as KEY=VALUE lines instead of
# the report. It is how the fish front end imports them.
if [ -n "${OFLM_ENV_DUMP:-}" ]; then
    for oflm_env_var in "${oflm_env_exported[@]}"; do
        if [ -n "${!oflm_env_var+x}" ]; then
            printf '%s=%s\n' "$oflm_env_var" "${!oflm_env_var}"
        fi
    done
    if [ "$oflm_env_sourced" = "1" ]; then
        oflm_env_leave="return 0"
    else
        oflm_env_leave="exit 0"
    fi
    unset oflm_env_exported oflm_env_sourced oflm_env_self oflm_env_self_dir \
          oflm_env_xrt_setup oflm_env_xrt_env oflm_env_var oflm_env_val \
          oflm_env_c oflm_env_npu oflm_env_rocm oflm_env_rocm_core \
          oflm_env_rocm_version oflm_env_venv oflm_env_site
    unset -f oflm_env_prepend oflm_env_append oflm_env_root oflm_env_newest \
          oflm_env_dedup oflm_env_merge oflm_env_report 2>/dev/null || true
    eval "$oflm_env_leave"
fi

oflm_env_report() {
    cat <<EOF

  project   $OFLM_ROOT
  XRT       ${XILINX_XRT:-$(oflm_env_c="$(command -v xrt-smi)" && echo "system install (${oflm_env_c%/*})" || echo "not found")}${XRT_ROOT:+ (root $XRT_ROOT)}
  ROCm      ${ROCM_PATH:-not found}${ROCM_VERSION:+ (version $ROCM_VERSION)}
  mlir-aie  ${MLIR_AIE_INSTALL_DIR:-not found}
  Peano     ${PEANO_INSTALL_DIR:-not found}
  venv      ${VIRTUAL_ENV:-none}
  registry  ${OFLM_CONFIG_PATH:-none}
  kernels   ${OFLM_XCLBIN_PATH:-none}
  models    ${OFLM_MODEL_PATH:-none}$( [ -n "${OFLM_MODEL_PATH:-}" ] && echo " (models in $OFLM_MODEL_PATH/models)" )
  settings  ${OFLM_SETTINGS_PATH:-none}
  oflm      ${OFLM_BIN:-not built}

EOF
}

if [ "$oflm_env_sourced" = "1" ]; then
    if [ -z "${OFLM_ENV_QUIET:-}" ]; then
        oflm_env_report
    fi
else
    # Executed, not sourced: nothing can be exported to the caller, so print
    # what was found instead.
    oflm_env_report
fi

# Clean up the locals this script used; the exported variables stay.
unset oflm_env_sourced oflm_env_self oflm_env_self_dir oflm_env_xrt_setup \
      oflm_env_xrt_env oflm_env_var oflm_env_val oflm_env_c oflm_env_npu \
      oflm_env_rocm oflm_env_rocm_core oflm_env_rocm_version \
      oflm_env_venv oflm_env_site
unset -f oflm_env_prepend oflm_env_append oflm_env_root oflm_env_newest oflm_env_dedup oflm_env_merge oflm_env_report 2>/dev/null || true
