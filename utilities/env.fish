#!/usr/bin/env fish
# =============================================================================
# env.fish -- the environment this project needs, for fish.
#
#   source utilities/env.fish        # from the repo root
#   source ./env.fish                # from anywhere
#   utilities/env.fish --check       # report what would be set
#
# It is the same environment utilities/env.sh sets: XRT, ROCm/HIP, MLIR-AIE +
# Peano and this project's own config, kernels, models and settings.
#
# The discovery lives in the bash script -- one place, one behaviour -- and
# this is the front end: bash runs it, prints the variables it exported, and
# fish imports them. Nothing is hardcoded here either: env.sh is looked for
# next to this file, and the paths it exports are the ones it found.
#
# bash is required, like everywhere else in this repository.
# =============================================================================

# --help, when executed. When sourced, the help goes to stdout and the
# script returns without touching the environment.
switch "$argv[1]"
    case -h --help
        sed -n '3,25p' (status --current-filename) | sed 's/^#\{1,\} \{0,1\}//'
        return 0
end

set -l script_dir (dirname (status --current-filename))
set -l bash_script $script_dir/env.sh

if not test -f $bash_script
    echo "env.fish: env.sh not found next to this script ($script_dir)" >&2
    exit 1
end

if not command -q bash
    echo "env.fish: bash is required (it runs utilities/env.sh)." >&2
    exit 1
end

# The variables that hold a colon-separated list. Fish keeps those as lists,
# so they are split on import; everything else stays a plain string.
set -l list_vars PATH LD_LIBRARY_PATH PYTHONPATH CPATH CMAKE_PREFIX_PATH

# What the bash script exported, one KEY=VALUE per line. OFLM_ENV_DUMP makes
# it print those instead of its report.
set -l dump (env OFLM_ENV_DUMP=1 bash -c 'source "$1"' _ $bash_script 2>/dev/null)

if test (count $dump) -eq 0
    echo "env.fish: nothing came back from $bash_script." >&2
    exit 1
end

for line in $dump
    set -l kv (string split -m1 '=' -- $line)
    if test (count $kv) -lt 2
        continue
    end
    set -l name $kv[1]
    set -l value $kv[2]
    if contains -- $name $list_vars
        if test -z "$value"
            set -gx $name ""
        else
            set -gx $name (string split ':' -- $value)
        end
    else
        set -gx $name $value
    end
end
set -e dump

# -----------------------------------------------------------------------------
# the summary
# -----------------------------------------------------------------------------

if test -z "$OFLM_ENV_QUIET"
    printf "\n"
    printf "  project   %s\n" $OFLM_ROOT
    printf "  XRT       %s\n" (test -n "$XILINX_XRT"; and echo $XILINX_XRT; or echo "not found")
    printf "  ROCm      %s\n" (test -n "$ROCM_PATH"; and echo "$ROCM_PATH (version $ROCM_VERSION)"; or echo "not found")
    printf "  mlir-aie  %s\n" (test -n "$MLIR_AIE_INSTALL_DIR"; and echo $MLIR_AIE_INSTALL_DIR; or echo "not found")
    printf "  Peano     %s\n" (test -n "$PEANO_INSTALL_DIR"; and echo $PEANO_INSTALL_DIR; or echo "not found")
    printf "  venv      %s\n" (test -n "$VIRTUAL_ENV"; and echo $VIRTUAL_ENV; or echo "none")
    printf "  registry  %s\n" (test -n "$OFLM_CONFIG_PATH"; and echo $OFLM_CONFIG_PATH; or echo "none")
    printf "  kernels   %s\n" (test -n "$OFLM_XCLBIN_PATH"; and echo $OFLM_XCLBIN_PATH; or echo "none")
    printf "  models    %s%s\n" $OFLM_MODEL_PATH (test -n "$OFLM_MODEL_PATH"; and echo " (models in $OFLM_MODEL_PATH/models)")
    printf "  settings  %s\n" (test -n "$OFLM_SETTINGS_PATH"; and echo $OFLM_SETTINGS_PATH; or echo "none")
    printf "  oflm      %s\n" (test -n "$OFLM_BIN"; and echo $OFLM_BIN; or echo "not built")
    printf "\n"
end
