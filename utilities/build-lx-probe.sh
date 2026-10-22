#!/bin/bash
# build-lx-probe.sh — build utilities/lx-probe, the standalone layer_x/dense
# stream dispatcher (see the header of utilities/lx-probe.cpp for usage,
# fixtures, and the -g generic mode).
#
#   source utilities/env.sh && utilities/build-lx-probe.sh
#
# Rebuilding is rare: the probe changes only when the stream launch law does.
# The binary lands beside its source, so a wedging stream probe lives at
# utilities/lx-probe rather than in a scratch dir.
set -euo pipefail
cd "$(dirname "$0")/.."

# env.sh locates XRT (its --check prints what it found); without it the
# headers/lib are wherever the user's own environment says.
if [ -f utilities/env.sh ]; then
    # shellcheck disable=SC1091
    source utilities/env.sh >/dev/null
fi

g++ -O2 -std=c++20 -o utilities/lx-probe utilities/lx-probe.cpp \
    -I"${XILINX_XRT:?XILINX_XRT is not set - source utilities/env.sh first}"/include \
    -L"$XILINX_XRT/lib" -lxrt_coreutil
echo "built utilities/lx-probe"
