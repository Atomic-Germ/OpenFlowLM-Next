#!/usr/bin/env bash
# Bring the NPU back after a process was killed while it was loaded.
#
# The NPU is a PCI device (vendor 1022, device 17f0). When a process dies
# while the NPU is loaded, the device stays in a state the driver will not
# take back: it stays unbound, /dev/accel does not appear, and xrt-smi reports
# "0 devices found". A PCI rescan makes the driver probe it again.
#
# Usage:  utilities/fix-npu.sh
#
# Needs sudo for the rescan. Without it the script says so and prints the
# commands to run by hand. It only repairs the device: it does not run any
# model and does not touch the NPU beyond making it usable again.

# Re-exec under bash if it was started with sh: the script uses bash features.
if [ -z "${BASH_VERSION:-}" ]; then
  exec bash "$0" "$@"
fi

set -uo pipefail
exec 2>&1   # never let a diagnostic vanish

echo "fix-npu: starting"

# ---- the XRT environment --------------------------------------------------
# setup.sh exports XILINX_XRT, LD_LIBRARY_PATH, PATH (xrt-smi, xclbinutil,
# aiebu-asm) and PYTHONPATH (pyxrt). It must be sourced, not executed.
#
# It is sourced in a subshell on purpose: it ends by sourcing bash_completion
# and the xrt-smi completion, and those can take the shell down with them. A
# subshell cannot take this script with it, and the four variables it sets
# are copied back.
XRT_SETUP="${XRT_SETUP:-/opt/xilinx/xrt/setup.sh}"
if [ -f "$XRT_SETUP" ]; then
  xrt_env="$(bash -c ". '$XRT_SETUP' >/dev/null 2>&1; env" 2>/dev/null || true)"
  for var in XILINX_XRT LD_LIBRARY_PATH PATH PYTHONPATH; do
    val="$(printf '%s\n' "$xrt_env" | sed -n "s/^${var}=//p" | tail -n 1)"
    if [ -n "$val" ]; then
      export "$var=$val"
    fi
  done
  echo "fix-npu: XRT environment loaded from $XRT_SETUP (XILINX_XRT=${XILINX_XRT:-unset})"
else
  echo "fix-npu: NOTE: $XRT_SETUP not found; continuing without it."
fi

# ---- is the NPU there? ----------------------------------------------------
# The accel device node is the thing that matters. xrt-smi is a cross-check:
# its wording for a single device has been seen both as "1 device found" and
# as "1 devices found", so match either -- and do not rely on it alone.
npu_present() {
  if [ -e /dev/accel/accel0 ]; then
    return 0
  fi
  xrt-smi examine 2>/dev/null | grep -qE '^[[:space:]]+1 devices? found'
}

npu_sysfs_path() {
  local d
  for d in /sys/bus/pci/devices/*/; do
    if [ "$(cat "$d/device" 2>/dev/null)" = "0x17f0" ]; then
      echo "${d%/}"
      return 0
    fi
  done
  return 1
}

if npu_present; then
  echo
  echo "fix-npu: the NPU is already there:"
  xrt-smi examine 2>/dev/null | grep -E 'devices? found' | tr -s ' ' | sed 's/^/  /'
  ls -1 /dev/accel/ 2>/dev/null | sed 's|^|  /dev/accel/|'
  exit 0
fi

echo
echo "fix-npu: the NPU is not there. What xrt-smi says:"
xrt-smi examine 2>/dev/null | tail -4 | sed 's/^/  /'
ls -l /dev/accel/ 2>&1 | sed 's/^/  /'
echo "fix-npu: trying to bring it back"

if ! command -v sudo >/dev/null 2>&1; then
  echo "fix-npu:   no sudo here; run the commands below by hand."
else
  DEVPATH="$(npu_sysfs_path || true)"
  if [ -z "$DEVPATH" ]; then
    echo "fix-npu:   the NPU is not in PCI any more -- a reboot is needed."
  else
    echo "fix-npu:   device: $DEVPATH"
    # Ask for the password once, then reuse the credential cache.
    if ! sudo -v; then
      echo "fix-npu:   sudo failed; run the commands below by hand."
    else
      echo "fix-npu:   1/2 per-device rescan"
      sudo sh -c "echo 1 > '$DEVPATH/rescan'" 2>/dev/null
      sleep 2

      if ! npu_present; then
        echo "fix-npu:   2/2 remove the device and rescan it"
        sudo sh -c "echo 1 > '$DEVPATH/remove'" 2>/dev/null
        sleep 2
        sudo sh -c "echo 1 > /sys/bus/pci/rescan" 2>/dev/null
        sleep 3
      fi
    fi
  fi
fi

echo
if npu_present; then
  echo "fix-npu: the NPU is back:"
  xrt-smi examine 2>/dev/null | grep -E 'devices? found' | tr -s ' ' | sed 's/^/  /'
  ls -1 /dev/accel/ 2>/dev/null | sed 's|^|  /dev/accel/|'
  exit 0
fi

cat <<EOF

fix-npu: the NPU is still not enumerated. By hand:

  export PATH=/opt/xilinx/xrt/bin:\$PATH
  xrt-smi examine | tail -3          # expect "1 device found"

  sudo sh -c 'echo 1 > /sys/bus/pci/devices/0000:c5:00.1/rescan'

  # if that is not enough:
  sudo sh -c 'echo 1 > /sys/bus/pci/devices/0000:c5:00.1/remove'
  sleep 2
  sudo sh -c 'echo 1 > /sys/bus/pci/rescan'

  # and if that is not enough either, a reboot.

EOF
exit 1
