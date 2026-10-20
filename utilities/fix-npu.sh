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

set -uo pipefail

# ---- the XRT environment --------------------------------------------------
# setup.sh exports XILINX_XRT, LD_LIBRARY_PATH, PATH (xrt-smi, xclbinutil,
# aiebu-asm) and PYTHONPATH (pyxrt). It must be sourced, not executed.
XRT_SETUP="${XRT_SETUP:-/opt/xilinx/xrt/setup.sh}"
if [ -f "$XRT_SETUP" ]; then
  # shellcheck disable=SC1090
  . "$XRT_SETUP" >/dev/null 2>&1
  echo "XRT environment loaded from $XRT_SETUP"
else
  echo "NOTE: $XRT_SETUP not found; continuing without it."
fi

# ---- is the NPU there? ----------------------------------------------------
npu_present() {
  xrt-smi examine 2>/dev/null | grep -qE '^[[:space:]]+1 devices found'
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
  echo "The NPU is already there:"
  xrt-smi examine 2>/dev/null | grep -E 'devices found' | tr -s ' '
  ls -1 /dev/accel/ 2>/dev/null | sed 's/^/  \/dev\/accel\//'
  exit 0
fi

echo
echo "The NPU is not there. Trying to bring it back..."

if ! command -v sudo >/dev/null 2>&1; then
  echo "  no sudo here; run the commands below by hand."
else
  DEVPATH="$(npu_sysfs_path || true)"
  if [ -z "$DEVPATH" ]; then
    echo "  the NPU is not in PCI any more -- a reboot is needed."
  else
    echo "  device: $DEVPATH"
    # Ask for the password once, then reuse the credential cache.
    if ! sudo -v; then
      echo "  sudo failed; run the commands below by hand."
    else
      echo "  1/2 per-device rescan"
      sudo sh -c "echo 1 > '$DEVPATH/rescan'" 2>/dev/null
      sleep 2

      if ! npu_present; then
        echo "  2/2 remove the device and rescan it"
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
  echo "The NPU is back:"
  xrt-smi examine 2>/dev/null | grep -E 'devices found' | tr -s ' '
  ls -1 /dev/accel/ 2>/dev/null | sed 's/^/  \/dev\/accel\//'
  exit 0
fi

cat <<EOF

The NPU is still not enumerated. By hand:

  export PATH=/opt/xilinx/xrt/bin:\$PATH
  xrt-smi examine | tail -3          # expect "1 devices found"

  sudo sh -c 'echo 1 > /sys/bus/pci/devices/0000:c5:00.1/rescan'

  # if that is not enough:
  sudo sh -c 'echo 1 > /sys/bus/pci/devices/0000:c5:00.1/remove'
  sleep 2
  sudo sh -c 'echo 1 > /sys/bus/pci/rescan'

  # and if that is not enough either, a reboot.

EOF
exit 1
