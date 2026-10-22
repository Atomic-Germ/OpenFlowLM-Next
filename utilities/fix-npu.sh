#!/usr/bin/env bash
# Bring the NPU back after a process was killed while it was loaded.
#
# The NPU is a PCI device (vendor 1022, device 17f0). When a process dies
# while the NPU is loaded, the device stays in a state the driver will not
# take back: it stays unbound, /dev/accel does not appear, and xrt-smi reports
# "0 devices found".
#
# Two things are tried, in order:
#
#   1. A PCI rescan, several times. The amdxdna driver probes the device
#      again on every rescan, and the probe fails on the first attempt after
#      the device was power cycled (amd/xdna-driver#1257), so the cycle is
#      repeated.
#
#   2. With --firmware-old, the firmware the device loads is swapped for the
#      previous generation. The 1.1.2.64 blob answers the SMU POWER_OFF
#      precheck with 0xff and the probe aborts with -22; the 1.0.0.63 blob
#      (mailbox protocol 6) probes cleanly. This is "update FW to get the old
#      FW back" from the XRT maintainer, and it edits a system file, so it is
#      opt-in and reversible.
#
# Usage:  utilities/fix-npu.sh [--firmware-old]
#
# Needs sudo for the rescan and for the firmware swap. Without it the script
# says so and prints the commands to run by hand. It only repairs the device:
# it does not run any model and does not touch the NPU beyond making it
# usable again.

# Re-exec under bash if it was started with sh: the script uses bash features.
if [ -z "${BASH_VERSION:-}" ]; then
  exec bash "$0" "$@"
fi

set -uo pipefail
exec 2>&1   # never let a diagnostic vanish

FIRMWARE_OLD=0
for arg in "$@"; do
  case "$arg" in
    --firmware-old) FIRMWARE_OLD=1 ;;
    -h|--help)
      sed -n '2,30p' "$0" | sed 's/^# \{0,1\}//'
      exit 0 ;;
  esac
done

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

FWROOT=/usr/lib/firmware/amdnpu

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

# The NPU's PCI device, and the firmware directory that goes with it
# (amdnpu/<device>_<revision>, e.g. 17f0_20).
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

npu_firmware_dir() {
  local devpath="$1" id rev
  id="$(cat "$devpath/device" 2>/dev/null)"      # 0x17f0
  rev="$(cat "$devpath/revision" 2>/dev/null)"    # 0x20
  id="${id#0x}"; rev="${rev#0x}"
  # 0x20 -> 20
  printf '%s/%s_%s\n' "$FWROOT" "$id" "$rev"
}

report() {
  echo
  echo "fix-npu: xrt-smi says:"
  xrt-smi examine 2>/dev/null | tail -4 | sed 's/^/  /'
  ls -l /dev/accel/ 2>&1 | sed 's/^/  /'
}

if npu_present; then
  echo
  echo "fix-npu: the NPU is already there:"
  xrt-smi examine 2>/dev/null | grep -E 'devices? found' | tr -s ' ' | sed 's/^/  /'
  ls -1 /dev/accel/ 2>/dev/null | sed 's|^|  /dev/accel/|'
  exit 0
fi

echo
echo "fix-npu: the NPU is not there."
report
echo "fix-npu: trying to bring it back"

if ! command -v sudo >/dev/null 2>&1; then
  echo "fix-npu:   no sudo here; run the commands below by hand."
  HAVE_SUDO=0
elif ! sudo -v; then
  echo "fix-npu:   sudo failed; run the commands below by hand."
  HAVE_SUDO=0
else
  HAVE_SUDO=1
  DEVPATH="$(npu_sysfs_path || true)"
  if [ -z "$DEVPATH" ]; then
    echo "fix-npu:   the NPU is not in PCI any more -- a reboot is needed."
    HAVE_SUDO=0
  else
    echo "fix-npu:   device: $DEVPATH"
  fi
fi

# ---- attempt 1: rescan, then remove + rescan, several times --------------
# The probe fails on the first attempt after the device was power cycled, so
# the cycle is repeated before giving up.
if [ "$HAVE_SUDO" = "1" ] && [ -n "${DEVPATH:-}" ]; then
  for attempt in 1 2 3; do
    echo "fix-npu:   attempt $attempt: rescan"
    sudo sh -c "echo 1 > '$DEVPATH/rescan'" 2>/dev/null
    sleep 2
    if npu_present; then
      break
    fi
    echo "fix-npu:   attempt $attempt: remove the device and rescan"
    sudo sh -c "echo 1 > '$DEVPATH/remove'" 2>/dev/null
    sleep 2
    sudo sh -c "echo 1 > /sys/bus/pci/rescan" 2>/dev/null
    sleep 3
    if npu_present; then
      break
    fi
  done
fi

# ---- attempt 2: the previous generation of firmware -----------------------
# The 1.1.2.64 blob answers the SMU POWER_OFF precheck with 0xff and the
# probe aborts with -22. The 1.0.0.63 blob (mailbox protocol 6) probes
# cleanly. The device loads npu.dev.sbin, so pointing that at the old blob
# is the whole change.
if ! npu_present && [ "$FIRMWARE_OLD" = "1" ] && [ "$HAVE_SUDO" = "1" ] && [ -n "${DEVPATH:-}" ]; then
  FWDIR="$(npu_firmware_dir "$DEVPATH")"
  echo
  echo "fix-npu: the rescan did not help. Swapping the firmware."
  echo "fix-npu:   firmware dir: $FWDIR"
  ls -la "$FWDIR" 2>/dev/null | sed 's/^/fix-npu:     /'
  OLD_BLOB="$(ls "$FWROOT"/17f0_10/npu.sbin.1.0.0.63.xz 2>/dev/null | head -1)"
  if [ -z "$OLD_BLOB" ]; then
    echo "fix-npu:   the previous generation blob (npu.sbin.1.0.0.63.xz) is not there; skipping."
  else
    CUR_TARGET="$(readlink "$FWDIR/npu.dev.sbin" 2>/dev/null || true)"
    echo "fix-npu:   npu.dev.sbin currently -> ${CUR_TARGET:-?}"
    echo "fix-npu:   repointing it at $OLD_BLOB"
    sudo ln -sfn "$(realpath --relative-to="$FWDIR" "$OLD_BLOB")" "$FWDIR/npu.dev.sbin" 2>/dev/null
    ls -la "$FWDIR/npu.dev.sbin" 2>/dev/null | sed 's/^/fix-npu:     /'
    echo "fix-npu:   rescanning"
    sudo sh -c "echo 1 > '$DEVPATH/remove'" 2>/dev/null
    sleep 2
    sudo sh -c "echo 1 > /sys/bus/pci/rescan" 2>/dev/null
    sleep 3
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

fix-npu: the NPU is still not enumerated.

  What it is:
    the device is a PCI function the driver is not bound to, and the amdxdna
    probe fails on this firmware (amd/xdna-driver#1257):

      aie2_smu_exec: smu cmd 4 failed, 0xff
      aie2_smu_init: Access power failed, ret -22
      amdxdna_probe: Hardware init failed, ret -22

    The 1.1.2.64 firmware answers the SMU POWER_OFF precheck with 0xff; the
    previous generation (1.0.0.63, mailbox protocol 6) probes cleanly.

  By hand:

    export PATH=/opt/xilinx/xrt/bin:\$PATH

    # 1. rescan, several times -- the first probe after a power cycle fails
    for i in 1 2 3; do
      sudo sh -c 'echo 1 > /sys/bus/pci/devices/0000:c5:00.1/remove'
      sleep 2
      sudo sh -c 'echo 1 > /sys/bus/pci/rescan'
      sleep 3
      xrt-smi examine | tail -1
    done

    # 2. the previous generation of firmware
    cd /usr/lib/firmware/amdnpu/17f0_20
    sudo ln -sfn ../17f0_10/npu.sbin.1.0.0.63.xz npu.dev.sbin
    sudo sh -c 'echo 1 > /sys/bus/pci/devices/0000:c5:00.1/remove'
    sleep 2
    sudo sh -c 'echo 1 > /sys/bus/pci/rescan'

    # 3. and if neither works, a reboot.

EOF
exit 1
