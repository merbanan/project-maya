#!/bin/bash
# lockclk.sh on|off: a steady GPU clock for A/B measurements and back.  on: 250 W, graphics clock locked at ${LOCK_MHZ:-1230}
# (power sweep 2026-10-16: sd 9 MHz, power-capped 8 % of a load, decode -4 % vs free boost); off: the standing setting,
# 175 W, clocks free, persistence on.  Needs /etc/sudoers.d/nvidia-smi-power.
S="sudo -n /usr/bin/nvidia-smi"
case "$1" in
  on)  $S -pm 1 >/dev/null; $S -pl 250 >/dev/null; $S -lgc ${LOCK_MHZ:-1230},${LOCK_MHZ:-1230} >/dev/null ;;
  off) $S -rgc >/dev/null; $S -pl 175 >/dev/null; $S -pm 1 >/dev/null ;;
  *) echo "usage: lockclk.sh on|off"; exit 2 ;;
esac
nvidia-smi --query-gpu=power.limit,clocks.applications.graphics,persistence_mode --format=csv,noheader
