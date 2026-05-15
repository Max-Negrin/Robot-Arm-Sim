#!/bin/bash
# Reload arm_vismach.py without restarting LinuxCNC.
# Run from the config directory: bash reload_vismach.sh

pkill -f arm_vismach.py
sleep 0.3

python3 "$(dirname "$0")/arm_vismach.py" &

# Wait for the HAL component to register before wiring it
sleep 1

halcmd net j0-vis arm-vismach.joint0
halcmd net j1-vis arm-vismach.joint1
halcmd net j2-vis arm-vismach.joint2
halcmd net j3-vis arm-vismach.joint3
halcmd net j4-vis arm-vismach.joint4
