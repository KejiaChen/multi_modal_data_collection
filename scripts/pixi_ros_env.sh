#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PACKAGE_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
WORKSPACE_ROOT="$(cd -- "${PACKAGE_ROOT}/../.." && pwd)"

export PACKAGE_ROOT
export WORKSPACE_ROOT

if [[ ! -f /opt/ros/humble/setup.bash ]]; then
    echo "ROS 2 Humble was not found at /opt/ros/humble." >&2
    echo "Install ROS 2 Humble on the workstation before using this Pixi environment." >&2
    exit 1
fi

# Bridge the Pixi Python environment with the system ROS installation.
set +u
source /opt/ros/humble/setup.bash

if [[ -f "${WORKSPACE_ROOT}/install/setup.bash" ]]; then
    source "${WORKSPACE_ROOT}/install/setup.bash"
fi
set -u

if [[ $# -eq 0 ]]; then
    exec bash --noprofile --norc
fi

exec "$@"
