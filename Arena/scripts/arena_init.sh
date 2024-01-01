#!/bin/bash

sudo apt update
sudo apt install libpq-dev ffmpeg libsm6 libxext6 gcc gstreamer1.0-plugins-bad

# creating configurations files
CONFIGURATIONS_DIR="$(dirname $(dirname "$0"))/configurations"
if ! test -f "${CONFIGURATIONS_DIR}/cam_config.yaml"; then
  echo "cam_config.yaml does not exist. Creating blank config..."
  echo "{}" > "${CONFIGURATIONS_DIR}/cam_config.yaml"
fi
if ! test -f "${CONFIGURATIONS_DIR}/periphery_config.json"; then
  echo "periphery_config.json does not exist. Creating blank config..."
  echo "{}" > "${CONFIGURATIONS_DIR}/periphery_config.json"
fi