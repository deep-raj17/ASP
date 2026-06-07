#!/usr/bin/env bash
# edge_deploy/deploy.sh — post-copy update on Raspberry Pi 5
#
# Usage (on Pi):
#   INSTALL_DIR=/opt/mimii ./deploy.sh
#   INSTALL_DIR=~/edge_ai ./deploy.sh
#
set -euo pipefail

INSTALL_DIR="${INSTALL_DIR:-/opt/mimii}"
cd "$INSTALL_DIR"

echo "==> MIMII edge deploy @ $INSTALL_DIR"

if [[ -d .git ]]; then
  echo "Pulling latest code..."
  git pull
fi

if [[ ! -d venv ]]; then
  echo "Creating venv..."
  python3 -m venv venv
fi

source venv/bin/activate

REQ="requirements_rpi.txt"
if [[ ! -f "$REQ" ]]; then
  REQ="requirements_edge.txt"
fi
echo "Installing Python deps from $REQ..."
pip install --upgrade pip -q
pip install -r "$REQ" -q

# ARM thread tuning (Pi 5: reserve core 0 for audio/OS)
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-3}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-3}"

if systemctl is-active --quiet mimii-detector 2>/dev/null; then
  echo "Restarting mimii-detector..."
  sudo systemctl restart mimii-detector
fi

if systemctl is-active --quiet mimii-api 2>/dev/null; then
  echo "Restarting mimii-api..."
  sudo systemctl restart mimii-api
fi

echo "Deployment complete."
echo "  Streaming: systemctl status mimii-detector"
echo "  HTTP API:  systemctl status mimii-api  (or curl http://localhost:8000/health)"
