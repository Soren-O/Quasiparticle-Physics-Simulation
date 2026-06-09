#!/bin/zsh
set -euo pipefail

SCRIPT_DIR="${0:A:h}"
REPO_DIR="${SCRIPT_DIR:h}"

cd "${REPO_DIR}"

OUT_DIR="outputs/prelim_readout_heating_overnight"
WALL_HOURS="${1:-12}"

mkdir -p "${OUT_DIR}"
echo "$$" > "${OUT_DIR}/run.pid"

exec .venv/bin/python -u scripts/run_prelim_readout_heating_overnight.py \
  --preset overnight \
  --out-dir "${OUT_DIR}" \
  --wall-hours "${WALL_HOURS}" \
  > "${OUT_DIR}/run.log" 2>&1
