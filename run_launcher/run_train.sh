#!/usr/bin/env bash
# Train PGFS paper-style (§4.3) on bundled Bi reaction data.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${ROOT}"

CONDA_ROOT="${CONDA_ROOT:-/home/bz317/rds/hpc-work/miniconda3}"
# shellcheck source=/dev/null
source "${CONDA_ROOT}/etc/profile.d/conda.sh"
CONDA_ENV="${CONDA_ENV:-pgfs}"
conda activate "${CONDA_ENV}"

export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export WANDB_PROJECT="${WANDB_PROJECT:-PGFS_Bi}"

if [[ -z "${WANDB_API_KEY:-}" && -f "${ROOT}/../wandb_api_key.txt" ]]; then
  export WANDB_API_KEY="$(tr -d '\n\r ' < "${ROOT}/../wandb_api_key.txt")"
fi

CONFIG="${CONFIG:-configs/paper_style_delta_qed.yaml}"

echo "ROOT=${ROOT}"
echo "CONDA_ENV=${CONDA_ENV}"
echo "CONFIG=${CONFIG}"

python -m pgfs.scripts.train --config "${CONFIG}" "$@"
