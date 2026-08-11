#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "usage: $0 OUTPUT_DIR EPOCHS [GPU_INDEX]" >&2
  exit 2
fi

output_dir=$1
epochs=$2
gpu_index=${3:-0}
project_root=${PROTEIN_QUANTA_ROOT:-/mnt/localDisk3/weizian/Protein-Quanta}
python_bin=${NEURALMD_PYTHON:-/mnt/localDisk3/weizian/conda_envs/protein-quanta-neuralmd/bin/python}

export PYTHONPATH="$project_root${PYTHONPATH:+:$PYTHONPATH}"
exec "$python_bin" -m scripts.run_neuralmd_training \
  "$output_dir" "$epochs" "$gpu_index" --max-grad-norm 1
