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
neuralmd_root=${NEURALMD_ROOT:-/mnt/localDisk3/weizian/external/NeuralMD}
python_bin=${NEURALMD_PYTHON:-/mnt/localDisk3/weizian/conda_envs/protein-quanta-neuralmd/bin/python}
data_root=${MISATO_DATA_ROOT:-/mnt/localDisk3/weizian/datasets/misato}

mkdir -p "$output_dir"
cd "$neuralmd_root/examples"
export PYTHONPATH="$project_root:$neuralmd_root${PYTHONPATH:+:$PYTHONPATH}"

exec "$python_bin" main_MISATO_multi_traj_NeuralMD.py \
  --device "$gpu_index" \
  --seed 42 \
  --input_data_dir "$data_root" \
  --dataset MISATO_100 \
  --batch_size 8 \
  --num_workers 0 \
  --epochs "$epochs" \
  --print_every_epoch 5 \
  --lr 1e-4 \
  --NeuralMD_Binding_frame_num 20 \
  --NeuralMD_step_size 5 \
  --NeuralMD_scaling 100 \
  --NeuralMD_velocity_refined_value_coefficient 0 \
  --ODE_method euler \
  --FrameNet_num_radial 100 \
  --max_grad_norm 1 \
  --output_model_dir "$output_dir"
