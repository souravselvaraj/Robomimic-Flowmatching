#!/bin/bash
# Train one flow-matching config through robomimic's full pipeline.
#   sbatch scripts/train_transformer.sbatch configs/transformer/fm_transformer_square.json
# Resumes automatically from last.pth if the experiment dir already has one.
#SBATCH -J fm_tf
#SBATCH -A mmaghelih
#SBATCH -p short,long,quick
#SBATCH --gres=gpu:1
#SBATCH --constraint=RTX6000B|A100|H100
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 11:59:00
#SBATCH -o /scratch/sselvaraj/projects/robomimic-cfm/output/%x_%j.out

set -u
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

cd /scratch/sselvaraj/projects/robomimic-cfm
VENV=/scratch/sselvaraj/projects/Kuromoto-Diffusion-Policy/robomimic_base/.train_venv
mkdir -p output

CONFIG="$1"
NAME=$(basename "$CONFIG" .json)
EXP_DIR=/scratch/sselvaraj/projects/robomimic-cfm/output/transformer/"$NAME"
# route train.py stdout to a regular file (not the SLURM pipe) to avoid the tqdm
# non-blocking-write BlockingIOError seen on this cluster's stdout pipes
LOG=output/"${NAME}"_train.log

if [[ -n "$(find "$EXP_DIR" -maxdepth 2 -name last.pth 2>/dev/null)" ]]; then
  echo "Found existing checkpoint under $EXP_DIR - resuming."
  PYTHONPATH=$PWD "$VENV/bin/python" -u scripts/train.py --config "$CONFIG" --resume > "$LOG" 2>&1
else
  printf 'y\n' | PYTHONPATH=$PWD "$VENV/bin/python" -u scripts/train.py --config "$CONFIG" > "$LOG" 2>&1
fi
echo "done: $NAME (train log: $LOG)"
