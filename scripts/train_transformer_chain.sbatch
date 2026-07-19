#!/bin/bash
# Self-chaining resume: keep resuming one flow-matching config from its checkpoint
# until it reaches train.num_epochs, recovering from crashes / time limits.
#   sbatch scripts/train_transformer_chain.sbatch configs/transformer/fm_transformer_square.json
# Each window queues its own successor (afterany), so the chain survives a crash or
# timeout; the successor re-checks the epoch and stops the chain once the target is hit.
#SBATCH -J fm_tf_chain
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

# Offscreen rendering for image-obs rollouts, on the GPU via EGL (~100x faster than
# software OSMesa: ~1265 vs ~11 frames/sec). GPU nodes have the NVIDIA EGL driver,
# libGLdispatch and the vendor ICD, but are missing only libEGL.so.1 (the GLVND
# loader) that MuJoCo dlopens; output/eglshim holds a copy taken from the login node
# (same Ubuntu 24.04 ABI), which dispatches to the node-local libEGL_nvidia.so.0.
# Harmless for low_dim runs, which never render.
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export LD_LIBRARY_PATH="/scratch/sselvaraj/projects/robomimic-cfm/output/eglshim:${LD_LIBRARY_PATH:-}"

REPO=/scratch/sselvaraj/projects/robomimic-cfm
VENV=/scratch/sselvaraj/projects/Kuromoto-Diffusion-Policy/robomimic_base/.train_venv
SBATCH=$(command -v sbatch || echo /cm/shared/apps/slurm/current/bin/sbatch)
MAX_ATTEMPTS=12
cd "$REPO"
mkdir -p output

CONFIG="$1"
NAME=$(basename "$CONFIG" .json)
EXP_DIR=$REPO/output/transformer/$NAME
LOG=output/${NAME}_train.log
ATT_FILE=$REPO/output/.chain_${NAME}.attempt

TARGET=$("$VENV/bin/python" -c "import json;print(json.load(open('$CONFIG'))['train']['num_epochs'])")

max_epoch() {
  local lg
  lg=$(find "$EXP_DIR" -maxdepth 3 -name log.txt 2>/dev/null | head -1)
  if [[ -n "$lg" ]]; then grep -a 'Train Epoch' "$lg" | grep -oE '[0-9]+' | sort -n | tail -1; else echo 0; fi
}

EP=$(max_epoch); EP=${EP:-0}
echo "chain[$NAME]: epoch=$EP target=$TARGET job=${SLURM_JOB_ID:-?}"

# stop the chain once training is complete
if (( EP >= TARGET )); then
  echo "chain[$NAME]: reached target ($EP >= $TARGET); chain complete."
  rm -f "$ATT_FILE"
  exit 0
fi

# runaway guard
ATT=$(cat "$ATT_FILE" 2>/dev/null || echo 0); ATT=$((ATT + 1)); echo "$ATT" > "$ATT_FILE"
if (( ATT > MAX_ATTEMPTS )); then
  echo "chain[$NAME]: hit MAX_ATTEMPTS=$MAX_ATTEMPTS at epoch $EP; stopping chain."
  exit 1
fi

# queue the successor to run after THIS window ends (crash, timeout, or success)
"$SBATCH" --dependency=afterany:${SLURM_JOB_ID} "$0" "$CONFIG"

# resume from last.pth if present, else start fresh (append to preserve history)
if [[ -n "$(find "$EXP_DIR" -maxdepth 2 -name last.pth 2>/dev/null)" ]]; then
  echo "chain[$NAME]: resuming from epoch $EP (attempt $ATT/$MAX_ATTEMPTS)"
  PYTHONPATH=$REPO "$VENV/bin/python" -u scripts/train.py --config "$CONFIG" --resume >> "$LOG" 2>&1
else
  echo "chain[$NAME]: fresh start (attempt $ATT/$MAX_ATTEMPTS)"
  printf 'y\n' | PYTHONPATH=$REPO "$VENV/bin/python" -u scripts/train.py --config "$CONFIG" >> "$LOG" 2>&1
fi
echo "chain[$NAME]: window ended at epoch $(max_epoch)"
