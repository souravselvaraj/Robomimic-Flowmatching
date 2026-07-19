#!/bin/bash
# Render a 2-camera image dataset from a low_dim dataset's sim states.
#   sbatch scripts/render_image_dataset.sbatch <task> <camera1> <camera2>
# e.g. sbatch scripts/render_image_dataset.sbatch tool_hang sideview robot0_eye_in_hand
# Written uncompressed on purpose: gzip chunks produced unreadable demos previously.
#SBATCH -J fm_dsrender
#SBATCH -A mmaghelih
#SBATCH -p short,long
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=24G
#SBATCH -t 11:59:00
#SBATCH -o /scratch/sselvaraj/projects/robomimic-cfm/output/%x_%j.out

set -u
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

# Hardware GPU rendering via EGL (~100x faster than software OSMesa).
# GPU nodes ship the NVIDIA EGL driver + libGLdispatch + the vendor ICD, but are
# missing only libEGL.so.1 (the GLVND loader) that MuJoCo dlopens. output/eglshim
# holds a copy of that loader taken from the login node (same Ubuntu 24.04 ABI);
# it then dispatches to the node-local libEGL_nvidia.so.0.
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export LD_LIBRARY_PATH="/scratch/sselvaraj/projects/robomimic-cfm/output/eglshim:${LD_LIBRARY_PATH:-}"

RM=/scratch/sselvaraj/projects/Kuromoto-Diffusion-Policy/robomimic_base/robomimic
VENV=/scratch/sselvaraj/projects/Kuromoto-Diffusion-Policy/robomimic_base/.train_venv
ROOT=/scratch/sselvaraj/projects/Flowmatch/robomimic_data

TASK="$1"; CAM1="$2"; CAM2="$3"
echo "render[$TASK]: cameras=$CAM1,$CAM2 job=${SLURM_JOB_ID:-?} start=$(date +%T)"

"$VENV/bin/python" -u "$RM/scripts/dataset_states_to_obs.py" \
  --dataset "$ROOT/$TASK/ph/low_dim_v15.hdf5" \
  --output_name image_v15.hdf5 \
  --camera_names "$CAM1" "$CAM2" \
  --camera_height 84 --camera_width 84 \
  --done_mode 2 --exclude-next-obs

echo "render[$TASK]: done=$(date +%T) size=$(ls -la "$ROOT/$TASK/ph/image_v15.hdf5" 2>/dev/null | awk '{print $5}')"
