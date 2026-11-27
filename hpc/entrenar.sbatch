#!/bin/bash -l
#SBATCH --job-name=mjd
#SBATCH --account=2026-ii-fs0432
#SBATCH --qos=gpu
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -e
cd "$SLURM_SUBMIT_DIR"

eval "$(micromamba shell hook --shell bash)"
micromamba activate mjd-hpc
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"

srun python -c "import torch; assert torch.cuda.is_available(), 'CUDA no disponible'; print('GPU:', torch.cuda.get_device_name(0))"
srun --cpus-per-task="$SLURM_CPUS_PER_TASK" python -u main.py
