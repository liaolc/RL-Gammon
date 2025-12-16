#!/bin/bash
#SBATCH --job-name=agent2_train_cpu
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1 --cpus-per-task=16
#SBATCH --partition=medium
#SBATCH --mem=32g
#SBATCH --time=10:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=zhangdjr@bc.edu

cd /home/zhangdjr/projects/RL-Gammon
source /home/zhangdjr/projects/RL-Gammon/venv/bin/activate

export JAX_PLATFORMS=cpu
python3 -c "import jax; print(f'JAX devices: {jax.devices()}')"

python3 -u agent2_tdl.py
