#!/bin/bash
#SBATCH --job-name=agent2_train          # Job name
#SBATCH --output=%x_%j.out               # Output file
#SBATCH --error=%x_%j.err                # Error file
#SBATCH --nodes=1                        # Number of nodes
#SBATCH --ntasks=1 --cpus-per-task=4     # 4 CPUs on a single node
#SBATCH --partition=medium              # GPU partition (adjust if needed)
#SBATCH --gres=gpu:1                     # 1 GPU
#SBATCH --mem=25g                        # Memory request
#SBATCH --time=24:00:00                  # Time limit (24 hours for 50k iterations)
#SBATCH --mail-type=BEGIN,END,FAIL       # Mail events
#SBATCH --mail-user=zhangdjr@bc.edu      # Your BC email

# Change to project directory
cd /home/zhangdjr/projects/RL-Gammon
source /home/zhangdjr/projects/RL-Gammon/venv/bin/activate
python3 agent2_tdl.py
