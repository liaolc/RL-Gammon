#!/bin/bash
#SBATCH --job-name=rl_gammon_agent4_muzero                        # Job name
#SBATCH --output=rl_gammon_agent4_muzero_%j.out                   # Standard output
#SBATCH --error=rl_gammon_agent4_muzero_%j.err                    # Standard error
#SBATCH --nodes=1                                   # Number of nodes
#SBATCH --ntasks=1                                  # Number of tasks
#SBATCH --cpus-per-task=4                           # Number of CPU cores per task
#SBATCH --gres=gpu:1                                # Request 1 GPU
#SBATCH --mem=49G                                   # Total memory
#SBATCH --partition=medium                          # Partition
#SBATCH --time=48:00:00                              # Wall time (hh:mm:ss)
#SBATCH --mail-type=BEGIN,END,FAIL                  # Email on start, end, fail
#SBATCH --mail-user=zhangdjr@bc.edu                   # Your email

# Change to the project directory
cd /home/zhangdjr/projects/RL-Gammon

# Load conda module if needed (adjust based on your cluster setup)
# module load anaconda3  # Uncomment if needed

# Check if conda environment exists
if conda env list | grep -q "^rl-gammon "; then
    echo "[INFO] Conda environment 'rl-gammon' found."
    echo "[INFO] Activating..."
    source activate rl-gammon
else
    echo "[WARN] Conda environment 'rl-gammon' NOT found."
    echo "[INFO] Creating environment from environment.yml..."

    conda env create -f environment.yml

    if [ $? -ne 0 ]; then
        echo "[ERROR] Failed to create conda environment"
        exit 1
    fi

    echo "[INFO] Activating environment..."
    source activate rl-gammon
fi

# Print node name and GPU status for debugging
echo "Running on node: $(hostname)"
nvidia-smi

# Verify JAX can see the GPU
echo "[INFO] Checking JAX devices..."
python3 -c "import jax; print(f'JAX devices: {jax.devices()}')"

# Run training script with unbuffered output
echo "[INFO] Starting agent4_muzero.py..."
python3 -u agent4_muzero.py train

echo "[INFO] Job completed."
