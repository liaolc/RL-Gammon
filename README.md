# RL-Gammon

Backgammon RL agents for CS final project.

## Setup

```bash
conda env create -f environment.yml
conda activate rl-gammon
```

## Training

### Agent 1
```bash
python agent1_td0.py
```

### Agent 3 (GPU)
```bash
sbatch train_agent3.sl
```

### Agent 4 (GPU)
```bash
sbatch train_agent4.sl
```

### Agent 2 (GPU)
```bash
sbatch run_gammon.sl
```
Note: You may run into issues if the machine's cuda has compatibility issues with JAX. Also, you will have to change the paths inside the code to match up with your own machine, as we used the paths on our laptops/clusters.

## Loading Weights

### Agent 1
```python
from load_agent1 import load_agent1, get_value
weights = load_agent1()
value = get_value(state, player, weights)
```

### Agent 2
Unzip `checkpoints/agent_2/agent2_b32_long.zip` (longer training) first, then:
```python
from load_agent2 import load_agent2, get_value
params, model = load_agent2("<path_to_checkpoint>")
value = get_value(params, model, board_features, aux_features)
```

### Agent 3
```python
from load_agent3 import load_agent3, get_value_and_policy
params, model = load_agent3()
value, policy_logits = get_value_and_policy(params, model, board_features, aux_features)
```

### Agent 4
```python
from load_agent4 import load_agent4, get_initial_inference
params, model = load_agent4()
state, policy, value = get_initial_inference(params, model, observation)
```

## Reflection

Tian-hao Zhang made agents 1, 3, 4 while Roy Liao made agent 2. The trained weights may not be the most advanced, because we received late notice that the cluster we initially used(NSF Delta) had an incompatible CUDA with JAX. Furthermore, memory issues during the middle of train jobs made it so that we only had 24 hours to train our agents. Some of the code used LLM assistance, mainly for generating some of the helper functions used in the agents.
