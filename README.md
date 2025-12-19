# RL-Gammon

Backgammon RL agents for CS final project.

## Setup

```bash
conda create -n rl-gammon python=3.12
conda activate rl-gammon
pip install -r requirements.txt
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

## Loading Weights

### Agent 1
```python
from load_agent1 import load_agent1, get_value
weights = load_agent1()
value = get_value(state, player, weights)
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
