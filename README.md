# RL-Gammon

Backgammon RL agents for CS final project. Environment code by Prof. Carl McTague.

## Structure

```
RL-Gammon/
├── agent1_td0.py              # Agent 1 implementation
├── backgammon_engine.py       # Game engine (professor)
├── backgammon_value_net.py    # For Agent 2 (professor)
├── backgammon_ppo_net.py      # For Agent 3 (professor)
├── jax_tutorial.py            # JAX examples
└── requirements.txt
```

## Quick Start

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python agent1_td0.py
# Weights saved to agent1_weights.npy
```

## Agent 1: TD(0) Linear

**Implementation:**
- 52 handcrafted features (blots, primes, pip count, etc.)
- TD(0) learning with self-play
- Epsilon-greedy exploration (0.3 → 0.01)
- 2-ply search with batch evaluation
- Canonical state representation
- Linear value function: V(s) = w·f(s)

**File:** `agent1_td0.py` - Complete implementation (features, training, evaluation)

**Training:** 50k iterations, batch size 256, saves to `agent1_weights.npy`

## Agents 2 

### Quick Start:
- Set up anaconda virtual environment (or similar), e.g. running 
`conda env create -f environment.yml` 
- Edit the SLURM script `run_gammon.sl` to reflect your file organization, environment path, etc.
- Hyperparameters like Batch Size, Num. Iterations, can be edited in the `__main__` part at the bottom of agents2_tdl.py (run flags to be implemented)
## Agent 3

Not yet implemented.
