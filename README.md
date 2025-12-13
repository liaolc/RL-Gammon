# RL-Gammon

Backgammon RL agents for CS final project. Environment code by Prof. Carl McTague.

## Structure

```
RL-Gammon/
├── src/
│   ├── agent_td0_linear.py      # Core implementation
│   └── agent_td0_vectorized.py  # Training script
├── docs/
│   └── AGENT1_GUIDE.md          # Implementation guide
├── backgammon_engine.py         # Game engine (professor)
├── backgammon_value_net.py      # For Agent 2 (professor)
├── backgammon_ppo_net.py        # For Agent 3 (professor)
├── load_agent1.py               # Load trained agent
└── requirements.txt
```

## Quick Start

```bash
# Setup
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Train agent (2-10 hours)
./venv/bin/python src/agent_td0_vectorized.py

# Load trained agent
python load_agent1.py
```

## Agent 1: TD(0) Linear ✅

**Implementation:**
- 52 handcrafted features
- TD(0) learning with self-play
- 2-ply minimax search
- Linear value function: V(s) = w·f(s)

**Files:**
- `src/agent_td0_linear.py` - Features & value function
- `src/agent_td0_vectorized.py` - Batch training
- `load_agent1.py` - Load & use agent

**Training:** Runs 50k iterations with batch size 256. Saves to `agent1_weights.npy`.

## Agents 2 & 3

Not yet implemented.
