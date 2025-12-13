# Agent 1: TD(0) Linear - Quick Guide

## What It Is

TD(0) agent with 52 handcrafted features and 2-ply search.

**Value function:** `V(s) = w · f(s)` (linear)

**Learning:** TD(0) updates: `w ← w + α·δ·f(S)` where `δ = R + γ·V(S') - V(S)`

**Search:** 2-ply minimax (considers opponent's best response to all dice rolls)

## Features (52 total)

- Raw state (28): Board positions
- Blot counts (2): Vulnerable checkers
- Stacked in home (2): Checkers ≥3 deep
- Prime length (4): Consecutive made points
- Trapped checkers (2): Behind opponent's prime
- Made points in home (2): Safe home board points
- Attackers (2): Checkers within 6 pips of blots
- Contact phase (1): Racing vs contact (0-1)
- Pip difference (1): Distance advantage
- Products (6): Feature interactions

## Usage

```bash
# Setup
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Train agent (2-10 hours depending on hardware)
./venv/bin/python src/agent_td0_vectorized.py

# Load and test trained agent
python load_agent1.py
```

## Training Parameters

Edit bottom of `src/agent_td0_vectorized.py`:

```python
train_vectorized(
    batch_size=256,        # Parallel games
    num_iterations=50000,  # Total iterations
    alpha=0.001,           # Learning rate
    gamma=1.0,             # Discount factor
    verbose_every=500      # Progress updates
)
```

## Key Concepts

**Self-Play:** Agent plays both sides using same weights. Board converted to canonical form each turn.

**TD(0):** Updates after every move (faster than Monte Carlo which waits for game end).

**2-Ply Search:** Considers opponent's best response to all possible dice rolls.

**Linear:** Fast, interpretable, but limited to handcrafted features.

## Tips

- Win rate ~50% in self-play is normal (both players improve together)
- Learning rate sweet spot: 0.001-0.01
- Use vectorized version for speed (10-100x faster)
- Training time: ~2-10 hours for 50k iterations depending on hardware
