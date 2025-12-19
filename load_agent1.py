"""Load Agent 1 (TD(0) Linear) weights."""

import numpy as np
import pathlib

from agent1_td0 import extract_features, value_function


def load_agent1(weights_path=None):
    """Load weights. Returns NumPy array (52,)."""
    if weights_path is None:
        weights_path = pathlib.Path(__file__).parent / 'agent1_weights.npy'
    else:
        weights_path = pathlib.Path(weights_path)
    
    if not weights_path.exists():
        raise FileNotFoundError(f"Weights file not found: {weights_path}")
    
    weights = np.load(weights_path)
    return weights


def get_value(state, player, weights):
    """Get value estimate for a state."""
    return value_function(state, player, weights)


if __name__ == '__main__':
    weights = load_agent1()
    print(f"Loaded weights: shape={weights.shape}, mean={weights.mean():.4f}")
