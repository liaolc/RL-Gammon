"""
Load Agent 1 (TD(0) Linear) trained weights.

Usage:
    from load_agent1 import load_agent1, get_value
    
    weights = load_agent1('agent1_weights.npy')
    # or
    weights = load_agent1()  # loads default path
    
    value = get_value(state, player, weights)
"""

import numpy as np
import pathlib

from agent1_td0 import extract_features, value_function


def load_agent1(weights_path=None):
    """
    Load Agent 1 (TD(0) Linear) trained weights.
    
    Args:
        weights_path: Path to .npy weights file. If None, loads 'agent1_weights.npy'.
    
    Returns:
        weights: NumPy array of shape (52,)
    """
    if weights_path is None:
        weights_path = pathlib.Path(__file__).parent / 'agent1_weights.npy'
    else:
        weights_path = pathlib.Path(weights_path)
    
    if not weights_path.exists():
        raise FileNotFoundError(f"Weights file not found: {weights_path}")
    
    print(f"Loading weights from {weights_path}...")
    weights = np.load(weights_path)
    print(f"Agent 1 loaded successfully! Shape: {weights.shape}")
    return weights


def get_value(state, player, weights):
    """
    Get value estimate for a state.
    
    Args:
        state: Backgammon state array (28,)
        player: Current player (+1 for white, -1 for black)
        weights: Loaded weights array (52,)
    
    Returns:
        value: Scalar value estimate
    """
    return value_function(state, player, weights)


if __name__ == '__main__':
    try:
        weights = load_agent1()
        print(f"\nWeight statistics:")
        print(f"  Shape: {weights.shape}")
        print(f"  Mean: {weights.mean():.4f}")
        print(f"  Std: {weights.std():.4f}")
        print(f"  Min: {weights.min():.4f}")
        print(f"  Max: {weights.max():.4f}")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Run agent1_td0.py to train and save weights first.")
