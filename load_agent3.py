"""
Load Agent 3 (PPO) trained weights.

Usage:
    from load_agent3 import load_agent3
    
    params, model = load_agent3('checkpoints/agent3/checkpoint_100')
    # or
    params, model = load_agent3()  # loads latest checkpoint
"""

import jax
import jax.numpy as jnp
from jax import random
import orbax.checkpoint as ocp
import pathlib
import os

from backgammon_ppo_net import BackgammonPPONet, BOARD_LENGTH, CONV_INPUT_CHANNELS, AUX_INPUT_SIZE


def load_agent3(checkpoint_path=None):
    """
    Load Agent 3 (PPO) model and trained weights.
    
    Args:
        checkpoint_path: Path to checkpoint directory. If None, loads the latest
                        checkpoint from 'checkpoints/agent3/'.
    
    Returns:
        params: The loaded model parameters (JAX pytree)
        model: The BackgammonPPONet model instance
    """
    # Default checkpoint directory
    base_dir = pathlib.Path(__file__).parent / 'checkpoints' / 'agent3'
    
    if checkpoint_path is None:
        # Find the latest checkpoint
        if not base_dir.exists():
            raise FileNotFoundError(f"Checkpoint directory not found: {base_dir}")
        
        checkpoints = [d for d in base_dir.iterdir() if d.is_dir() and d.name.startswith('checkpoint_')]
        if not checkpoints:
            raise FileNotFoundError(f"No checkpoints found in {base_dir}")
        
        # Sort by iteration number and get the latest
        checkpoints.sort(key=lambda x: int(x.name.split('_')[1]))
        checkpoint_path = checkpoints[-1]
        print(f"Loading latest checkpoint: {checkpoint_path}")
    else:
        checkpoint_path = pathlib.Path(checkpoint_path)
    
    # Create model
    model = BackgammonPPONet()
    
    # Initialize model with dummy input to get parameter structure
    rng = random.PRNGKey(0)
    dummy_board = jnp.zeros((1, BOARD_LENGTH, CONV_INPUT_CHANNELS))
    dummy_aux = jnp.zeros((1, AUX_INPUT_SIZE))
    init_variables = model.init(rng, dummy_board, dummy_aux)
    # Checkpoint saves just params (not wrapped in {'params': ...})
    init_params = init_variables['params']
    
    # Load checkpoint
    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpointer = ocp.StandardCheckpointer()
    params = checkpointer.restore(checkpoint_path, init_params)
    checkpointer.close()
    
    # Wrap params for model.apply()
    params = {'params': params}
    
    print("Agent 3 loaded successfully!")
    return params, model


def get_value_and_policy(params, model, board_features, aux_features):
    """
    Get value estimate and policy logits for a given state.
    
    Args:
        params: Model parameters
        model: BackgammonPPONet model
        board_features: Board features array of shape (batch, 24, 15)
        aux_features: Auxiliary features array of shape (batch, 6)
    
    Returns:
        value: Value estimate (batch,)
        policy_logits: Policy logits (batch, 625)
    """
    value, policy_logits = model.apply(params, board_features, aux_features)
    return value.squeeze(-1), policy_logits


if __name__ == '__main__':
    # Test loading
    try:
        params, model = load_agent3()
        print(f"\nModel architecture: BackgammonPPONet")
        print(f"  - Board input: ({BOARD_LENGTH}, {CONV_INPUT_CHANNELS})")
        print(f"  - Aux input: ({AUX_INPUT_SIZE},)")
        print(f"  - Policy output: 625 logits (25x25 source-dest pairs)")
        print(f"  - Value output: scalar in [-3, 3]")
        
        # Test forward pass
        dummy_board = jnp.zeros((1, BOARD_LENGTH, CONV_INPUT_CHANNELS))
        dummy_aux = jnp.zeros((1, AUX_INPUT_SIZE))
        value, policy = get_value_and_policy(params, model, dummy_board, dummy_aux)
        print(f"\nTest forward pass:")
        print(f"  Value: {float(value[0]):.4f}")
        print(f"  Policy shape: {policy.shape}")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Make sure you have trained Agent 3 and saved checkpoints.")
