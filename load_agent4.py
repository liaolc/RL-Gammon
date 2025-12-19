"""
Load Agent 4 (Stochastic MuZero) trained weights.

Usage:
    from load_agent4 import load_agent4
    
    params, model = load_agent4('checkpoints/agent4/final_params')
    # or
    params, model = load_agent4()  # loads final_params by default
"""

import jax
import jax.numpy as jnp
from jax import random
import orbax.checkpoint as ocp
import pathlib

from backgammon_muzero_net import StochasticMuZeroNetwork, HIDDEN_SIZE, NUM_DICE_OUTCOMES

MAX_SUBMOVES = 4  # Same as in agent4_muzero.py


def load_agent4(checkpoint_path=None):
    """
    Load Agent 4 (Stochastic MuZero) model and trained weights.
    
    Args:
        checkpoint_path: Path to checkpoint directory. If None, loads 
                        'checkpoints/agent4/final_params'.
    
    Returns:
        params: The loaded model parameters (JAX pytree)
        model: The StochasticMuZeroNetwork model instance
    """
    # Default checkpoint directory
    base_dir = pathlib.Path(__file__).parent / 'checkpoints' / 'agent4'
    
    if checkpoint_path is None:
        # Try final_params first, then latest checkpoint
        final_path = base_dir / 'final_params'
        if final_path.exists():
            checkpoint_path = final_path
            print(f"Loading final params: {checkpoint_path}")
        else:
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
    model = StochasticMuZeroNetwork(hidden_size=HIDDEN_SIZE, max_moves=500)
    
    # Initialize model with dummy inputs matching training code
    rng = random.PRNGKey(0)
    dummy_obs = jnp.zeros((1, 28))
    dummy_move = jnp.zeros((1, MAX_SUBMOVES * 2))
    dummy_dice = jnp.zeros((1, NUM_DICE_OUTCOMES))
    init_variables = model.init(rng, dummy_obs, dummy_move, dummy_dice)
    init_params = init_variables['params']
    
    # Load checkpoint with target structure for proper device placement
    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpointer = ocp.StandardCheckpointer()
    try:
        # Try with target structure first
        params = checkpointer.restore(checkpoint_path, init_params)
    except Exception as e:
        print(f"Warning: Could not restore with target structure: {e}")
        print("Trying without target structure...")
        # Fall back to restoring without target (may fail on CPU if saved on GPU)
        params = checkpointer.restore(checkpoint_path)
    checkpointer.close()
    
    # Wrap params for model.apply() if not already wrapped
    if isinstance(params, dict) and 'params' not in params:
        params = {'params': params}
    
    print("Agent 4 loaded successfully!")
    return params, model


def get_initial_inference(params, model, observation):
    """
    Run initial inference: observation -> (state, policy, value)
    
    Args:
        params: Model parameters (already wrapped as {'params': ...})
        model: StochasticMuZeroNetwork model
        observation: Observation array of shape (batch, 28)
    
    Returns:
        state: Latent state representation (batch, hidden_size)
        policy: Policy logits over moves (batch, max_moves)
        value: Value estimate (batch,)
    """
    state, policy, value = model.apply(
        params, observation, method=model.initial_inference
    )
    return state, policy, value


def encode_observation(state, player):
    """
    Encode backgammon state into observation vector for Agent 4.
    
    Args:
        state: Backgammon state array (28,)
        player: Current player (+1 for white, -1 for black)
    
    Returns:
        observation: Encoded observation (28,)
    """
    import numpy as np
    from backgammon_engine import _to_canonical
    
    canonical = _to_canonical(state, player)
    return canonical.astype(np.float32) / 15.0  # Normalize


if __name__ == '__main__':
    # Test loading
    try:
        params, model = load_agent4()
        print(f"\nModel architecture: StochasticMuZeroNetwork")
        print(f"  - Hidden size: {HIDDEN_SIZE}")
        print(f"  - Observation input: (28,)")
        print(f"  - Policy output: 500 logits (max legal moves)")
        print(f"  - Value output: scalar")
        
        # Test forward pass
        dummy_obs = jnp.zeros((1, 28))
        state, policy, value = get_initial_inference(params, model, dummy_obs)
        print(f"\nTest forward pass:")
        print(f"  Latent state shape: {state.shape}")
        print(f"  Policy shape: {policy.shape}")
        print(f"  Value: {float(value[0]):.4f}")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Make sure you have trained Agent 4 and saved checkpoints.")
