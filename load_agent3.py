"""Load Agent 3 (PPO) weights."""

import jax
import jax.numpy as jnp
from jax import random
import orbax.checkpoint as ocp
import pathlib
import os

from backgammon_ppo_net import BackgammonPPONet, BOARD_LENGTH, CONV_INPUT_CHANNELS, AUX_INPUT_SIZE


def load_agent3(checkpoint_path=None):
    """Load model and weights. Returns (params, model)."""
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
    
    checkpointer = ocp.StandardCheckpointer()
    params = checkpointer.restore(checkpoint_path, init_params)
    checkpointer.close()
    
    params = {'params': params}
    return params, model


def get_value_and_policy(params, model, board_features, aux_features):
    """Get value and policy logits for a state."""
    value, policy_logits = model.apply(params, board_features, aux_features)
    return value.squeeze(-1), policy_logits


if __name__ == '__main__':
    params, model = load_agent3()
    num_params = sum(x.size for x in jax.tree_util.tree_leaves(params))
    print(f"Loaded Agent 3: {num_params:,} parameters")
