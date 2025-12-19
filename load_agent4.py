"""Load Agent 4 (Stochastic MuZero) weights."""

import jax
import jax.numpy as jnp
from jax import random
import orbax.checkpoint as ocp
import pathlib

from backgammon_muzero_net import StochasticMuZeroNetwork, HIDDEN_SIZE, NUM_DICE_OUTCOMES

MAX_SUBMOVES = 4  # Same as in agent4_muzero.py


def load_agent4(checkpoint_path=None):
    """Load model and weights. Returns (params, model)."""
    # Default checkpoint directory
    base_dir = pathlib.Path(__file__).parent / 'checkpoints' / 'agent4'
    
    if checkpoint_path is None:
        # Try final_params first, then latest checkpoint
        final_path = base_dir / 'final_params'
        if final_path.exists():
            checkpoint_path = final_path
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
    
    checkpointer = ocp.StandardCheckpointer()
    try:
        params = checkpointer.restore(checkpoint_path, init_params)
    except Exception:
        params = checkpointer.restore(checkpoint_path)
    checkpointer.close()
    
    if isinstance(params, dict) and 'params' not in params:
        params = {'params': params}
    return params, model


def get_initial_inference(params, model, observation):
    """Run initial inference: observation -> (state, policy, value)."""
    state, policy, value = model.apply(
        params, observation, method=model.initial_inference
    )
    return state, policy, value


def encode_observation(state, player):
    """Encode backgammon state into observation vector."""
    import numpy as np
    from backgammon_engine import _to_canonical
    
    canonical = _to_canonical(state, player)
    return canonical.astype(np.float32) / 15.0  # Normalize


if __name__ == '__main__':
    params, model = load_agent4()
    dummy_obs = jnp.zeros((1, 28))
    state, policy, value = get_initial_inference(params, model, dummy_obs)
    print(f"Loaded Agent 4: value={float(value[0]):.4f}, policy_shape={policy.shape}")
