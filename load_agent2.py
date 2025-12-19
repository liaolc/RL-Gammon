"""Load Agent 2 (TD-lambda) weights."""

import jax
import jax.numpy as jnp
import orbax.checkpoint as ocp
import pathlib

from backgammon_value_net import BackgammonValueNet, BOARD_LENGTH, CONV_INPUT_CHANNELS, AUX_INPUT_SIZE


def load_agent2(checkpoint_path):
    """Load model and weights. Returns (params, model)."""
    model = BackgammonValueNet()
    rng_key = jax.random.key(0)
    dummy_planes = jnp.zeros((1, BOARD_LENGTH, CONV_INPUT_CHANNELS))
    dummy_aux = jnp.zeros((1, AUX_INPUT_SIZE))
    init_params = model.init(rng_key, dummy_planes, dummy_aux)['params']
    
    checkpoint_path = pathlib.Path(checkpoint_path).resolve()  # Convert to absolute
    checkpointer = ocp.StandardCheckpointer()
    params = checkpointer.restore(checkpoint_path, init_params)
    checkpointer.close()
    
    params = {'params': params}
    return params, model


def get_value(params, model, board_features, aux_features):
    """Get value estimate for a state."""
    value = model.apply(params, board_features, aux_features)
    return value.squeeze(-1)


if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print("Usage: python load_agent2.py <checkpoint_path>")
        sys.exit(1)
    params, model = load_agent2(sys.argv[1])
    dummy_board = jnp.zeros((1, BOARD_LENGTH, CONV_INPUT_CHANNELS))
    dummy_aux = jnp.zeros((1, AUX_INPUT_SIZE))
    value = get_value(params, model, dummy_board, dummy_aux)
    print(f"Loaded Agent 2: value={float(value[0]):.4f}")
