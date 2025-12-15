import jax
import jax.numpy as jnp
from jax.tree_util import tree_map
import optax
import numpy as np
from backgammon_engine import (
    _new_game, _roll_dice, _apply_move, _reward,
    _vectorized_new_game, _vectorized_roll_dice, 
    _vectorized_apply_move, _vectorized_2_ply_search,
    _to_canonical,
    W_BAR, B_BAR, W_OFF, B_OFF, NUM_POINTS, NUM_CHECKERS
)
from backgammon_value_net import BackgammonValueNet, BOARD_LENGTH, CONV_INPUT_CHANNELS, AUX_INPUT_SIZE
import orbax.checkpoint as ocp
import pathlib

# Agent 2: TD(lambda) with Neural Network
# Uses 1D ConvNet + ResNets, eligibility traces, and 2-ply search

LEARNING_RATE = 1e-4
LAMBDA = 0.7
GAMMA = 1.0
BATCH_SIZE = 256
NUM_ITERATIONS = 50000
CHECKPOINT_DIR = './checkpoints/agent2'

def encode_state(state, player):
    """
    Encode a single state into 15 feature planes (24, 15) + 6 aux features.
    
    Feature planes (15 total):
    - Plane 0: Empty (shared, binary)
    - Planes 1-7: My checkers (blot, made, builder, basic anchor, deep anchor, permanent anchor, overflow)
    - Planes 8-14: Opponent checkers (same pattern)
    
    Auxiliary features (6 total):
    - My bar active, my bar scale, my borne-off
    - Opp bar active, opp bar scale, opp borne-off
    """
    canonical_state = _to_canonical(state, player)
    
    planes = np.zeros((BOARD_LENGTH, CONV_INPUT_CHANNELS), dtype=np.float32)
    
    for point in range(1, NUM_POINTS + 1):
        my_count = max(0, canonical_state[point])
        opp_count = max(0, -canonical_state[point])
        
        # Plane 0: Empty (shared)
        if canonical_state[point] == 0:
            planes[point - 1, 0] = 1.0
        
        # My checkers (planes 1-7)
        if my_count == 1:
            planes[point - 1, 1] = 1.0  # Blot
        elif my_count == 2:
            planes[point - 1, 2] = 1.0  # Made
        elif my_count == 3:
            planes[point - 1, 3] = 1.0  # Builder
        elif my_count == 4:
            planes[point - 1, 4] = 1.0  # Basic Anchor
        elif my_count == 5:
            planes[point - 1, 5] = 1.0  # Deep Anchor
        elif my_count == 6:
            planes[point - 1, 6] = 1.0  # Permanent Anchor
        elif my_count > 6:
            planes[point - 1, 7] = (my_count - 6) / 9.0  # Overflow (max 9)
        
        # Opponent checkers (planes 8-14)
        if opp_count == 1:
            planes[point - 1, 8] = 1.0
        elif opp_count == 2:
            planes[point - 1, 9] = 1.0
        elif opp_count == 3:
            planes[point - 1, 10] = 1.0
        elif opp_count == 4:
            planes[point - 1, 11] = 1.0
        elif opp_count == 5:
            planes[point - 1, 12] = 1.0
        elif opp_count == 6:
            planes[point - 1, 13] = 1.0
        elif opp_count > 6:
            planes[point - 1, 14] = (opp_count - 6) / 9.0
    
    # Auxiliary features
    aux = np.array([
        1.0 if canonical_state[W_BAR] > 0 else 0.0,
        canonical_state[W_BAR] / 15.0,
        canonical_state[W_OFF] / 15.0,
        1.0 if canonical_state[B_BAR] < 0 else 0.0,
        abs(canonical_state[B_BAR]) / 15.0,
        abs(canonical_state[B_OFF]) / 15.0
    ], dtype=np.float32)
    
    return planes, aux

def batch_encode_states(states, players):
    """Encode a batch of states into feature planes and aux features."""
    batch_size = len(states)
    planes_batch = np.zeros((batch_size, BOARD_LENGTH, CONV_INPUT_CHANNELS), dtype=np.float32)
    aux_batch = np.zeros((batch_size, AUX_INPUT_SIZE), dtype=np.float32)
    
    for i in range(batch_size):
        planes_batch[i], aux_batch[i] = encode_state(states[i], players[i])
    
    return jnp.array(planes_batch), jnp.array(aux_batch)

def create_batch_value_function(model, params):
    """
    Create a batch value function for 2-ply search.
    
    The 2-ply search expects a function that takes a list of states
    and returns their values from the current player's perspective.
    """
    def batch_value_fn(states):
        # Convert states to feature representation
        batch_size = len(states)
        planes = np.zeros((batch_size, BOARD_LENGTH, CONV_INPUT_CHANNELS), dtype=np.float32)
        aux = np.zeros((batch_size, AUX_INPUT_SIZE), dtype=np.float32)
        
        for i in range(batch_size):
            state = np.array(states[i], dtype=np.int8)
            planes[i], aux[i] = encode_state(state, 1)
        
        # Evaluate with neural network
        planes_jax = jnp.array(planes)
        aux_jax = jnp.array(aux)
        values = model.apply({'params': params}, planes_jax, aux_jax)
        
        # Scale from [-1, 1] to [-3, 3]
        values = values * 3.0
        
        return np.array(values.flatten(), dtype=np.float32)
    
    return batch_value_fn

def loss_fn(params, model, planes, aux, targets):
    """
    Compute MSE loss between predicted values and TD targets.
    """
    predictions = model.apply({'params': params}, planes, aux)
    predictions = predictions * 3.0  # Scale to [-3, 3]
    
    td_errors = targets - predictions.flatten()
    loss = jnp.mean(jnp.square(td_errors))
    
    return loss

def td_lambda_update(params, opt_state, optimizer, traces, grads, td_errors, lambda_param, gamma):
    """
    Classical TD(lambda) update.
    
    z_t = γλ z_{t-1} + g_t
    w_{t+1} = w_t + α·δ_t·z_t
    """
    # Update traces: z_t = γλ z_{t-1} + g_t
    new_traces = tree_map(
        lambda z, g: gamma * lambda_param * z + g,
        traces,
        grads
    )
    
    # Scale traces by TD error for weight update
    scaled_traces = tree_map(lambda z: td_errors * z, new_traces)
    
    # Apply optimizer update using scaled traces
    updates, new_opt_state = optimizer.update(scaled_traces, opt_state, params)
    new_params = optax.apply_updates(params, updates)
    
    return new_params, new_opt_state, new_traces

def train_agent2(batch_size=BATCH_SIZE, num_iterations=NUM_ITERATIONS, 
                 learning_rate=LEARNING_RATE, lambda_param=LAMBDA,
                 verbose_every=100, checkpoint_every=5000):
    """
    Train Agent 2 using TD(lambda) with neural network and eligibility traces.
    """
    print(f"Training Agent 2 (TD(λ) Neural Network)")
    print(f"Batch size: {batch_size}")
    print(f"Iterations: {num_iterations}")
    print(f"Learning rate: {learning_rate}, λ: {lambda_param}, γ: {GAMMA}")
    print("-" * 60)
    
    # Initialize model
    model = BackgammonValueNet()
    rng_key = jax.random.key(0)
    
    dummy_planes = jnp.zeros((1, BOARD_LENGTH, CONV_INPUT_CHANNELS))
    dummy_aux = jnp.zeros((1, AUX_INPUT_SIZE))
    
    init_variables = model.init(rng_key, dummy_planes, dummy_aux)
    params = init_variables['params']
    
    # Initialize optimizer
    optimizer = optax.adam(learning_rate=learning_rate)
    opt_state = optimizer.init(params)
    
    # Initialize eligibility traces (same structure as params, all zeros)
    traces = tree_map(lambda p: jnp.zeros_like(p), params)
    
    # Initialize games
    states, players, dices = _vectorized_new_game(batch_size)
    
    total_games = 0
    white_wins = 0
    black_wins = 0
    
    # Setup checkpointing
    checkpoint_path = pathlib.Path(CHECKPOINT_DIR)
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    checkpointer = ocp.StandardCheckpointer()
    
    for iteration in range(num_iterations):
        # Store current states for TD update
        prev_states = states.copy()
        prev_players = players.copy()
        
        # Encode previous states
        prev_planes, prev_aux = batch_encode_states(prev_states, prev_players)
        
        # Select moves using 2-ply search
        batch_value_fn = create_batch_value_function(model, params)
        moves = _vectorized_2_ply_search(states, players, dices, batch_value_fn)
        
        # Apply moves
        new_states = _vectorized_apply_move(states, players, moves)
        
        # Compute rewards
        rewards = np.zeros(batch_size, dtype=np.float32)
        game_over = np.zeros(batch_size, dtype=bool)
        
        for i in range(batch_size):
            reward = _reward(new_states[i], players[i])
            rewards[i] = reward
            if reward != 0:
                game_over[i] = True
                total_games += 1
                if reward > 0:
                    white_wins += 1
                else:
                    black_wins += 1
        
        # Compute next state values (0 for terminal states)
        next_values = np.zeros(batch_size, dtype=np.float32)
        for i in range(batch_size):
            if not game_over[i]:
                # encode_state already does canonicalization internally
                next_planes, next_aux = encode_state(new_states[i], players[i])
                next_planes_jax = jnp.array(next_planes[np.newaxis, :, :])
                next_aux_jax = jnp.array(next_aux[np.newaxis, :])
                next_value = model.apply({'params': params}, next_planes_jax, next_aux_jax)
                next_values[i] = float(next_value[0, 0]) * 3.0
        
        # Compute current state values
        prev_values_pred = model.apply({'params': params}, prev_planes, prev_aux)
        prev_values_pred = (prev_values_pred * 3.0).flatten()
        prev_values_pred = np.array(prev_values_pred, dtype=np.float32)
        
        # Compute TD errors: δ = r + γ·V(s') - V(s)
        td_errors = rewards + GAMMA * next_values - prev_values_pred
        
        # Average TD error across batch
        avg_td_error = float(np.mean(td_errors))
        
        # Compute gradients
        td_targets = rewards + GAMMA * next_values
        td_targets_jax = jnp.array(td_targets)
        loss, grads = jax.value_and_grad(loss_fn)(params, model, prev_planes, prev_aux, td_targets_jax)
        
        # Perform Classical TD(lambda) update
        params, opt_state, traces = td_lambda_update(
            params, opt_state, optimizer, traces, grads, avg_td_error, lambda_param, GAMMA
        )
        
        # Reset traces for finished games
        if np.any(game_over):
            decay_factor = 1.0 - (np.sum(game_over) / batch_size)
            traces = tree_map(lambda z: z * decay_factor, traces)
        
        loss = float(loss)
        
        # Update states and switch players for continuing games
        states = new_states
        players = -players
        dices = _vectorized_roll_dice(batch_size)
        
        # Reset finished games (after updating, so they don't get overwritten)
        for i in range(batch_size):
            if game_over[i]:
                new_state, new_player, new_dice = _vectorized_new_game(1)
                states[i] = new_state[0]
                players[i] = new_player[0]
                dices[i] = new_dice[0]
        
        # Progress report
        if (iteration + 1) % verbose_every == 0:
            if total_games > 0:
                white_win_rate = white_wins / total_games * 100
                print(f"Iter {iteration + 1}/{num_iterations} | Loss: {loss:.6f} | "
                      f"Games: {total_games} | White: {white_win_rate:.1f}% "
                      f"({white_wins}W-{black_wins}L)")
                white_wins = 0
                black_wins = 0
                total_games = 0
            else:
                print(f"Iter {iteration + 1}/{num_iterations} | Loss: {loss:.6f} | "
                      f"No games finished yet")
        
        # Save checkpoint
        if (iteration + 1) % checkpoint_every == 0:
            checkpoint_name = f"checkpoint_{iteration + 1}"
            checkpointer.save(
                checkpoint_path / checkpoint_name,
                params,
                force=True
            )
            print(f"Saved checkpoint: {checkpoint_name}")
    
    print("-" * 60)
    print("Training complete!")
    
    # Save final checkpoint
    checkpointer.save(checkpoint_path / "final", params, force=True)
    print(f"Final checkpoint saved to {checkpoint_path / 'final'}")
    
    checkpointer.close()
    
    return params

if __name__ == "__main__":
    params = train_agent2(
        batch_size=256,
        num_iterations=50000,
        learning_rate=1e-4,
        lambda_param=0.7,
        verbose_every=100,
        checkpoint_every=5000
    )
