import jax
import jax.numpy as jnp
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
from datetime import datetime

LEARNING_RATE = 1e-4
LAMBDA = 0.7
GAMMA = 1.0
BATCH_SIZE = 64
NUM_ITERATIONS = 50000
CHECKPOINT_DIR = '/scratch/liaolc/RL-Gammon/checkpoints/agent2'

def encode_state(state, player):
    """Encode state into 15 feature planes (24, 15) + 6 aux features."""
    canonical_state = _to_canonical(state, player)
    planes = np.zeros((BOARD_LENGTH, CONV_INPUT_CHANNELS), dtype=np.float32)
    
    # Encode each point: plane 0 = empty, planes 1-7 = my checkers, planes 8-14 = opponent
    for point in range(1, NUM_POINTS + 1):
        my_count = max(0, canonical_state[point])
        opp_count = max(0, -canonical_state[point])
        
        if canonical_state[point] == 0:
            planes[point - 1, 0] = 1.0
        if my_count == 1:
            planes[point - 1, 1] = 1.0
        elif my_count == 2:
            planes[point - 1, 2] = 1.0
        elif my_count == 3:
            planes[point - 1, 3] = 1.0
        elif my_count == 4:
            planes[point - 1, 4] = 1.0
        elif my_count == 5:
            planes[point - 1, 5] = 1.0
        elif my_count == 6:
            planes[point - 1, 6] = 1.0
        elif my_count > 6:
            planes[point - 1, 7] = (my_count - 6) / 9.0
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
    """Create batch value function for 2-ply search."""
    def batch_value_fn(states):
        batch_size = len(states)
        planes = np.zeros((batch_size, BOARD_LENGTH, CONV_INPUT_CHANNELS), dtype=np.float32)
        aux = np.zeros((batch_size, AUX_INPUT_SIZE), dtype=np.float32)
        
        for i in range(batch_size):
            state = np.array(states[i], dtype=np.int8)
            planes[i], aux[i] = encode_state(state, 1)
        
        planes_jax = jnp.array(planes)
        aux_jax = jnp.array(aux)
        values = model.apply({'params': params}, planes_jax, aux_jax)
        values = values * 3.0
        
        return np.array(values.flatten(), dtype=np.float32)
    
    return batch_value_fn

def compute_value_gradients(params, model, planes, aux):
    """Compute per-sample ∇V̂(S_t, w_t) for Classical TD(λ)."""
    def single_value_fn(p, plane, aux_single):
        pred = model.apply({'params': p}, plane[None, ...], aux_single[None, ...])
        return (pred * 3.0)[0, 0]
    
    # Compute gradient for each game in the batch
    batched_grad = jax.vmap(
        lambda pl, au: jax.grad(lambda p: single_value_fn(p, pl, au))(params),
        in_axes=(0, 0)
    )
    grads = batched_grad(planes, aux)
    return grads

def loss_fn(params, model, planes, aux, targets):
    """Compute MSE loss for monitoring."""
    predictions = model.apply({'params': params}, planes, aux)
    predictions = predictions * 3.0
    td_errors = targets - predictions.flatten()
    loss = jnp.mean(jnp.square(td_errors))
    
    return loss

def load_checkpoint(checkpoint_name):
    """Load a checkpoint and return the model and params.

    Args:
        checkpoint_name: Name of checkpoint (e.g., 'checkpoint_30') or full path

    Returns:
        tuple: (model, params)
    """
    model = BackgammonValueNet()
    rng_key = jax.random.key(0)
    dummy_planes = jnp.zeros((1, BOARD_LENGTH, CONV_INPUT_CHANNELS))
    dummy_aux = jnp.zeros((1, AUX_INPUT_SIZE))
    params = model.init(rng_key, dummy_planes, dummy_aux)['params']

    checkpoint_path = pathlib.Path(CHECKPOINT_DIR)
    checkpointer = ocp.StandardCheckpointer()

    # Handle both relative and absolute paths
    if not pathlib.Path(checkpoint_name).is_absolute():
        restore_path = checkpoint_path / checkpoint_name
    else:
        restore_path = pathlib.Path(checkpoint_name)

    params = checkpointer.restore(restore_path)
    return model, params
def true_online_td_lambda_update(params, traces, grads, values, next_values, rewards, 
                                 alpha, lambda_param, gamma):
    """
    Implements the specific True Online TD(lambda) updates from the PDF.
    
    Args:
        values: V(S_t)
        next_values: V(S_{t+1})
        grads: ∇V(S_t)
    """
    # 1. Calculate TD Error (delta)
    # δ_t = R_{t+1} + γV(S_{t+1}) - V(S_t)
    td_errors = rewards + gamma * next_values - values

    # 2. Update Eligibility Traces (z_t)
    # z_t = γλ z_{t-1} + g_t - αγλ(z_{t-1} · g_t)g_t
    # We need the dot product (z_{t-1} · g_t) for each sample in batch
    def update_trace_single(z, g):
        dot_prod = jnp.sum(z * g) # Dot product of weights and gradients
        term1 = gamma * lambda_param * z
        term2 = g
        term3 = alpha * gamma * lambda_param * dot_prod * g
        return term1 + term2 - term3
    
    # Apply over batch
    new_traces = jax.tree.map(lambda z, g: jax.vmap(update_trace_single)(z, g), traces, grads)

    # 3. Calculate Weight Update (Δw)
    # w_{t+1} = w_t + α δ_t z_t + α(V(S_t) - g_t · w_t)(z_t - g_t)
    # Note: For non-linear V, (V - g·w) is non-zero.
    
    # We need to compute (g_t · w_t). 
    # This is tricky in JAX with pytrees. We sum dot products across all leaves.
    def compute_gw_dot(g_tree, w_tree):
        # Flatten trees to vectors and dot them
        leaves_g, _ = jax.tree_util.tree_flatten(g_tree)
        leaves_w, _ = jax.tree_util.tree_flatten(w_tree)
        total = 0.
        for g, w in zip(leaves_g, leaves_w):
             # w is (params), g is (batch, params). Broadcast w.
             total += jnp.sum(g * w[None, ...], axis=list(range(1, g.ndim)))
        return total

    gw_dot = compute_gw_dot(grads, params)
    
    # Correction scalar: α(V(S_t) - g_t · w_t)
    correction_scalar = alpha * (values - gw_dot)
    
    def compute_weight_delta(z, g, delta, corr):
        term1 = alpha * delta * z
        term2 = corr * (z - g)
        return term1 + term2

    # Note: delta and correction_scalar are vectors of size (batch,)
    # reshape them to broadcast against z and g
    updates = jax.tree.map(
        lambda z, g: jax.vmap(compute_weight_delta)(
            z, g, td_errors, correction_scalar
        ),
        new_traces, grads
    )
    
    final_updates = jax.tree.map(lambda u: jnp.sum(u, axis=0), updates)
    new_params = optax.apply_updates(params, final_updates) # Or just params + final_updates

    return new_params, new_traces

def train_agent2(batch_size=BATCH_SIZE, num_iterations=NUM_ITERATIONS,
                 learning_rate=LEARNING_RATE, lambda_param=LAMBDA,
                 verbose_every=100, checkpoint_every=5000, resume_from=None):
    """Train Agent 2 using TD(λ) with neural network.

    Args:
        batch_size: Number of parallel games
        num_iterations: Number of training iterations
        learning_rate: SGD learning rate
        lambda_param: TD(λ) parameter
        verbose_every: Print progress every N iterations
        checkpoint_every: Save checkpoint every N iterations
        resume_from: Path to checkpoint to resume from (e.g., 'checkpoint_30' or full path)
    """
    print(f"Training Agent 2: batch={batch_size}, iters={num_iterations}, lr={learning_rate}, λ={lambda_param}")

    # Create unique run directory with timestamp
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_checkpoint_dir = pathlib.Path(CHECKPOINT_DIR) / f"run_{run_id}"

    # Initialize model and optimizer
    model = BackgammonValueNet()
    rng_key = jax.random.key(0)
    dummy_planes = jnp.zeros((1, BOARD_LENGTH, CONV_INPUT_CHANNELS))
    dummy_aux = jnp.zeros((1, AUX_INPUT_SIZE))
    params = model.init(rng_key, dummy_planes, dummy_aux)['params']

    # Load checkpoint if resuming
    if resume_from is not None:
        checkpoint_path = pathlib.Path(CHECKPOINT_DIR)
        checkpointer = ocp.StandardCheckpointer()

        # Handle both relative and absolute paths
        if not pathlib.Path(resume_from).is_absolute():
            resume_path = checkpoint_path / resume_from
        else:
            resume_path = pathlib.Path(resume_from)

        print(f"Loading checkpoint from {resume_path}")
        params = checkpointer.restore(resume_path)
        print("Checkpoint loaded successfully")

    # optimizer = optax.sgd(learning_rate=learning_rate)
    # opt_state = optimizer.init(params)
    
    # Initialize per-game eligibility traces
    traces = jax.tree.map(lambda p: jnp.zeros((batch_size,) + p.shape, dtype=p.dtype), params)
    
    states, players, dices = _vectorized_new_game(batch_size)
    total_games = white_wins = black_wins = 0
    
    checkpoint_path = pathlib.Path(CHECKPOINT_DIR)
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    checkpointer = ocp.StandardCheckpointer()
    
    for iteration in range(num_iterations):
        # Store current states and encode
        prev_states = states.copy()
        prev_players = players.copy()
        prev_planes, prev_aux = batch_encode_states(prev_states, prev_players)
        
        # Select moves using 2-ply search and apply
        batch_value_fn = create_batch_value_function(model, params)
        moves = _vectorized_2_ply_search(states, players, dices, batch_value_fn)
        new_states = _vectorized_apply_move(states, players, moves)
        
        # Compute rewards and track game endings
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
        
        # Compute next state values in one batched forward pass (0 for terminal states)
        next_planes, next_aux = batch_encode_states(new_states, players)
        next_pred = model.apply({'params': params}, next_planes, next_aux)
        next_values = (next_pred * 3.0).flatten()
        next_values = np.array(next_values, dtype=np.float32)
        next_values[game_over] = 0.0
        
        prev_values_pred = model.apply({'params': params}, prev_planes, prev_aux)
        prev_values_pred = (prev_values_pred * 3.0).flatten()
        prev_values_pred = np.array(prev_values_pred, dtype=np.float32)
        
        # TD update
        grads = compute_value_gradients(params, model, prev_planes, prev_aux)
        params, traces = true_online_td_lambda_update(
            params, 
            traces, 
            grads, 
            jnp.array(prev_values_pred),  
            jnp.array(next_values),      
            jnp.array(rewards), 
            learning_rate, 
            lambda_param, 
            GAMMA
        )
        
        # Reset traces for finished games
        if np.any(game_over):
            game_over_jax = jnp.array(game_over)
            def reset_finished_traces(z):
                mask = (1.0 - game_over_jax).reshape((batch_size,) + (1,) * (z.ndim - 1))
                return z * mask
            traces = jax.tree.map(reset_finished_traces, traces)
        
        td_targets = rewards + GAMMA * next_values
        td_targets_jax = jnp.array(td_targets)
        loss = float(loss_fn(params, model, prev_planes, prev_aux, td_targets_jax))
        
        # Update states and switch players
        states = new_states
        players = -players
        dices = _vectorized_roll_dice(batch_size)
        
        # Reset finished games
        for i in range(batch_size):
            if game_over[i]:
                new_state, new_player, new_dice = _vectorized_new_game(1)
                states[i] = new_state[0]
                players[i] = new_player[0]
                dices[i] = new_dice[0]
        
        # Progress reporting
        if (iteration + 1) % verbose_every == 0:
            if total_games > 0:
                white_win_rate = white_wins / total_games * 100
                print(f"[{iteration + 1}/{num_iterations}] Loss: {loss:.6f} | Games: {total_games} | White: {white_win_rate:.1f}% ({white_wins}W-{black_wins}L)")
                white_wins = black_wins = total_games = 0
            else:
                print(f"[{iteration + 1}/{num_iterations}] Loss: {loss:.6f}")
        
        if (iteration + 1) % checkpoint_every == 0 or iteration == 1:
            checkpointer.save(checkpoint_path / f"checkpoint_{iteration + 1}", params, force=True)
            print(f"Checkpoint saved: {iteration + 1}")
    
    checkpointer.save(checkpoint_path / "final", params, force=True)
    print("Training complete")
    checkpointer.close()
    
    return params

if __name__ == "__main__":
    params = train_agent2(
        batch_size=16,
        num_iterations=300,
        learning_rate=1e-4,
        lambda_param=0.7,
        verbose_every=1,
        checkpoint_every=30,
        #resume_from="/scratch/liaolc/RL-Gammon/checkpoints/agent2/checkpoint_30"
    )