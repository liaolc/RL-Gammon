import jax
import jax.numpy as jnp
from jax import random
import numpy as np
import optax
from flax.training import train_state
import orbax.checkpoint as ocp
import pathlib
from numba import njit, prange, types
from numba.typed import List
from typing import Tuple, List as PyList, Dict

from backgammon_engine import (
    _new_game, _roll_dice, _actions, _apply_move, _reward,
    _to_canonical, _vectorized_new_game, _vectorized_roll_dice,
    _vectorized_apply_move, _move_afterstate_dict, _state_to_tuple,
    W_BAR, B_BAR, W_OFF, B_OFF, NUM_POINTS, NUM_CHECKERS, HOME_BOARD_SIZE,
    State, Action, NUM_SORTED_ROLLS
)
from backgammon_ppo_net import BackgammonPPONet, BOARD_LENGTH, CONV_INPUT_CHANNELS, AUX_INPUT_SIZE

LEARNING_RATE = 3e-4
GAMMA = 1.0              # Discount factor
LAMBDA = 0.95            # GAE lambda
EPSILON_CLIP = 0.2       # PPO clipping
C1 = 0.5                 # Value loss coefficient
C2 = 0.01                # Entropy bonus coefficient
BATCH_SIZE = 512
BUFFER_SIZE = 2048
NUM_EPOCHS = 4
MINIBATCH_SIZE = 256
TOP_K_MOVES = 5

@njit
def encode_state_features(state, player):
    """Encode state into 15 feature planes + 6 aux features."""
    board_features = np.zeros((BOARD_LENGTH, CONV_INPUT_CHANNELS), dtype=np.float32)
    canonical_state = _to_canonical(state, player)
    
    for point_idx in range(1, NUM_POINTS + 1):
        checkers = canonical_state[point_idx]
        
        if checkers == 0:
            board_features[point_idx - 1, 0] = 1.0
        
        # Planes 1-7: Player checkers (Blot, Made, Builder, Anchors, Overflow)
        if checkers > 0:
            board_features[point_idx - 1, 1] = 1.0 if checkers == 1 else 0.0
            board_features[point_idx - 1, 2] = 1.0 if checkers == 2 else 0.0
            board_features[point_idx - 1, 3] = 1.0 if checkers == 3 else 0.0
            board_features[point_idx - 1, 4] = 1.0 if checkers == 4 else 0.0
            board_features[point_idx - 1, 5] = 1.0 if checkers == 5 else 0.0
            board_features[point_idx - 1, 6] = 1.0 if checkers == 6 else 0.0
            board_features[point_idx - 1, 7] = max(0.0, (checkers - 6) / 9.0)
        
        # Planes 8-14: Opponent checkers (same structure)
        if checkers < 0:
            abs_checkers = abs(checkers)
            board_features[point_idx - 1, 8] = 1.0 if abs_checkers == 1 else 0.0
            board_features[point_idx - 1, 9] = 1.0 if abs_checkers == 2 else 0.0
            board_features[point_idx - 1, 10] = 1.0 if abs_checkers == 3 else 0.0
            board_features[point_idx - 1, 11] = 1.0 if abs_checkers == 4 else 0.0
            board_features[point_idx - 1, 12] = 1.0 if abs_checkers == 5 else 0.0
            board_features[point_idx - 1, 13] = 1.0 if abs_checkers == 6 else 0.0
            board_features[point_idx - 1, 14] = max(0.0, (abs_checkers - 6) / 9.0)
    
    aux_features = np.zeros(AUX_INPUT_SIZE, dtype=np.float32)
    player_bar_count = canonical_state[W_BAR]
    aux_features[0] = 1.0 if player_bar_count > 0 else 0.0
    aux_features[1] = player_bar_count / 15.0
    opponent_bar_count = abs(canonical_state[B_BAR])
    aux_features[2] = 1.0 if opponent_bar_count > 0 else 0.0
    aux_features[3] = opponent_bar_count / 15.0
    aux_features[4] = canonical_state[W_OFF] / 15.0
    aux_features[5] = abs(canonical_state[B_OFF]) / 15.0
    
    return board_features, aux_features

@njit(parallel=True)
def batch_encode_states(states, players):
    batch_size = len(states)
    board_batch = np.zeros((batch_size, BOARD_LENGTH, CONV_INPUT_CHANNELS), dtype=np.float32)
    aux_batch = np.zeros((batch_size, AUX_INPUT_SIZE), dtype=np.float32)
    for i in prange(batch_size):
        board_batch[i], aux_batch[i] = encode_state_features(states[i], players[i])
    return board_batch, aux_batch

def encode_move_to_canonical(move, state, player):
    """Encode move to canonical (source, dest) pairs."""
    encoded_submoves = []
    for from_point, roll in move:
        if player == 1:
            source = 0 if from_point == W_BAR else from_point
            dest = from_point + roll
            if dest > NUM_POINTS:
                dest = 25
        else:
            source = 0 if from_point == B_BAR else (NUM_POINTS + 1 - from_point)
            dest = source + roll
            if dest > NUM_POINTS:
                dest = 25
        source = min(25, max(0, source))
        dest = min(25, max(0, dest))
        encoded_submoves.append((source, dest))
    return encoded_submoves

def get_first_submove_logits(policy_logits, legal_moves, state, player):
    """Get logits for the first submove of each legal move."""
    first_submove_logits = []
    
    for move in legal_moves:
        if len(move) == 0:
            first_submove_logits.append(-1e9)
            continue
        
        # Get first submove
        from_point, roll = int(move[0][0]), int(move[0][1])
        
        # Convert to canonical coordinates
        if player == 1:
            source = 0 if from_point == W_BAR else from_point
            dest = from_point + roll
            if dest > NUM_POINTS:
                dest = 25
        else:
            source = 0 if from_point == B_BAR else (NUM_POINTS + 1 - from_point)
            dest = source + roll
            if dest > NUM_POINTS:
                dest = 25
        
        source = min(25, max(0, source))
        dest = min(25, max(0, dest))
        idx = source * 25 + dest
        
        logit = policy_logits[idx] if idx < len(policy_logits) else -1e9
        first_submove_logits.append(float(logit))
    
    return np.array(first_submove_logits, dtype=np.float32)

def select_top_k_moves_by_first_submove(policy_logits, legal_moves, state, player, k):
    """Select moves by grouping by first submove until >= k moves accumulated."""
    if len(legal_moves) <= k:
        return np.arange(len(legal_moves))
    
    submove_to_move_indices = {}
    submove_logits = {}
    
    for move_idx, move in enumerate(legal_moves):
        if len(move) == 0:
            continue
        
        from_point, roll = int(move[0][0]), int(move[0][1])
        
        # Convert to canonical coordinates
        if player == 1:
            source = 0 if from_point == W_BAR else from_point
            dest = from_point + roll
            if dest > NUM_POINTS:
                dest = 25
        else:
            source = 0 if from_point == B_BAR else (NUM_POINTS + 1 - from_point)
            dest = source + roll
            if dest > NUM_POINTS:
                dest = 25
        
        source = min(25, max(0, source))
        dest = min(25, max(0, dest))
        submove_idx = source * 25 + dest
        
        if submove_idx not in submove_to_move_indices:
            submove_to_move_indices[submove_idx] = []
            submove_logits[submove_idx] = policy_logits[submove_idx] if submove_idx < 625 else -1e9
        
        submove_to_move_indices[submove_idx].append(move_idx)
    
    sorted_submoves = sorted(submove_logits.keys(), key=lambda x: submove_logits[x], reverse=True)
    selected_indices = []
    for submove_idx in sorted_submoves:
        selected_indices.extend(submove_to_move_indices[submove_idx])
        if len(selected_indices) >= k:
            break
    
    return np.array(selected_indices, dtype=np.int64)

@njit
def select_top_k_moves_by_logits(move_logits, k):
    if len(move_logits) <= k:
        return np.arange(len(move_logits))
    return np.argsort(move_logits)[-k:][::-1]

def sample_move_sequentially(state, player, dice, params, model, rng_key):
    """Sample move by sequentially sampling submoves with masking."""
    dice_list = [dice[0], dice[1]]
    if dice[0] == dice[1]:
        dice_list = [dice[0], dice[0], dice[0], dice[0]]
    
    current_state = state.copy()
    selected_submoves = []
    total_log_prob = 0.0
    
    for die_idx, die_value in enumerate(dice_list):
        legal_submoves = []
        bar_index = W_BAR if player == 1 else B_BAR
        if current_state[bar_index] * player > 0:
            from_point = bar_index
            from backgammon_engine import _is_move_legal, _get_target_index
            target = _get_target_index(from_point, die_value, player)
            if _is_move_legal(current_state, player, from_point, target):
                legal_submoves.append((from_point, die_value))
        else:
            for from_point in range(1, NUM_POINTS + 1):
                if current_state[from_point] * player > 0:
                    from backgammon_engine import _is_move_legal, _get_target_index
                    target = _get_target_index(from_point, die_value, player)
                    if _is_move_legal(current_state, player, from_point, target):
                        legal_submoves.append((from_point, die_value))
        
        if len(legal_submoves) == 0:
            break
        
        board_features, aux_features = encode_state_features(current_state, player)
        board_batch = jnp.array([board_features])
        aux_batch = jnp.array([aux_features])
        _, policy_logits = model.apply({'params': params}, board_batch, aux_batch)
        policy_logits = np.array(policy_logits[0])
        
        submove_logits = []
        for from_point, die_val in legal_submoves:
            from_point = int(from_point)
            die_val = int(die_val)
            # Convert to canonical coordinates
            if player == 1:
                source = 0 if from_point == W_BAR else from_point
                dest = from_point + die_val
                if dest > NUM_POINTS:
                    dest = 25
            else:
                source = 0 if from_point == B_BAR else (NUM_POINTS + 1 - from_point)
                dest = source + die_val
                if dest > NUM_POINTS:
                    dest = 25
            
            source = int(min(25, max(0, source)))
            dest = int(min(25, max(0, dest)))
            idx = source * 25 + dest
            submove_logits.append(policy_logits[idx] if idx < 625 else -1e9)
        
        submove_logits_jax = jnp.array(submove_logits)
        log_probs = jax.nn.log_softmax(submove_logits_jax)
        
        rng_key, subkey = random.split(rng_key)
        submove_idx = random.categorical(subkey, log_probs)
        
        selected_submove = legal_submoves[int(submove_idx)]
        total_log_prob += float(log_probs[submove_idx])
        selected_submoves.append(selected_submove)
        
        from backgammon_engine import _apply_sub_move, _get_target_index
        from_point, die_val = selected_submove
        target = _get_target_index(from_point, die_val, player)
        current_state = _apply_sub_move(current_state, player, from_point, target)
        
        if current_state is None:
            break
    
    move = List.empty_list(types.UniTuple(types.int8, 2))
    for from_point, roll in selected_submoves:
        move.append((np.int8(from_point), np.int8(roll)))
    
    encoded_move = encode_move_to_canonical(move, state, player)
    
    return move, total_log_prob, encoded_move

def pruned_2_ply_search(state, player, dice, params, model, rng_key=None, training=False):
    """2-ply search with policy pruning. Returns (move, value, log_prob, encoded_move)."""
    player_moves, player_afterstates = _actions(state, player, dice)
    
    if len(player_moves) == 0:
        no_op = List.empty_list(types.UniTuple(types.int8, 2))
        return no_op, 0.0, None, None
    
    board_features, aux_features = encode_state_features(state, player)
    board_batch = jnp.array([board_features])
    aux_batch = jnp.array([aux_features])
    value_pred, policy_logits = model.apply({'params': params}, board_batch, aux_batch)
    value_estimate = float(value_pred[0, 0])
    policy_logits = np.array(policy_logits[0])
    
    if training and rng_key is not None:
        selected_move, log_prob, encoded_move = sample_move_sequentially(
            state, player, dice, params, model, rng_key
        )
        return selected_move, value_estimate, log_prob, encoded_move
    
    top_k_indices = select_top_k_moves_by_first_submove(
        policy_logits, player_moves, state, player, TOP_K_MOVES
    )
    
    states_to_evaluate = []
    move_dice_to_states = []
    
    for move_idx in top_k_indices:
        player_afterstate = player_afterstates[move_idx]
        dice_to_states = []
        
        for r1 in range(1, 7):
            for r2 in range(1, r1 + 1):
                opponent_dice = np.array([r1, r2], dtype=np.int8)
                opponent_moves, opponent_afterstates_list = _actions(
                    player_afterstate, -player, opponent_dice
                )
                
                dice_state_indices = []
                
                if len(opponent_moves) > 0:
                    opp_board, opp_aux = encode_state_features(player_afterstate, -player)
                    opp_board_batch = jnp.array([opp_board])
                    opp_aux_batch = jnp.array([opp_aux])
                    _, opp_policy_logits = model.apply(
                        {'params': params}, opp_board_batch, opp_aux_batch
                    )
                    opp_policy_logits = np.array(opp_policy_logits[0])
                    
                    opp_move_logits = get_first_submove_logits(
                        opp_policy_logits, opponent_moves, player_afterstate, -player
                    )
                    opp_top_k = select_top_k_moves_by_logits(opp_move_logits, TOP_K_MOVES)
                    
                    for opp_idx in opp_top_k:
                        dice_state_indices.append(len(states_to_evaluate))
                        states_to_evaluate.append(opponent_afterstates_list[opp_idx])
                
                dice_to_states.append(dice_state_indices)
        
        move_dice_to_states.append(dice_to_states)
    
    if len(states_to_evaluate) > 0:
        eval_players = np.ones(len(states_to_evaluate), dtype=np.int8) * player
        board_batch, aux_batch = batch_encode_states(
            np.array(states_to_evaluate), eval_players
        )
        board_batch = jnp.array(board_batch)
        aux_batch = jnp.array(aux_batch)
        
        values, _ = model.apply({'params': params}, board_batch, aux_batch)
        values = np.array(values).flatten()
    else:
        values = np.array([])
    
    # E[V] = Σ P(dice) * min_{opp_move} V(afterstate)
    best_value = -np.inf
    best_move_idx = top_k_indices[0]
    
    for i, move_idx in enumerate(top_k_indices):
        expected_value = 0.0
        dice_idx = 0
        
        for r1 in range(1, 7):
            for r2 in range(1, r1 + 1):
                p_dice = 1.0 / 36.0 if r1 == r2 else 2.0 / 36.0
                state_indices = move_dice_to_states[i][dice_idx]
                
                if len(state_indices) == 0:
                    min_value = value_estimate
                else:
                    min_value = np.min(values[state_indices])
                
                expected_value += p_dice * min_value
                dice_idx += 1
        
        if expected_value > best_value:
            best_value = expected_value
            best_move_idx = move_idx
    
    return player_moves[best_move_idx], value_estimate, None, None

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.states = []
        self.players = []
        self.dices = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []
        
    def add(self, state, player, dice, action, log_prob, reward, value, done):
        self.states.append(state.copy())
        self.players.append(player)
        self.dices.append(dice.copy())
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)
    
    def get_batch(self):
        return {
            'states': np.array(self.states),
            'players': np.array(self.players),
            'dices': np.array(self.dices),
            'actions': self.actions,
            'log_probs': np.array(self.log_probs),
            'rewards': np.array(self.rewards),
            'values': np.array(self.values),
            'dones': np.array(self.dones)
        }
    
    def clear(self):
        self.states = []
        self.players = []
        self.dices = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []
    
    def __len__(self):
        return len(self.states)

def compute_gae(rewards, values, dones, gamma=GAMMA, lambda_=LAMBDA):
    """Compute GAE advantages and returns."""
    advantages = np.zeros_like(rewards, dtype=np.float32)
    last_advantage = 0.0
    
    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            next_value = 0.0
        else:
            next_value = values[t + 1] if not dones[t] else 0.0
        
        delta = rewards[t] + gamma * next_value - values[t]
        advantages[t] = delta + gamma * lambda_ * last_advantage * (1.0 - dones[t])
        last_advantage = advantages[t]
    
    returns = advantages + values
    
    return advantages, returns

def compute_action_log_probs(policy_logits, actions, old_log_probs):
    """Compute log probabilities for taken actions (JAX-compatible)."""
    batch_size = policy_logits.shape[0]
    
    # Pre-compute action indices outside JAX tracing
    action_indices = []
    for i in range(len(actions)):
        action = actions[i]
        if action is None or len(action) == 0:
            action_indices.append([])
        else:
            indices = []
            for source, dest in action:
                source, dest = int(source), int(dest)
                idx = source * 25 + dest
                if idx < 625:
                    indices.append(idx)
            action_indices.append(indices)
    
    # Compute log probs using JAX operations
    log_probs_list = []
    for i in range(batch_size):
        indices = action_indices[i]
        if len(indices) == 0:
            log_probs_list.append(old_log_probs[i])
        else:
            log_softmax_all = jax.nn.log_softmax(policy_logits[i])
            total = sum(log_softmax_all[idx] for idx in indices)
            log_probs_list.append(total)
    
    return jnp.stack(log_probs_list)

def compute_legal_move_mask(state, player, dice):
    """Create mask for legal submoves in policy grid."""
    mask = np.zeros(625, dtype=bool)
    legal_moves, _ = _actions(state, player, dice)
    
    for move in legal_moves:
        if len(move) > 0:
            from_point, roll = int(move[0][0]), int(move[0][1])
            
            if player == 1:
                source = 0 if from_point == W_BAR else from_point
                dest = from_point + roll
                if dest > NUM_POINTS:
                    dest = 25
            else:
                source = 0 if from_point == B_BAR else (NUM_POINTS + 1 - from_point)
                dest = source + roll
                if dest > NUM_POINTS:
                    dest = 25
            
            source = int(min(25, max(0, source)))
            dest = int(min(25, max(0, dest)))
            idx = source * 25 + dest
            if idx < 625:
                mask[idx] = True
    
    return mask

def compute_masked_entropy(policy_logits, states, players, dices):
    """Compute entropy over legal moves only (JAX-compatible)."""
    batch_size = policy_logits.shape[0]
    
    # Pre-compute masks outside JAX tracing
    masks = []
    for i in range(len(states)):
        mask = compute_legal_move_mask(states[i], players[i], dices[i])
        masks.append(mask)
    
    # Compute entropy using masking that works with JAX
    total_entropy = 0.0
    valid_count = 0
    
    for i in range(batch_size):
        mask = masks[i]
        if not np.any(mask):
            continue
        
        logits_i = policy_logits[i]
        # Mask illegal actions with large negative value
        masked_logits = jnp.where(jnp.array(mask), logits_i, -1e9)
        log_probs = jax.nn.log_softmax(masked_logits)
        probs = jax.nn.softmax(masked_logits)
        entropy_i = -jnp.sum(probs * log_probs)
        total_entropy = total_entropy + entropy_i
        valid_count += 1
    
    if valid_count == 0:
        return jnp.array(0.0)
    return total_entropy / valid_count

def ppo_loss(params, apply_fn, board_batch, aux_batch, actions, states, players, dices,
             old_log_probs, advantages, returns, epsilon=EPSILON_CLIP):
    """Compute PPO loss with clipped surrogate objective."""
    values_pred, policy_logits = apply_fn({'params': params}, board_batch, aux_batch)
    values_pred = values_pred.flatten()
    
    # PPO clipped surrogate objective
    new_log_probs = compute_action_log_probs(policy_logits, actions, old_log_probs)
    old_log_probs_jax = jnp.array(old_log_probs)
    ratio = jnp.exp(new_log_probs - old_log_probs_jax)
    
    advantages_jax = jnp.array(advantages)
    surr1 = ratio * advantages_jax
    surr2 = jnp.clip(ratio, 1.0 - epsilon, 1.0 + epsilon) * advantages_jax
    policy_loss = -jnp.mean(jnp.minimum(surr1, surr2))
    
    returns_jax = jnp.array(returns)
    value_loss = jnp.mean((values_pred - returns_jax) ** 2)
    
    entropy = compute_masked_entropy(policy_logits, states, players, dices)
    # L = L_policy + c1*L_value - c2*entropy
    total_loss = policy_loss + C1 * value_loss - C2 * entropy
    
    # Return JAX arrays for metrics - convert to float outside grad
    return total_loss, (policy_loss, value_loss, entropy, total_loss)

def train_step(state, board_batch, aux_batch, actions, states, players, dices,
               old_log_probs, advantages, returns):
    """Single training step."""
    def loss_fn(params):
        return ppo_loss(params, state.apply_fn, board_batch, aux_batch, actions,
                        states, players, dices, old_log_probs, advantages, returns)
    
    (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    state = state.apply_gradients(grads=grads)
    
    # Convert metrics to dict after grad computation
    policy_loss, value_loss, entropy, total_loss = aux
    metrics = {
        'policy_loss': float(policy_loss),
        'value_loss': float(value_loss),
        'entropy': float(entropy),
        'total_loss': float(total_loss)
    }
    
    return state, metrics

def train_ppo(num_iterations=10000, batch_size=BATCH_SIZE, buffer_size=BUFFER_SIZE,
              checkpoint_dir='/home/zhangdjr/Desktop/RL-Gammon/checkpoints/agent3', verbose_every=10):
    """Train Agent 3 using PPO."""
    print(f"Training Agent 3 (PPO) | batch={batch_size}, buffer={buffer_size}")
    print("-" * 60)
    
    model = BackgammonPPONet()
    rng = random.PRNGKey(0)
    rng, init_rng = random.split(rng)
    
    dummy_board = jnp.zeros((1, BOARD_LENGTH, CONV_INPUT_CHANNELS))
    dummy_aux = jnp.zeros((1, AUX_INPUT_SIZE))
    params = model.init(init_rng, dummy_board, dummy_aux)['params']
    
    optimizer = optax.adam(learning_rate=LEARNING_RATE)
    train_state_obj = train_state.TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=optimizer
    )
    
    buffer = ReplayBuffer(buffer_size)
    states, players, dices = _vectorized_new_game(batch_size)
    
    total_games = 0
    white_wins = 0
    black_wins = 0
    
    for iteration in range(num_iterations):
        for step in range(buffer_size // batch_size):
            rng, *step_rngs = random.split(rng, batch_size + 1)
            
            for i in range(batch_size):
                move, value, log_prob, encoded_move = pruned_2_ply_search(
                    states[i], players[i], dices[i],
                    train_state_obj.params, model, step_rngs[i], training=True
                )
                
                new_state = _apply_move(states[i], players[i], move)
                reward = _reward(new_state, players[i])
                done = (reward != 0)
                
                buffer.add(states[i], players[i], dices[i], encoded_move, log_prob, reward, value, done)
                
                if done:
                    total_games += 1
                    if reward > 0:
                        white_wins += 1
                    else:
                        black_wins += 1
                    new_player, new_dice, new_state = _new_game()
                    states[i] = new_state
                    players[i] = new_player
                    dices[i] = new_dice
                else:
                    states[i] = new_state
                    players[i] = -players[i]
                    dices[i] = _roll_dice()
        
        if len(buffer) >= buffer_size:
            batch_data = buffer.get_batch()
            advantages, returns = compute_gae(
                batch_data['rewards'], batch_data['values'], batch_data['dones']
            )
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            
            for epoch in range(NUM_EPOCHS):
                indices = np.random.permutation(len(buffer))
                for start_idx in range(0, len(buffer), MINIBATCH_SIZE):
                    end_idx = min(start_idx + MINIBATCH_SIZE, len(buffer))
                    mb_indices = indices[start_idx:end_idx]
                    
                    mb_states = batch_data['states'][mb_indices]
                    mb_players = batch_data['players'][mb_indices]
                    mb_dices = batch_data['dices'][mb_indices]
                    mb_actions = [batch_data['actions'][i] for i in mb_indices]
                    mb_old_log_probs = batch_data['log_probs'][mb_indices]
                    mb_advantages = advantages[mb_indices]
                    mb_returns = returns[mb_indices]
                    
                    # Encode batch BEFORE calling train_step (Numba can't run inside JAX grad)
                    mb_board, mb_aux = batch_encode_states(mb_states, mb_players)
                    mb_board = jnp.array(mb_board)
                    mb_aux = jnp.array(mb_aux)
                    
                    train_state_obj, metrics = train_step(
                        train_state_obj, mb_board, mb_aux, mb_actions,
                        mb_states, mb_players, mb_dices,
                        mb_old_log_probs, mb_advantages, mb_returns
                    )
            
            buffer.clear()
        
        if (iteration + 1) % verbose_every == 0:
            if total_games > 0:
                white_win_rate = white_wins / total_games * 100
                print(f"Iter {iteration + 1}/{num_iterations} | Games: {total_games} | "
                      f"White: {white_win_rate:.1f}% ({white_wins}W-{black_wins}L)")
                white_wins = 0
                black_wins = 0
                total_games = 0
        
        # Periodic checkpointing every 100 iterations
        if (iteration + 1) % 100 == 0:
            checkpoint_path = pathlib.Path(checkpoint_dir)
            checkpoint_path.mkdir(parents=True, exist_ok=True)
            checkpointer = ocp.StandardCheckpointer()
            checkpointer.save(checkpoint_path / f'checkpoint_{iteration + 1}', train_state_obj.params, force=True)
            checkpointer.close()
            print(f"Checkpoint saved at iteration {iteration + 1}")
    
    print("-" * 60)
    print("Training complete!")
    
    checkpoint_path = pathlib.Path(checkpoint_dir)
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    checkpointer = ocp.StandardCheckpointer()
    checkpointer.save(checkpoint_path / 'final_params', train_state_obj.params, force=True)
    checkpointer.close()
    print(f"Model saved to {checkpoint_path / 'final_params'}")
    
    return train_state_obj.params

def load_agent(checkpoint_path):
    """Load a trained PPO agent from checkpoint."""
    model = BackgammonPPONet()
    checkpointer = ocp.StandardCheckpointer()
    params = checkpointer.restore(pathlib.Path(checkpoint_path))
    checkpointer.close()
    
    return params, model

def evaluate_agent(params, model, num_games=100, opponent_params=None, opponent_model=None):
    """Evaluate agent via self-play. Returns (win_rate, avg_score)."""
    if opponent_params is None:
        opponent_params = params
        opponent_model = model
    
    wins = 0
    total_score = 0
    
    for game_idx in range(num_games):
        player, dice, state = _new_game()
        
        while True:
            if player == 1:
                move, _, _, _ = pruned_2_ply_search(
                    state, player, dice, params, model, training=False
                )
            else:
                move, _, _, _ = pruned_2_ply_search(
                    state, player, dice, opponent_params, opponent_model, training=False
                )
            
            state = _apply_move(state, player, move)
            reward = _reward(state, player)
            
            if reward != 0:
                if player == 1:
                    total_score += reward
                    if reward > 0:
                        wins += 1
                else:
                    total_score -= reward
                    if reward < 0:
                        wins += 1
                break
            
            player = -player
            dice = _roll_dice()
    
    win_rate = wins / num_games
    avg_score = total_score / num_games
    
    return win_rate, avg_score

def play_single_game(params, model, verbose=True):
    """Play a single game. Returns (winner, score, num_moves)."""
    player, dice, state = _new_game()
    num_moves = 0
    
    if verbose:
        print(f"Starting player: {'White' if player == 1 else 'Black'}")
        print(f"Initial dice: {dice}")
    
    while True:
        move, value, _, _ = pruned_2_ply_search(
            state, player, dice, params, model, training=False
        )
        
        if verbose:
            print(f"\nMove {num_moves + 1}: {'White' if player == 1 else 'Black'}")
            print(f"Dice: {dice}, Value estimate: {value:.3f}")
            print(f"Move: {[(f, r) for f, r in move]}")
        
        state = _apply_move(state, player, move)
        reward = _reward(state, player)
        num_moves += 1
        
        if reward != 0:
            winner = player if reward > 0 else -player
            score = abs(reward)
            
            if verbose:
                print(f"\nGame over! {'White' if winner == 1 else 'Black'} wins!")
                print(f"Score: {score} ({'Backgammon' if score == 3 else 'Gammon' if score == 2 else 'Normal'})")
                print(f"Total moves: {num_moves}")
            
            return winner, score, num_moves
        
        player = -player
        dice = _roll_dice()

if __name__ == "__main__":
    import sys
    mode = sys.argv[1] if len(sys.argv) > 1 else "train"
    
    if mode == "train":
        print("Starting PPO training...")
        params = train_ppo(
            num_iterations=30000,
            batch_size=128,
            buffer_size=2048,
            checkpoint_dir='/home/zhangdjr/Desktop/RL-Gammon/checkpoints/agent3',
            verbose_every=1000
        )
        print("\nTraining complete!")
    
    elif mode == "eval":
        checkpoint_path = sys.argv[2] if len(sys.argv) > 2 else '/tmp/agent3_checkpoints/final_params'
        print(f"Loading agent from {checkpoint_path}...")
        params, model = load_agent(checkpoint_path)
        
        print("\nEvaluating agent (100 self-play games)...")
        win_rate, avg_score = evaluate_agent(params, model, num_games=100)
        print(f"Win rate (as White): {win_rate:.1%}")
        print(f"Average score: {avg_score:.3f}")
    
    elif mode == "play":
        checkpoint_path = sys.argv[2] if len(sys.argv) > 2 else '/tmp/agent3_checkpoints/final_params'
        print(f"Loading agent from {checkpoint_path}...")
        params, model = load_agent(checkpoint_path)
        
        print("\nPlaying a single game...\n")
        winner, score, num_moves = play_single_game(params, model, verbose=True)
    
    else:
        print("Usage: python agent3_ppo.py [train|eval|play] [checkpoint_path]")
        print("  train: Train a new agent")
        print("  eval: Evaluate a trained agent")
        print("  play: Play and display a single game")
