import numpy as np
import numba
from numba import njit, prange
from numba.typed import List
from backgammon_engine import (
    _new_game, _roll_dice, _actions, _apply_move, _reward,
    _to_canonical, _2_ply_search, _vectorized_new_game,
    _vectorized_roll_dice, _vectorized_2_ply_search,
    _vectorized_2_ply_search_epsilon_greedy, _vectorized_apply_move,
    W_BAR, B_BAR, W_OFF, B_OFF, NUM_POINTS, NUM_CHECKERS, HOME_BOARD_SIZE
)

@njit
def count_blots(state, player):
    blot_count = 0
    for i in range(1, NUM_POINTS + 1):
        checkers = state[i] * player
        if checkers == 1:
            blot_count += 1
    return blot_count

@njit
def count_stacked_in_home(state, player):
    stacked = 0
    if player == 1:
        # White's home: P19-P24
        for i in range(NUM_POINTS - HOME_BOARD_SIZE + 1, NUM_POINTS + 1):
            checkers = state[i] * player
            if checkers >= 3:
                stacked += checkers - 2
    else:
        # Black's home: P1-P6
        for i in range(1, HOME_BOARD_SIZE + 1):
            checkers = state[i] * player
            if checkers >= 3:
                stacked += checkers - 2
    return stacked

@njit
def longest_prime(state, player):
    max_length = 0
    current_length = 0
    
    for i in range(1, NUM_POINTS + 1):
        checkers = state[i] * player
        if checkers >= 2:
            current_length += 1
            max_length = max(max_length, current_length)
        else:
            current_length = 0
    
    return max_length

@njit
def longest_prime_in_home(state, player):
    max_length = 0
    current_length = 0
    
    if player == 1:
        # White's home: P19-P24
        for i in range(NUM_POINTS - HOME_BOARD_SIZE + 1, NUM_POINTS + 1):
            checkers = state[i] * player
            if checkers >= 2:
                current_length += 1
                max_length = max(max_length, current_length)
            else:
                current_length = 0
    else:
        # Black's home: P1-P6
        for i in range(1, HOME_BOARD_SIZE + 1):
            checkers = state[i] * player
            if checkers >= 2:
                current_length += 1
                max_length = max(max_length, current_length)
            else:
                current_length = 0
    
    return max_length

@njit
def checkers_trapped_behind_prime(state, player):
    max_length = 0
    current_length = 0
    prime_end = 0
    
    for i in range(1, NUM_POINTS + 1):
        checkers = state[i] * player
        if checkers >= 2:
            current_length += 1
            if current_length > max_length:
                max_length = current_length
                prime_end = i
        else:
            current_length = 0
    
    if max_length == 0:
        return 0
    
    # Count opponent checkers behind this prime
    trapped = 0
    prime_start = prime_end - max_length + 1
    
    if player == 1:
        # Count black checkers before prime_start
        for i in range(1, prime_start):
            if state[i] < 0:
                trapped += abs(state[i])
    else:
        # Count white checkers after prime_end
        for i in range(prime_end + 1, NUM_POINTS + 1):
            if state[i] > 0:
                trapped += state[i]
    
    return trapped

@njit
def made_points_in_home(state, player):
    made_points = 0
    
    if player == 1:
        # White's home: P19-P24
        for i in range(NUM_POINTS - HOME_BOARD_SIZE + 1, NUM_POINTS + 1):
            if state[i] * player >= 2:
                made_points += 1
    else:
        # Black's home: P1-P6
        for i in range(1, HOME_BOARD_SIZE + 1):
            if state[i] * player >= 2:
                made_points += 1
    
    return made_points

@njit
def checkers_within_6_pips_of_blots(state, player):
    count = 0
    
    # Find opponent blots
    from numba import types
    opponent_blots = List.empty_list(types.int64)
    for i in range(1, NUM_POINTS + 1):
        if state[i] * player == -1:
            opponent_blots.append(i)
    
    if len(opponent_blots) == 0:
        return 0
    
    # Count player checkers within 6 pips of any blot
    for i in range(1, NUM_POINTS + 1):
        if state[i] * player > 0:
            for blot_pos in opponent_blots:
                distance = abs(blot_pos - i)
                if distance <= 6:
                    count += state[i] * player
                    break
    
    return count

@njit
def contact_phase_feature(state, player):
    checkers_past_barrier = 0
    
    if player == 1:
        # White: count checkers on P13-P24
        for i in range(13, NUM_POINTS + 1):
            if state[i] > 0:
                checkers_past_barrier += state[i]
    else:
        # Black: count checkers on P1-P12
        for i in range(1, 13):
            if state[i] < 0:
                checkers_past_barrier += abs(state[i])
    
    return checkers_past_barrier / 15.0

@njit
def pip_count(state, player):
    total_pips = 0
    
    if player == 1:
        # White moves from low to high, bears off at 25
        for i in range(1, NUM_POINTS + 1):
            if state[i] > 0:
                total_pips += state[i] * (NUM_POINTS + 1 - i)
        # Checkers on bar
        if state[W_BAR] > 0:
            total_pips += state[W_BAR] * (NUM_POINTS + 1)
    else:
        # Black moves from high to low, bears off at 0
        for i in range(1, NUM_POINTS + 1):
            if state[i] < 0:
                total_pips += abs(state[i]) * i
        # Checkers on bar
        if state[B_BAR] < 0:
            total_pips += abs(state[B_BAR]) * (NUM_POINTS + 1)
    
    return total_pips

@njit
def extract_features(state, player):
    """Extract 52 handcrafted features from state."""
    features = np.zeros(52, dtype=np.float32)
    idx = 0
    
    for i in range(28):
        features[idx] = state[i] * player / 15.0
        idx += 1
    player_blots = count_blots(state, player)
    opponent_blots = count_blots(state, -player)
    features[idx] = player_blots / 15.0
    idx += 1
    features[idx] = opponent_blots / 15.0
    idx += 1
    player_stacked = count_stacked_in_home(state, player)
    opponent_stacked = count_stacked_in_home(state, -player)
    features[idx] = player_stacked / 15.0
    idx += 1
    features[idx] = opponent_stacked / 15.0
    idx += 1
    player_prime = longest_prime(state, player)
    opponent_prime = longest_prime(state, -player)
    features[idx] = player_prime / 24.0
    idx += 1
    features[idx] = opponent_prime / 24.0
    idx += 1
    player_home_prime = longest_prime_in_home(state, player)
    opponent_home_prime = longest_prime_in_home(state, -player)
    features[idx] = player_home_prime / 6.0
    idx += 1
    features[idx] = opponent_home_prime / 6.0
    idx += 1
    player_trapped = checkers_trapped_behind_prime(state, -player)
    opponent_trapped = checkers_trapped_behind_prime(state, player)
    features[idx] = player_trapped / 15.0
    idx += 1
    features[idx] = opponent_trapped / 15.0
    idx += 1
    player_made = made_points_in_home(state, player)
    opponent_made = made_points_in_home(state, -player)
    features[idx] = player_made / 6.0
    idx += 1
    features[idx] = opponent_made / 6.0
    idx += 1
    player_attackers = checkers_within_6_pips_of_blots(state, player)
    opponent_attackers = checkers_within_6_pips_of_blots(state, -player)
    features[idx] = player_attackers / 15.0
    idx += 1
    features[idx] = opponent_attackers / 15.0
    idx += 1
    x_race = contact_phase_feature(state, player)
    features[idx] = x_race
    idx += 1
    player_pips = pip_count(state, player)
    opponent_pips = pip_count(state, -player)
    pip_diff = (opponent_pips - player_pips) / 100.0
    features[idx] = pip_diff
    idx += 1
    features[idx] = (opponent_blots / 15.0) * (player_attackers / 15.0)
    idx += 1
    features[idx] = (player_blots / 15.0) * (opponent_attackers / 15.0)
    idx += 1
    features[idx] = (player_prime / 24.0) * (opponent_trapped / 15.0)
    idx += 1
    features[idx] = (opponent_prime / 24.0) * (player_trapped / 15.0)
    idx += 1
    features[idx] = x_race * pip_diff
    idx += 1
    features[idx] = (player_made / 6.0) ** 2
    
    return features

@njit
def value_function(state, player, weights):
    features = extract_features(state, player)
    return np.dot(weights, features)

@njit(parallel=True)
def batch_value_function_numba(states, weights):
    n = len(states)
    values = np.empty(n, dtype=np.float32)
    
    for i in prange(n):
        state = np.array(states[i], dtype=np.int8)
        values[i] = np.dot(weights, extract_features(state, 1))
    
    return values

def create_batch_value_function(weights):
    def batch_eval(states):
        return batch_value_function_numba(states, weights)
    return batch_eval

@njit(parallel=True)
def batch_extract_features(states, players):
    n = len(states)
    num_features = 52
    features_batch = np.empty((n, num_features), dtype=np.float32)
    
    for i in prange(n):
        canonical_state = _to_canonical(states[i], players[i])
        features_batch[i] = extract_features(canonical_state, 1)
    
    return features_batch

@njit(parallel=True)
def batch_value_estimates(states, players, weights):
    n = len(states)
    values = np.empty(n, dtype=np.float32)
    
    for i in prange(n):
        canonical_state = _to_canonical(states[i], players[i])
        features = extract_features(canonical_state, 1)
        values[i] = np.dot(weights, features)
    
    return values

def train_vectorized(batch_size=256, num_iterations=1000, alpha=0.01, gamma=1.0, 
                     epsilon_start=0.3, epsilon_end=0.01, epsilon_decay_steps=None,
                     verbose_every=10):
    num_features = 52
    weights = np.random.randn(num_features).astype(np.float32) * 0.01
    if epsilon_decay_steps is None:
        epsilon_decay_steps = int(0.8 * num_iterations)
    
    print(f"Training Agent 1: batch={batch_size}, iters={num_iterations}, lr={alpha}")
    states, players, dices = _vectorized_new_game(batch_size)
    
    total_games = 0
    white_wins = 0
    black_wins = 0
    
    for iteration in range(num_iterations):
        if iteration < epsilon_decay_steps:
            epsilon = epsilon_start - (epsilon_start - epsilon_end) * (iteration / epsilon_decay_steps)
        else:
            epsilon = epsilon_end
        prev_states = states.copy()
        prev_players = players.copy()
        batch_value_fn = create_batch_value_function(weights)
        moves = _vectorized_2_ply_search_epsilon_greedy(states, players, dices, batch_value_fn, epsilon)
        new_states = _vectorized_apply_move(states, players, moves)
        rewards = np.zeros(batch_size, dtype=np.float32)
        game_over = np.zeros(batch_size, dtype=np.bool_)
        
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
        
        prev_features = batch_extract_features(prev_states, prev_players)
        prev_values = np.dot(prev_features, weights)
        next_values = np.zeros(batch_size, dtype=np.float32)
        for i in range(batch_size):
            if not game_over[i]:
                canonical_next = _to_canonical(new_states[i], players[i])
                next_features = extract_features(canonical_next, 1)
                next_values[i] = np.dot(weights, next_features)
        
        td_errors = rewards + gamma * next_values - prev_values
        gradient = np.zeros(num_features, dtype=np.float32)
        for i in range(batch_size):
            gradient += td_errors[i] * prev_features[i]
        
        weights += alpha * gradient / batch_size
        states = new_states
        players = -players
        dices = _vectorized_roll_dice(batch_size)
        for i in range(batch_size):
            if game_over[i]:
                new_state, new_player, new_dice = _vectorized_new_game(1)
                states[i] = new_state[0]
                players[i] = new_player[0]
                dices[i] = new_dice[0]
        
        if (iteration + 1) % verbose_every == 0:
            if total_games > 0:
                white_win_rate = white_wins / total_games * 100
                print(f"Iter {iteration + 1}/{num_iterations} | ε={epsilon:.3f} | Games: {total_games} | "
                      f"White: {white_win_rate:.1f}% ({white_wins}W-{black_wins}L)")
                white_wins = 0
                black_wins = 0
                total_games = 0
    print("Training complete")
    return weights

if __name__ == "__main__":
    weights = train_vectorized(
        batch_size=256,
        num_iterations=50000,
        alpha=0.001,
        gamma=1.0,
        epsilon_start=0.3,
        epsilon_end=0.01,
        epsilon_decay_steps=40000,
        verbose_every=500
    )
    
    # Save final weights
    np.save("agent1_weights.npy", weights)
    print("\nWeights saved to agent1_weights.npy")
    print(f"Weight statistics: mean={weights.mean():.4f}, std={weights.std():.4f}")
