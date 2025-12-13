import numpy as np
import numba
from numba import njit, prange
from backgammon_engine import (
    _new_game, _roll_dice, _actions, _apply_move, _reward,
    _to_canonical, _2_ply_search, _vectorized_new_game,
    _vectorized_roll_dice, _vectorized_2_ply_search,
    W_BAR, B_BAR, W_OFF, B_OFF, NUM_POINTS, NUM_CHECKERS, HOME_BOARD_SIZE
)

# Agent 1: TD(0) with Linear Function Approximation
# Uses handcrafted features and 2-ply search

# Feature extraction functions
@njit
def count_blots(state, player):
    """Count number of blots (single exposed checkers) for player"""
    blot_count = 0
    for i in range(1, NUM_POINTS + 1):
        checkers = state[i] * player
        if checkers == 1:
            blot_count += 1
    return blot_count

@njit
def count_stacked_in_home(state, player):
    """Count checkers stacked >= 3 in home quadrant"""
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
    """Length of longest consecutive sequence of points with >= 2 checkers"""
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
    """Length of longest consecutive sequence in home quadrant with >= 2 checkers"""
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
    """Count opponent checkers trapped behind longest prime"""
    # Find longest prime and its position
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
    """Count points in home board with at least 2 checkers"""
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
    """Count player's checkers within 6 pips of any opponent blot"""
    count = 0
    
    # Find opponent blots
    opponent_blots = []
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
    """Racing phase indicator: 0 (full contact) to 1 (pure race)"""
    # Count player's checkers past the 12-point barrier
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
    """Calculate pip count (total distance to bear off)"""
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
    """
    Extract all handcrafted features from state.
    Returns feature vector of length 52.
    """
    # Pre-allocate array (28 + 2 + 2 + 2 + 2 + 2 + 2 + 2 + 1 + 1 + 6 = 52 features)
    features = np.zeros(52, dtype=np.float32)
    idx = 0
    
    # 1. Raw state (28 features) - normalized
    for i in range(28):
        features[idx] = state[i] * player / 15.0
        idx += 1
    
    # 2. Blot counts (2 features)
    player_blots = count_blots(state, player)
    opponent_blots = count_blots(state, -player)
    features[idx] = player_blots / 15.0
    idx += 1
    features[idx] = opponent_blots / 15.0
    idx += 1
    
    # 3. Stacked in home (2 features)
    player_stacked = count_stacked_in_home(state, player)
    opponent_stacked = count_stacked_in_home(state, -player)
    features[idx] = player_stacked / 15.0
    idx += 1
    features[idx] = opponent_stacked / 15.0
    idx += 1
    
    # 4. Longest prime (2 features)
    player_prime = longest_prime(state, player)
    opponent_prime = longest_prime(state, -player)
    features[idx] = player_prime / 24.0
    idx += 1
    features[idx] = opponent_prime / 24.0
    idx += 1
    
    # 5. Longest prime in home (2 features)
    player_home_prime = longest_prime_in_home(state, player)
    opponent_home_prime = longest_prime_in_home(state, -player)
    features[idx] = player_home_prime / 6.0
    idx += 1
    features[idx] = opponent_home_prime / 6.0
    idx += 1
    
    # 6. Trapped checkers (2 features)
    player_trapped = checkers_trapped_behind_prime(state, -player)
    opponent_trapped = checkers_trapped_behind_prime(state, player)
    features[idx] = player_trapped / 15.0
    idx += 1
    features[idx] = opponent_trapped / 15.0
    idx += 1
    
    # 7. Made points in home (2 features)
    player_made = made_points_in_home(state, player)
    opponent_made = made_points_in_home(state, -player)
    features[idx] = player_made / 6.0
    idx += 1
    features[idx] = opponent_made / 6.0
    idx += 1
    
    # 8. Checkers within 6 pips of blots (2 features)
    player_attackers = checkers_within_6_pips_of_blots(state, player)
    opponent_attackers = checkers_within_6_pips_of_blots(state, -player)
    features[idx] = player_attackers / 15.0
    idx += 1
    features[idx] = opponent_attackers / 15.0
    idx += 1
    
    # 9. Contact phase (1 feature)
    x_race = contact_phase_feature(state, player)
    features[idx] = x_race
    idx += 1
    
    # 10. Pip count difference (1 feature)
    player_pips = pip_count(state, player)
    opponent_pips = pip_count(state, -player)
    pip_diff = (opponent_pips - player_pips) / 100.0
    features[idx] = pip_diff
    idx += 1
    
    # 11. Product features (6 features)
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
    """Compute V(s) = w · f(s)"""
    features = extract_features(state, player)
    return np.dot(weights, features)

@njit(parallel=True)
def batch_value_function_numba(states, weights):
    """Evaluate value function for a batch of states (for 2-ply search)"""
    n = len(states)
    values = np.empty(n, dtype=np.float32)
    
    for i in prange(n):
        # Convert tuple back to array
        state = np.array(states[i], dtype=np.int8)
        # Always evaluate from current player's perspective (player=1 in canonical form)
        values[i] = value_function(state, 1, weights)
    
    return values

def create_batch_value_function(weights):
    """Create a closure that captures weights for 2-ply search"""
    def batch_eval(states):
        return batch_value_function_numba(states, weights)
    return batch_eval

# TD(0) Learning
def td_update(state, next_state, reward, player, weights, alpha, gamma):
    """
    Perform TD(0) update.
    
    δ = R + γ·V(S') - V(S)
    w ← w + α·δ·f(S)
    """
    features = extract_features(state, player)
    v_current = np.dot(weights, features)
    
    if reward != 0:
        # Terminal state
        v_next = 0.0
    else:
        v_next = value_function(next_state, player, weights)
    
    td_error = reward + gamma * v_next - v_current
    weights += alpha * td_error * features
    
    return weights, td_error

# Single game training
def play_single_game(weights, alpha=0.01, gamma=1.0, verbose=False):
    """Play one game of self-play with TD(0) learning"""
    
    player, dice, state = _new_game()
    game_history = []
    
    while True:
        # Convert to canonical form (current player's perspective)
        canonical_state = _to_canonical(state, player)
        
        # Select move using 2-ply search
        batch_value_fn = create_batch_value_function(weights)
        move = _2_ply_search(state, player, dice, batch_value_fn)
        
        # Apply move
        new_state = _apply_move(state, player, move)
        reward = _reward(new_state, player)
        
        # Store experience
        game_history.append((canonical_state, player, reward))
        
        # Check if game over
        if reward != 0:
            # Backpropagate final reward through game history
            for i in range(len(game_history) - 1, -1, -1):
                hist_state, hist_player, _ = game_history[i]
                
                # Reward from this player's perspective
                player_reward = reward * hist_player
                
                if i == len(game_history) - 1:
                    # Terminal state
                    features = extract_features(hist_state, 1)  # Canonical form
                    v_current = np.dot(weights, features)
                    td_error = player_reward - v_current
                    weights += alpha * td_error * features
                else:
                    # Non-terminal: use next state
                    next_hist_state, _, _ = game_history[i + 1]
                    features = extract_features(hist_state, 1)
                    v_current = np.dot(weights, features)
                    v_next = value_function(next_hist_state, 1, weights)
                    td_error = gamma * v_next - v_current
                    weights += alpha * td_error * features
            
            if verbose:
                winner = "White" if reward > 0 else "Black"
                print(f"Game over! Winner: {winner}, Reward: {reward}")
            
            return weights, reward
        
        # Continue game
        state = new_state
        player = -player
        dice = _roll_dice()

# Training loop
def train_agent(num_games=10000, alpha=0.01, gamma=1.0, verbose_every=1000):
    """Train Agent 1 using TD(0) with self-play"""
    
    # Initialize weights
    num_features = len(extract_features(np.zeros(28, dtype=np.int8), 1))
    weights = np.random.randn(num_features).astype(np.float32) * 0.01
    
    print(f"Training Agent 1 (TD(0) Linear)")
    print(f"Number of features: {num_features}")
    print(f"Number of games: {num_games}")
    print(f"Learning rate: {alpha}, Gamma: {gamma}")
    print("-" * 50)
    
    wins = 0
    losses = 0
    
    for game_num in range(num_games):
        weights, reward = play_single_game(weights, alpha, gamma, verbose=False)
        
        if reward > 0:
            wins += 1
        else:
            losses += 1
        
        if (game_num + 1) % verbose_every == 0:
            win_rate = wins / (wins + losses) * 100
            print(f"Game {game_num + 1}/{num_games} | Win Rate: {win_rate:.1f}% | Wins: {wins}, Losses: {losses}")
            wins = 0
            losses = 0
    
    print("-" * 50)
    print("Training complete!")
    
    return weights

if __name__ == "__main__":
    # Train the agent
    weights = train_agent(num_games=1000, alpha=0.01, gamma=1.0, verbose_every=100)
    
    # Save weights
    np.save("agent1_weights.npy", weights)
    print("Weights saved to agent1_weights.npy")
