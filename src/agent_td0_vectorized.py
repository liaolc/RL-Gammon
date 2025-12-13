import numpy as np
import numba
from numba import njit, prange
from numba.typed import List
from backgammon_engine import (
    _new_game, _roll_dice, _apply_move, _reward,
    _vectorized_new_game, _vectorized_roll_dice, 
    _vectorized_apply_move, _vectorized_2_ply_search,
    _vectorized_2_ply_search_epsilon_greedy,
    _to_canonical,
    W_BAR, B_BAR, W_OFF, B_OFF, NUM_POINTS, NUM_CHECKERS, HOME_BOARD_SIZE
)

# Import feature extraction from the linear agent
from agent_td0_linear import extract_features

# Vectorized TD(0) Agent with Batch Training

@njit(parallel=True)
def batch_value_function_numba(states, weights):
    """Evaluate value function for a batch of states"""
    n = len(states)
    values = np.empty(n, dtype=np.float32)
    
    for i in prange(n):
        state = np.array(states[i], dtype=np.int8)
        values[i] = np.dot(weights, extract_features(state, 1))
    
    return values

def create_batch_value_function(weights):
    """Create closure for 2-ply search"""
    def batch_eval(states):
        return batch_value_function_numba(states, weights)
    return batch_eval

@njit(parallel=True)
def batch_extract_features(states, players):
    """Extract features for a batch of states"""
    n = len(states)
    num_features = 52
    features_batch = np.empty((n, num_features), dtype=np.float32)
    
    for i in prange(n):
        canonical_state = _to_canonical(states[i], players[i])
        features_batch[i] = extract_features(canonical_state, 1)
    
    return features_batch

@njit(parallel=True)
def batch_value_estimates(states, players, weights):
    """Compute value estimates for a batch of states"""
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
    """
    Train Agent 1 using vectorized self-play with epsilon-greedy exploration.
    
    Args:
        batch_size: Number of games to play in parallel
        num_iterations: Number of training iterations
        alpha: Learning rate
        gamma: Discount factor
        epsilon_start: Initial exploration rate (default 0.3)
        epsilon_end: Final exploration rate (default 0.01)
        epsilon_decay_steps: Steps to decay epsilon (default: 80% of num_iterations)
        verbose_every: Print progress every N iterations
    """
    
    # Initialize weights
    num_features = 52
    weights = np.random.randn(num_features).astype(np.float32) * 0.01
    
    # Set epsilon decay schedule
    if epsilon_decay_steps is None:
        epsilon_decay_steps = int(0.8 * num_iterations)
    
    print(f"Training Agent 1 (TD(0) Linear) - Vectorized with Epsilon-Greedy")
    print(f"Batch size: {batch_size}")
    print(f"Number of iterations: {num_iterations}")
    print(f"Features: {num_features}, Alpha: {alpha}, Gamma: {gamma}")
    print(f"Epsilon: {epsilon_start:.3f} → {epsilon_end:.3f} over {epsilon_decay_steps} steps")
    print("-" * 60)
    
    # Initialize games
    states, players, dices = _vectorized_new_game(batch_size)
    
    total_games = 0
    white_wins = 0
    black_wins = 0
    
    for iteration in range(num_iterations):
        # Compute current epsilon (linear decay)
        if iteration < epsilon_decay_steps:
            epsilon = epsilon_start - (epsilon_start - epsilon_end) * (iteration / epsilon_decay_steps)
        else:
            epsilon = epsilon_end
        
        # Store current states for TD update
        prev_states = states.copy()
        prev_players = players.copy()
        
        # Select moves using epsilon-greedy 2-ply search
        batch_value_fn = create_batch_value_function(weights)
        moves = _vectorized_2_ply_search_epsilon_greedy(states, players, dices, batch_value_fn, epsilon)
        
        # Apply moves
        new_states = _vectorized_apply_move(states, players, moves)
        
        # Compute rewards
        rewards = np.zeros(batch_size, dtype=np.float32)
        game_over = np.zeros(batch_size, dtype=np.bool_)
        
        for i in range(batch_size):
            reward = _reward(new_states[i], players[i])
            rewards[i] = reward
            if reward != 0:
                game_over[i] = True
                total_games += 1
                # Track wins from white's perspective (reward > 0 means white won)
                if reward > 0:
                    white_wins += 1
                else:
                    black_wins += 1
        
        # TD(0) update for all games
        prev_features = batch_extract_features(prev_states, prev_players)
        prev_values = np.dot(prev_features, weights)
        
        # Compute next values (0 for terminal states)
        next_values = np.zeros(batch_size, dtype=np.float32)
        for i in range(batch_size):
            if not game_over[i]:
                canonical_next = _to_canonical(new_states[i], players[i])
                next_features = extract_features(canonical_next, 1)
                next_values[i] = np.dot(weights, next_features)
        
        # TD errors (from each player's perspective)
        td_errors = rewards + gamma * next_values - prev_values
        
        # Batch gradient update
        gradient = np.zeros(num_features, dtype=np.float32)
        for i in range(batch_size):
            gradient += td_errors[i] * prev_features[i]
        
        weights += alpha * gradient / batch_size
        
        # Reset finished games
        for i in range(batch_size):
            if game_over[i]:
                new_state, new_player, new_dice = _vectorized_new_game(1)
                states[i] = new_state[0]
                players[i] = new_player[0]
                dices[i] = new_dice[0]
        
        # Update states and switch players for continuing games
        states = new_states
        players = -players
        dices = _vectorized_roll_dice(batch_size)
        
        # Progress report
        if (iteration + 1) % verbose_every == 0:
            if total_games > 0:
                white_win_rate = white_wins / total_games * 100
                print(f"Iter {iteration + 1}/{num_iterations} | ε={epsilon:.3f} | Games: {total_games} | "
                      f"White: {white_win_rate:.1f}% ({white_wins}W-{black_wins}L)")
                white_wins = 0
                black_wins = 0
                total_games = 0
            else:
                print(f"Iter {iteration + 1}/{num_iterations} | ε={epsilon:.3f} | No games finished yet")
    
    print("-" * 60)
    print("Training complete!")
    
    return weights

if __name__ == "__main__":
    # Full training configuration with epsilon-greedy exploration
    # Adjust these parameters based on your compute resources
    weights = train_vectorized(
        batch_size=256,         # Increase for SLURM (more parallel games)
        num_iterations=50000,   # Long training (~10-20M games)
        alpha=0.001,            # Learning rate
        gamma=1.0,              # No discounting
        epsilon_start=0.3,      # Start with 30% random exploration
        epsilon_end=0.01,       # End with 1% exploration
        epsilon_decay_steps=40000,  # Decay over 80% of training
        verbose_every=500       # Print progress every 500 iterations
    )
    
    # Save final weights
    np.save("agent1_weights.npy", weights)
    print("\nWeights saved to agent1_weights.npy")
    print(f"Weight statistics: mean={weights.mean():.4f}, std={weights.std():.4f}")
