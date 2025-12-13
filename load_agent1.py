"""
Load and use Agent 1 (TD(0) with Linear Features)

This script provides a simple interface to load the trained agent
and use it to select moves during gameplay or tournaments.
"""

import numpy as np
from src.agent_td0_linear import extract_features
from backgammon_engine import _2_ply_search
from numba import njit

class Agent1:
    """Agent 1: TD(0) with Linear Function Approximation"""
    
    def __init__(self, weights_file="agent1_weights.npy"):
        """Load trained weights"""
        self.weights = np.load(weights_file)
        print(f"Agent 1 loaded from {weights_file}")
        print(f"Weight statistics: mean={self.weights.mean():.4f}, std={self.weights.std():.4f}")
    
    def select_move(self, state, player, dice):
        """
        Select best move using 2-ply search with learned value function.
        
        Args:
            state: Board state (28-element array)
            player: Current player (+1 for white, -1 for black)
            dice: Dice roll (2-element array)
        
        Returns:
            move: Best move (list of (from, roll) tuples)
        """
        # Create batch value function for 2-ply search
        @njit
        def batch_value_fn(states):
            n = len(states)
            values = np.empty(n, dtype=np.float32)
            for i in range(n):
                state_arr = np.array(states[i], dtype=np.int8)
                features = extract_features(state_arr, 1)  # Canonical form
                values[i] = np.dot(self.weights, features)
            return values
        
        # Use 2-ply search to select move
        move = _2_ply_search(state, player, dice, batch_value_fn)
        return move

# Example usage
if __name__ == "__main__":
    from backgammon_engine import _new_game, _apply_move, _reward, _roll_dice
    
    # Load agent
    agent = Agent1("agent1_weights.npy")
    
    # Play one game
    print("\nPlaying one game as demonstration...")
    player, dice, state = _new_game()
    
    move_count = 0
    while move_count < 500:
        # Agent selects move
        move = agent.select_move(state, player, dice)
        
        # Apply move
        state = _apply_move(state, player, move)
        reward = _reward(state, player)
        
        if reward != 0:
            winner = "White" if reward > 0 else "Black"
            print(f"Game over after {move_count} moves. Winner: {winner}")
            break
        
        # Next turn
        player = -player
        dice = _roll_dice()
        move_count += 1
    
    print("Agent 1 demonstration complete!")
