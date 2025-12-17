"""Stochastic MuZero Network for Backgammon (Agent 4).

Corrected implementation:
- No learned chance codebook (dice are known in backgammon)
- 21 explicit dice outcomes
- Proper afterstate formulation
"""

import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Tuple

# Network dimensions
HIDDEN_SIZE = 256
NUM_RES_BLOCKS = 10
NUM_DICE_OUTCOMES = 21  # Explicit dice outcomes, not learned


class ResBlockV2(nn.Module):
    """Pre-activation ResNet v2 block with LayerNorm."""
    features: int
    
    @nn.compact
    def __call__(self, x):
        residual = x
        x = nn.LayerNorm()(x)
        x = nn.relu(x)
        x = nn.Dense(self.features)(x)
        x = nn.LayerNorm()(x)
        x = nn.relu(x)
        x = nn.Dense(self.features)(x)
        return x + residual


class RepresentationNetwork(nn.Module):
    """h: observation -> latent state s^0"""
    hidden_size: int = HIDDEN_SIZE
    num_blocks: int = NUM_RES_BLOCKS
    
    @nn.compact
    def __call__(self, observation):
        x = nn.Dense(self.hidden_size)(observation)
        for _ in range(self.num_blocks):
            x = ResBlockV2(self.hidden_size)(x)
        return x


class AfterstateDynamicsNetwork(nn.Module):
    """φ: (state, action_embedding) -> afterstate
    
    In backgammon: state after player moves, before opponent rolls dice.
    """
    hidden_size: int = HIDDEN_SIZE
    num_blocks: int = NUM_RES_BLOCKS // 2
    
    @nn.compact
    def __call__(self, state, action_embedding):
        x = jnp.concatenate([state, action_embedding], axis=-1)
        x = nn.Dense(self.hidden_size)(x)
        for _ in range(self.num_blocks):
            x = ResBlockV2(self.hidden_size)(x)
        return x


class DynamicsNetwork(nn.Module):
    """g: (afterstate, dice_embedding) -> (next_state, reward)
    
    In backgammon: dice is KNOWN (21 outcomes), not learned.
    This is deterministic given afterstate + dice.
    """
    hidden_size: int = HIDDEN_SIZE
    num_blocks: int = NUM_RES_BLOCKS // 2
    
    @nn.compact
    def __call__(self, afterstate, dice_embedding):
        x = jnp.concatenate([afterstate, dice_embedding], axis=-1)
        x = nn.Dense(self.hidden_size)(x)
        for _ in range(self.num_blocks):
            x = ResBlockV2(self.hidden_size)(x)
        
        # Reward head (scalar, range [-3, 3] for backgammon)
        reward = nn.Dense(1)(x)
        reward = reward.squeeze(-1)
        
        return x, reward


class PredictionNetwork(nn.Module):
    """f: state -> (policy_logits, value)
    
    Policy is over move indices (not micro-actions).
    """
    hidden_size: int = HIDDEN_SIZE
    max_moves: int = 500  # Max legal moves to consider
    
    @nn.compact
    def __call__(self, state):
        # Policy head - outputs logits for each possible move
        policy = nn.Dense(self.hidden_size)(state)
        policy = nn.relu(policy)
        policy = nn.Dense(self.max_moves)(policy)
        
        # Value head
        value = nn.Dense(self.hidden_size)(state)
        value = nn.relu(value)
        value = nn.Dense(1)(value)
        value = value.squeeze(-1)
        
        return policy, value


class AfterstatePredictionNetwork(nn.Module):
    """ψ: afterstate -> Q_value
    
    For backgammon: No σ network needed since dice probabilities are known.
    Just predict Q(s, a) = V(afterstate).
    """
    hidden_size: int = HIDDEN_SIZE
    
    @nn.compact
    def __call__(self, afterstate):
        q_value = nn.Dense(self.hidden_size)(afterstate)
        q_value = nn.relu(q_value)
        q_value = nn.Dense(1)(q_value)
        q_value = q_value.squeeze(-1)
        return q_value


class MoveEncoder(nn.Module):
    """Encode a move (sequence of submoves) into an embedding."""
    hidden_size: int = HIDDEN_SIZE
    max_submoves: int = 4
    
    @nn.compact
    def __call__(self, move_features):
        # move_features: (batch, max_submoves * 2) - flattened (from, die) pairs
        x = nn.Dense(self.hidden_size)(move_features)
        x = nn.relu(x)
        return x


class DiceEncoder(nn.Module):
    """Encode dice roll into an embedding."""
    hidden_size: int = HIDDEN_SIZE
    
    @nn.compact
    def __call__(self, dice_onehot):
        # dice_onehot: (batch, 21) - one-hot over 21 dice outcomes
        x = nn.Dense(self.hidden_size)(dice_onehot)
        x = nn.relu(x)
        return x


class StochasticMuZeroNetwork(nn.Module):
    """Stochastic MuZero for Backgammon with explicit dice modeling."""
    hidden_size: int = HIDDEN_SIZE
    max_moves: int = 500
    
    def setup(self):
        self.representation = RepresentationNetwork(self.hidden_size)
        self.afterstate_dynamics = AfterstateDynamicsNetwork(self.hidden_size)
        self.dynamics = DynamicsNetwork(self.hidden_size)
        self.prediction = PredictionNetwork(self.hidden_size, self.max_moves)
        self.afterstate_prediction = AfterstatePredictionNetwork(self.hidden_size)
        self.move_encoder = MoveEncoder(self.hidden_size)
        self.dice_encoder = DiceEncoder(self.hidden_size)
    
    def initial_inference(self, observation):
        """h(o) -> s, f(s) -> (p, v)"""
        state = self.representation(observation)
        policy, value = self.prediction(state)
        return state, policy, value
    
    def recurrent_inference_afterstate(self, state, move_features):
        """φ(s, a) -> as, ψ(as) -> Q"""
        action_emb = self.move_encoder(move_features)
        afterstate = self.afterstate_dynamics(state, action_emb)
        q_value = self.afterstate_prediction(afterstate)
        return afterstate, q_value
    
    def recurrent_inference_state(self, afterstate, dice_onehot):
        """g(as, dice) -> (s', r), f(s') -> (p, v)"""
        dice_emb = self.dice_encoder(dice_onehot)
        next_state, reward = self.dynamics(afterstate, dice_emb)
        policy, value = self.prediction(next_state)
        return next_state, reward, policy, value
    
    def __call__(self, observation, move_features=None, dice_onehot=None):
        """Forward pass for initialization."""
        state, policy, value = self.initial_inference(observation)
        
        if move_features is not None:
            afterstate, q_value = self.recurrent_inference_afterstate(state, move_features)
            if dice_onehot is not None:
                next_state, reward, next_policy, next_value = self.recurrent_inference_state(
                    afterstate, dice_onehot)
                return (state, policy, value, afterstate, q_value,
                        next_state, reward, next_policy, next_value)
            return state, policy, value, afterstate, q_value
        
        return state, policy, value


# Dice utilities
def dice_to_index(d1, d2):
    """Convert dice roll to index (0-20)."""
    high, low = max(d1, d2), min(d1, d2)
    # Index formula for upper triangular enumeration
    idx = 0
    for h in range(1, high):
        idx += h
    idx += (high - low)
    return idx


def index_to_dice(idx):
    """Convert index to dice roll."""
    count = 0
    for high in range(1, 7):
        for low in range(1, high + 1):
            if count == idx:
                return (high, low)
            count += 1
    return (6, 6)


def dice_probability(idx):
    """Get probability of dice outcome by index."""
    high, low = index_to_dice(idx)
    if high == low:
        return 1.0 / 36.0  # Doubles
    else:
        return 2.0 / 36.0  # Non-doubles


# All 21 dice probabilities
DICE_PROBS = jnp.array([dice_probability(i) for i in range(NUM_DICE_OUTCOMES)])