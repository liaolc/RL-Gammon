import gc
import jax
import jax.numpy as jnp
from jax import random
import numpy as np
import optax
import orbax.checkpoint as ocp
import pathlib
from numba import njit
from numba.typed import List
from typing import Dict, List as PyList, Optional, Tuple, Any
import math

from backgammon_engine import (
    _new_game, _roll_dice, _actions, _apply_move, _reward,
    _to_canonical, W_BAR, B_BAR, W_OFF, B_OFF, NUM_POINTS
)
from backgammon_muzero_net import (
    StochasticMuZeroNetwork, HIDDEN_SIZE, NUM_DICE_OUTCOMES,
    dice_to_index, index_to_dice, DICE_PROBS
)

LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4
DISCOUNT = 1.0
NUM_SIMULATIONS = 25
PB_C_BASE = 19652
PB_C_INIT = 1.25
ROOT_DIRICHLET_FRACTION = 0.25
BATCH_SIZE = 128
BUFFER_SIZE = 10000
NUM_UNROLL_STEPS = 5
MAX_SUBMOVES = 4
KNOWN_BOUNDS = (-3.0, 3.0)


@njit
def encode_observation(state, player):
    canonical = _to_canonical(state, player)
    obs = np.zeros(28, dtype=np.float32)
    for i in range(28):
        obs[i] = float(canonical[i])
    return obs


def encode_move_features(move):
    features = np.zeros(MAX_SUBMOVES * 2, dtype=np.float32)
    if move is None:
        return features
    
    for i, (from_point, die) in enumerate(move):
        if i >= MAX_SUBMOVES:
            break
        features[i * 2] = float(from_point) / 25.0  # Normalize
        features[i * 2 + 1] = float(die) / 6.0
    
    return features


def encode_dice_onehot(dice):
    onehot = np.zeros(NUM_DICE_OUTCOMES, dtype=np.float32)
    idx = dice_to_index(dice[0], dice[1])
    onehot[idx] = 1.0
    return onehot


class MinMaxStats:
    def __init__(self):
        self.minimum = KNOWN_BOUNDS[0]
        self.maximum = KNOWN_BOUNDS[1]
    
    def update(self, value: float):
        self.minimum = min(self.minimum, value)
        self.maximum = max(self.maximum, value)
    
    def normalize(self, value: float) -> float:
        if self.maximum > self.minimum:
            return (value - self.minimum) / (self.maximum - self.minimum)
        return value


class Node:
    def __init__(self, prior: float, is_chance: bool = False):
        self.visit_count = 0
        self.prior = prior
        self.value_sum = 0.0
        self.children: Dict[Any, 'Node'] = {}
        self.state = None
        self.is_chance = is_chance
        self.reward = 0.0
        self.to_play = 0
    
    def expanded(self) -> bool:
        return len(self.children) > 0
    
    def value(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count


def ucb_score(parent: Node, child: Node, min_max_stats: MinMaxStats) -> float:
    pb_c = math.log((parent.visit_count + PB_C_BASE + 1) / PB_C_BASE) + PB_C_INIT
    pb_c *= math.sqrt(parent.visit_count) / (child.visit_count + 1)
    prior_score = pb_c * child.prior
    
    if child.visit_count > 0:
        value_score = min_max_stats.normalize(child.reward + DISCOUNT * child.value())
    else:
        value_score = 0.0
    
    return prior_score + value_score


def select_child(node: Node, min_max_stats: MinMaxStats) -> Tuple[Any, 'Node']:
    if node.is_chance:
        best_score = -float('inf')
        best_outcome = None
        best_child = None
        
        for outcome, child in node.children.items():
            score = child.prior / (child.visit_count + 1)
            if score > best_score:
                best_score = score
                best_outcome = outcome
                best_child = child
        
        return best_outcome, best_child
    
    best_score = -float('inf')
    best_action = None
    best_child = None
    
    for action, child in node.children.items():
        score = ucb_score(node, child, min_max_stats)
        if score > best_score:
            best_score = score
            best_action = action
            best_child = child
    
    return best_action, best_child


def backpropagate(search_path: PyList[Node], value: float, to_play: int,
                  min_max_stats: MinMaxStats):
    for node in reversed(search_path):
        node.value_sum += value if node.to_play == to_play else -value
        node.visit_count += 1
        min_max_stats.update(node.value())
        value = node.reward + DISCOUNT * value


def add_exploration_noise(node: Node):
    actions = list(node.children.keys())
    if len(actions) == 0:
        return
    alpha = 1.0 / math.sqrt(len(actions))
    noise = np.random.dirichlet([alpha] * len(actions))
    
    frac = ROOT_DIRICHLET_FRACTION
    for i, action in enumerate(actions):
        node.children[action].prior = (
            node.children[action].prior * (1 - frac) + noise[i] * frac
        )


def run_mcts(state, player, dice, params, model, num_simulations=NUM_SIMULATIONS):
    min_max_stats = MinMaxStats()
    legal_moves, legal_afterstates = _actions(state, player, dice)
    if len(legal_moves) == 0:
        return None, 0.0, {}
    
    obs = encode_observation(state, player)
    obs_batch = jnp.array([obs])
    
    latent_state, policy_logits, value = model.apply(
        {'params': params}, obs_batch,
        method=model.initial_inference
    )
    latent_state = np.array(latent_state[0])
    policy_logits = np.array(policy_logits[0])
    value = float(value[0])
    num_legal = len(legal_moves)
    if num_legal > len(policy_logits):
        # Extend with zeros if needed
        policy_logits = np.concatenate([
            policy_logits, 
            np.zeros(num_legal - len(policy_logits))
        ])
    
    legal_logits = policy_logits[:num_legal]
    legal_logits = legal_logits - np.max(legal_logits)
    exp_logits = np.exp(legal_logits)
    policy_probs = exp_logits / exp_logits.sum()
    root = Node(prior=0.0, is_chance=False)
    root.state = latent_state
    root.to_play = player
    for move_idx, prob in enumerate(policy_probs):
        root.children[move_idx] = Node(prior=prob, is_chance=True)
    
    backpropagate([root], value, player, min_max_stats)
    add_exploration_noise(root)
    
    for _ in range(num_simulations):
        node = root
        search_path = [node]
        sim_state = state.copy()
        sim_player = player
        sim_dice = dice
        sim_legal_moves = legal_moves
        
        while node.expanded():
            action_or_outcome, child = select_child(node, min_max_stats)
            if child is None:
                break
            node = child
            search_path.append(node)
            
            # Track game state through tree
            if not node.is_chance and len(search_path) >= 2:
                parent = search_path[-2]
                if parent.is_chance:
                    dice_idx = action_or_outcome
                    sim_dice = index_to_dice(dice_idx)
                    sim_legal_moves, _ = _actions(sim_state, sim_player, sim_dice)
        
        if node is None or len(search_path) < 2:
            continue
        
        parent = search_path[-2]
        
        if parent.is_chance:
            # Parent is chance node (afterstate) -> expand decision node
            # Get dice outcome that led here
            dice_idx = None
            for outcome, child in parent.children.items():
                if child is node:
                    dice_idx = outcome
                    break
            
            if dice_idx is None:
                continue
            
            dice_onehot = np.zeros(NUM_DICE_OUTCOMES, dtype=np.float32)
            dice_onehot[dice_idx] = 1.0
            
            afterstate_batch = jnp.array([parent.state])
            dice_batch = jnp.array([dice_onehot])
            
            next_state, reward, policy_logits, value = model.apply(
                {'params': params}, afterstate_batch, dice_batch,
                method=model.recurrent_inference_state
            )
            
            node.state = np.array(next_state[0])
            node.reward = float(reward[0])
            node.to_play = -sim_player  # Opponent's turn
            node.is_chance = False
            
            # Get opponent's legal moves for this dice
            opp_dice = index_to_dice(dice_idx)
            opp_legal_moves, _ = _actions(sim_state, -sim_player, opp_dice)
            
            if len(opp_legal_moves) > 0:
                policy_logits = np.array(policy_logits[0])
                num_opp_legal = len(opp_legal_moves)
                if num_opp_legal > len(policy_logits):
                    policy_logits = np.concatenate([
                        policy_logits,
                        np.zeros(num_opp_legal - len(policy_logits))
                    ])
                
                opp_logits = policy_logits[:num_opp_legal]
                opp_logits = opp_logits - np.max(opp_logits)
                exp_opp = np.exp(opp_logits)
                opp_probs = exp_opp / exp_opp.sum()
                
                for move_idx, prob in enumerate(opp_probs):
                    node.children[move_idx] = Node(prior=prob, is_chance=True)
            
            value = float(value[0])
        
        else:
            move_idx = None
            for act, child in parent.children.items():
                if child is node:
                    move_idx = act
                    break
            
            if move_idx is None or move_idx >= len(sim_legal_moves):
                continue
            
            move = sim_legal_moves[move_idx]
            move_features = encode_move_features(move)
            
            state_batch = jnp.array([parent.state])
            move_batch = jnp.array([move_features])
            
            afterstate, q_value = model.apply(
                {'params': params}, state_batch, move_batch,
                method=model.recurrent_inference_afterstate
            )
            
            node.state = np.array(afterstate[0])
            node.is_chance = True
            node.to_play = sim_player
            for dice_idx in range(NUM_DICE_OUTCOMES):
                prob = float(DICE_PROBS[dice_idx])
                node.children[dice_idx] = Node(prior=prob, is_chance=False)
            
            value = float(q_value[0])
        
        backpropagate(search_path, value, player, min_max_stats)
    
    visit_counts = {move_idx: child.visit_count 
                    for move_idx, child in root.children.items()}
    total_visits = sum(visit_counts.values())
    
    if total_visits == 0:
        return legal_moves[0], root.value(), {}
    
    best_move_idx = max(visit_counts.keys(), key=lambda m: visit_counts[m])
    search_policy = {}
    for move_idx, count in visit_counts.items():
        search_policy[move_idx] = count / total_visits
    
    return legal_moves[best_move_idx], root.value(), search_policy


class ReplayBuffer:
    def __init__(self, capacity=BUFFER_SIZE):
        self.capacity = capacity
        self.buffer = []
    
    def save(self, trajectory):
        if len(self.buffer) >= self.capacity:
            self.buffer.pop(0)
        self.buffer.append(trajectory)
    
    def sample_batch(self, batch_size):
        if len(self.buffer) == 0:
            return []
        
        samples = []
        for _ in range(batch_size):
            traj_idx = np.random.randint(len(self.buffer))
            traj = self.buffer[traj_idx]
            if len(traj) == 0:
                continue
            state_idx = np.random.randint(len(traj))
            samples.append((traj, state_idx))
        
        return samples
    
    def __len__(self):
        return len(self.buffer)


def compute_value_target(trajectory, state_idx):
    if len(trajectory) == 0:
        return 0.0
    
    final_reward = trajectory[-1].get('reward', 0.0)
    player_at_idx = trajectory[state_idx].get('player', 1)
    final_player = trajectory[-1].get('player', 1)
    
    if final_player == player_at_idx:
        return final_reward
    else:
        return -final_reward


def train_step(params, model, optimizer, opt_state, batch):
    def loss_fn(p):
        total_loss = 0.0
        num_samples = 0
        
        for traj, state_idx in batch:
            if state_idx >= len(traj):
                continue
            
            step = traj[state_idx]
            obs_batch = jnp.array([step['observation']])
            
            state, policy_logits, value = model.apply(
                {'params': p}, obs_batch,
                method=model.initial_inference
            )
            
            value_target = compute_value_target(traj, state_idx)
            total_loss += (value[0] - value_target) ** 2
            
            if 'search_policy' in step and step['search_policy']:
                log_probs = jax.nn.log_softmax(policy_logits[0])
                for move_idx, prob in step['search_policy'].items():
                    if move_idx < len(log_probs):
                        total_loss -= prob * log_probs[move_idx]
            
            for k in range(min(NUM_UNROLL_STEPS, len(traj) - state_idx - 1)):
                next_step = traj[state_idx + k + 1]
                
                move_features = jnp.array([step['move_features']])
                afterstate, q_value = model.apply(
                    {'params': p}, state, move_features,
                    method=model.recurrent_inference_afterstate
                )
                
                total_loss += (q_value[0] - value_target) ** 2
                dice_onehot = jnp.array([next_step['dice_onehot']])
                state, reward, policy_logits, value = model.apply(
                    {'params': p}, afterstate, dice_onehot,
                    method=model.recurrent_inference_state
                )
                
                actual_reward = next_step.get('reward', 0.0)
                total_loss += (reward[0] - actual_reward) ** 2
                
                step = next_step
                value_target = compute_value_target(traj, state_idx + k + 1)
            
            num_samples += 1
        
        return total_loss / max(num_samples, 1)
    
    loss, grads = jax.value_and_grad(loss_fn)(params)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    
    return params, opt_state, float(loss)


def train_stochastic_muzero(num_iterations=1000, games_per_iter=5,
                            checkpoint_dir='/home/zhangdjr/Desktop/RL-Gammon/checkpoints/agent4',
                            verbose_every=10):
    print(f"Training Agent 4: sims={NUM_SIMULATIONS}, batch={BATCH_SIZE}")
    model = StochasticMuZeroNetwork(hidden_size=HIDDEN_SIZE, max_moves=500)
    rng = random.PRNGKey(42)
    dummy_obs = jnp.zeros((1, 28))
    dummy_move = jnp.zeros((1, MAX_SUBMOVES * 2))
    dummy_dice = jnp.zeros((1, NUM_DICE_OUTCOMES))
    
    params = model.init(rng, dummy_obs, dummy_move, dummy_dice)['params']
    optimizer = optax.adamw(learning_rate=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    opt_state = optimizer.init(params)
    buffer = ReplayBuffer()
    
    total_games = 0
    white_wins = 0
    
    for iteration in range(num_iterations):
        for _ in range(games_per_iter):
            trajectory = []
            player, dice, state = _new_game()
            
            while True:
                obs = encode_observation(state, player)
                move, search_value, search_policy = run_mcts(
                    state, player, dice, params, model,
                    num_simulations=min(100, NUM_SIMULATIONS)
                )
                
                move_features = encode_move_features(move)
                dice_onehot = encode_dice_onehot(dice)
                
                trajectory.append({
                    'observation': obs,
                    'player': player,
                    'move_features': move_features,
                    'dice_onehot': dice_onehot,
                    'search_policy': search_policy,
                    'search_value': search_value,
                    'reward': 0.0
                })
                
                if move is None:
                    move = List()
                new_state = _apply_move(state, player, move)
                reward = _reward(new_state, player)
                
                if reward != 0:
                    trajectory[-1]['reward'] = reward
                    total_games += 1
                    if (reward > 0 and player == 1) or (reward < 0 and player == -1):
                        white_wins += 1
                    break
                
                state = new_state
                player = -player
                dice = _roll_dice()
            
            buffer.save(trajectory)
        
        if len(buffer) >= 10:
            batch = buffer.sample_batch(min(BATCH_SIZE, len(buffer) * 5))
            if batch:
                params, opt_state, loss = train_step(
                    params, model, optimizer, opt_state, batch
                )
        
        if (iteration + 1) % verbose_every == 0:
            win_rate = white_wins / max(total_games, 1)
            print(f"Iter {iteration + 1}/{num_iterations} | "
                  f"Games: {total_games} | White: {win_rate:.1%} | "
                  f"Buffer: {len(buffer)}")
            jax.clear_caches()
            gc.collect()
            checkpoint_path = pathlib.Path(checkpoint_dir)
            checkpoint_path.mkdir(parents=True, exist_ok=True)
            checkpointer = ocp.StandardCheckpointer()
            checkpointer.save(checkpoint_path / f'checkpoint_{iteration + 1}', params, force=True)
            checkpointer.close()
    
    print("Training complete")
    checkpoint_path = pathlib.Path(checkpoint_dir)
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    checkpointer = ocp.StandardCheckpointer()
    checkpointer.save(checkpoint_path / 'final_params', params, force=True)
    checkpointer.close()
    print(f"Model saved to {checkpoint_path / 'final_params'}")
    
    return params


def load_agent(checkpoint_path):
    model = StochasticMuZeroNetwork(hidden_size=HIDDEN_SIZE, max_moves=500)
    checkpointer = ocp.StandardCheckpointer()
    params = checkpointer.restore(pathlib.Path(checkpoint_path))
    checkpointer.close()
    return params, model


def select_move(state, player, dice, params, model):
    move, _, _ = run_mcts(state, player, dice, params, model, NUM_SIMULATIONS)
    return move


if __name__ == "__main__":
    import sys
    mode = sys.argv[1] if len(sys.argv) > 1 else "train"
    
    if mode == "train":
        params = train_stochastic_muzero(
            num_iterations=5000,
            games_per_iter=3,
            checkpoint_dir='/home/zhangdjr/Desktop/RL-Gammon/checkpoints/agent4',
            verbose_every=1000
        )
    
    elif mode == "eval":
        checkpoint_path = sys.argv[2] if len(sys.argv) > 2 else '/tmp/agent4_checkpoints/final_params'
        print(f"Loading agent from {checkpoint_path}...")
        params, model = load_agent(checkpoint_path)
        print("Agent loaded.")
    
    else:
        print("Usage: python agent4_muzero.py [train|eval] [checkpoint_path]")
