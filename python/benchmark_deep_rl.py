# python/benchmark_deep_rl.py
# Unified deep RL benchmark: DQN, SAC, PPO, and LSTM PPO trained with a shared
# 4-stage curriculum, evaluated on a held-out maze, averaged over 3 seeds.
#
# CONCEPT — Why a curriculum?
#   Throwing a freshly-initialised agent into a dense maze (35% obstacles) and
#   expecting it to find the goal is like teaching driving on a motorway on day one.
#   A curriculum starts with easy mazes (low obstacle density) where the goal is
#   reachable quickly — the agent gets reward signal early and can bootstrap a
#   useful policy. Subsequent stages increase density progressively. Weights are
#   never reset between stages, so each agent transfers everything it has learned.
#   The held-out maze (density=0.35, fixed seed=999) is never seen during training
#   — it measures genuine generalisation, not memorisation.
#
# CONCEPT — Why compare DQN, SAC, PPO, and LSTM PPO together?
#   Each algorithm occupies a different point in the design space:
#     DQN      — off-policy, discrete, epsilon-greedy exploration, replay buffer.
#     SAC      — off-policy, entropy-regularised, learned temperature, twin critics.
#     PPO      — on-policy, clipped surrogate objective, no replay buffer.
#     LSTM PPO — PPO + recurrent hidden state for memory in partial observability.
#   Comparing them on identical mazes with identical curricula isolates algorithmic
#   differences from environment differences.
#
# CONCEPT — Adapter pattern
#   Each algorithm has a different internal API (different update signatures,
#   different state management). The adapter classes wrap each algorithm behind
#   a uniform interface: train_stage / swap_maze / greedy_action. The benchmark
#   loop calls only this interface, so adding a fifth algorithm means only writing
#   a new adapter — the loop does not change.
#
# CONCEPT — Multi-seed averaging
#   One run of any RL algorithm is noisy — the policy might get lucky with a maze
#   layout that happens to be easy to solve by chance. Running three seeds with
#   freshly-initialised weights each time gives mean ± std. The std measures
#   how stable the algorithm is: low std means the algorithm reliably learns
#   regardless of initialisation; high std means results depend heavily on luck.
#
# Self-driving analog:
#   The curriculum is a driving school programme — empty car parks first, then
#   quiet streets, then busy roads. The held-out maze is the final road test on
#   a route the driver has never seen. Multi-seed averaging is like testing many
#   student drivers (same school, same lessons, different individuals).

import sys
import os

# CONCEPT — sys.path manipulation
#   Python's import system searches sys.path in order. The pybind11 .so lives in
#   the project root and python/ lives one level up from this file. Adding both
#   ensures imports resolve correctly regardless of which directory the script is
#   invoked from.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import copy
import csv
import math
import random
import time

import torch
import torch.nn as nn
from torch.distributions import Categorical

import pathplanning
from dqn_network        import DQNNetwork
from replay_buffer      import ReplayBuffer
from networks.sac_network  import SACActor, SACCritic
from agents.sac_agent      import SACAgent
from networks.actor_critic import ActorCriticNetwork
from agents.ppo_agent      import PPOAgent
from networks.lstm_policy  import LSTMActorCriticNetwork

# ===========================================================================
# Hyperparameters
# ===========================================================================

GRID_WIDTH    = 41
GRID_HEIGHT   = 41
START_X       = 0
START_Y       = 0
GOAL_X        = 38
GOAL_Y        = 40
STATE_SIZE    = 6          # [x/width, y/height, wall_up, wall_down, wall_left, wall_right]
ACTION_SIZE   = 4          # 0=UP 1=DOWN 2=LEFT 3=RIGHT
HIDDEN_SIZE   = 128
MAX_STEPS     = GRID_WIDTH * GRID_HEIGHT * 4
GAMMA         = 0.95
LR            = 0.001
SEEDS         = 3
EVAL_EPISODES = 100

# DQN-specific
BATCH_SIZE        = 64
BUFFER_CAPACITY   = 10_000
EPSILON_START     = 1.0
EPSILON_MIN       = 0.05
EPSILON_DECAY     = 0.995
TARGET_UPDATE_FREQ = 100

# SAC-specific
SAC_LR  = 0.001
SAC_TAU = 0.005

# PPO-specific
PPO_EPOCHS       = 4
PPO_CLIP         = 0.2
ENTROPY_COEF     = 0.01
VALUE_LOSS_COEF  = 0.5
GAE_LAMBDA       = 0.95

# CONCEPT — Curriculum stages
#   Four mazes of increasing difficulty. The same (density, seed) pair always
#   produces the same labyrinth — same obstacles, same topology — making
#   multi-seed experiments reproducible at the maze level. Only the agent weights
#   vary across seeds.
CURRICULUM_STAGES = [
    {"obstacle_density": 0.10, "episodes": 500,  "maze_seed": 11},
    {"obstacle_density": 0.20, "episodes": 500,  "maze_seed": 22},
    {"obstacle_density": 0.30, "episodes": 1000, "maze_seed": 33},
    {"obstacle_density": 0.35, "episodes": 1000, "maze_seed": 44},
]

EVAL_DENSITY  = 0.35
EVAL_SEED     = 999


# ===========================================================================
# Shared utility
# ===========================================================================

def state_to_tensor(state: list, env) -> torch.Tensor:
    """Encode raw [x, y] state into a normalised 6-dim feature tensor.

    CONCEPT — Normalisation
        x and y can range from 0 to GRID_WIDTH-1 / GRID_HEIGHT-1.  Raw pixel
        coordinates have different magnitudes than the binary wall flags (0 or 1).
        Dividing by grid dimensions puts everything in [0, 1] so no input feature
        dominates the others in the network's first layer.
    """
    normalised_position = [state[0] / GRID_WIDTH, state[1] / GRID_HEIGHT]
    wall_flags = list(env.getLineOfSight(int(state[0]), int(state[1])))
    return torch.FloatTensor(normalised_position + wall_flags).unsqueeze(0)


def make_env(obstacle_density: float, maze_seed: int) -> pathplanning.GridEnvironment:
    """Construct a fresh GridEnvironment with the given maze parameters."""
    return pathplanning.GridEnvironment(
        GRID_WIDTH, GRID_HEIGHT,
        START_X, START_Y,
        GOAL_X, GOAL_Y,
        obstacle_density,
        maze_seed,
    )


def make_eval_env() -> pathplanning.GridEnvironment:
    """Construct the held-out evaluation environment. Never used during training."""
    return make_env(EVAL_DENSITY, EVAL_SEED)


# ===========================================================================
# Evaluation
# ===========================================================================

def evaluate(adapter, eval_env: pathplanning.GridEnvironment, num_episodes: int):
    """Run num_episodes greedy episodes and return (success_rate, avg_steps_on_success).

    CONCEPT — Greedy evaluation
        During training the agent explores (epsilon > 0 for DQN, stochastic
        sampling for policy-gradient methods). Evaluation must measure the LEARNED
        policy without exploration noise, so we ask each adapter for its greedy
        action: argmax for DQN, mode/argmax of softmax for PPO/SAC, same for LSTM.
        This cleanly separates "how well did it explore" from "how well did it learn."
    """
    success_count = 0
    total_steps_on_success = 0

    for _ in range(num_episodes):
        raw_state    = eval_env.reset()
        state_tensor = state_to_tensor(raw_state, eval_env)
        reached_goal = False

        for step_index in range(MAX_STEPS):
            action = adapter.greedy_action(state_tensor)
            result = eval_env.step(action)
            raw_state    = [int(result[0]), int(result[1])]
            reward       = float(result[2])
            done         = bool(result[3] > 0.5)
            state_tensor = state_to_tensor(raw_state, eval_env)

            if done and reward > 0:
                reached_goal = True
                total_steps_on_success += step_index + 1
                break

        if reached_goal:
            success_count += 1

    success_rate = success_count / num_episodes
    avg_steps    = (total_steps_on_success / success_count) if success_count > 0 else float("nan")
    return success_rate, avg_steps


# ===========================================================================
# DQN adapter
# ===========================================================================

class DQNAdapter:
    """Wraps DQN training behind the uniform adapter interface.

    CONCEPT — Double DQN
        The main network selects the best next action; the frozen target network
        evaluates it. Decoupling selection from evaluation dampens the systematic
        overestimation bias that single-network DQN accumulates over time.
    """

    def __init__(self):
        self.main_network   = DQNNetwork(STATE_SIZE, ACTION_SIZE, HIDDEN_SIZE)
        self.target_network = DQNNetwork(STATE_SIZE, ACTION_SIZE, HIDDEN_SIZE)
        self.target_network.load_state_dict(self.main_network.state_dict())
        self.optimizer  = torch.optim.Adam(self.main_network.parameters(), lr=LR)
        self.buffer     = ReplayBuffer(BUFFER_CAPACITY)
        self.epsilon    = EPSILON_START
        self.total_steps = 0
        self.env        = None

    def swap_maze(self, new_env: pathplanning.GridEnvironment) -> None:
        """Update the internal environment reference; keep all weights intact."""
        self.env = new_env

    def greedy_action(self, state_tensor: torch.Tensor) -> int:
        """Select the action with the highest Q-value (no exploration noise)."""
        with torch.no_grad():
            return int(self.main_network(state_tensor).argmax().item())

    def _select_action_epsilon_greedy(self, state_tensor: torch.Tensor) -> int:
        if random.random() < self.epsilon:
            return random.randint(0, ACTION_SIZE - 1)
        return self.greedy_action(state_tensor)

    def _update_network(self) -> None:
        if not self.buffer.is_ready(BATCH_SIZE):
            return
        states, actions, rewards, next_states, dones = self.buffer.sample(BATCH_SIZE)

        predicted_q = self.main_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            # CONCEPT — Double DQN decoupling
            best_next_actions   = self.main_network(next_states).argmax(1)
            evaluated_q_values  = self.target_network(next_states).gather(
                1, best_next_actions.unsqueeze(1)
            ).squeeze(1)
        bellman_target = rewards + GAMMA * evaluated_q_values * (1.0 - dones)

        loss = nn.functional.mse_loss(predicted_q, bellman_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def train_stage(self, env: pathplanning.GridEnvironment, num_episodes: int) -> int:
        """Train for num_episodes on env. Returns first episode where goal was reached, or -1."""
        self.env = env
        first_goal_episode = -1

        for episode_number in range(1, num_episodes + 1):
            raw_state    = env.reset()
            state_tensor = state_to_tensor(raw_state, env)
            reached_goal = False

            for _ in range(MAX_STEPS):
                action      = self._select_action_epsilon_greedy(state_tensor)
                result      = env.step(action)
                next_raw    = [int(result[0]), int(result[1])]
                reward      = float(result[2])
                done        = bool(result[3] > 0.5)
                next_tensor = state_to_tensor(next_raw, env)

                self.buffer.push(state_tensor, action, reward, next_tensor, done)
                state_tensor = next_tensor
                self.total_steps += 1

                self._update_network()

                if self.total_steps % TARGET_UPDATE_FREQ == 0:
                    self.target_network.load_state_dict(self.main_network.state_dict())

                if done:
                    if reward > 0:
                        reached_goal = True
                    break

            self.epsilon = max(EPSILON_MIN, self.epsilon * EPSILON_DECAY)

            if reached_goal and first_goal_episode == -1:
                first_goal_episode = episode_number

        return first_goal_episode


# ===========================================================================
# SAC adapter
# ===========================================================================

class SACAdapter:
    """Wraps SAC behind the uniform adapter interface.

    CONCEPT — Off-policy with entropy regularisation
        SAC reuses experience from a replay buffer (off-policy), just like DQN.
        The key difference is the actor is trained to maximise BOTH expected return
        AND entropy — the agent learns to explore appropriately without a hand-tuned
        epsilon schedule. The learned temperature alpha auto-tunes the exploration
        intensity throughout training.
    """

    def __init__(self):
        actor      = SACActor(STATE_SIZE, ACTION_SIZE, HIDDEN_SIZE)
        critic_one = SACCritic(STATE_SIZE, ACTION_SIZE, HIDDEN_SIZE)
        critic_two = SACCritic(STATE_SIZE, ACTION_SIZE, HIDDEN_SIZE)
        self.agent  = SACAgent(actor, critic_one, critic_two, SAC_LR, GAMMA, SAC_TAU, ACTION_SIZE)
        self.buffer = ReplayBuffer(BUFFER_CAPACITY)

    def swap_maze(self, new_env: pathplanning.GridEnvironment) -> None:
        pass  # SAC holds no env reference internally — env is passed to train_stage

    def greedy_action(self, state_tensor: torch.Tensor) -> int:
        """Pick the action with the highest softmax probability (no sampling)."""
        with torch.no_grad():
            logits = self.agent.actor(state_tensor)
            return int(torch.softmax(logits, dim=-1).argmax().item())

    def train_stage(self, env: pathplanning.GridEnvironment, num_episodes: int) -> int:
        """Train for num_episodes on env. Returns first episode where goal was reached, or -1."""
        first_goal_episode = -1

        for episode_number in range(1, num_episodes + 1):
            raw_state    = env.reset()
            state_tensor = state_to_tensor(raw_state, env)
            reached_goal = False

            for _ in range(MAX_STEPS):
                action      = self.agent.select_action(state_tensor)
                result      = env.step(action)
                next_raw    = [int(result[0]), int(result[1])]
                reward      = float(result[2])
                done        = bool(result[3] > 0.5)
                next_tensor = state_to_tensor(next_raw, env)

                self.buffer.push(state_tensor, action, reward, next_tensor, done)
                state_tensor = next_tensor

                if self.buffer.is_ready(BATCH_SIZE):
                    batch = self.buffer.sample(BATCH_SIZE)
                    self.agent.update(batch)

                if done:
                    if reward > 0:
                        reached_goal = True
                    break

            if reached_goal and first_goal_episode == -1:
                first_goal_episode = episode_number

        return first_goal_episode


# ===========================================================================
# PPO adapter
# ===========================================================================

class PPOAdapter:
    """Wraps PPO behind the uniform adapter interface.

    CONCEPT — On-policy episode collection
        PPO collects one full episode with the current policy, computes advantages
        using the discounted returns and the critic's baseline, runs PPO_EPOCHS
        gradient steps on that data, then discards it. No replay buffer — each
        update strictly uses data from the current policy. This is wasteful in
        terms of data efficiency but keeps the policy update mathematically sound
        (importance sampling ratio starts at 1 when data is fresh).
    """

    def __init__(self):
        network    = ActorCriticNetwork(STATE_SIZE, ACTION_SIZE, HIDDEN_SIZE)
        self.agent = PPOAgent(network, LR, GAMMA, PPO_EPOCHS)

    def swap_maze(self, new_env: pathplanning.GridEnvironment) -> None:
        pass  # PPO holds no env reference

    def greedy_action(self, state_tensor: torch.Tensor) -> int:
        """Pick the action with the highest policy probability (no sampling)."""
        with torch.no_grad():
            action_probs, _ = self.agent.network(state_tensor)
            return int(action_probs.argmax().item())

    def train_stage(self, env: pathplanning.GridEnvironment, num_episodes: int) -> int:
        """Train for num_episodes on env. Returns first episode where goal was reached, or -1."""
        first_goal_episode = -1

        for episode_number in range(1, num_episodes + 1):
            states_list    = []
            actions_list   = []
            log_probs_list = []
            rewards_list   = []
            dones_list     = []
            values_list    = []

            raw_state    = env.reset()
            state_tensor = state_to_tensor(raw_state, env)
            reached_goal = False

            for _ in range(MAX_STEPS):
                action, log_prob, value = self.agent.select_action(state_tensor)
                result      = env.step(action)
                next_raw    = [int(result[0]), int(result[1])]
                reward      = float(result[2])
                done        = bool(result[3] > 0.5)
                next_tensor = state_to_tensor(next_raw, env)

                states_list.append(state_tensor)
                actions_list.append(action)
                log_probs_list.append(log_prob)
                rewards_list.append(reward)
                dones_list.append(done)
                values_list.append(value.item())

                state_tensor = next_tensor

                if done:
                    if reward > 0:
                        reached_goal = True
                    break

            if len(rewards_list) == 0:
                continue

            returns    = self.agent.compute_returns(rewards_list, dones_list)
            states_tensor    = torch.cat(states_list, dim=0)
            actions_tensor   = torch.LongTensor(actions_list)
            log_probs_tensor = torch.stack(log_probs_list)
            values_tensor    = torch.FloatTensor(values_list)
            advantages       = (returns - values_tensor)
            if advantages.std() > 1e-8:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            self.agent.update(states_tensor, actions_tensor, log_probs_tensor, returns, advantages)

            if reached_goal and first_goal_episode == -1:
                first_goal_episode = episode_number

        return first_goal_episode


# ===========================================================================
# LSTM PPO adapter
# ===========================================================================

def _lstm_ppo_update(
    network:             LSTMActorCriticNetwork,
    optimizer:           torch.optim.Optimizer,
    episode_states:      list,
    episode_actions:     list,
    old_log_probs:       torch.Tensor,
    returns:             torch.Tensor,
    advantages:          torch.Tensor,
) -> None:
    """PPO update with BPTT over the full episode sequence.

    CONCEPT — BPTT (Backpropagation Through Time)
        Rather than treating each time-step independently, we stack the full
        episode into one tensor [1, T, state_size] and run the LSTM in a single
        forward pass. PyTorch's autograd unrolls the computation graph across all
        T LSTM cells, so gradients flow back through every hidden state update.
        This lets the network learn long-range dependencies — e.g., "the dead end
        I entered 20 steps ago is why I'm stuck now."

    CONCEPT — Why start h_0 at zeros each PPO epoch?
        During rollout, the hidden state evolves step by step. During the update
        we re-run the same episode, but the weights have changed since the rollout.
        Using the stored hidden states from rollout would be inconsistent with the
        new weights. Starting from h_0=zeros each update epoch is a standard
        approximation used in OpenAI's original PPO code and CleanRL — the error
        is small for short episodes and far simpler than the alternative.
    """
    episode_sequence = torch.cat(episode_states, dim=0).unsqueeze(0)  # [1, T, state_size]
    actions_tensor   = torch.LongTensor(episode_actions)               # [T]

    for _ in range(PPO_EPOCHS):
        projected_sequence   = network.relu(network.input_projection(episode_sequence))
        initial_hidden_state = network.get_initial_hidden_state(batch_size=1)
        lstm_output, _       = network.lstm(projected_sequence, initial_hidden_state)
        per_step_features    = lstm_output.squeeze(0)  # [T, hidden_size]

        new_action_probs = torch.softmax(network.policy_head(per_step_features), dim=-1)
        new_values       = network.value_head(per_step_features).squeeze(-1)  # [T]

        distribution  = Categorical(new_action_probs)
        new_log_probs = distribution.log_prob(actions_tensor)
        entropy       = distribution.entropy().mean()

        ratio         = torch.exp(new_log_probs - old_log_probs)
        clipped_ratio = torch.clamp(ratio, 1.0 - PPO_CLIP, 1.0 + PPO_CLIP)
        actor_loss    = -torch.min(ratio * advantages, clipped_ratio * advantages).mean()
        value_loss    = nn.functional.mse_loss(new_values, returns)
        total_loss    = actor_loss + VALUE_LOSS_COEF * value_loss - ENTROPY_COEF * entropy

        optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(network.parameters(), max_norm=0.5)
        optimizer.step()


def _compute_gae(
    rewards: list,
    values:  list,
    dones:   list,
) -> tuple:
    """Generalised Advantage Estimation — returns (advantages, returns).

    CONCEPT — GAE blends bias and variance
        Simple Monte-Carlo returns have zero bias but high variance.
        One-step TD has low variance but high bias (bootstraps from an imperfect critic).
        GAE interpolates between them via lambda:
            A_t = sum_{l=0}^{inf} (gamma * lambda)^l * delta_{t+l}
        lambda=0.95 (empirically best from the original PPO paper) sits near the
        Monte-Carlo end — low bias, moderate variance.
    """
    episode_length = len(rewards)
    values_flat    = [value_tensor.item() for value_tensor in values]

    advantages_reversed = []
    running_gae_sum     = 0.0

    for time_index in reversed(range(episode_length)):
        done_mask   = 1.0 - float(dones[time_index])
        next_value  = (values_flat[time_index + 1] if time_index + 1 < episode_length else 0.0) * done_mask
        td_residual = rewards[time_index] + GAMMA * next_value - values_flat[time_index]
        running_gae_sum = td_residual + GAMMA * GAE_LAMBDA * done_mask * running_gae_sum
        advantages_reversed.append(running_gae_sum)

    advantages_reversed.reverse()
    raw_advantages = torch.FloatTensor(advantages_reversed)
    values_tensor  = torch.FloatTensor(values_flat)
    returns        = raw_advantages + values_tensor

    advantages = raw_advantages
    if advantages.std() > 1e-8:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    return advantages, returns


class LSTMPPOAdapter:
    """Wraps LSTM PPO behind the uniform adapter interface.

    CONCEPT — Recurrent memory for partial observability
        The 6-dim state encodes only the agent's current position and its four
        immediate neighbours. Two positions with identical wall patterns (e.g.
        two corridor junctions) are indistinguishable from a single state snapshot.
        The LSTM hidden state carries a compressed history of the episode, allowing
        the agent to distinguish "junction I came from the left" from "junction I
        came from the right" — information that matters for not revisiting dead ends.
    """

    def __init__(self):
        self.network   = LSTMActorCriticNetwork(STATE_SIZE, ACTION_SIZE, HIDDEN_SIZE)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=LR)

    def swap_maze(self, new_env: pathplanning.GridEnvironment) -> None:
        pass  # LSTM PPO holds no env reference

    def greedy_action(self, state_tensor: torch.Tensor) -> int:
        """Pick the action with the highest probability (stateless — no hidden state carried).

        Note: during greedy eval we do not carry hidden state between calls because
        the evaluate() function is stateless (each call is a separate query). For a
        more faithful evaluation, callers should maintain hidden state across an
        episode — see evaluate_lstm() below. The standard evaluate() function is used
        here for interface uniformity; the impact is small on short eval episodes.
        """
        with torch.no_grad():
            action_probs, _, _ = self.network.forward(state_tensor, None)
            return int(action_probs.argmax().item())

    def train_stage(self, env: pathplanning.GridEnvironment, num_episodes: int) -> int:
        """Train for num_episodes on env. Returns first episode where goal was reached, or -1."""
        first_goal_episode = -1

        for episode_number in range(1, num_episodes + 1):
            episode_states    = []
            episode_actions   = []
            episode_log_probs = []
            episode_rewards   = []
            episode_dones     = []
            episode_values    = []

            raw_state    = env.reset()
            state_tensor = state_to_tensor(raw_state, env)
            hidden_state = self.network.get_initial_hidden_state(batch_size=1)
            reached_goal = False

            for _ in range(MAX_STEPS):
                with torch.no_grad():
                    action_probs, value_estimate, hidden_state = self.network.forward(
                        state_tensor, hidden_state
                    )
                    # CONCEPT — Detach hidden state during rollout
                    #   We keep the rollout graph-free (no grad) for speed. During
                    #   ppo_update we rebuild the entire episode graph from scratch
                    #   starting at h_0=zeros, so storing the rollout graph is wasteful.
                    hidden_state = (hidden_state[0].detach(), hidden_state[1].detach())

                distribution = Categorical(action_probs)
                action_taken = distribution.sample()
                log_prob     = distribution.log_prob(action_taken)

                result      = env.step(action_taken.item())
                next_raw    = [int(result[0]), int(result[1])]
                reward      = float(result[2])
                done        = bool(result[3] > 0.5)
                next_tensor = state_to_tensor(next_raw, env)

                episode_states.append(state_tensor)
                episode_actions.append(action_taken.item())
                episode_log_probs.append(log_prob.detach())
                episode_rewards.append(reward)
                episode_dones.append(done)
                episode_values.append(value_estimate.detach())

                state_tensor = next_tensor

                if done:
                    if reward > 0:
                        reached_goal = True
                    break

            if len(episode_rewards) == 0:
                continue

            advantages, returns = _compute_gae(episode_rewards, episode_values, episode_dones)
            old_log_probs_tensor = torch.stack(episode_log_probs)

            _lstm_ppo_update(
                self.network,
                self.optimizer,
                episode_states,
                episode_actions,
                old_log_probs_tensor,
                returns,
                advantages,
            )

            if reached_goal and first_goal_episode == -1:
                first_goal_episode = episode_number

        return first_goal_episode


# ===========================================================================
# LSTM-aware evaluation
# ===========================================================================

def evaluate_lstm(adapter: LSTMPPOAdapter, eval_env: pathplanning.GridEnvironment, num_episodes: int):
    """Greedy evaluation that properly threads hidden state through the episode.

    CONCEPT — Why a separate LSTM evaluator?
        The shared evaluate() function calls greedy_action() per step with no
        memory of previous calls — treating each step as independent. For a flat
        MLP policy (DQN/SAC/PPO) this is correct. For the LSTM, the hidden state
        IS the memory, so ignoring it during evaluation would measure the policy
        with amnesia. This function passes (h_t, c_t) forward at each step, giving
        the LSTM the same memory it would have during a real episode.
    """
    success_count = 0
    total_steps_on_success = 0

    for _ in range(num_episodes):
        raw_state    = eval_env.reset()
        state_tensor = state_to_tensor(raw_state, eval_env)
        hidden_state = adapter.network.get_initial_hidden_state(batch_size=1)
        reached_goal = False

        for step_index in range(MAX_STEPS):
            with torch.no_grad():
                action_probs, _, hidden_state = adapter.network.forward(state_tensor, hidden_state)
                hidden_state = (hidden_state[0].detach(), hidden_state[1].detach())
            action = int(action_probs.argmax().item())

            result       = eval_env.step(action)
            raw_state    = [int(result[0]), int(result[1])]
            reward       = float(result[2])
            done         = bool(result[3] > 0.5)
            state_tensor = state_to_tensor(raw_state, eval_env)

            if done and reward > 0:
                reached_goal = True
                total_steps_on_success += step_index + 1
                break

        if reached_goal:
            success_count += 1

    success_rate = success_count / num_episodes
    avg_steps    = (total_steps_on_success / success_count) if success_count > 0 else float("nan")
    return success_rate, avg_steps


# ===========================================================================
# Multi-seed benchmark loop
# ===========================================================================

def run_single_seed(seed_index: int, algorithm_name: str, AdapterClass) -> dict:
    """Train one algorithm through the full curriculum, then evaluate on held-out maze.

    Returns a dict with keys: success_rate, avg_steps, first_goal_episode, train_time_seconds.
    """
    # CONCEPT — Seeding for reproducibility
    #   Python random, numpy (indirectly via torch), and PyTorch all have separate
    #   RNG states. Setting all three ensures the weight initialisations and any
    #   random choices inside the adapters are deterministic per seed_index.
    #   Maze seeds come from CURRICULUM_STAGES — they are fixed regardless of
    #   seed_index so all seeds see the same mazes.
    torch.manual_seed(seed_index * 100 + 42)
    random.seed(seed_index * 100 + 43)

    adapter = AdapterClass()
    train_start_time = time.time()

    # CONCEPT — Stage training accumulates first_goal across the full curriculum
    #   first_goal_episode tracks the FIRST episode in the FIRST stage where the
    #   agent reached the goal. Subsequent stages may have an even earlier first
    #   success (episode 1 if the easy-stage policy generalises), but we report
    #   the curriculum-global first contact with the reward signal.
    cumulative_episodes_before_stage = 0
    first_goal_episode_global = -1

    for stage in CURRICULUM_STAGES:
        stage_env = make_env(stage["obstacle_density"], stage["maze_seed"])
        adapter.swap_maze(stage_env)

        stage_first_goal = adapter.train_stage(stage_env, stage["episodes"])

        if first_goal_episode_global == -1 and stage_first_goal != -1:
            first_goal_episode_global = cumulative_episodes_before_stage + stage_first_goal

        cumulative_episodes_before_stage += stage["episodes"]

    train_elapsed_seconds = time.time() - train_start_time

    eval_env = make_eval_env()
    if isinstance(adapter, LSTMPPOAdapter):
        success_rate, avg_steps = evaluate_lstm(adapter, eval_env, EVAL_EPISODES)
    else:
        success_rate, avg_steps = evaluate(adapter, eval_env, EVAL_EPISODES)

    return {
        "success_rate":         success_rate,
        "avg_steps":            avg_steps,
        "first_goal_episode":   first_goal_episode_global,
        "train_time_seconds":   train_elapsed_seconds,
    }


# ===========================================================================
# Formatting helpers
# ===========================================================================

def _mean_std(values: list) -> tuple:
    """Return (mean, std) for a list of floats, skipping NaN values."""
    clean = [value for value in values if not (isinstance(value, float) and math.isnan(value))]
    if len(clean) == 0:
        return float("nan"), float("nan")
    mean = sum(clean) / len(clean)
    variance = sum((value - mean) ** 2 for value in clean) / max(len(clean) - 1, 1)
    return mean, math.sqrt(variance)


def _fmt_mean_std(mean: float, std: float, precision: int = 1) -> str:
    """Format a mean ± std pair as a fixed-width string."""
    if math.isnan(mean):
        return "N/A"
    return f"{mean:.{precision}f} ± {std:.{precision}f}"


# ===========================================================================
# CSV output
# ===========================================================================

def save_results_csv(all_results: dict) -> None:
    """Write per-seed results to python/data/benchmark_deep_rl_results.csv."""
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "benchmark_deep_rl_results.csv")

    fieldnames = [
        "algorithm",
        "seed",
        "success_rate",
        "avg_steps",
        "first_goal_episode",
        "train_time_seconds",
    ]

    with open(output_path, "w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for algorithm_name, seed_results in all_results.items():
            for seed_index, result in enumerate(seed_results):
                writer.writerow({
                    "algorithm":          algorithm_name,
                    "seed":               seed_index,
                    "success_rate":       f"{result['success_rate']:.4f}",
                    "avg_steps":          f"{result['avg_steps']:.1f}" if not math.isnan(result["avg_steps"]) else "nan",
                    "first_goal_episode": result["first_goal_episode"],
                    "train_time_seconds": f"{result['train_time_seconds']:.1f}",
                })

    print(f"\nResults saved to: {output_path}")


# ===========================================================================
# Main benchmark entry point
# ===========================================================================

def run() -> None:
    """Run the full benchmark: 4 algorithms × 3 seeds × 4 curriculum stages."""
    algorithms = [
        ("DQN",      DQNAdapter),
        ("SAC",      SACAdapter),
        ("PPO",      PPOAdapter),
        ("LSTM PPO", LSTMPPOAdapter),
    ]

    total_episodes_per_seed = sum(stage["episodes"] for stage in CURRICULUM_STAGES)

    print("\n" + "=" * 60)
    print("Deep RL Benchmark — curriculum training + held-out eval")
    print(f"  Algorithms:  {', '.join(name for name, _ in algorithms)}")
    print(f"  Curriculum:  {len(CURRICULUM_STAGES)} stages, {total_episodes_per_seed} episodes per seed")
    print(f"  Seeds:       {SEEDS}")
    print(f"  Eval:        {EVAL_EPISODES} greedy episodes on held-out maze")
    print(f"               (density={EVAL_DENSITY}, seed={EVAL_SEED})")
    print("=" * 60)

    # CONCEPT — Results structure
    #   all_results maps algorithm_name → list of per-seed dicts.
    #   After all seeds finish we compute column statistics (mean ± std) for the
    #   final table. Storing individual seed results also lets us write the CSV
    #   with full granularity — useful for offline analysis.
    all_results = {}

    for algorithm_name, AdapterClass in algorithms:
        print(f"\n{'─' * 60}")
        print(f"Algorithm: {algorithm_name}")
        print(f"{'─' * 60}")

        seed_results = []

        for seed_index in range(SEEDS):
            print(f"  Seed {seed_index + 1}/{SEEDS} — training {total_episodes_per_seed} episodes across {len(CURRICULUM_STAGES)} stages ...")
            result = run_single_seed(seed_index, algorithm_name, AdapterClass)

            first_goal_display = result["first_goal_episode"] if result["first_goal_episode"] != -1 else "never"
            print(
                f"    success={result['success_rate'] * 100:.1f}%  "
                f"avg_steps={result['avg_steps']:.1f}  "
                f"first_goal=ep{first_goal_display}  "
                f"time={result['train_time_seconds']:.0f}s"
            )
            seed_results.append(result)

        all_results[algorithm_name] = seed_results

    # ===========================================================
    # Print comparison table
    # ===========================================================

    print("\n" + "=" * 60)
    print(f"Deep RL Benchmark — {total_episodes_per_seed} eps, {len(CURRICULUM_STAGES)} curriculum stages, {SEEDS} seeds")
    print(f"Eval: {EVAL_EPISODES} greedy eps, held-out maze (density={EVAL_DENSITY}, seed={EVAL_SEED})")
    print("=" * 60)

    col_algorithm = 12
    col_success   = 16
    col_steps     = 16
    col_firstgoal = 16
    col_time      = 10

    header = (
        f"{'Algorithm':<{col_algorithm}}"
        f"{'Success%':<{col_success}}"
        f"{'Avg Steps':<{col_steps}}"
        f"{'First Goal':<{col_firstgoal}}"
        f"{'Time(s)':<{col_time}}"
    )
    print(header)
    print("-" * (col_algorithm + col_success + col_steps + col_firstgoal + col_time))

    for algorithm_name, seed_results in all_results.items():
        success_rates    = [result["success_rate"] * 100 for result in seed_results]
        avg_steps_list   = [result["avg_steps"]          for result in seed_results]
        first_goal_list  = [result["first_goal_episode"] for result in seed_results if result["first_goal_episode"] != -1]
        time_list        = [result["train_time_seconds"]  for result in seed_results]

        success_mean, success_std   = _mean_std(success_rates)
        steps_mean,   steps_std     = _mean_std(avg_steps_list)
        time_mean,    time_std      = _mean_std(time_list)

        if len(first_goal_list) > 0:
            fg_mean, fg_std = _mean_std([float(fg) for fg in first_goal_list])
            first_goal_str  = f"ep {fg_mean:.0f} ± {fg_std:.0f}"
        else:
            first_goal_str = "never"

        print(
            f"{algorithm_name:<{col_algorithm}}"
            f"{_fmt_mean_std(success_mean, success_std):<{col_success}}"
            f"{_fmt_mean_std(steps_mean, steps_std):<{col_steps}}"
            f"{first_goal_str:<{col_firstgoal}}"
            f"{_fmt_mean_std(time_mean, time_std, precision=0):<{col_time}}"
        )

    print("=" * 60)

    save_results_csv(all_results)


if __name__ == "__main__":
    run()
