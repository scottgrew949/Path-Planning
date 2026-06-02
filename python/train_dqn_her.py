# python/train_dqn_her.py
# DQN training with Prioritized Experience Replay (PER) + Hindsight Experience Replay (HER).
#
# CONCEPT — Why HER for sparse reward?
#   The grid gives reward +100 only at the goal. Early DQN sees almost nothing
#   but -1 and -10 — sparse signal makes learning slow. HER fixes this by
#   relabelling failed episodes: "you didn't reach (38,40) but you DID reach
#   (12,7) — count that as a success with goal=(12,7)." The agent suddenly
#   receives positive signal from every episode, regardless of outcome.
#   This is exactly how a child learns — any goal achieved counts for something.
#
# CONCEPT — Goal-conditioned policy
#   State now includes [x/W, y/H, goal_x/W, goal_y/H, wall_up, ..., wall_right].
#   The network learns Q(state, goal, action) — a single network that can
#   navigate to ANY goal, not just the one hardcoded at training time.
#   This is far more general and maps directly to multi-agent CBS: each CBS
#   agent has its own goal, and a goal-conditioned policy handles all of them.
#
# CONCEPT — HER "future" strategy
#   For each real transition at timestep t in an episode of length T:
#     sample HER_K random future timesteps f ∈ [t+1, T-1]
#     use the position reached at f as a relabelled goal
#     compute sparse reward: +1 if next_pos == relabelled_goal, else -1
#     push the relabelled (state, action, reward, next_state, done) to buffer
#   This generates HER_K extra training examples per real step at zero env cost.
#
# CONCEPT — PER + HER interaction
#   HER transitions start with max_priority (same as real transitions).
#   After training, their TD error updates their priority just like real ones.
#   High-error hindsight transitions get sampled more — both mechanisms
#   reinforce each other toward the most informative experiences.
#
# Self-driving analog:
#   HER = a driver who learns from every trip, even the ones where they got
#   lost. "I didn't reach downtown, but I did successfully navigate to the
#   pharmacy — file that as a win and learn from it."

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import random
import torch
import torch.nn as nn
import torch.optim as optim
import pathplanning

from replay_buffer import PrioritizedReplayBuffer
from dqn_network   import DQNNetwork

# ---- Hyperparameters --------------------------------------------------------

GRID_WIDTH         = 41
GRID_HEIGHT        = 41
GOAL_X             = 38
GOAL_Y             = 40
STATE_SIZE         = 8      # [x/W, y/H, goal_x/W, goal_y/H, wall×4]
ACTION_SIZE        = 4
HIDDEN_SIZE        = 128

EPISODES           = 2000
MAX_STEPS          = GRID_WIDTH * GRID_HEIGHT * 4
BATCH_SIZE         = 64
BUFFER_CAPACITY    = 50000

LEARNING_RATE      = 0.001
GAMMA              = 0.95
EPSILON_START      = 1.0
EPSILON_MIN        = 0.05
EPSILON_DECAY      = 0.995
TARGET_UPDATE_FREQ = 100

PER_ALPHA          = 0.6    # priority exponent: 0=uniform, 1=full priority
BETA_START         = 0.4    # IS correction start: annealed → 1.0 over training
BETA_END           = 1.0

HER_K              = 4      # hindsight goals per real transition (future strategy)

# ---- Helpers ----------------------------------------------------------------

def state_to_tensor(pos_x: int, pos_y: int,
                    goal_x: int, goal_y: int,
                    width: int, height: int,
                    env=None) -> torch.Tensor:
    normalised = [
        pos_x  / width,
        pos_y  / height,
        goal_x / width,
        goal_y / height,
    ]
    if env is not None:
        normalised += env.getLineOfSight(pos_x, pos_y)
    else:
        normalised += [0.0, 0.0, 0.0, 0.0]
    return torch.FloatTensor(normalised).unsqueeze(0)


def select_action(state_tensor: torch.Tensor,
                  main_network: DQNNetwork,
                  epsilon: float) -> int:
    if random.random() < epsilon:
        return random.randint(0, ACTION_SIZE - 1)
    with torch.no_grad():
        return main_network(state_tensor).argmax().item()


def compute_loss(batch,
                 main_network:   DQNNetwork,
                 target_network: DQNNetwork,
                 gamma:          float):
    states, actions, rewards, next_states, dones, indices, weights = batch

    q_values = main_network(states)
    q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

    with torch.no_grad():
        best_actions  = main_network(next_states).argmax(1)
        next_q_values = target_network(next_states).gather(
                            1, best_actions.unsqueeze(1)).squeeze(1)

    targets   = rewards + gamma * next_q_values * (1 - dones)
    td_errors = (targets - q_values).detach().abs().cpu().numpy()

    # Weighted MSE: importance sampling weights correct PER's sampling bias.
    element_loss = nn.functional.mse_loss(q_values, targets, reduction='none')
    loss         = (weights * element_loss).mean()
    return loss, td_errors


def apply_her(trajectory: list, buffer: PrioritizedReplayBuffer,
              width: int, height: int, env) -> None:
    # trajectory: list of (pos_x, pos_y, action, next_x, next_y)
    n = len(trajectory)
    for t, (px, py, action, npx, npy) in enumerate(trajectory):
        # Sample up to HER_K future positions as relabelled goals.
        future_count  = min(HER_K, n - t - 1)
        if future_count <= 0:
            continue
        future_indices = random.sample(range(t + 1, n), future_count)
        for f in future_indices:
            _, _, _, goal_x, goal_y = trajectory[f]
            state      = state_to_tensor(px,  py,  goal_x, goal_y, width, height, env)
            next_state = state_to_tensor(npx, npy, goal_x, goal_y, width, height, env)
            her_done   = (npx == goal_x and npy == goal_y)
            her_reward = 1.0 if her_done else -1.0
            buffer.push(state, action, her_reward, next_state, her_done)

# ---- Training loop ----------------------------------------------------------

def train():
    env            = pathplanning.GridEnvironment(
                         GRID_WIDTH, GRID_HEIGHT, 0, 0, GOAL_X, GOAL_Y, 0.3)
    main_network   = DQNNetwork(STATE_SIZE, ACTION_SIZE, HIDDEN_SIZE)
    target_network = DQNNetwork(STATE_SIZE, ACTION_SIZE, HIDDEN_SIZE)
    target_network.load_state_dict(main_network.state_dict())
    optimizer      = optim.Adam(main_network.parameters(), lr=LEARNING_RATE)
    buffer         = PrioritizedReplayBuffer(BUFFER_CAPACITY, alpha=PER_ALPHA)

    epsilon    = EPSILON_START
    beta       = BETA_START
    total_steps = 0
    beta_increment = (BETA_END - BETA_START) / EPISODES

    for episode in range(1, EPISODES + 1):
        state      = env.reset()
        cur_x, cur_y = int(state[0]), int(state[1])
        state_tensor = state_to_tensor(cur_x, cur_y, GOAL_X, GOAL_Y,
                                       GRID_WIDTH, GRID_HEIGHT, env)
        episode_reward = 0.0
        trajectory     = []  # (pos_x, pos_y, action, next_x, next_y)

        for _ in range(MAX_STEPS):
            action = select_action(state_tensor, main_network, epsilon)
            result = env.step(action)
            next_x, next_y = int(result[0]), int(result[1])
            reward = float(result[2])
            done   = bool(result[3])

            next_state_tensor = state_to_tensor(next_x, next_y, GOAL_X, GOAL_Y,
                                                GRID_WIDTH, GRID_HEIGHT, env)

            # Push real transition to PER buffer.
            buffer.push(state_tensor, action, reward, next_state_tensor, done)
            trajectory.append((cur_x, cur_y, action, next_x, next_y))

            episode_reward += reward
            cur_x, cur_y  = next_x, next_y
            state_tensor  = next_state_tensor
            total_steps   += 1

            if buffer.is_ready(BATCH_SIZE):
                batch = buffer.sample(BATCH_SIZE, beta=beta)
                loss, td_errors = compute_loss(batch, main_network,
                                               target_network, GAMMA)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                buffer.update_priorities(batch[5], td_errors)

            if total_steps % TARGET_UPDATE_FREQ == 0:
                target_network.load_state_dict(main_network.state_dict())

            if done:
                break

        # Hindsight relabeling: generate HER_K extra transitions per real step.
        apply_her(trajectory, buffer, GRID_WIDTH, GRID_HEIGHT, env)

        epsilon = max(EPSILON_MIN, epsilon * EPSILON_DECAY)
        beta    = min(BETA_END, beta + beta_increment)

        if episode % 10 == 0 or episode <= 20:
            print(f"Episode {episode:4d} | Reward: {episode_reward:7.1f} "
                  f"| Epsilon: {epsilon:.3f} | Beta: {beta:.3f} "
                  f"| Buffer: {len(buffer)}")

    print("Training complete.")


if __name__ == "__main__":
    train()
