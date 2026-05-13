# python/train_dqn.py
# DQN training loop — ties together GridEnvironment, ReplayBuffer, DQNNetwork.
#
# CONCEPT — DQN training loop (one episode)
#   1. reset()              → initial state
#   2. select action        → epsilon-greedy on main_network output
#   3. step(action)         → (next_state, reward, done)
#   4. push to buffer       → store experience
#   5. sample batch         → random experiences from buffer
#   6. compute Bellman target → reward + gamma * max(target_network(next_state))
#   7. compute loss         → MSE between main_network(state)[action] and target
#   8. gradient step        → optimizer.zero_grad(); loss.backward(); optimizer.step()
#   9. update target net    → every TARGET_UPDATE_FREQ steps, copy main → target
#
# CONCEPT — Loss function
#   Mean Squared Error between predicted Q-value and Bellman target.
#   MSE = (Q_predicted - Q_target)^2
#   Gradient descent minimises this — pushes Q-values toward the truth.
#
# CONCEPT — Why copy main → target periodically?
#   If the target updates every step, the Bellman target moves every step —
#   the network is chasing a moving goalpost. Freezing target for N steps
#   stabilises training. Main network converges toward a fixed reference.

from replay_buffer import ReplayBuffer
from dqn_network   import DQNNetwork
from value_heatmap import plot_value_heatmap

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.optim as optim
import random
import pathplanning

# ---- Hyperparameters --------------------------------------------------------

GRID_WIDTH        = 41
GRID_HEIGHT       = 41
STATE_SIZE        = 6        # normalised [x, y] + [wall_up, wall_down, wall_left, wall_right]
ACTION_SIZE       = 4        # UP DOWN LEFT RIGHT
HIDDEN_SIZE       = 128

EPISODES          = 2000
MAX_STEPS         = GRID_WIDTH * GRID_HEIGHT * 4
BATCH_SIZE        = 64
BUFFER_CAPACITY   = 10000

LEARNING_RATE     = 0.001
GAMMA             = 0.95     # discount factor
EPSILON_START     = 1.0
EPSILON_MIN       = 0.05
EPSILON_DECAY     = 0.995
TARGET_UPDATE_FREQ = 100     # copy main → target every N steps

# ---- Helpers ----------------------------------------------------------------

def state_to_tensor(state: list, width: int, height: int, env=None) -> torch.Tensor:
    normalised = [state[0] / width, state[1] / height]
    if env is not None:
        normalised += env.getLineOfSight(state[0], state[1])
    else:
        normalised += [0, 0, 0, 0]
    return torch.FloatTensor(normalised).unsqueeze(0)

def select_action(state_tensor: torch.Tensor,
                  main_network: DQNNetwork,
                  epsilon: float) -> int:
    
    if random.random() < epsilon:
        return random.randint(0, ACTION_SIZE - 1)
    with torch.no_grad():
        q_values = main_network(state_tensor)
    return q_values.argmax().item()
    #
    # CONCEPT — torch.no_grad()
    #   Disables gradient tracking during inference — faster and uses less memory.
    #   Gradients are only needed during loss.backward(), not during action selection.

def compute_loss(batch,
                 main_network:   DQNNetwork,
                 target_network: DQNNetwork,
                 gamma:          float) -> torch.Tensor:
    
    states, actions, rewards, next_states, dones = batch
    q_values = main_network(states)
    q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
    with torch.no_grad():
        best_actions  = main_network(next_states).argmax(1)
        next_q_values = target_network(next_states).gather(1, best_actions.unsqueeze(1)).squeeze(1)
    target = rewards + gamma * next_q_values * (1 - dones)
    return nn.MSELoss()(q_values, target)

# ---- Training loop ----------------------------------------------------------

def train():
    env            = pathplanning.GridEnvironment(GRID_WIDTH, GRID_HEIGHT, 0, 0, 38, 40, 0.3)
    main_network   = DQNNetwork(STATE_SIZE, ACTION_SIZE, HIDDEN_SIZE)
    target_network = DQNNetwork(STATE_SIZE, ACTION_SIZE, HIDDEN_SIZE)
    target_network.load_state_dict(main_network.state_dict())
    optimizer      = optim.Adam(main_network.parameters(), lr=LEARNING_RATE)
    buffer         = ReplayBuffer(BUFFER_CAPACITY)
    epsilon        = EPSILON_START
    total_steps    = 0

    # EPISODE LOOP
    for episode in range(1, EPISODES + 1):
        state = env.reset()
        state_tensor = state_to_tensor(state, GRID_WIDTH, GRID_HEIGHT, env)
        episode_reward = 0.0
        for step in range(MAX_STEPS):
            action = select_action(state_tensor, main_network, epsilon)
            result = env.step(action)
            next_state = [int(result[0]), int(result[1])]
            reward = result[2]
            done = bool(result[3])

            buffer.push(state_tensor, action, reward,
            state_to_tensor(next_state, GRID_WIDTH, GRID_HEIGHT, env), done)
            episode_reward += reward
            state_tensor = state_to_tensor(next_state, GRID_WIDTH, GRID_HEIGHT, env)
            total_steps += 1     

            if buffer.is_ready(BATCH_SIZE):
                batch = buffer.sample(BATCH_SIZE)
                loss = compute_loss(batch, main_network, target_network, GAMMA)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step() 
            if total_steps % TARGET_UPDATE_FREQ == 0:
                target_network.load_state_dict(main_network.state_dict())
            if done:
                break
        epsilon = max(EPSILON_MIN, epsilon * EPSILON_DECAY)
        if episode % 10 == 0 or episode <= 20:
            print(f"Episode {episode} | Reward: {episode_reward:.1f} | Epsilon: {epsilon:.3f}")
    
    plot_value_heatmap(main_network, env, GRID_WIDTH, GRID_HEIGHT)


if __name__ == "__main__":
    train()
