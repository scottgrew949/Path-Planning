# python/train_ppo.py
# PPO training loop — collects full episodes, then updates the policy.
#
# CONCEPT — PPO training loop vs DQN training loop
#   DQN:  step → push to replay buffer → sample random batch → update.
#         Off-policy: trains on old experience from buffer.
#   PPO:  collect full episode (or N steps) → update on THAT data → discard.
#         On-policy: policy must be trained on data collected by the CURRENT policy.
#         Can't reuse old experience — policy changed, old data is stale.
#
# CONCEPT — Episode collection
#   Run one full episode, storing at each step:
#     state, action, log_prob, reward, done, value
#   After episode ends: compute returns and advantages, then call agent.update().
#
# CONCEPT — No epsilon in PPO
#   DQN needed epsilon-greedy because argmax is deterministic — no exploration.
#   PPO's actor outputs a probability distribution — sampling IS exploration.
#   Entropy bonus in the loss further encourages the distribution to stay spread out.

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import pathplanning
from networks.actor_critic import ActorCriticNetwork
from agents.ppo_agent       import PPOAgent

# ---- Hyperparameters --------------------------------------------------------

GRID_WIDTH    = 41
GRID_HEIGHT   = 41
STATE_SIZE    = 6       # [x/w, y/h, wall_up, wall_down, wall_left, wall_right]
ACTION_SIZE   = 4       # UP DOWN LEFT RIGHT
HIDDEN_SIZE   = 128

EPISODES      = 2000
MAX_STEPS     = GRID_WIDTH * GRID_HEIGHT * 4

LEARNING_RATE = 0.0003  # lower than DQN — PPO updates are already conservative
GAMMA         = 0.95
PPO_EPOCHS    = 4       # gradient steps per collected episode

# ---- Helpers ----------------------------------------------------------------

def state_to_tensor(state: list, width: int, height: int, env=None) -> torch.Tensor:
    normalised = [state[0] / width, state[1] / height]
    if env is not None:
        normalised += env.getLineOfSight(state[0], state[1])
    else:
        normalised += [0, 0, 0, 0]
    return torch.FloatTensor(normalised).unsqueeze(0)

# ---- Training loop ----------------------------------------------------------

def train():
    env     = pathplanning.GridEnvironment(GRID_WIDTH, GRID_HEIGHT, 0, 0, 38, 40, 0.3)
    network = ActorCriticNetwork(STATE_SIZE, ACTION_SIZE, HIDDEN_SIZE)
    agent   = PPOAgent(network, LEARNING_RATE, GAMMA, PPO_EPOCHS)

    for episode in range(1, EPISODES + 1):

        # -- Trajectory buffers (reset each episode) --------------------------
        states_list    = []
        actions_list   = []
        log_probs_list = []
        rewards_list   = []
        dones_list     = []
        values_list    = []

        # -- Collect one episode ----------------------------------------------
        state        = env.reset()
        state_tensor = state_to_tensor(state, GRID_WIDTH, GRID_HEIGHT, env)
        episode_reward = 0.0

        for step in range(MAX_STEPS):
            # TODO: call agent.select_action(state_tensor) → action, log_prob, value
            # TODO: call env.step(action) → result
            # TODO: unpack result into next_state, reward, done
            # TODO: append state_tensor, action, log_prob, reward, done, value to lists
            # TODO: update episode_reward
            # TODO: advance state_tensor
            # TODO: if done: break
            pass

        # -- Compute returns and advantages -----------------------------------
        # TODO: returns    = agent.compute_returns(rewards_list, dones_list)
        # TODO: values_tensor = torch.cat(values_list).squeeze()
        # TODO: advantages = returns - values_tensor.detach()
        # TODO: normalise advantages: (advantages - mean) / (std + 1e-8)

        # -- Stack trajectory into tensors ------------------------------------
        # TODO: states_tensor    = torch.cat(states_list)
        # TODO: actions_tensor   = torch.LongTensor(actions_list)
        # TODO: log_probs_tensor = torch.stack(log_probs_list).detach()

        # -- Update policy ----------------------------------------------------
        # TODO: loss = agent.update(states_tensor, actions_tensor,
        #                           log_probs_tensor, returns, advantages)

        if episode % 10 == 0 or episode <= 20:
            print(f"Episode {episode} | Reward: {episode_reward:.1f}")


if __name__ == "__main__":
    train()
