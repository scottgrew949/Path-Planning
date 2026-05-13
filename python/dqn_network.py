# python/dqn_network.py
# DQN neural network — replaces the Q-table with a function approximator.
#
# CONCEPT — Why a neural net instead of a table?
#   The Q-table works for a 41x41 grid (1681 states). But a real autonomous
#   vehicle has millions of possible states (position + velocity + nearby
#   obstacles + sensor readings). A table can't scale.
#   A neural net approximates Q(s,a) for ANY state — including ones never
#   seen during training. It generalises.
#
# CONCEPT — Dueling network architecture
#   Input:  state vector — [x/width, y/height, wall_up, wall_down, wall_left, wall_right]
#           Normalised to [0,1] so large values don't dominate gradients.
#   Trunk:  two shared fully connected layers with ReLU activation.
#   Split into two heads:
#     Value head:     single output V(s) — how good is this state regardless of action?
#     Advantage head: 4 outputs A(s,a)  — how much better is each action vs the average?
#   Combined: Q(s,a) = V(s) + A(s,a) - mean(A(s))
#     Subtracting mean(A) prevents V and A drifting in opposite directions (identifiability).
#   Output: 4 Q-values — agent picks argmax (greedy action).
#
# CONCEPT — Target network
#   DQN uses TWO networks with identical architecture:
#     main_network   — updated every step via gradient descent
#     target_network — frozen copy, updated every N steps by copying main weights
#   The Bellman target uses target_network to compute max Q(s').
#   Without this, the target moves every step — like chasing a moving goalpost.
#   The target network holds the goalpost still for N steps at a time.
#
# Self-driving analog:
#   The network is the car's learned intuition — "this intersection looks like
#   ones where turning left was bad, so don't turn left." Generalises from
#   past experience to new situations.

import torch
import torch.nn as nn


class DQNNetwork(nn.Module):

    def __init__(self, state_size: int, action_size: int, hidden_size: int = 128):
        super().__init__()
        self.shared_layer_one = nn.Linear(state_size, hidden_size)
        self.shared_layer_two = nn.Linear(hidden_size, hidden_size)
        self.value_head       = nn.Linear(hidden_size, 1)
        self.advantage_head   = nn.Linear(hidden_size, action_size)

    def forward(self, state_tensor: torch.Tensor) -> torch.Tensor:
        features = torch.relu(self.shared_layer_one(state_tensor))
        features = torch.relu(self.shared_layer_two(features))
        value     = self.value_head(features)      # shape [batch, 1]
        advantage = self.advantage_head(features)  # shape [batch, 4]
        return value + advantage - advantage.mean(dim=1, keepdim=True)