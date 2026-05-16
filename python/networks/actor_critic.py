# python/networks/actor_critic.py
# Actor-Critic neural network for PPO.
#
# CONCEPT — Why two heads instead of one?
#   DQN output: Q(s,a) — value of taking action a in state s.
#   Actor-Critic splits this into two separate questions:
#     Actor (policy head): given state s, what probability should I assign to each action?
#                          Output: 4 probabilities that sum to 1.0 (softmax).
#     Critic (value head): given state s, how good is this state overall?
#                          Output: single scalar V(s).
#   The critic's job is to tell the actor whether it did better or worse than expected.
#
# CONCEPT — Shared trunk
#   Both heads share the same feature extraction layers (trunk).
#   Features learned for value estimation are useful for policy too — no duplication.
#   Only the final layer differs per head.
#
# CONCEPT — Softmax vs argmax
#   DQN: argmax over Q-values — always pick the best known action (greedy).
#   Actor: softmax over logits — outputs a probability distribution.
#          Agent samples from it — naturally balances exploration and exploitation.
#          No epsilon needed.
#
# Self-driving analog:
#   Actor = steering/throttle decision given current sensor reading.
#   Critic = "was that a good situation to be in?" — safety score for the state.

import torch
import torch.nn as nn

class ActorCriticNetwork(nn.Module):

    def __init__(self, state_size: int, action_size: int, hidden_size: int = 128):
        super().__init__()
        # Shared trunk — extracts features from state before the split.
        self.shared_layer_one = nn.Linear(state_size, hidden_size)
        self.shared_layer_two = nn.Linear(hidden_size, hidden_size)

        # Policy head (actor) — outputs logits over actions, softmax applied in forward().
        self.policy_head = nn.Linear(hidden_size, action_size)

        # Value head (critic) — outputs single scalar V(s).
        self.value_head  = nn.Linear(hidden_size, 1)

        self.relu = nn.ReLU()

    def forward(self, state_tensor: torch.Tensor):
        features     = self.relu(self.shared_layer_one(state_tensor))
        features     = self.relu(self.shared_layer_two(features))
        action_probs = torch.softmax(self.policy_head(features), dim=-1)
        value        = self.value_head(features)
        return action_probs, value
