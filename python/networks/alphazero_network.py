# python/networks/alphazero_network.py
# AlphaZero-style neural network — shared trunk, policy head, value head.
#
# CONCEPT — Why AlphaZero uses tanh instead of sigmoid on the value head
#   PPO/A2C value heads predict V(s) — an unbounded cumulative return,
#   so no output activation is used (raw linear).
#   AlphaZero reframes value as a game outcome: +1 = win, -1 = loss.
#   tanh squashes any real number into (-1, 1), which matches that contract.
#   Every backup in MCTS is interpreted as "how close to a win is this state?"
#   not "how much total reward do I expect?" — a subtle but important shift.
#
# CONCEPT — Why this architecture mirrors ActorCriticNetwork
#   The shared trunk + two-head split is identical to ActorCriticNetwork by
#   design. The key differences are:
#     1. Value activation: tanh here vs none in ActorCriticNetwork.
#     2. Training signal: MCTS visit counts (policy) + episode outcome (value)
#        instead of GAE advantages.
#   Keeping the shape identical means the two can be swapped in experiments
#   with a single import change.
#
# Self-driving analog:
#   Policy head = probability distribution over steering decisions.
#   Value head  = "how safe / how close to the destination is this situation?"
#                 scaled to [-1, 1] rather than raw expected future reward.

import torch
import torch.nn as nn


class AlphaZeroNetwork(nn.Module):

    def __init__(self, state_size: int, action_size: int, hidden_size: int = 128):
        super().__init__()

        # Shared trunk — extracts state features before the two heads diverge.
        self.shared_layer_one = nn.Linear(state_size, hidden_size)
        self.shared_layer_two = nn.Linear(hidden_size, hidden_size)

        # Policy head — outputs a probability distribution over actions.
        # Softmax is applied in forward() rather than baked into the layer so
        # the raw logits remain available for cross-entropy loss if needed.
        self.policy_head = nn.Linear(hidden_size, action_size)

        # Value head — outputs a single scalar in (-1, 1).
        # tanh enforces the AlphaZero value contract: win/loss outcome signal.
        self.value_head = nn.Linear(hidden_size, 1)

        self.relu = nn.ReLU()

    def forward(self, state_tensor: torch.Tensor):
        # CONCEPT — Why we don't separate trunk from heads with explicit calls
        #   A single forward() keeps the computation graph intact for
        #   autograd. Splitting it into helper methods would require careful
        #   detach() management — unnecessary complexity.
        features     = self.relu(self.shared_layer_one(state_tensor))
        features     = self.relu(self.shared_layer_two(features))
        policy_probs = torch.softmax(self.policy_head(features), dim=-1)
        value        = torch.tanh(self.value_head(features))
        return policy_probs, value
