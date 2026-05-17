# python/agents/bc_agent.py
# Behavioral Cloning agent — pure supervised learning, no RL.
#
# CONCEPT — What is Behavioral Cloning?
#   The simplest form of imitation learning: treat the expert's trajectory as a
#   labelled dataset and train a classifier to reproduce those actions.
#   State s → action a, just like image → label in image classification.
#   No reward signal, no environment interaction during training.
#   The agent learns to mimic the expert directly from demonstrations.
#
# CONCEPT — Why Cross-Entropy loss?
#   The policy network outputs raw logits over 4 actions (UP/DOWN/LEFT/RIGHT).
#   We want to maximise the probability assigned to the expert's chosen action.
#   Cross-entropy = -log P(correct_action) — minimising it pushes the network
#   to assign higher probability to the action the expert actually took.
#   Same loss as any multi-class classifier (e.g. digit recognition in MNIST).
#
# CONCEPT — Limitation: distributional shift
#   BC is trained on states visited by the EXPERT. At test time the agent makes
#   mistakes and ends up in states the expert never visited — the policy has no
#   guidance there and errors compound. DAgger (train_dagger.py) fixes this by
#   iteratively adding expert labels on states the AGENT visits.

import torch
import torch.nn as nn


def encode_state(x: float, y: float,
                 gx: float, gy: float,
                 width: int, height: int,
                 los: list) -> torch.FloatTensor:
    # State = [los_up, los_down, los_left, los_right, dx_to_goal, dy_to_goal].
    # Line-of-sight encodes local wall topology — the same (x,y) in different
    # mazes has different walls, so including los makes the state Markov.
    # Goal delta tells the agent which direction reduces distance to goal.
    # Returns shape (1, 6) — batch dim included so tensors stack correctly.
    return torch.FloatTensor([
        los[0], los[1], los[2], los[3],
        (gx - x) / width,
        (gy - y) / height,
    ]).unsqueeze(0)


class BCAgent:

    def __init__(self,
                 state_size:  int   = 6,
                 action_size: int   = 4,
                 hidden_size: int   = 128,
                 lr:          float = 1e-3):
        # 3-layer MLP: state → hidden → hidden → logits over actions.
        # ReLU between layers; no activation on output (CE loss wants raw logits).
        self.network = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_size),
        )
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=lr,
                                          weight_decay=1e-4)
        self._loss_fn  = nn.CrossEntropyLoss()

    def select_action(self, state_tensor: torch.Tensor) -> int:
        # Greedy inference — no gradient needed, no exploration noise.
        with torch.no_grad():
            logits = self.network(state_tensor)
        return logits.argmax(dim=-1).item()

    def update(self,
               states_tensor:  torch.Tensor,
               actions_tensor: torch.Tensor) -> float:
        # Cross-entropy between predicted logits and integer expert action labels.
        logits = self.network(states_tensor)
        loss   = self._loss_fn(logits, actions_tensor)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()
