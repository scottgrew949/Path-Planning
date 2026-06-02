# python/networks/world_model.py
# Neural networks that form the World Model: dynamics, reward prediction, and policy.
#
# CONCEPT — What is a World Model?
#   A World Model is a neural approximation of the environment's transition function.
#   Instead of asking "what happens when I take action a in state s?" by actually
#   doing it in the real world, the agent asks its learned model — which runs in
#   milliseconds and has no real cost.
#
#   Three separate networks each answer one question:
#     DynamicsNetwork  — "if I am in state s and take action a, what state do I land in?"
#     RewardNetwork    — "if I am in state s and take action a, what reward do I receive?"
#     PolicyNetwork    — "given state s, which action should I take?"
#
#   The policy is then trained entirely inside the model's "imagination" — no real
#   environment steps required during policy learning.
#
# CONCEPT — Why separate dynamics and reward networks?
#   In principle one network could predict (next_state, reward) jointly, but splitting
#   them keeps each task simple and lets you tune capacity independently.
#   Dynamics is a harder function (6-dim output); reward is a scalar regression — a
#   smaller network suffices.
#
# CONCEPT — State + action concatenation
#   The dynamics and reward networks both receive [state | action_one_hot] as input.
#   Concatenation is the standard way to condition a network on two inputs when they
#   live in different spaces (continuous state vs discrete action index).
#   One-hot encoding of the action avoids giving the integer index a spurious
#   ordinal meaning (e.g. action 3 is NOT "bigger" than action 1).
#
# Self-driving analog:
#   DynamicsNetwork = physics simulator: given current speed/heading and throttle,
#                     predict where the car ends up.
#   RewardNetwork   = collision detector: given current position and steering, predict
#                     whether this move is safe.
#   PolicyNetwork   = the planner that picks manoeuvres using the simulator's output,
#                     not live road tests.

import torch
import torch.nn as nn


def action_to_onehot(action_index: int, action_size: int = 4) -> torch.FloatTensor:
    # CONCEPT — One-hot encoding
    #   Converts a discrete integer (e.g. action=2) into a vector with a single 1.0
    #   at that position and 0.0 everywhere else: [0, 0, 1, 0].
    #   This removes any implicit ordering between action indices that raw integers
    #   would introduce in the network's weight matrix.
    one_hot = torch.zeros(1, action_size)
    one_hot[0, action_index] = 1.0
    return one_hot


class DynamicsNetwork(nn.Module):
    # CONCEPT — Transition model f(s, a) → position'
    #   This network learns the movement part of the transition function:
    #     given (current_state, action), predict the normalised next position (x', y').
    #
    #   Output is POSITION ONLY (2-dim), not the full state vector.
    #   Why? The other 4 state dimensions are wall flags — those depend on the maze
    #   structure, not on the agent's movement. Predicting them would require the
    #   network to memorise the specific obstacle layout of the training maze, which
    #   makes the model maze-specific and prevents any generalisation.
    #   Instead, wall flags are looked up from a pre-built wall map after each step
    #   (see build_wall_map() in train_world_model.py) — the maze structure is stored
    #   exactly once and queried rather than approximated.
    #
    #   Input:  state (6-dim) concatenated with action one-hot (4-dim) → 10-dim
    #   Output: predicted normalised next position [x/width, y/height] (2-dim)

    def __init__(self, state_size: int, action_size: int, hidden_size: int = 128,
                 position_size: int = 2):
        super().__init__()
        input_size = state_size + action_size  # 6 + 4 = 10

        self.hidden_layer_one = nn.Linear(input_size, hidden_size)
        self.hidden_layer_two = nn.Linear(hidden_size, hidden_size)
        self.output_layer     = nn.Linear(hidden_size, position_size)
        self.relu             = nn.ReLU()

    def forward(self, state_tensor: torch.Tensor, action_onehot: torch.Tensor) -> torch.Tensor:
        combined_input      = torch.cat([state_tensor, action_onehot], dim=1)
        hidden_features     = self.relu(self.hidden_layer_one(combined_input))
        hidden_features     = self.relu(self.hidden_layer_two(hidden_features))
        predicted_position  = self.output_layer(hidden_features)
        return predicted_position  # shape: [batch, 2] — normalised (x', y')


class RewardNetwork(nn.Module):
    # CONCEPT — Reward model r(s, a) → r
    #   Learns to predict the scalar reward the environment returns for a given
    #   (state, action) pair. Trained alongside the dynamics network on the same
    #   real experience buffer.
    #
    #   Simpler than DynamicsNetwork: reward is a single scalar so one hidden
    #   layer with half the width is sufficient capacity.
    #
    #   During imagination, the agent uses this network instead of querying the
    #   real environment for reward — making rollouts fully self-contained.

    def __init__(self, state_size: int, action_size: int, hidden_size: int = 64):
        super().__init__()
        input_size = state_size + action_size  # 6 + 4 = 10

        self.hidden_layer = nn.Linear(input_size, hidden_size)
        self.output_layer = nn.Linear(hidden_size, 1)
        self.relu         = nn.ReLU()

    def forward(self, state_tensor: torch.Tensor, action_onehot: torch.Tensor) -> torch.Tensor:
        combined_input    = torch.cat([state_tensor, action_onehot], dim=1)
        hidden_features   = self.relu(self.hidden_layer(combined_input))
        predicted_reward  = self.output_layer(hidden_features)
        return predicted_reward  # shape: [batch, 1]


class PolicyNetwork(nn.Module):
    # CONCEPT — Policy network π(s) → action logits
    #   A lightweight policy that maps state → distribution over actions.
    #   This is the network being *trained* via imagined rollouts.
    #
    #   Architecture mirrors the actor head in ActorCriticNetwork: two shared
    #   hidden layers feeding into an action logit output. Softmax is applied
    #   externally (in the training loop) so raw logits are returned — this
    #   avoids numerical issues when computing log_softmax for REINFORCE.
    #
    #   CONCEPT — Why train the policy in imagination?
    #   Interacting with the real environment is expensive (each step may involve
    #   physical hardware, simulation time, or API calls). Once the world model
    #   is accurate, you can generate thousands of (state, action, reward)
    #   sequences from imagination in the time a single real episode would take.
    #   This is the core efficiency gain of model-based RL.

    def __init__(self, state_size: int, action_size: int, hidden_size: int = 128):
        super().__init__()

        self.hidden_layer_one = nn.Linear(state_size, hidden_size)
        self.hidden_layer_two = nn.Linear(hidden_size, hidden_size)
        self.output_layer     = nn.Linear(hidden_size, action_size)
        self.relu             = nn.ReLU()

    def forward(self, state_tensor: torch.Tensor) -> torch.Tensor:
        hidden_features = self.relu(self.hidden_layer_one(state_tensor))
        hidden_features = self.relu(self.hidden_layer_two(hidden_features))
        action_logits   = self.output_layer(hidden_features)
        return action_logits  # shape: [batch, action_size] — raw logits, no softmax
