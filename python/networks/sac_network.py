# python/networks/sac_network.py
# SAC actor and twin critic networks for discrete-action environments.
#
# CONCEPT — Why SAC over DQN or PPO?
#   DQN:  off-policy, discrete, epsilon-greedy exploration (heuristic, not learned).
#   PPO:  on-policy, discards the replay buffer after each update — sample-inefficient.
#   SAC:  off-policy (reuses buffer like DQN) + entropy baked into the objective:
#
#       J(pi) = E[ sum_t  reward_t  +  alpha * H(pi(·|s_t)) ]
#
#   H(pi) is the entropy of the action distribution — high entropy means the policy
#   stays spread across actions rather than collapsing to one. Alpha (temperature)
#   controls the exploration-exploitation tradeoff and is itself learned.
#   No epsilon needed: the agent is trained to be appropriately uncertain.
#
# CONCEPT — Twin critics
#   SAC keeps TWO independent Q-networks. Every Bellman target uses min(Q1, Q2).
#   Why? A single critic tends to overestimate — it grades its own predictions.
#   Taking the minimum of two independent networks systematically dampens that bias.
#   This is the same trick used in TD3 for continuous control.
#
# CONCEPT — Categorical actor (discrete version)
#   Continuous SAC uses a Gaussian + reparameterisation trick.
#   This environment has 4 discrete actions, so we use a categorical actor:
#     forward() → logits → softmax → Categorical(probs) → sample
#   Entropy of Categorical: H = -sum(p * log p). Maximising H spreads probability
#   mass evenly — the agent explores by default, exploits as evidence accumulates.
#
# CONCEPT — Temperature alpha
#   alpha is a learned scalar. Its gradient comes from:
#     alpha_loss = -log_alpha * (log_prob + target_entropy).detach()
#   If current log_prob < -target_entropy (agent is too greedy): alpha grows,
#   pushing the actor loss toward higher entropy (more exploration).
#   If current log_prob > -target_entropy (agent is too random): alpha shrinks.
#   target_entropy = log(action_size) — entropy of the uniform distribution over actions.
#
# Self-driving analog:
#   SAC is how a car stays "humble" — it maintains uncertainty across lane choices
#   until enough evidence forces it to commit. Prevents overconfident crashes.

import torch
import torch.nn as nn
from torch.distributions import Categorical


class SACActor(nn.Module):
    """
    Categorical policy: state → action logits.
    get_action_and_log_prob() samples an action and returns its log probability
    and the full distribution entropy — both are needed by the SAC actor loss.
    """

    def __init__(self, state_size: int, action_size: int, hidden_size: int = 128):
        super().__init__()

        # CONCEPT — Shared trunk
        #   Two Linear+ReLU layers extract features from the raw state vector before
        #   the output head. Same architecture as ActorCriticNetwork so the learned
        #   representations are comparable across agents in this project.
        self.trunk_layer_one = nn.Linear(state_size, hidden_size)
        self.trunk_layer_two = nn.Linear(hidden_size, hidden_size)

        # Output: raw logits — softmax is applied inside get_action_and_log_prob()
        # rather than here, so we can pass logits to Categorical directly.
        self.logits_layer = nn.Linear(hidden_size, action_size)

        self.relu = nn.ReLU()

    def forward(self, state_tensor: torch.Tensor) -> torch.Tensor:
        """Raw logits of shape [batch, action_size]. Softmax is NOT applied here."""
        features = self.relu(self.trunk_layer_one(state_tensor))
        features = self.relu(self.trunk_layer_two(features))
        return self.logits_layer(features)

    def get_action_and_log_prob(self, state_tensor: torch.Tensor):
        """
        Sample one action from the policy distribution and return its log probability
        plus the full distribution entropy.

        Returns:
            action_index  (int)           — sampled discrete action
            log_prob      (Tensor scalar) — log pi(action | state)
            entropy       (Tensor scalar) — H(pi(·|state)), averaged over batch

        CONCEPT — Why entropy here?
            The SAC actor loss is:  alpha * log_prob - min(Q1, Q2)[action]
            Equivalently it minimises negative entropy + Q-weighted term.
            Returning entropy separately lets the training loop log it as a
            diagnostic: entropy dropping to near-zero means the policy is
            collapsing to one action — a warning sign of premature convergence.
        """
        logits = self.forward(state_tensor)
        action_probs = torch.softmax(logits, dim=-1)

        # CONCEPT — Categorical distribution
        #   Categorical(probs) creates a distribution over {0, 1, 2, 3}.
        #   sample() draws one action index proportional to the probabilities.
        #   log_prob(action) returns log pi(action) — used in the SAC loss.
        #   entropy() returns H = -sum(p * log p) — the full distribution entropy.
        distribution = Categorical(action_probs)
        sampled_action = distribution.sample()
        log_prob = distribution.log_prob(sampled_action)
        entropy = distribution.entropy()

        return sampled_action.item(), log_prob, entropy


class SACCritic(nn.Module):
    """
    Q-network: state → Q-value for every action.

    Returns a vector of shape [batch, action_size] — one Q-value per action.
    Instantiate twice (critic_one, critic_two) in the agent. They have
    independent weights and are trained independently; only their outputs
    are combined via min() when computing Bellman targets.

    CONCEPT — Why not a single Q(s,a) with action as input?
        Passing the action as part of the input requires a separate forward pass
        per action (slow). Outputting all Q-values at once means one forward pass
        gives Q-values for all actions — efficient for both critic training and
        actor loss computation.
    """

    def __init__(self, state_size: int, action_size: int, hidden_size: int = 128):
        super().__init__()
        self.trunk_layer_one = nn.Linear(state_size, hidden_size)
        self.trunk_layer_two = nn.Linear(hidden_size, hidden_size)
        self.q_values_layer  = nn.Linear(hidden_size, action_size)
        self.relu = nn.ReLU()

    def forward(self, state_tensor: torch.Tensor) -> torch.Tensor:
        """Q-values of shape [batch, action_size]."""
        features = self.relu(self.trunk_layer_one(state_tensor))
        features = self.relu(self.trunk_layer_two(features))
        return self.q_values_layer(features)
