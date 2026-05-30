# python/agents/ppo_agent.py
# PPO update logic — collects trajectories, computes advantages, clips updates.
#
# CONCEPT — Why PPO over vanilla Actor-Critic?
#   Vanilla policy gradient: take one big gradient step on collected experience,
#   then throw the data away. Problem: a bad step can destroy the policy —
#   no recovery from a too-large update.
#   PPO fix: clip the update ratio so the policy can't move too far from its
#   previous version in a single step. Stable, sample-efficient, industry standard.
#
# CONCEPT — The clip
#   ratio = new_policy(a|s) / old_policy(a|s)
#   If ratio > 1+clip, action is more likely now than before — don't reward further.
#   If ratio < 1-clip, action is less likely now — don't penalise further.
#   Clipped loss = min(ratio * advantage, clip(ratio, 1-ε, 1+ε) * advantage)
#   This keeps updates conservative — no single batch destroys the policy.
#
# CONCEPT — Advantage A(s,a)
#   A(s,a) = reward_to_go - V(s)
#   "How much better was this action than what the critic expected?"
#   Positive advantage → do this action more.
#   Negative advantage → do this action less.
#
# CONCEPT — Reward to go (discounted return)
#   G_t = r_t + gamma*r_{t+1} + gamma^2*r_{t+2} + ...
#   Future rewards worth less than immediate (discount factor gamma < 1).
#   Same Bellman intuition as Q-learning — delayed gratification discounted.

import torch
import torch.nn as nn
from torch.distributions import Categorical
from networks.actor_critic import ActorCriticNetwork

CLIP_EPSILON = 0.2   # standard PPO clip range
ENTROPY_COEF = 0.01  # entropy bonus — encourages exploration

class PPOAgent:

    def __init__(self,
                 network:       ActorCriticNetwork,
                 learning_rate: float,
                 gamma:         float,
                 epochs:        int):
        self.network       = network
        self.gamma         = gamma
        self.epochs        = epochs
        self.optimizer     = torch.optim.Adam(network.parameters(), lr=learning_rate)

    def select_action(self, state_tensor: torch.Tensor):
        action_probs, value = self.network(state_tensor)
        dist                = Categorical(action_probs)
        action              = dist.sample()
        log_prob            = dist.log_prob(action)
        return action.item(), log_prob, value

    def compute_returns(self, rewards: list, dones: list) -> torch.Tensor:
        returns = []
        G = 0.0
        for reward, done in zip(reversed(rewards), reversed(dones)):
            if done:
                G = 0.0
            G = reward + self.gamma * G
            returns.insert(0, G)
        return torch.FloatTensor(returns)

    def update(self,
               states:     torch.Tensor,
               actions:    torch.Tensor,
               log_probs:  torch.Tensor,
               returns:    torch.Tensor,
               advantages: torch.Tensor) -> float:
        total_loss = 0.0
        for _ in range(self.epochs):
            new_action_probs, new_values = self.network(states)
            dist          = Categorical(new_action_probs)
            new_log_probs = dist.log_prob(actions)
            entropy       = dist.entropy().mean()

            ratio       = torch.exp(new_log_probs - log_probs)
            clipped     = torch.clamp(ratio, 1.0 - CLIP_EPSILON, 1.0 + CLIP_EPSILON)
            actor_loss  = -torch.min(ratio * advantages, clipped * advantages).mean()

            critic_loss = nn.functional.mse_loss(new_values.squeeze(-1), returns)

            loss = actor_loss + 0.5 * critic_loss - ENTROPY_COEF * entropy

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss = loss.item()

        return total_loss
