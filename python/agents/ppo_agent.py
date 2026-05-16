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
        self.epochs        = epochs   # how many gradient steps per collected trajectory
        self.optimizer     = torch.optim.Adam(network.parameters(), lr=learning_rate)

    def select_action(self, state_tensor: torch.Tensor):
        # TODO: call network(state_tensor) → action_probs, value
        # TODO: create torch.distributions.Categorical(action_probs)
        # TODO: sample action from distribution
        # TODO: compute log_prob of sampled action
        # TODO: return action.item(), log_prob, value
        pass

    def compute_returns(self, rewards: list, dones: list) -> torch.Tensor:
        # TODO: walk rewards list in reverse, accumulate discounted return
        #       G = 0
        #       for each step from last to first:
        #           if done: G = 0
        #           G = reward + gamma * G
        #           prepend G to returns list
        
        return torch.FloatTensor(returns)

    def update(self,
               states:     torch.Tensor,
               actions:    torch.Tensor,
               log_probs:  torch.Tensor,
               returns:    torch.Tensor,
               advantages: torch.Tensor) -> float:
        # TODO: loop self.epochs times:
        #   new_action_probs, new_values = network(states)
        #   dist = Categorical(new_action_probs)
        #   new_log_probs = dist.log_prob(actions)
        #   entropy = dist.entropy().mean()
        #
        #   ratio = exp(new_log_probs - old log_probs)
        #   clipped = clip(ratio, 1-CLIP_EPSILON, 1+CLIP_EPSILON)
        #   actor_loss = -min(ratio * advantages, clipped * advantages).mean()
        #
        #   critic_loss = MSE(new_values.squeeze(), returns)
        #
        #   loss = actor_loss + 0.5 * critic_loss - ENTROPY_COEF * entropy
        #   optimizer step
        # TODO: return loss.item()
        pass
