# python/agents/sac_agent.py
# SAC update logic — twin critics, soft policy improvement, learned temperature.
#
# CONCEPT — Why four critic networks?
#   Two main critics (critic_one, critic_two) are trained directly via gradient descent.
#   Two target critics (target_critic_one, target_critic_two) are their slowly-moving
#   copies, updated only via polyak averaging, never by gradients.
#   The targets provide stable Bellman bootstrapping — training the main critics against
#   their own outputs would chase a moving goalpost and diverge. Freezing a lagged copy
#   gives a stable reference.
#
# CONCEPT — Polyak (soft) update vs hard copy
#   DQN copies main → target every N steps (hard update).
#   SAC updates every step with:
#     theta_target = tau * theta_main + (1 - tau) * theta_target
#   tau = 0.005 means the target moves 0.5% toward the main network each step.
#   Smoother than hard copies — less variance in the Bellman target.
#
# CONCEPT — Critic update (Bellman backup with entropy)
#   Standard Bellman: y = r + gamma * max_a Q(s', a) * (1 - done)
#   SAC Bellman adds the entropy bonus:
#     y = r + gamma * (min(Q1_target(s', a*), Q2_target(s', a*)) - alpha * log_pi(a*|s'))
#   where a* ~ pi(·|s') is sampled from the CURRENT actor on the NEXT state.
#   The subtraction of alpha * log_prob makes high-entropy actions more attractive —
#   the agent is rewarded for staying uncertain in uncertain states.
#
# CONCEPT — Actor update (entropy-regularised policy improvement)
#   Actor loss: E[ alpha * log_pi(a|s) - min(Q1(s,a), Q2(s,a)) ]
#   Minimising this = maximise Q-values AND maximise entropy simultaneously.
#   The actor wants actions that are both high-value AND the policy stays spread.
#
# CONCEPT — Alpha (temperature) update
#   alpha_loss = -log_alpha * (log_prob + target_entropy).detach()
#   If log_prob > -target_entropy (too deterministic): gradient pushes alpha up.
#   If log_prob < -target_entropy (too random):        gradient pushes alpha down.
#   Using log_alpha instead of alpha directly ensures alpha stays positive (exp(x) > 0).
#
# Self-driving analog:
#   The twin critics are like two independent safety evaluators — a car uses the
#   more conservative score to avoid overconfident manoeuvres. The temperature
#   auto-tunes how cautious the car is: in familiar territory it commits; in
#   novel situations it keeps options open.

import math
import torch
import torch.nn as nn
from networks.sac_network import SACActor, SACCritic


class SACAgent:

    def __init__(self,
                 actor:           SACActor,
                 critic_one:      SACCritic,
                 critic_two:      SACCritic,
                 learning_rate:   float,
                 gamma:           float,
                 tau:             float,
                 action_size:     int):

        self.actor      = actor
        self.critic_one = critic_one
        self.critic_two = critic_two
        self.gamma      = gamma
        self.tau        = tau

        # CONCEPT — Target critics
        #   Deep copies of the main critics — same architecture, same initial weights.
        #   They receive no gradient updates; only polyak averaging touches them.
        import copy
        self.target_critic_one = copy.deepcopy(critic_one)
        self.target_critic_two = copy.deepcopy(critic_two)

        # Freeze target parameters — no accidental gradient updates.
        for parameter in self.target_critic_one.parameters():
            parameter.requires_grad = False
        for parameter in self.target_critic_two.parameters():
            parameter.requires_grad = False

        # CONCEPT — Learned temperature (log_alpha)
        #   We optimise log_alpha rather than alpha to guarantee alpha > 0 at all
        #   times (exp of any real number is strictly positive). The alpha property
        #   below converts back for use in loss computations.
        self.log_alpha = torch.tensor(0.0, requires_grad=True)

        # CONCEPT — Target entropy
        #   Set to the entropy of a uniform distribution over all actions:
        #   H_uniform = log(action_size)
        #   This is the entropy the agent would have if it had no preference — a
        #   reasonable lower bound for how exploratory we want the policy to stay.
        self.target_entropy = math.log(action_size)

        # Separate optimisers — actor, critics, and alpha are independent problems.
        self.actor_optimizer   = torch.optim.Adam(actor.parameters(),      lr=learning_rate)
        self.critic_optimizer  = torch.optim.Adam(
            list(critic_one.parameters()) + list(critic_two.parameters()),
            lr=learning_rate
        )
        self.alpha_optimizer   = torch.optim.Adam([self.log_alpha],        lr=learning_rate)

    @property
    def alpha(self) -> torch.Tensor:
        """Current temperature scalar (always positive)."""
        return self.log_alpha.exp()

    def select_action(self, state_tensor: torch.Tensor) -> int:
        """
        Sample an action from the current policy. Used during environment interaction,
        not during gradient updates — no gradients needed here.
        """
        with torch.no_grad():
            action_index, _log_prob, _entropy = self.actor.get_action_and_log_prob(state_tensor)
        return action_index

    def update(self, batch) -> tuple:
        """
        One full SAC update step: critic → actor → alpha → soft target sync.

        Args:
            batch: (states, actions, rewards, next_states, dones) from ReplayBuffer

        Returns:
            (critic_loss_value, actor_loss_value, alpha_loss_value) as Python floats
        """
        states, actions, rewards, next_states, dones = batch

        # ---- 1. Critic update -------------------------------------------------
        #
        # CONCEPT — Bellman target with entropy
        #   We need the actor's distribution over NEXT states to compute the
        #   entropy-augmented bootstrap value. The actor runs on next_states,
        #   returns log_prob for the sampled action, then we look up the Q-values
        #   that the TARGET critics assign to those next states.
        #   Using torch.no_grad() here because the target is treated as a constant
        #   label — we do not want gradients flowing back into actor or target critics
        #   through this computation.
        with torch.no_grad():
            # Sample next actions from the current policy (not the old policy).
            next_logits = self.actor(next_states)
            next_action_probs = torch.softmax(next_logits, dim=-1)
            next_distribution = torch.distributions.Categorical(next_action_probs)
            next_actions = next_distribution.sample()
            next_log_probs = next_distribution.log_prob(next_actions)

            # Target Q-values for the sampled next actions.
            next_q_one = self.target_critic_one(next_states).gather(
                1, next_actions.unsqueeze(1)
            ).squeeze(1)
            next_q_two = self.target_critic_two(next_states).gather(
                1, next_actions.unsqueeze(1)
            ).squeeze(1)

            # CONCEPT — min of twin targets
            #   Taking the minimum dampens overestimation. The entropy term subtracts
            #   alpha * log_prob — high-probability (low-entropy) actions have their
            #   bootstrap value reduced, discouraging premature greediness.
            next_q_target = torch.min(next_q_one, next_q_two)
            bellman_target = rewards + self.gamma * (next_q_target - self.alpha.detach() * next_log_probs) * (1.0 - dones)

        # Predicted Q-values from main critics for the ACTIONS TAKEN.
        predicted_q_one = self.critic_one(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        predicted_q_two = self.critic_two(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # MSE loss for each critic independently, summed so both get equal gradient signal.
        critic_loss = nn.functional.mse_loss(predicted_q_one, bellman_target) \
                    + nn.functional.mse_loss(predicted_q_two, bellman_target)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # ---- 2. Actor update --------------------------------------------------
        #
        # CONCEPT — Actor uses MAIN critics, not target critics
        #   The actor loss needs gradients to flow back into the actor network.
        #   Target critics have requires_grad=False on their parameters, so
        #   we use the main critics here — their Q-values are part of the
        #   actor's training signal, but actor gradients do NOT update the critics
        #   (critic_optimizer is called separately above, and actor_optimizer only
        #   touches actor parameters).
        current_logits      = self.actor(states)
        current_probs       = torch.softmax(current_logits, dim=-1)
        current_dist        = torch.distributions.Categorical(current_probs)
        current_actions     = current_dist.sample()
        current_log_probs   = current_dist.log_prob(current_actions)

        q_one_for_actor = self.critic_one(states).gather(1, current_actions.unsqueeze(1)).squeeze(1)
        q_two_for_actor = self.critic_two(states).gather(1, current_actions.unsqueeze(1)).squeeze(1)
        min_q_for_actor = torch.min(q_one_for_actor, q_two_for_actor)

        # CONCEPT — Actor loss sign
        #   We minimise  alpha * log_prob - Q.
        #   Minimising -Q = maximise Q (seek high-value actions).
        #   Minimising alpha * log_prob = maximise entropy (stay exploratory).
        #   The two terms are in tension — alpha balances them.
        actor_loss = (self.alpha.detach() * current_log_probs - min_q_for_actor).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ---- 3. Alpha (temperature) update ------------------------------------
        #
        # CONCEPT — Detach log_prob from actor graph
        #   log_prob was computed from the actor (which we just updated).
        #   We detach it here so the alpha gradient only updates log_alpha —
        #   not the actor weights. Alpha and the actor are updated by separate
        #   optimisers acting on separate parameters; they should not cross-contaminate.
        alpha_loss = -(self.log_alpha * (current_log_probs + self.target_entropy).detach()).mean()

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        # ---- 4. Soft update target critics ------------------------------------
        self.soft_update_target_critics()

        return critic_loss.item(), actor_loss.item(), alpha_loss.item()

    def soft_update_target_critics(self) -> None:
        """
        Polyak averaging: nudge target parameters a small fraction tau toward the
        main network parameters each step. This keeps the Bellman target stable
        without the abrupt jumps of a hard copy.
        """
        # CONCEPT — zip over named parameter pairs
        #   torch.no_grad() prevents autograd from recording this assignment as
        #   a computation — target parameters are not meant to accumulate gradients.
        with torch.no_grad():
            for main_param, target_param in zip(
                self.critic_one.parameters(), self.target_critic_one.parameters()
            ):
                target_param.data.copy_(
                    self.tau * main_param.data + (1.0 - self.tau) * target_param.data
                )
            for main_param, target_param in zip(
                self.critic_two.parameters(), self.target_critic_two.parameters()
            ):
                target_param.data.copy_(
                    self.tau * main_param.data + (1.0 - self.tau) * target_param.data
                )
