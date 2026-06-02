# python/train_sac.py
# SAC training loop — ties together GridEnvironment, ReplayBuffer, SACActor,
# SACCritic, and SACAgent.
#
# CONCEPT — SAC training loop vs DQN training loop
#   DQN loop:  collect step → push buffer → update main network →
#              every N steps: copy main → target.
#   SAC loop:  collect step → push buffer → update actor + twin critics + alpha →
#              every step: polyak-average target critics.
#   SAC does more work per step (three separate gradient updates) but eliminates
#   epsilon scheduling entirely — the temperature alpha auto-tunes exploration.
#
# CONCEPT — No epsilon in SAC
#   DQN needs epsilon-greedy because the network is not trained to be
#   appropriately uncertain — it always outputs its best guess and epsilon
#   forces random exploration externally.
#   SAC trains the actor to maintain entropy: the distribution naturally spreads
#   over actions when the agent is uncertain, then sharpens as it gains experience.
#   Removing a hand-tuned hyperparameter (epsilon) makes SAC more self-contained.
#
# CONCEPT — Warm-up period
#   We do not update until the buffer has at least BATCH_SIZE experiences.
#   Training on too few experiences gives noisy, biased gradients.
#   Same guard as DQN (buffer.is_ready).
#
# Self-driving analog:
#   The training loop is a driving school: each episode is a full run from start
#   to goal. Crashes and detours are stored in the dashcam archive (replay buffer).
#   After every second of driving, we review a random clip and adjust the policy.
#   The temperature alpha decides how cautiously the driver approaches intersections
#   it hasn't seen before.

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import pathplanning

from replay_buffer          import ReplayBuffer
from networks.sac_network   import SACActor, SACCritic
from agents.sac_agent       import SACAgent

# ---- Hyperparameters --------------------------------------------------------

GRID_WIDTH       = 41
GRID_HEIGHT      = 41
STATE_SIZE       = 6          # [x/width, y/height, wall_up, wall_down, wall_left, wall_right]
ACTION_SIZE      = 4          # UP DOWN LEFT RIGHT
HIDDEN_SIZE      = 128

EPISODES         = 2000
MAX_STEPS        = GRID_WIDTH * GRID_HEIGHT * 4
BATCH_SIZE       = 64
BUFFER_CAPACITY  = 10000

LEARNING_RATE    = 0.0003
GAMMA            = 0.95       # discount factor
TAU              = 0.005      # polyak averaging coefficient

# ---- Helpers ----------------------------------------------------------------

def state_to_tensor(state: list, width: int, height: int, env=None) -> torch.Tensor:
    """
    Convert a raw [x, y] state list into a normalised 6-dim float tensor.
    Appends wall sensor readings from getLineOfSight() when an env is provided.

    CONCEPT — Normalisation
        Neural networks converge faster when inputs are in a consistent range.
        Dividing x by width and y by height maps both into [0, 1] regardless of
        grid size — the same network architecture works on any size grid.
    """
    normalised = [state[0] / width, state[1] / height]
    if env is not None:
        normalised += env.getLineOfSight(state[0], state[1])
    else:
        normalised += [0, 0, 0, 0]
    return torch.FloatTensor(normalised).unsqueeze(0)

# ---- Training loop ----------------------------------------------------------

def train():
    print("\n" + "=" * 60)
    print("SAC — Soft Actor-Critic")
    print("=" * 60)
    print("DQN explores via epsilon-greedy — a heuristic decay schedule.")
    print("PPO is on-policy: every update discards the replay buffer.")
    print("SAC combines the best of both: off-policy buffer reuse (like DQN)")
    print("with exploration built into the objective via entropy maximisation.")
    print("The agent is rewarded for being uncertain — no epsilon needed.")
    print("")
    print("Twin critics (Q1, Q2): min(Q1,Q2) prevents Q-value overestimation.")
    print("Learned temperature alpha: auto-tunes how much entropy matters.")
    print("Polyak target updates: smoother than hard-copying every N steps.")
    print("")
    print("Self-driving analog: SAC keeps the car 'humble' about lane choice")
    print("until evidence forces a commit — prevents overconfident lane-change crashes.")
    print("")
    print("What to watch: alpha value decaying as policy learns — the agent")
    print("becoming less exploratory as it gains confidence in its Q-values.")
    print("")
    print("  Episode      = one full run start-to-goal (or timeout)")
    print("  Reward       = total reward accumulated this episode")
    print("  Critic loss  = MSE on twin Q-networks — should decrease over time")
    print("  Actor loss   = entropy-weighted policy improvement signal")
    print("  Alpha loss   = temperature self-adjustment signal")
    print("  Alpha        = current exploration temperature")
    print("")

    env = pathplanning.GridEnvironment(GRID_WIDTH, GRID_HEIGHT, 0, 0, 38, 40, 0.3)

    actor      = SACActor(STATE_SIZE, ACTION_SIZE, HIDDEN_SIZE)
    critic_one = SACCritic(STATE_SIZE, ACTION_SIZE, HIDDEN_SIZE)
    critic_two = SACCritic(STATE_SIZE, ACTION_SIZE, HIDDEN_SIZE)

    # CONCEPT — Two separate SACCritic instances
    #   critic_one and critic_two are independent objects with independent weight
    #   initialisations. They are both passed into SACAgent, which trains them
    #   separately and deep-copies them into frozen target critics.
    agent  = SACAgent(
        actor=actor,
        critic_one=critic_one,
        critic_two=critic_two,
        learning_rate=LEARNING_RATE,
        gamma=GAMMA,
        tau=TAU,
        action_size=ACTION_SIZE,
    )
    buffer = ReplayBuffer(BUFFER_CAPACITY)

    # Running totals for printing — averaged over the last 10 episodes.
    recent_critic_losses = []
    recent_actor_losses  = []
    recent_alpha_losses  = []

    for episode in range(1, EPISODES + 1):
        raw_state    = env.reset()
        state_tensor = state_to_tensor(raw_state, GRID_WIDTH, GRID_HEIGHT, env)
        episode_reward = 0.0

        episode_critic_loss = 0.0
        episode_actor_loss  = 0.0
        episode_alpha_loss  = 0.0
        update_count        = 0

        for _step in range(MAX_STEPS):
            action = agent.select_action(state_tensor)
            result = env.step(action)

            next_raw_state = [int(result[0]), int(result[1])]
            reward         = float(result[2])
            done           = bool(result[3])

            next_state_tensor = state_to_tensor(next_raw_state, GRID_WIDTH, GRID_HEIGHT, env)

            buffer.push(state_tensor, action, reward, next_state_tensor, done)
            episode_reward += reward
            state_tensor    = next_state_tensor

            if buffer.is_ready(BATCH_SIZE):
                batch = buffer.sample(BATCH_SIZE)
                critic_loss, actor_loss, alpha_loss = agent.update(batch)
                episode_critic_loss += critic_loss
                episode_actor_loss  += actor_loss
                episode_alpha_loss  += alpha_loss
                update_count        += 1

            if done:
                break

        # Accumulate rolling averages for the print block.
        if update_count > 0:
            recent_critic_losses.append(episode_critic_loss / update_count)
            recent_actor_losses.append(episode_actor_loss  / update_count)
            recent_alpha_losses.append(episode_alpha_loss  / update_count)

        if episode % 10 == 0:
            avg_critic = sum(recent_critic_losses[-10:]) / max(len(recent_critic_losses[-10:]), 1)
            avg_actor  = sum(recent_actor_losses[-10:])  / max(len(recent_actor_losses[-10:]),  1)
            avg_alpha_loss = sum(recent_alpha_losses[-10:]) / max(len(recent_alpha_losses[-10:]), 1)
            current_alpha  = agent.alpha.item()

            print(
                f"Episode {episode:4d} | "
                f"Reward: {episode_reward:7.1f} | "
                f"Critic loss: {avg_critic:7.4f} | "
                f"Actor loss: {avg_actor:7.4f} | "
                f"Alpha loss: {avg_alpha_loss:7.4f} | "
                f"Alpha: {current_alpha:.4f}"
            )


if __name__ == "__main__":
    train()
