# python/train_world_model.py
# World Model training: learn environment dynamics and reward, then train a
# policy entirely through imagined (model-generated) experience.
#
# CONCEPT — World Models vs Dyna-Q
#   Dyna-Q (Sutton 1990) is the simplest model-based RL algorithm: after each
#   real step, run K simulated steps using a tabular model of (s, a) → (s', r).
#   World Models (Ha & Schmidhuber 2018) extend this idea to continuous states
#   and neural function approximators:
#     - Tabular model → DynamicsNetwork + RewardNetwork (neural regressors)
#     - K one-step simulations → H-step imagined rollouts (longer horizon)
#     - Tabular Q-update → REINFORCE on imagined trajectory returns
#
#   The key upgrade: neural models generalise across states never seen before,
#   while tabular models can only replay states that were explicitly visited.
#
# CONCEPT — Three-phase training
#   Phase 1 — Data collection:   random policy fills replay buffer with real (s,a,r,s') tuples.
#   Phase 2 — Model learning:    DynamicsNetwork and RewardNetwork are trained on that buffer.
#   Phase 3 — Policy learning:   PolicyNetwork is trained *entirely inside the learned model*
#                                 using REINFORCE on imagined rollout returns.
#
#   After Phase 3 the agent has never made a decision in the real world — only a
#   random policy touched the environment. The learned policy's quality depends on
#   how accurate the world model is.
#
# Self-driving analog:
#   Phase 1 = human driver collecting dashcam footage (random/manual exploration).
#   Phase 2 = training a physics simulator from that footage.
#   Phase 3 = running thousands of simulated test drives to train the autopilot,
#             with zero real-world risk.

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pathplanning

from replay_buffer import ReplayBuffer
from networks.world_model import DynamicsNetwork, RewardNetwork, PolicyNetwork, action_to_onehot

# ---- Hyperparameters ---------------------------------------------------------

GRID_WIDTH           = 41
GRID_HEIGHT          = 41
STATE_SIZE           = 6     # [x/w, y/h, wall_up, wall_down, wall_left, wall_right]
ACTION_SIZE          = 4     # 0=UP 1=DOWN 2=LEFT 3=RIGHT
HIDDEN_SIZE          = 128

COLLECTION_EPISODES  = 200   # real episodes to fill the buffer with random exploration
MODEL_EPOCHS         = 50    # supervised training epochs for dynamics + reward networks
PLANNING_EPISODES    = 500   # episodes of imagined rollouts used to train the policy
IMAGINATION_HORIZON  = 15    # steps per imagined rollout
IMAGINED_ROLLOUTS    = 10    # imagined rollouts generated per planning step

BATCH_SIZE           = 64
LEARNING_RATE        = 0.001
GAMMA                = 0.95  # discount factor for imagined returns
BUFFER_CAPACITY      = 20000

MAX_REAL_STEPS       = GRID_WIDTH * GRID_HEIGHT * 4  # safety limit per episode
EVAL_EPISODES        = 10    # greedy episodes used to measure final success rate

# ---- Wall map ----------------------------------------------------------------
# CONCEPT — Why pre-build a wall map instead of predicting wall flags?
#   Wall flags are a property of the maze layout, not the agent's movement.
#   Asking DynamicsNetwork to predict them would force it to memorise obstacle
#   positions for a specific maze — baking maze structure into the network weights.
#   Instead, we snapshot the full maze wall layout once at training start.
#   During imagined rollouts, we predict only the next position and look up the
#   correct wall flags from this snapshot. Dynamics stays general; maze is exact.

def build_wall_map(env, width: int, height: int) -> list:
    """Query getLineOfSight for every cell and store as a 2D list.

    wall_map[x][y] = [wall_up, wall_down, wall_left, wall_right]
    """
    wall_map = []
    for cell_x in range(width):
        column = []
        for cell_y in range(height):
            column.append(list(env.getLineOfSight(cell_x, cell_y)))
        wall_map.append(column)
    return wall_map


def reconstruct_state_tensor(
    position_tensor: torch.Tensor,
    wall_map: list,
    width: int,
    height: int,
) -> torch.Tensor:
    """Convert a predicted normalised position back to a full 6-dim state tensor.

    Clamps predicted position to grid bounds, rounds to nearest cell,
    then appends the real wall flags from the wall map.
    """
    predicted_x_norm = float(position_tensor[0, 0].item())
    predicted_y_norm = float(position_tensor[0, 1].item())

    cell_x = int(round(predicted_x_norm * width))
    cell_y = int(round(predicted_y_norm * height))
    cell_x = max(0, min(width  - 1, cell_x))
    cell_y = max(0, min(height - 1, cell_y))

    walls = wall_map[cell_x][cell_y]
    state = [cell_x / width, cell_y / height] + walls
    return torch.FloatTensor(state).unsqueeze(0)


# ---- Helpers -----------------------------------------------------------------

def state_to_tensor(state: list, width: int, height: int, env=None) -> torch.Tensor:
    # CONCEPT — State normalisation
    #   Raw grid coordinates (e.g. x=32 out of 41) are normalised to [0, 1].
    #   Neural networks train faster and more stably when inputs are on the same
    #   scale — avoids one feature dominating gradients simply because its range is larger.
    normalised = [state[0] / width, state[1] / height]
    if env is not None:
        normalised += env.getLineOfSight(int(state[0]), int(state[1]))
    else:
        normalised += [0.0, 0.0, 0.0, 0.0]
    return torch.FloatTensor(normalised).unsqueeze(0)  # shape: [1, STATE_SIZE]


def actions_to_onehot_batch(action_indices: torch.Tensor, action_size: int) -> torch.Tensor:
    # Convert a batch of action integer indices to a batch of one-hot vectors.
    # action_indices shape: [batch]  →  output shape: [batch, action_size]
    batch_size = action_indices.shape[0]
    one_hot = torch.zeros(batch_size, action_size)
    one_hot.scatter_(1, action_indices.unsqueeze(1), 1.0)
    return one_hot


# ---- Phase 1: Collect real experience ----------------------------------------

def collect_real_experience(
    env,
    buffer: ReplayBuffer,
    num_episodes: int,
) -> None:
    # CONCEPT — Random exploration policy
    #   The world model needs a diverse dataset of (s, a, s', r) transitions so it
    #   can learn the dynamics of the full grid — not just the path a good policy
    #   would take. A purely random policy visits states broadly and ensures the
    #   model sees walls, dead ends, and open corridors.
    #
    #   This is analogous to a test driver deliberately exploring every street in a
    #   city before handing the data to the simulator training team.
    print(f"[Phase 1] Collecting {num_episodes} episodes of random experience...")

    for episode in range(1, num_episodes + 1):
        state = env.reset()
        state_tensor = state_to_tensor(state, GRID_WIDTH, GRID_HEIGHT, env)

        for _ in range(MAX_REAL_STEPS):
            action = random.randint(0, ACTION_SIZE - 1)
            result = env.step(action)

            next_state      = [int(result[0]), int(result[1])]
            reward          = float(result[2])
            done            = bool(result[3])
            next_state_tensor = state_to_tensor(next_state, GRID_WIDTH, GRID_HEIGHT, env)

            buffer.push(state_tensor, action, reward, next_state_tensor, done)
            state_tensor = next_state_tensor

            if done:
                break

        if episode % 50 == 0:
            print(f"  Episode {episode}/{num_episodes} | Buffer size: {len(buffer)}")

    print(f"[Phase 1] Done. Buffer contains {len(buffer)} transitions.\n")


# ---- Phase 2: Train world models ---------------------------------------------

def train_world_models(
    dynamics_net:   DynamicsNetwork,
    reward_net:     RewardNetwork,
    buffer:         ReplayBuffer,
    num_epochs:     int,
    dynamics_optimizer: optim.Optimizer,
    reward_optimizer:   optim.Optimizer,
) -> tuple:
    # CONCEPT — Supervised regression on real experience
    #   Both the dynamics and reward networks are trained as standard regression
    #   models using Mean Squared Error loss:
    #
    #     dynamics_loss = MSE(f_θ(s, a),  s')
    #     reward_loss   = MSE(g_φ(s, a),  r)
    #
    #   where f_θ and g_φ are the dynamics and reward networks respectively.
    #   This is pure supervised learning — no RL signal involved yet.
    #
    # CONCEPT — Why MSE for state prediction?
    #   The next state is a continuous vector (normalised coordinates + wall bits).
    #   MSE penalises large prediction errors quadratically, which pushes the
    #   network to be accurate on every dimension of the state simultaneously.
    print(f"[Phase 2] Training world models for {num_epochs} epochs...")

    mse_loss = nn.MSELoss()
    final_dynamics_loss = 0.0
    final_reward_loss   = 0.0

    for epoch in range(1, num_epochs + 1):
        states, actions, rewards, next_states, _ = buffer.sample(BATCH_SIZE)
        action_onehot_batch = actions_to_onehot_batch(actions, ACTION_SIZE)

        # --- Dynamics network update ---
        # Target is position only (first 2 dims of next_states: x_norm, y_norm).
        # Wall flags (dims 2-5) are maze-specific and not predicted by the network.
        predicted_next_positions = dynamics_net(states, action_onehot_batch)
        dynamics_loss            = mse_loss(predicted_next_positions, next_states[:, :2])

        dynamics_optimizer.zero_grad()
        dynamics_loss.backward()
        dynamics_optimizer.step()

        # --- Reward network update ---
        predicted_rewards = reward_net(states, action_onehot_batch).squeeze(1)
        reward_loss       = mse_loss(predicted_rewards, rewards)

        reward_optimizer.zero_grad()
        reward_loss.backward()
        reward_optimizer.step()

        final_dynamics_loss = dynamics_loss.item()
        final_reward_loss   = reward_loss.item()

        if epoch % 10 == 0:
            print(f"  Epoch {epoch:3d}/{num_epochs} | "
                  f"Dynamics loss: {final_dynamics_loss:.5f} | "
                  f"Reward loss:   {final_reward_loss:.5f}")

    print(f"[Phase 2] Done. Final losses — "
          f"Dynamics: {final_dynamics_loss:.5f}, Reward: {final_reward_loss:.5f}\n")
    return final_dynamics_loss, final_reward_loss


# ---- Phase 3 helpers ---------------------------------------------------------

def imagine_rollout(
    start_state_tensor: torch.Tensor,
    dynamics_net:       DynamicsNetwork,
    reward_net:         RewardNetwork,
    policy_net:         PolicyNetwork,
    wall_map:           list,
    horizon:            int,
) -> list:
    # CONCEPT — Imagined rollout (hallucination)
    #   Starting from a real state, the agent runs forward H steps using ONLY its
    #   learned models — no real environment interaction.
    #
    #   At each step:
    #     1. Policy network proposes an action (argmax of logits — greedy during imagination).
    #     2. Dynamics network predicts the next state.
    #     3. Reward network predicts the reward.
    #
    #   The result is a sequence of (state, action, reward) tuples that represent
    #   a "dream" trajectory. These trajectories are used to compute returns and
    #   train the policy via REINFORCE.
    #
    #   CONCEPT — Why argmax during imagination rollouts?
    #   The policy has not been trained yet (or is being trained). Using the greedy
    #   action during imagination keeps the rollout deterministic, which simplifies
    #   the gradient computation and is standard in early World Model implementations.
    #   More advanced methods (Dreamer) use stochastic sampling throughout.
    #
    # Returns: list of (state_tensor [1, STATE_SIZE], action_index int, reward float)
    rollout_transitions = []
    current_state_tensor = start_state_tensor.detach().clone()

    with torch.no_grad():
        for _ in range(horizon):
            action_logits  = policy_net(current_state_tensor)
            action_index   = action_logits.argmax(dim=1).item()
            action_onehot  = action_to_onehot(action_index, ACTION_SIZE)

            predicted_position = dynamics_net(current_state_tensor, action_onehot)
            predicted_reward   = reward_net(current_state_tensor, action_onehot).item()

            rollout_transitions.append((
                current_state_tensor,
                action_index,
                predicted_reward,
            ))
            # Reconstruct full state: predicted position + real wall flags from map.
            current_state_tensor = reconstruct_state_tensor(
                predicted_position, wall_map, GRID_WIDTH, GRID_HEIGHT
            )

    return rollout_transitions


def compute_discounted_returns(rewards: list, gamma: float) -> list:
    # CONCEPT — Discounted return G_t
    #   G_t = r_t + γ*r_{t+1} + γ²*r_{t+2} + ... = Σ_{k=0}^{H-t} γ^k * r_{t+k}
    #   Future rewards are worth less than immediate rewards — γ < 1 encodes this.
    #   Returns are computed backward from the end of the rollout for efficiency.
    discounted_returns = []
    cumulative_return  = 0.0
    for reward in reversed(rewards):
        cumulative_return = reward + gamma * cumulative_return
        discounted_returns.insert(0, cumulative_return)
    return discounted_returns


# ---- Phase 3: Train policy on imagination ------------------------------------

def train_policy_on_imagination(
    policy_net:        PolicyNetwork,
    dynamics_net:      DynamicsNetwork,
    reward_net:        RewardNetwork,
    real_buffer:       ReplayBuffer,
    wall_map:          list,
    planning_episodes: int,
    policy_optimizer:  optim.Optimizer,
) -> None:
    # CONCEPT — REINFORCE on imagined rollouts (model-based policy gradient)
    #   REINFORCE is the simplest policy gradient algorithm:
    #     Loss = -Σ_t log π(a_t | s_t) * G_t
    #   where G_t is the discounted return from time t, and π(a|s) is the policy.
    #
    #   Maximising return = minimising this loss.
    #   The log probability of the chosen action is multiplied by its return:
    #     - If G_t is large (good trajectory), increase the probability of those actions.
    #     - If G_t is small (bad trajectory), decrease it.
    #
    # CONCEPT — Baseline subtraction (variance reduction)
    #   Raw returns have high variance — one lucky rollout might have G_0 = 10,
    #   another G_0 = -5. The gradient update jumps around.
    #   Subtracting the mean return across rollouts from each G_t centres the signal
    #   around zero, dramatically reducing gradient variance with no bias introduced.
    #   This is the simplest form of baseline; the critic in Actor-Critic is a
    #   learned, more powerful version.
    #
    # CONCEPT — Why sample start states from the real buffer?
    #   The imagined rollouts start from states the agent has actually visited.
    #   This anchors imagination to the real state distribution — if we started from
    #   random noise the rollouts would be meaningless.
    print(f"[Phase 3] Training policy on imagination for {planning_episodes} planning episodes...")

    for planning_episode in range(1, planning_episodes + 1):
        all_log_probs      = []
        all_returns        = []

        # Generate multiple imagined rollouts to reduce variance.
        for _ in range(IMAGINED_ROLLOUTS):
            # Sample a real start state from the buffer.
            states_batch, _, _, _, _ = real_buffer.sample(1)
            start_state_tensor = states_batch  # shape: [1, STATE_SIZE]

            rollout_transitions = imagine_rollout(
                start_state_tensor,
                dynamics_net,
                reward_net,
                policy_net,
                wall_map,
                IMAGINATION_HORIZON,
            )

            # Recompute log probs WITH gradients for the policy update.
            # (imagine_rollout used torch.no_grad — we must redo the forward pass here.)
            imagined_rewards = [transition[2] for transition in rollout_transitions]
            discounted_returns = compute_discounted_returns(imagined_rewards, GAMMA)

            current_state_tensor = rollout_transitions[0][0].detach().clone()
            for step_index, (_, action_index, _) in enumerate(rollout_transitions):
                action_logits    = policy_net(current_state_tensor)
                log_probabilities = F.log_softmax(action_logits, dim=1)
                log_prob_chosen   = log_probabilities[0, action_index]

                all_log_probs.append(log_prob_chosen)
                all_returns.append(discounted_returns[step_index])

                # Advance state through dynamics (no grad needed here — only policy updates).
                with torch.no_grad():
                    action_onehot      = action_to_onehot(action_index, ACTION_SIZE)
                    predicted_position = dynamics_net(current_state_tensor, action_onehot)
                    current_state_tensor = reconstruct_state_tensor(
                        predicted_position, wall_map, GRID_WIDTH, GRID_HEIGHT
                    )

        if not all_log_probs:
            continue

        return_tensor = torch.FloatTensor(all_returns)

        # CONCEPT — Baseline: subtract mean return to centre the gradient signal.
        baseline         = return_tensor.mean()
        advantages       = return_tensor - baseline

        # REINFORCE loss: negative because optimizers minimize, REINFORCE maximises return.
        log_prob_tensor  = torch.stack(all_log_probs)
        policy_loss      = -(log_prob_tensor * advantages).mean()

        policy_optimizer.zero_grad()
        policy_loss.backward()
        policy_optimizer.step()

        if planning_episode % 100 == 0:
            mean_imagined_return = return_tensor.mean().item()
            print(f"  Planning episode {planning_episode:4d}/{planning_episodes} | "
                  f"Policy loss: {policy_loss.item():.4f} | "
                  f"Mean imagined return: {mean_imagined_return:.3f}")

    print("[Phase 3] Done.\n")


# ---- Evaluation --------------------------------------------------------------

def evaluate_policy(policy_net: PolicyNetwork, env, num_episodes: int) -> float:
    # Run greedy episodes on the REAL environment to measure success rate.
    # The policy acts greedily (argmax) — no exploration.
    successful_episodes = 0

    for _ in range(num_episodes):
        state = env.reset()
        state_tensor = state_to_tensor(state, GRID_WIDTH, GRID_HEIGHT, env)

        for _ in range(MAX_REAL_STEPS):
            with torch.no_grad():
                action_logits = policy_net(state_tensor)
                action_index  = action_logits.argmax(dim=1).item()

            result     = env.step(action_index)
            next_state = [int(result[0]), int(result[1])]
            done       = bool(result[3])

            state_tensor = state_to_tensor(next_state, GRID_WIDTH, GRID_HEIGHT, env)

            if done:
                successful_episodes += 1
                break

    success_rate = successful_episodes / num_episodes
    return success_rate


# ---- Entry point -------------------------------------------------------------

def train() -> None:
    # CONCEPT — Full World Model pipeline
    #   1. Fill buffer with random transitions (Phase 1).
    #   2. Fit DynamicsNetwork and RewardNetwork to that buffer (Phase 2).
    #   3. Train PolicyNetwork via REINFORCE on imagined rollouts (Phase 3).
    #   4. Evaluate the policy on the real environment.
    print("\n" + "=" * 60)
    print("World Models — Neural Dyna-Q")
    print("=" * 60)
    print("Dyna-Q (menu option 4) reuses a tabular model of seen (state, action) pairs.")
    print("World Models extend this to neural networks that generalise to unseen states.")
    print("")
    print("Three phases:")
    print("  Phase 1 — Collect real transitions via random policy → fill replay buffer.")
    print("  Phase 2 — Train DynamicsNetwork: (state, action) → predicted next state.")
    print("            Train RewardNetwork:   (state, action) → predicted reward.")
    print("  Phase 3 — Train PolicyNetwork entirely on IMAGINED rollouts from the model.")
    print("            No further real environment interaction needed during policy training.")
    print("")
    print("The agent learns to plan in its own 'head' — same concept as how a human")
    print("mentally simulates routes before moving, rather than trial-and-erroring.")
    print("")
    print("Self-driving analog: the car builds an internal simulator of road physics")
    print("and other vehicles, then tests thousands of manoeuvres in imagination before")
    print("committing to one in the real world.")
    print("")
    print("What to watch: dynamics loss dropping in Phase 2 (model getting accurate),")
    print("then policy success rate rising in Phase 3 (planning improving in imagination).")
    print("")

    env = pathplanning.GridEnvironment(
        GRID_WIDTH, GRID_HEIGHT,
        0, 0, 38, 40,
        0.3,
    )

    # Networks
    dynamics_net = DynamicsNetwork(STATE_SIZE, ACTION_SIZE, HIDDEN_SIZE)
    reward_net   = RewardNetwork(STATE_SIZE, ACTION_SIZE, hidden_size=64)
    policy_net   = PolicyNetwork(STATE_SIZE, ACTION_SIZE, HIDDEN_SIZE)

    # Optimizers — one per network so each can be tuned independently.
    dynamics_optimizer = optim.Adam(dynamics_net.parameters(), lr=LEARNING_RATE)
    reward_optimizer   = optim.Adam(reward_net.parameters(),   lr=LEARNING_RATE)
    policy_optimizer   = optim.Adam(policy_net.parameters(),   lr=LEARNING_RATE)

    # Replay buffer stores real environment transitions.
    real_buffer = ReplayBuffer(BUFFER_CAPACITY)

    # Phase 1 — Random exploration: fill the buffer.
    collect_real_experience(env, real_buffer, num_episodes=COLLECTION_EPISODES)

    # Build wall map once — snapshot of the maze structure for use during imagination.
    wall_map = build_wall_map(env, GRID_WIDTH, GRID_HEIGHT)

    # Phase 2 — Supervised model training: fit dynamics and reward networks.
    train_world_models(
        dynamics_net,
        reward_net,
        real_buffer,
        num_epochs=MODEL_EPOCHS,
        dynamics_optimizer=dynamics_optimizer,
        reward_optimizer=reward_optimizer,
    )

    # Phase 3 — Imagined policy training: train policy via REINFORCE on rollouts.
    train_policy_on_imagination(
        policy_net,
        dynamics_net,
        reward_net,
        real_buffer,
        wall_map,
        planning_episodes=PLANNING_EPISODES,
        policy_optimizer=policy_optimizer,
    )

    # Evaluation — measure how often the policy reaches the goal on the real grid.
    print(f"[Evaluation] Running {EVAL_EPISODES} greedy episodes on the real environment...")
    success_rate = evaluate_policy(policy_net, env, num_episodes=EVAL_EPISODES)
    print(f"[Evaluation] Success rate: {success_rate * 100:.1f}% "
          f"({int(success_rate * EVAL_EPISODES)}/{EVAL_EPISODES} episodes reached goal)\n")

    print("=" * 60)
    print("Training complete.")
    print("=" * 60)


if __name__ == "__main__":
    train()
