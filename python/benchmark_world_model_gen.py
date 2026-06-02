# python/benchmark_world_model_gen.py
# World Model generalization test: train on one maze, evaluate on four.
#
# CONCEPT — What generalization testing reveals
#   A world model trained on a single maze learns three things simultaneously:
#     1. Movement physics: pressing UP from an open cell moves you upward.
#     2. Maze structure:   which positions have walls, which corridors connect.
#     3. Goal-seeking behaviour: which sequences of moves reach (GOAL_X, GOAL_Y).
#
#   Only (1) generalizes to a new maze. (2) is maze-specific — the new maze has
#   different walls in different places. (3) partially generalizes if the goal
#   position is the same but the path is different.
#
#   By training on maze A and evaluating on mazes B, C, D (same density, same
#   goal, different seeds), we isolate (1) from (2) and (3). A large success-rate
#   drop reveals how much of the policy depended on memorized maze structure.
#
# Self-driving analog:
#   A simulator trained on one city's road layout and then deployed in a different
#   city. The car's physics model transfers (braking, steering physics are the
#   same everywhere), but any "turn left at the fork" knowledge is layout-specific.
#   This test is the equivalent of asking: did the car learn to drive, or did it
#   memorise a route?

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

GRID_WIDTH  = 41
GRID_HEIGHT = 41
GOAL_X      = 38
GOAL_Y      = 40
STATE_SIZE  = 6      # [x/w, y/h, wall_up, wall_down, wall_left, wall_right]
ACTION_SIZE = 4      # 0=UP 1=DOWN 2=LEFT 3=RIGHT
HIDDEN_SIZE = 128

MAX_STEPS = GRID_WIDTH * GRID_HEIGHT * 4  # safety cap per episode

# Training
COLLECTION_EPISODES = 200
MODEL_EPOCHS        = 50
PLANNING_EPISODES   = 500
IMAGINATION_HORIZON = 15
IMAGINED_ROLLOUTS   = 10
BATCH_SIZE          = 64
LEARNING_RATE       = 0.001
GAMMA               = 0.95
BUFFER_CAPACITY     = 20000

# Evaluation
EVAL_EPISODES      = 50
TRAIN_MAZE_DENSITY = 0.3
TRAIN_MAZE_SEED    = 1
HELD_OUT_SEEDS     = [2, 3, 4]


# ---- Wall map helpers --------------------------------------------------------

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
    then appends the real wall flags from the wall map snapshot.
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


# ---- State helper ------------------------------------------------------------

def state_to_tensor(state: list, width: int, height: int, env=None) -> torch.Tensor:
    # CONCEPT — State normalisation
    #   Raw coordinates (e.g. x=32) are normalised to [0, 1].
    #   All 6 input dimensions are then on the same scale, preventing any one
    #   feature from dominating gradients simply because its numeric range is larger.
    normalised = [state[0] / width, state[1] / height]
    if env is not None:
        normalised += list(env.getLineOfSight(int(state[0]), int(state[1])))
    else:
        normalised += [0.0, 0.0, 0.0, 0.0]
    return torch.FloatTensor(normalised).unsqueeze(0)  # shape: [1, STATE_SIZE]


def actions_to_onehot_batch(action_indices: torch.Tensor, action_size: int) -> torch.Tensor:
    """Convert a batch of action integer indices to a batch of one-hot vectors."""
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
    """Fill the replay buffer with random-policy transitions from the training maze."""
    print(f"[Phase 1] Collecting {num_episodes} random episodes on training maze...")

    for episode in range(1, num_episodes + 1):
        state = env.reset()
        state_tensor = state_to_tensor(state, GRID_WIDTH, GRID_HEIGHT, env)

        for _ in range(MAX_STEPS):
            action = random.randint(0, ACTION_SIZE - 1)
            result = env.step(action)

            next_state    = [int(result[0]), int(result[1])]
            reward        = float(result[2])
            done          = bool(result[3])
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
    dynamics_net: DynamicsNetwork,
    reward_net:   RewardNetwork,
    buffer:       ReplayBuffer,
    dynamics_optimizer: optim.Optimizer,
    reward_optimizer:   optim.Optimizer,
    num_epochs:   int,
) -> None:
    """Fit dynamics and reward networks to real experience via MSE regression."""
    print(f"[Phase 2] Training world models for {num_epochs} epochs...")

    mse_loss = nn.MSELoss()

    for epoch in range(1, num_epochs + 1):
        states, actions, rewards, next_states, _ = buffer.sample(BATCH_SIZE)
        action_onehot_batch = actions_to_onehot_batch(actions, ACTION_SIZE)

        # Dynamics target: position only (first 2 dims). Wall flags are maze-specific
        # and are not predicted — they are looked up from the wall map snapshot instead.
        predicted_next_positions = dynamics_net(states, action_onehot_batch)
        dynamics_loss            = mse_loss(predicted_next_positions, next_states[:, :2])

        dynamics_optimizer.zero_grad()
        dynamics_loss.backward()
        dynamics_optimizer.step()

        predicted_rewards = reward_net(states, action_onehot_batch).squeeze(1)
        reward_loss       = mse_loss(predicted_rewards, rewards)

        reward_optimizer.zero_grad()
        reward_loss.backward()
        reward_optimizer.step()

        if epoch % 10 == 0:
            print(f"  Epoch {epoch:3d}/{num_epochs} | "
                  f"Dynamics loss: {dynamics_loss.item():.5f} | "
                  f"Reward loss:   {reward_loss.item():.5f}")

    print("[Phase 2] Done.\n")


# ---- Phase 3 helpers ---------------------------------------------------------

def imagine_rollout(
    start_state_tensor: torch.Tensor,
    dynamics_net:       DynamicsNetwork,
    reward_net:         RewardNetwork,
    policy_net:         PolicyNetwork,
    wall_map:           list,
    horizon:            int,
) -> list:
    """Run one imagined trajectory of length `horizon` from start_state_tensor.

    Returns a list of (state_tensor, action_index, predicted_reward) tuples.
    All forward passes use torch.no_grad — gradients are recomputed in the
    training loop where the policy update actually happens.
    """
    rollout_transitions  = []
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

            current_state_tensor = reconstruct_state_tensor(
                predicted_position, wall_map, GRID_WIDTH, GRID_HEIGHT
            )

    return rollout_transitions


def compute_discounted_returns(rewards: list, gamma: float) -> list:
    """Compute G_t = r_t + γ*r_{t+1} + ... backwards for efficiency."""
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
    """Train the policy via REINFORCE on imagined rollouts from the world model."""
    print(f"[Phase 3] Training policy on imagination for {planning_episodes} episodes...")

    for planning_episode in range(1, planning_episodes + 1):
        all_log_probs = []
        all_returns   = []

        for _ in range(IMAGINED_ROLLOUTS):
            states_batch, _, _, _, _ = real_buffer.sample(1)
            start_state_tensor = states_batch

            rollout_transitions = imagine_rollout(
                start_state_tensor,
                dynamics_net,
                reward_net,
                policy_net,
                wall_map,
                IMAGINATION_HORIZON,
            )

            imagined_rewards   = [transition[2] for transition in rollout_transitions]
            discounted_returns = compute_discounted_returns(imagined_rewards, GAMMA)

            current_state_tensor = rollout_transitions[0][0].detach().clone()
            for step_index, (_, action_index, _) in enumerate(rollout_transitions):
                action_logits     = policy_net(current_state_tensor)
                log_probabilities = F.log_softmax(action_logits, dim=1)
                log_prob_chosen   = log_probabilities[0, action_index]

                all_log_probs.append(log_prob_chosen)
                all_returns.append(discounted_returns[step_index])

                with torch.no_grad():
                    action_onehot      = action_to_onehot(action_index, ACTION_SIZE)
                    predicted_position = dynamics_net(current_state_tensor, action_onehot)
                    current_state_tensor = reconstruct_state_tensor(
                        predicted_position, wall_map, GRID_WIDTH, GRID_HEIGHT
                    )

        if not all_log_probs:
            continue

        return_tensor = torch.FloatTensor(all_returns)
        baseline      = return_tensor.mean()
        advantages    = return_tensor - baseline

        log_prob_tensor = torch.stack(all_log_probs)
        policy_loss     = -(log_prob_tensor * advantages).mean()

        policy_optimizer.zero_grad()
        policy_loss.backward()
        policy_optimizer.step()

        if planning_episode % 100 == 0:
            print(f"  Planning episode {planning_episode:4d}/{planning_episodes} | "
                  f"Policy loss: {policy_loss.item():.4f} | "
                  f"Mean imagined return: {return_tensor.mean().item():.3f}")

    print("[Phase 3] Done.\n")


# ---- Evaluation --------------------------------------------------------------

def evaluate_policy(policy_net: PolicyNetwork, eval_env, num_episodes: int) -> float:
    # CONCEPT — Why we use real wall flags from eval_env during evaluation
    #   During training, the policy learned to respond to a 6-dim state vector:
    #     [x_norm, y_norm, wall_up, wall_down, wall_left, wall_right]
    #   The wall flags were always sourced from the real training maze via getLineOfSight.
    #   The policy never saw DynamicsNetwork-predicted wall flags in the final state
    #   representation — those are looked up from the wall map snapshot.
    #
    #   On a held-out maze, we use eval_env.getLineOfSight to supply the NEW maze's
    #   wall flags. The policy sees the same input format it trained on, but now the
    #   wall flags describe a maze it has never encountered. This is the right way to
    #   test generalization: the input format is identical, only the content changes.
    #
    #   The DynamicsNetwork is NOT used here at all. During evaluation the real
    #   environment advances state (env.step). DynamicsNetwork was only needed for
    #   imagined rollouts during Phase 3 training.
    successful_episodes = 0

    for _ in range(num_episodes):
        state = eval_env.reset()
        state_tensor = state_to_tensor(state, GRID_WIDTH, GRID_HEIGHT, eval_env)

        for _ in range(MAX_STEPS):
            with torch.no_grad():
                action_logits = policy_net(state_tensor)
                action_index  = action_logits.argmax(dim=1).item()

            result     = eval_env.step(action_index)
            next_state = [int(result[0]), int(result[1])]
            done       = bool(result[3])

            state_tensor = state_to_tensor(next_state, GRID_WIDTH, GRID_HEIGHT, eval_env)

            if done:
                successful_episodes += 1
                break

    return successful_episodes / num_episodes


# ---- Entry point -------------------------------------------------------------

def run() -> None:
    print()
    print("=" * 48)
    print("World Model Generalization Test")
    print(f"Train maze: density={TRAIN_MAZE_DENSITY}, seed={TRAIN_MAZE_SEED}")
    print("=" * 48)
    print()
    print("Phase 1 — collect random transitions from training maze")
    print("Phase 2 — fit DynamicsNetwork + RewardNetwork via MSE")
    print("Phase 3 — train PolicyNetwork on imagined rollouts (REINFORCE)")
    print("Eval     — greedy policy tested on training maze + 3 held-out mazes")
    print()

    training_env = pathplanning.GridEnvironment(
        GRID_WIDTH, GRID_HEIGHT,
        0, 0,
        GOAL_X, GOAL_Y,
        TRAIN_MAZE_DENSITY,
        TRAIN_MAZE_SEED,
    )

    dynamics_net = DynamicsNetwork(STATE_SIZE, ACTION_SIZE, HIDDEN_SIZE)
    reward_net   = RewardNetwork(STATE_SIZE, ACTION_SIZE, hidden_size=64)
    policy_net   = PolicyNetwork(STATE_SIZE, ACTION_SIZE, HIDDEN_SIZE)

    dynamics_optimizer = optim.Adam(dynamics_net.parameters(), lr=LEARNING_RATE)
    reward_optimizer   = optim.Adam(reward_net.parameters(),   lr=LEARNING_RATE)
    policy_optimizer   = optim.Adam(policy_net.parameters(),   lr=LEARNING_RATE)

    real_buffer = ReplayBuffer(BUFFER_CAPACITY)

    # Phase 1
    collect_real_experience(training_env, real_buffer, num_episodes=COLLECTION_EPISODES)

    # Snapshot wall layout for use during imagined rollouts (training maze only).
    training_wall_map = build_wall_map(training_env, GRID_WIDTH, GRID_HEIGHT)

    # Phase 2
    train_world_models(
        dynamics_net,
        reward_net,
        real_buffer,
        dynamics_optimizer,
        reward_optimizer,
        num_epochs=MODEL_EPOCHS,
    )

    # Phase 3
    train_policy_on_imagination(
        policy_net,
        dynamics_net,
        reward_net,
        real_buffer,
        training_wall_map,
        planning_episodes=PLANNING_EPISODES,
        policy_optimizer=policy_optimizer,
    )

    # ---- Evaluation phase ----------------------------------------------------
    # CONCEPT — Interpreting the results
    #   The policy was trained exclusively inside the world model's imagination of
    #   maze A. Its weights encode "given these wall flags, take this action."
    #
    #   On mazes B, C, D the policy receives the correct wall flags for the new
    #   maze (via eval_env.getLineOfSight) so it has accurate local information.
    #   What it lacks is knowledge of the global structure: which corridors lead
    #   to the goal, which dead ends to avoid.
    #
    #   Expected result: a large success-rate drop (>30%).
    #   Why? DynamicsNetwork predicts position — its gradient signal was anchored
    #   to the specific corridors and junctions of maze A. The policy learned
    #   "in corridor X, go right" rather than "when wall_left=0 and wall_right=1
    #   and I'm below the goal, go up." True generalization would require the
    #   latter — much harder to extract from a single maze.
    #
    #   A small drop (<15%) would indicate the policy learned general navigation
    #   principles (wall-following, goal-directed movement) rather than routes.

    all_seeds     = [TRAIN_MAZE_SEED] + HELD_OUT_SEEDS
    success_rates = {}

    print("[Evaluation] Running greedy policy on each maze...")

    for maze_seed in all_seeds:
        eval_env = pathplanning.GridEnvironment(
            GRID_WIDTH, GRID_HEIGHT,
            0, 0,
            GOAL_X, GOAL_Y,
            TRAIN_MAZE_DENSITY,
            maze_seed,
        )
        success_rate = evaluate_policy(policy_net, eval_env, num_episodes=EVAL_EPISODES)
        success_rates[maze_seed] = success_rate

        label = "Training" if maze_seed == TRAIN_MAZE_SEED else f"Held-out (seed={maze_seed})"
        print(f"  {label}: {success_rate * 100:.1f}%")

    # ---- Report --------------------------------------------------------------

    training_success_rate = success_rates[TRAIN_MAZE_SEED]

    print()
    print("=" * 48)
    print("World Model Generalization Test")
    print(f"Train maze: density={TRAIN_MAZE_DENSITY}, seed={TRAIN_MAZE_SEED}")
    print("=" * 48)
    print(f"{'Maze':<14}{'Seed':<8}{'Success%':<12}{'vs Training'}")
    print("-" * 48)

    print(f"{'Training':<14}{TRAIN_MAZE_SEED:<8}{training_success_rate * 100:>6.1f}%     {'baseline':>10}")

    for held_out_seed in HELD_OUT_SEEDS:
        held_out_success_rate = success_rates[held_out_seed]
        delta_percent = (held_out_success_rate - training_success_rate) * 100
        delta_label   = f"{delta_percent:>+.1f}%"
        print(f"{'Held-out ' + chr(65 + HELD_OUT_SEEDS.index(held_out_seed) + 1):<14}"
              f"{held_out_seed:<8}"
              f"{held_out_success_rate * 100:>6.1f}%     {delta_label:>10}")

    print("=" * 48)

    # ---- Interpretation ------------------------------------------------------
    held_out_rates   = [success_rates[seed] for seed in HELD_OUT_SEEDS]
    mean_held_out    = sum(held_out_rates) / len(held_out_rates) if held_out_rates else 0.0
    mean_drop_points = (training_success_rate - mean_held_out) * 100

    print()
    print("Interpretation:")
    if mean_drop_points > 30:
        print(f"  Large drop ({mean_drop_points:.1f}pp average): policy relied on maze-specific")
        print("  structure — limited generalization.")
        print()
        print("  Why: DynamicsNetwork predicts position; the policy learned which")
        print("  positions lead to the goal in maze A. Position → good action is")
        print("  maze-specific knowledge. The new maze has the same physics but")
        print("  different corridors, so the learned position-action mapping fails.")
        print()
        print("  To improve: train on multiple mazes simultaneously (curriculum),")
        print("  or use a policy conditioned on local wall flags only (not position).")
    elif mean_drop_points < 15:
        print(f"  Small drop ({mean_drop_points:.1f}pp average): policy learned general")
        print("  navigation principles — good generalization.")
        print()
        print("  The policy responds to wall flags rather than memorised positions,")
        print("  suggesting it learned a general rule like wall-following or")
        print("  goal-directed movement based on local observations.")
    else:
        print(f"  Moderate drop ({mean_drop_points:.1f}pp average): partial generalization.")
        print()
        print("  The policy learned some general principles (e.g. which directions")
        print("  tend to be productive) but also memorised some maze-specific routes.")
        print("  Training on more diverse mazes would likely improve transfer.")
    print()


if __name__ == "__main__":
    run()
