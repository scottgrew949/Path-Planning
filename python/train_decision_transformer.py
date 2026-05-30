# python/train_decision_transformer.py
#
# PURPOSE: Train the Decision Transformer on A* expert demonstrations.
#          No environment interaction during training — pure supervised learning on episodes.
#
# CORE CONCEPT — Offline RL vs Online RL
#   Online RL (DQN, PPO): agent interacts with environment during training.
#   The policy improves WHILE collecting data. Environment must be available at all times.
#
#   Offline RL (Decision Transformer): train entirely on a fixed dataset of episodes.
#   No environment interaction during training. The dataset was collected ONCE by another policy.
#   This matters for real robotics:
#     - Real robots have hardware limits — you can't run 100,000 training episodes on a real arm.
#     - But you might have 10,000 human demonstrations or simulation rollouts.
#     - Decision Transformer extracts a policy from that fixed data.
#
# CORE CONCEPT — Why A* demonstrations are ideal training data
#   A* always finds the optimal path. Every demonstration has:
#     - Maximum return (fewest steps → highest cumulative reward)
#     - Correct actions at every state (A* never takes a suboptimal step)
#   This is "expert data" — the best possible label source.
#   DAgger (Phase 6) addresses the limitation that offline BC fails on unseen states.
#   Decision Transformer addresses this differently: the RTG conditioning lets the
#   model understand the CONTEXT of the decision, not just the current state.
#
# CORE CONCEPT — Building context windows from episodes
#   An A* episode is a sequence of (state, action, reward) tuples.
#   We sample context windows of length K from within each episode.
#   For each window starting at timestep t:
#     - return_to_go_t = sum of rewards from t to episode end
#     - state_t, action_t, timestep_t
#   We batch these windows from many episodes for training.
#
# CORE CONCEPT — RTG normalisation
#   Raw RTG values depend on maze difficulty (short mazes → high RTG, long → low).
#   We normalise by dividing by the maximum possible episode return.
#   This maps all RTG values to [0, 1], preventing large-return episodes from
#   dominating gradients. The model learns relative desirability, not absolute values.

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import numpy as np
import pathplanning
from torch.utils.data import Dataset, DataLoader
from networks.decision_transformer import DecisionTransformer, CONTEXT_LENGTH, STATE_DIM

# ---- Configuration ----------------------------------------------------------

GRID_HEIGHT         = 41
GRID_WIDTH          = 41
NUM_MAZES           = 1000      # number of A* expert episodes to collect
OBSTACLE_DENSITY    = 0.25
CONTEXT_LENGTH_K    = CONTEXT_LENGTH   # K=20 timesteps per training sample
BATCH_SIZE          = 64
NUM_EPOCHS          = 40
LEARNING_RATE       = 6e-4      # from the original DT paper for small-scale experiments
MAX_RETURN          = 100.0     # approximate maximum episode return (goal reward)

SAVE_PATH     = os.path.join(os.path.dirname(__file__), 'data', 'decision_transformer.pt')

# ---- State extraction -------------------------------------------------------

def extract_state_vector(
    environment:     pathplanning.GridEnvironment,
    position_x:      int,
    position_y:      int
) -> list:
    """
    CONCEPT — 8-dimensional flat state for the Decision Transformer:
    [x/W, y/H, goal_x/W, goal_y/H, wall_up, wall_down, wall_left, wall_right]
    Richer than Phase 5's 6-dim state (adds goal coordinates explicitly).
    The DT can attend to goal proximity across K timesteps — useful context.

    Implement:
    1. goal = environment.getGoal()  — returns [gx, gy]
    2. los  = environment.getLineOfSight(position_x, position_y)  — [up, down, left, right]
    3. return [position_x / GRID_WIDTH,
               position_y / GRID_HEIGHT,
               goal[0] / GRID_WIDTH,
               goal[1] / GRID_HEIGHT,
               los[0], los[1], los[2], los[3]]
    """
    goal = environment.getGoal()
    los  = environment.getLineOfSight(position_x, position_y)
    return [
        position_x / GRID_WIDTH,
        position_y / GRID_HEIGHT,
        goal[0]    / GRID_WIDTH,
        goal[1]    / GRID_HEIGHT,
        los[0], los[1], los[2], los[3]
    ]

# ---- Episode collection -----------------------------------------------------

def collect_expert_episodes(num_mazes: int) -> list:
    """
    CONCEPT — Using A* as the expert policy:
    getExpertTrajectory() runs A* internally and returns flat [x,y,action, x,y,action, ...]
    We use it to build complete episodes: list of (state, action, reward) tuples.

    Reward structure (matching RLEnvironment.cpp):
      +100.0  on reaching goal (done=True)
       -1.0   each valid step

    Implement:
    1. episodes = []  (will be list of dicts)
    2. For maze_index in range(num_mazes):
       a. Create GridEnvironment(GRID_WIDTH, GRID_HEIGHT, 1, 1, GRID_WIDTH-2, GRID_HEIGHT-2,
                                  OBSTACLE_DENSITY, seed=maze_index)
       b. trajectory = env.getExpertTrajectory()
          Returns flat [x0,y0,a0, x1,y1,a1, ...]. Decode:
              steps = len(trajectory) // 3
              positions = [(trajectory[3*i], trajectory[3*i+1]) for i in range(steps)]
              actions   = [trajectory[3*i+2] for i in range(steps)]
       c. Build states from positions:
              states = [extract_state_vector(env, x, y) for (x, y) in positions]
       d. Build rewards: [-1.0] * (steps - 1) + [100.0 - 1.0]  (last step reaches goal)
       e. Compute return_to_go (suffix sums):
              rtg = []
              running_sum = 0.0
              for reward in reversed(rewards):
                  running_sum += reward
                  rtg.insert(0, running_sum)
       f. Normalise RTG by MAX_RETURN: rtg = [r / MAX_RETURN for r in rtg]
       g. Append {'states': states, 'actions': actions, 'rtg': rtg,
                   'timesteps': list(range(steps))} to episodes
    3. Print progress every 100 mazes.
    4. return episodes
    """
    episodes = []

    for maze_index in range(num_mazes):
        env        = pathplanning.GridEnvironment(GRID_WIDTH, GRID_HEIGHT, 1, 1,
                                                  GRID_WIDTH - 2, GRID_HEIGHT - 2,
                                                  OBSTACLE_DENSITY, maze_index)
        trajectory = env.getExpertTrajectory()

        if len(trajectory) < 3:
            continue

        steps     = len(trajectory) // 3
        positions = [(trajectory[3 * i], trajectory[3 * i + 1]) for i in range(steps)]
        actions   = [trajectory[3 * i + 2]                       for i in range(steps)]

        states  = [extract_state_vector(env, x, y) for (x, y) in positions]
        rewards = [-1.0] * (steps - 1) + [99.0]

        running_sum = 0.0
        rtg         = []
        for reward in reversed(rewards):
            running_sum += reward
            rtg.insert(0, running_sum)
        rtg = [r / MAX_RETURN for r in rtg]

        episodes.append({
            'states':    states,
            'actions':   actions,
            'rtg':       rtg,
            'timesteps': list(range(steps)),
        })

        if (maze_index + 1) % 100 == 0:
            print(f"Collected {maze_index + 1}/{num_mazes} episodes")

    return episodes

# ---- Dataset ----------------------------------------------------------------

class TrajectoryContextDataset(Dataset):
    """
    CONCEPT — Sliding context window sampling:
    From each episode of length T, we can extract (T - K) context windows.
    Each sample covers K consecutive timesteps from some start position t_start.
    Padding with zeros is applied when the episode is shorter than K.

    __getitem__ returns one training sample:
      return_to_go_sequence: (K, 1)  — RTG values (normalised)
      state_sequence:        (K, 8)  — flat state vectors
      action_sequence:       (K,)    — integer action indices
      timestep_sequence:     (K,)    — episode timestep indices
      target_actions:        (K,)    — same as action_sequence (what we predict)

    Why store target_actions separately?
    Training: predict action_t from (RTG_t, state_t, context_0..t-1).
    The target actions are the ground truth for cross-entropy loss.
    """

    def __init__(self, episodes: list, context_length: int = CONTEXT_LENGTH_K):
        """
        CONCEPT — Pre-extracting all context windows at dataset init time:
        Rather than extracting windows in __getitem__ (slow with many small ops),
        we extract all windows during __init__ and store them as tensors.
        This shifts work to the dataset creation step, making training faster.

        Implement:
        1. all_samples = []
        2. For each episode in episodes:
               episode_length = len(episode['actions'])
               For t_start in range(episode_length):
                   end = min(t_start + context_length, episode_length)
                   window_length = end - t_start
                   Extract slice of rtg, states, actions, timesteps for t_start..end
                   Pad with zeros if window_length < context_length (beginning of short episodes)
                   Append as tensors to all_samples
        3. self.samples = all_samples
        """
        all_samples = []

        for episode in episodes:
            episode_length = len(episode['actions'])

            for t_start in range(episode_length):
                end           = min(t_start + context_length, episode_length)
                window_length = end - t_start
                padding       = context_length - window_length

                rtg_window    = episode['rtg'][t_start:end]
                state_window  = episode['states'][t_start:end]
                action_window = episode['actions'][t_start:end]
                time_window   = episode['timesteps'][t_start:end]

                rtg_tensor    = torch.zeros(context_length, 1,          dtype=torch.float32)
                state_tensor  = torch.zeros(context_length, STATE_DIM,  dtype=torch.float32)
                action_tensor = torch.zeros(context_length,             dtype=torch.long)
                time_tensor   = torch.zeros(context_length,             dtype=torch.long)

                rtg_tensor[padding:]    = torch.FloatTensor(rtg_window).unsqueeze(1)
                state_tensor[padding:]  = torch.FloatTensor(state_window)
                action_tensor[padding:] = torch.LongTensor(action_window)
                time_tensor[padding:]   = torch.LongTensor(time_window)

                all_samples.append((rtg_tensor, state_tensor, action_tensor,
                                    time_tensor, action_tensor.clone()))

        self.samples = all_samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple:
        return self.samples[index]

# ---- Training ---------------------------------------------------------------

def train_one_epoch(
    model:          DecisionTransformer,
    data_loader:    DataLoader,
    optimiser:      torch.optim.Optimizer,
    loss_function:  nn.Module
) -> float:
    """
    CONCEPT — Supervised training on action predictions:
    Cross-entropy loss compares predicted action logits to true expert actions.
    This is IDENTICAL to training a text language model — just predicting "next action"
    instead of "next word." The DT literature calls this "behaviour cloning in sequence space."

    CONCEPT — Why cross-entropy for actions (not MSE)?
    Actions are DISCRETE (0, 1, 2, 3). MSE between logits and class indices is meaningless.
    Cross-entropy correctly treats it as a classification problem:
      loss = -log(softmax(logits)[true_action])
    Low loss when the model confidently predicts the correct action.
    High loss when it assigns low probability to the true action.

    Implement:
    1. model.train()
    2. For each batch (rtg_seq, state_seq, action_seq, timestep_seq, target_actions):
       a. optimiser.zero_grad()
       b. action_logits = model(rtg_seq, state_seq, action_seq, timestep_seq)
          Shape: (batch_size, context_length, num_actions)
       c. Reshape for loss: logits_flat = action_logits.reshape(-1, 4)
                            targets_flat = target_actions.reshape(-1)
          Shape: (batch_size * context_length, 4) and (batch_size * context_length,)
       d. loss = loss_function(logits_flat, targets_flat)
       e. loss.backward()
       f. torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
          CONCEPT: gradient clipping prevents exploding gradients — common with Transformers.
          Clips the global norm of all gradients to ≤ 1.0.
       g. optimiser.step()
       h. Accumulate loss
    3. Return mean loss over all samples
    """
    model.train()
    total_loss   = 0.0
    total_batches = 0

    for rtg_seq, state_seq, action_seq, timestep_seq, target_actions in data_loader:
        optimiser.zero_grad()

        action_logits = model(rtg_seq, state_seq, action_seq, timestep_seq)

        logits_flat  = action_logits.reshape(-1, action_logits.shape[-1])
        targets_flat = target_actions.reshape(-1)

        loss = loss_function(logits_flat, targets_flat)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimiser.step()

        total_loss    += loss.item()
        total_batches += 1

    return total_loss / max(total_batches, 1)


def main() -> None:
    """
    Implement:
    1. episodes = collect_expert_episodes(NUM_MAZES)
       Print: f"Collected {len(episodes)} expert episodes"

    2. dataset    = TrajectoryContextDataset(episodes, CONTEXT_LENGTH_K)
       data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
       Print: f"Dataset: {len(dataset)} context windows"

    3. model         = DecisionTransformer()
       loss_function = nn.CrossEntropyLoss()
       optimiser     = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
       CONCEPT: AdamW (Adam with decoupled weight decay) is standard for Transformers.
       Weight decay regularises the weights, reducing overfitting on fixed datasets.

    4. Loop NUM_EPOCHS:
       a. train_loss = train_one_epoch(model, data_loader, optimiser, loss_function)
       b. Print: f"Epoch {epoch+1:3d}/{NUM_EPOCHS} | Loss: {train_loss:.4f}"

    5. torch.save(model.state_dict(), SAVE_PATH)
       Print: f"Decision Transformer saved to {SAVE_PATH}"
    """
    episodes = collect_expert_episodes(NUM_MAZES)
    print(f"Collected {len(episodes)} expert episodes")

    dataset     = TrajectoryContextDataset(episodes, CONTEXT_LENGTH_K)
    data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    print(f"Dataset: {len(dataset)} context windows")

    model         = DecisionTransformer()
    loss_function = nn.CrossEntropyLoss()
    optimiser     = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(NUM_EPOCHS):
        train_loss = train_one_epoch(model, data_loader, optimiser, loss_function)
        print(f"Epoch {epoch + 1:3d}/{NUM_EPOCHS} | Loss: {train_loss:.4f}")

    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
    torch.save(model.state_dict(), SAVE_PATH)
    print(f"Decision Transformer saved to {SAVE_PATH}")


if __name__ == '__main__':
    main()
