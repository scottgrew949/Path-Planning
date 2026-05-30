# python/train_diffuser.py
#
# PURPOSE: Train the trajectory diffusion model on A* path data.
#          Learn the distribution p(τ | start, goal) over valid navigation paths.
#          At inference: start from Gaussian noise, iteratively denoise → valid path.
#
# CORE CONCEPT — What is being optimised
#   The denoiser predicts ε (the noise added at step t), not the clean trajectory.
#   This is the key insight from DDPM (Ho et al. 2020): predicting noise is easier
#   than predicting the clean signal directly, and produces better samples.
#
#   Training objective (simplified DDPM loss):
#     L = E[||ε - ε_θ(τ_t, t, conditioning)||^2]
#   τ_t = sqrt(ᾱ_t) * τ_0 + sqrt(1 - ᾱ_t) * ε   (forward process, closed form)
#   At each training step:
#     1. Sample a clean trajectory τ_0 from the dataset
#     2. Sample a random diffusion step t ~ Uniform(0, T-1)
#     3. Sample noise ε ~ N(0, I)
#     4. Compute noisy trajectory τ_t using the closed-form forward formula
#     5. Ask the denoiser to predict ε from (τ_t, t, conditioning)
#     6. MSE loss between predicted ε and true ε
#
# CORE CONCEPT — Trajectory padding and masking
#   A* paths vary in length (short maze → 10 steps, long maze → 70 steps).
#   We pad shorter paths with the FINAL position (not zeros):
#     padding with the goal position means padded positions are "already there."
#     the denoiser will learn to keep them near the goal during denoising.
#   During loss computation, we mask out the padded positions — no gradient flows
#   through positions that were fabricated by padding.
#
# CORE CONCEPT — Inference procedure (DDPM reverse process)
#   Start: τ_T ~ N(0, I)  — pure Gaussian noise in path space
#   For t = T-1 down to 0:
#     1. ε_pred = denoiser(τ_t, t, conditioning)   — predict the noise
#     2. μ_t    = (τ_t - β_t/sqrt(1-ᾱ_t) * ε_pred) / sqrt(α_t)  — denoise one step
#     3. σ_t    = sqrt(β_t)  (only add noise for t > 0 — last step is deterministic)
#     4. τ_{t-1} = μ_t + σ_t * N(0, I)
#   Final τ_0 is the generated trajectory.
#   Post-process: snap (x/W * W, y/H * H) to nearest valid grid cells.

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import numpy as np
import pathplanning
from torch.utils.data import Dataset, DataLoader
from networks.trajectory_diffuser import (
    TrajectoryDenoiser, build_diffusion_schedule,
    MAX_PATH_LENGTH, WAYPOINT_DIM, CONDITIONING_DIM, DIFFUSION_TIMESTEPS
)

# ---- Configuration ----------------------------------------------------------

GRID_HEIGHT         = 41
GRID_WIDTH          = 41
NUM_TRAINING_PATHS  = 2000      # A* paths to collect for training
OBSTACLE_DENSITY    = 0.25
BATCH_SIZE          = 32        # smaller than DT — trajectories are longer sequences
NUM_EPOCHS          = 80
LEARNING_RATE       = 2e-4      # lower LR for diffusion stability

SAVE_PATH     = os.path.join(os.path.dirname(__file__), 'data', 'trajectory_diffuser.pt')

# ---- Data collection --------------------------------------------------------

def collect_astar_trajectories(num_paths: int) -> list:
    """
    CONCEPT — Using A* paths as training trajectories:
    A* provides clean, optimal, diverse paths — ideal training data.
    Each path is a sequence of (x/W, y/H) normalised waypoints.
    We record (trajectory, start, goal) triples for conditioning.

    Implement:
    1. trajectories = []
    2. For path_index in range(num_paths):
       a. seed = path_index
       b. Create GridEnvironment(GRID_WIDTH, GRID_HEIGHT, 1, 1, GRID_WIDTH-2, GRID_HEIGHT-2,
                                  OBSTACLE_DENSITY, seed=seed)
       c. raw = env.getExpertTrajectory()  — [x0,y0,a0, x1,y1,a1, ...]
       d. Extract positions: [(raw[3*i]/GRID_WIDTH, raw[3*i+1]/GRID_HEIGHT)
                              for i in range(len(raw)//3)]
       e. Pad or truncate to MAX_PATH_LENGTH:
            if len(positions) >= MAX_PATH_LENGTH: positions = positions[:MAX_PATH_LENGTH]
            else: pad with positions[-1] (repeat goal position)
            valid_length = min(original_length, MAX_PATH_LENGTH)
       f. Record conditioning: [1/GRID_WIDTH, 1/GRID_HEIGHT,
                                (GRID_WIDTH-2)/GRID_WIDTH, (GRID_HEIGHT-2)/GRID_HEIGHT]
       g. Append {'trajectory': positions_as_numpy,  shape (MAX_PATH_LENGTH, 2)
                   'conditioning': conditioning_vector,  shape (4,)
                   'valid_length': valid_length}
    3. Print progress every 200 paths.
    4. return trajectories
    """
    trajectories = []

    for path_index in range(num_paths):
        env = pathplanning.GridEnvironment(GRID_WIDTH, GRID_HEIGHT, 1, 1,
                                           GRID_WIDTH - 2, GRID_HEIGHT - 2,
                                           OBSTACLE_DENSITY, path_index)
        raw = env.getExpertTrajectory()

        if len(raw) < 3:
            continue

        original_length = len(raw) // 3
        positions = [(raw[3 * i] / GRID_WIDTH, raw[3 * i + 1] / GRID_HEIGHT)
                     for i in range(original_length)]

        valid_length = min(original_length, MAX_PATH_LENGTH)
        if len(positions) >= MAX_PATH_LENGTH:
            positions = positions[:MAX_PATH_LENGTH]
        else:
            positions = positions + [positions[-1]] * (MAX_PATH_LENGTH - len(positions))

        conditioning = [
            1.0 / GRID_WIDTH,
            1.0 / GRID_HEIGHT,
            (GRID_WIDTH  - 2) / GRID_WIDTH,
            (GRID_HEIGHT - 2) / GRID_HEIGHT,
        ]

        trajectories.append({
            'trajectory':  np.array(positions, dtype=np.float32),
            'conditioning': np.array(conditioning, dtype=np.float32),
            'valid_length': valid_length,
        })

        if (path_index + 1) % 200 == 0:
            print(f"Collected {path_index + 1}/{num_paths} trajectories")

    return trajectories


class TrajectoryDataset(Dataset):
    """
    CONCEPT — Trajectory dataset with padding mask:
    Returns (trajectory_tensor, conditioning_tensor, validity_mask).
    The validity_mask is a boolean tensor: True for real positions, False for padding.
    During training, loss is computed ONLY over valid positions.
    """

    def __init__(self, trajectory_data: list):
        """
        Implement:
        1. self.trajectories   = torch.tensor([d['trajectory'] for d in trajectory_data], dtype=torch.float32)
           Shape: (N, MAX_PATH_LENGTH, 2)
        2. self.conditionings  = torch.tensor([d['conditioning'] for d in trajectory_data], dtype=torch.float32)
           Shape: (N, 4)
        3. Build validity masks:
               self.validity_masks = torch.zeros(len(trajectory_data), MAX_PATH_LENGTH, dtype=torch.bool)
               for i, data_item in enumerate(trajectory_data):
                   self.validity_masks[i, :data_item['valid_length']] = True
        """
        self.trajectories  = torch.from_numpy(
            np.array([d['trajectory']   for d in trajectory_data], dtype=np.float32))
        self.conditionings = torch.from_numpy(
            np.array([d['conditioning'] for d in trajectory_data], dtype=np.float32))

        self.validity_masks = torch.zeros(len(trajectory_data), MAX_PATH_LENGTH, dtype=torch.bool)
        for i, data_item in enumerate(trajectory_data):
            self.validity_masks[i, :data_item['valid_length']] = True

    def __len__(self) -> int:
        return len(self.trajectories)

    def __getitem__(self, index: int) -> tuple:
        return (self.trajectories[index],
                self.conditionings[index],
                self.validity_masks[index])


def add_noise_to_trajectory(
    clean_trajectory: torch.Tensor,
    diffusion_step:   torch.Tensor,
    schedule:         dict
) -> tuple:
    """
    CONCEPT — Closed-form forward process sampling:
    Instead of iterating t steps of the Markov chain (slow), we sample τ_t DIRECTLY.
    The closed-form formula:
        τ_t = sqrt(ᾱ_t) * τ_0 + sqrt(1 - ᾱ_t) * ε
    This works because Gaussians compose multiplicatively.

    Input:
      clean_trajectory: (batch_size, MAX_PATH_LENGTH, 2) — τ_0
      diffusion_step:   (batch_size,)                    — integer t, one per sample
      schedule:         dict from build_diffusion_schedule()

    Output: (noisy_trajectory, true_noise) — both shape (batch_size, MAX_PATH_LENGTH, 2)

    Implement:
    1. true_noise = torch.randn_like(clean_trajectory)

    2. Extract schedule coefficients for the batch's diffusion steps:
           sqrt_alpha_bars           = schedule['sqrt_alpha_bars'][diffusion_step]
           sqrt_one_minus_alpha_bars = schedule['sqrt_one_minus_alpha_bars'][diffusion_step]
       Shape: (batch_size,) — one coefficient per sample in the batch.

    3. Reshape for broadcasting over (path_length, waypoint_dim):
           sqrt_alpha_bars           = sqrt_alpha_bars.view(-1, 1, 1)
           sqrt_one_minus_alpha_bars = sqrt_one_minus_alpha_bars.view(-1, 1, 1)

    4. noisy_trajectory = sqrt_alpha_bars * clean_trajectory
                         + sqrt_one_minus_alpha_bars * true_noise

    5. return (noisy_trajectory, true_noise)
    """
    true_noise = torch.randn_like(clean_trajectory)

    sqrt_alpha_bars           = schedule['sqrt_alpha_bars'][diffusion_step].view(-1, 1, 1)
    sqrt_one_minus_alpha_bars = schedule['sqrt_one_minus_alpha_bars'][diffusion_step].view(-1, 1, 1)

    noisy_trajectory = sqrt_alpha_bars * clean_trajectory + sqrt_one_minus_alpha_bars * true_noise
    return (noisy_trajectory, true_noise)


def train_one_epoch(
    denoiser:       TrajectoryDenoiser,
    data_loader:    DataLoader,
    optimiser:      torch.optim.Optimizer,
    schedule:       dict
) -> float:
    """
    CONCEPT — DDPM training loop:
    At each batch:
      1. Sample random diffusion steps (one per trajectory in the batch).
         This is the "random t" trick — the denoiser learns ALL noise levels simultaneously.
      2. Apply forward noise (closed-form) to get τ_t.
      3. Denoiser predicts ε from (τ_t, t, conditioning).
      4. MSE loss between predicted ε and true ε, MASKED to valid positions only.

    Why random t per sample?
      If all samples in a batch had the same t, the denoiser would specialise for
      one noise level and fail at others. Random t ensures uniform coverage of
      the full denoising trajectory from t=0 to t=T-1.

    Implement:
    1. denoiser.train()
    2. For each batch (trajectories, conditionings, validity_masks):
       a. optimiser.zero_grad()
       b. Sample random timesteps:
              diffusion_steps = torch.randint(0, DIFFUSION_TIMESTEPS, (batch_size,))
       c. noisy_trajectories, true_noise = add_noise_to_trajectory(
              trajectories, diffusion_steps, schedule)
       d. predicted_noise = denoiser(noisy_trajectories, diffusion_steps, conditionings)
       e. Compute masked MSE:
              element_wise_loss = (predicted_noise - true_noise) ** 2
              mask_expanded = validity_masks.unsqueeze(-1).float()  (broadcast over dim 2)
              masked_loss = (element_wise_loss * mask_expanded).sum() / mask_expanded.sum()
       f. masked_loss.backward()
       g. torch.nn.utils.clip_grad_norm_(denoiser.parameters(), max_norm=1.0)
       h. optimiser.step()
    3. Return mean masked loss over all batches
    """
    denoiser.train()
    total_loss    = 0.0
    total_batches = 0

    for trajectories, conditionings, validity_masks in data_loader:
        batch_size = trajectories.shape[0]
        optimiser.zero_grad()

        diffusion_steps = torch.randint(0, DIFFUSION_TIMESTEPS, (batch_size,))

        noisy_trajectories, true_noise = add_noise_to_trajectory(
            trajectories, diffusion_steps, schedule)

        predicted_noise = denoiser(noisy_trajectories, diffusion_steps, conditionings)

        element_wise_loss = (predicted_noise - true_noise) ** 2
        mask_expanded     = validity_masks.unsqueeze(-1).float()
        masked_loss       = (element_wise_loss * mask_expanded).sum() / mask_expanded.sum()

        masked_loss.backward()
        torch.nn.utils.clip_grad_norm_(denoiser.parameters(), max_norm=1.0)
        optimiser.step()

        total_loss    += masked_loss.item()
        total_batches += 1

    return total_loss / max(total_batches, 1)


def sample_trajectory(
    denoiser:     TrajectoryDenoiser,
    conditioning: torch.Tensor,
    schedule:     dict
) -> np.ndarray:
    """
    CONCEPT — DDPM reverse process (inference):
    Start from pure Gaussian noise and iteratively denoise T steps.
    At each step t (counting down from T-1 to 0):
        1. Predict noise: ε_pred = denoiser(τ_t, t, conditioning)
        2. Compute mean: μ_t = (τ_t - β_t / sqrt(1-ᾱ_t) * ε_pred) / sqrt(α_t)
        3. Add noise (except at t=0): τ_{t-1} = μ_t + sqrt(β_t) * N(0,I)

    Input:  conditioning of shape (1, CONDITIONING_DIM)
    Output: generated trajectory as numpy array, shape (MAX_PATH_LENGTH, 2)

    Implement:
    1. denoiser.eval()
    2. with torch.no_grad():
       a. Start: current_trajectory = torch.randn(1, MAX_PATH_LENGTH, 2)
       b. For t in range(DIFFUSION_TIMESTEPS - 1, -1, -1):  (T-1 down to 0)
              t_tensor = torch.tensor([t])
              epsilon_predicted = denoiser(current_trajectory, t_tensor, conditioning)

              alpha_t    = schedule['alphas'][t]
              alpha_bar_t = schedule['alpha_bars'][t]
              beta_t     = schedule['betas'][t]
              sqrt_one_minus_alpha_bar = schedule['sqrt_one_minus_alpha_bars'][t]

              # Reverse step mean:
              mean = (1.0 / torch.sqrt(alpha_t)) * (
                  current_trajectory - (beta_t / sqrt_one_minus_alpha_bar) * epsilon_predicted
              )

              if t > 0:
                  current_trajectory = mean + torch.sqrt(beta_t) * torch.randn_like(mean)
              else:
                  current_trajectory = mean  # final step: no added noise

    3. return current_trajectory.squeeze(0).numpy()  — shape (MAX_PATH_LENGTH, 2)
    """
    denoiser.eval()
    with torch.no_grad():
        current_trajectory = torch.randn(1, MAX_PATH_LENGTH, WAYPOINT_DIM)

        for t in range(DIFFUSION_TIMESTEPS - 1, -1, -1):
            t_tensor          = torch.tensor([t])
            epsilon_predicted = denoiser(current_trajectory, t_tensor, conditioning)

            alpha_t                  = schedule['alphas'][t]
            alpha_bar_t              = schedule['alpha_bars'][t]
            beta_t                   = schedule['betas'][t]
            sqrt_one_minus_alpha_bar = schedule['sqrt_one_minus_alpha_bars'][t]

            mean = (1.0 / torch.sqrt(alpha_t)) * (
                current_trajectory
                - (beta_t / sqrt_one_minus_alpha_bar) * epsilon_predicted
            )

            if t > 0:
                current_trajectory = mean + torch.sqrt(beta_t) * torch.randn_like(mean)
            else:
                current_trajectory = mean

    return current_trajectory.squeeze(0).numpy()


def main() -> None:
    """
    Implement:
    1. trajectory_data = collect_astar_trajectories(NUM_TRAINING_PATHS)

    2. dataset     = TrajectoryDataset(trajectory_data)
       data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
       Print: f"Dataset: {len(dataset)} trajectories"

    3. schedule  = build_diffusion_schedule()
    4. denoiser  = TrajectoryDenoiser()
       optimiser = torch.optim.Adam(denoiser.parameters(), lr=LEARNING_RATE)

    5. Loop NUM_EPOCHS:
       train_loss = train_one_epoch(denoiser, data_loader, optimiser, schedule)
       Print: f"Epoch {epoch+1:3d}/{NUM_EPOCHS} | Loss: {train_loss:.6f}"

    6. torch.save({'model_state': denoiser.state_dict(),
                   'schedule': {k: v.tolist() for k, v in schedule.items()}},
                  SAVE_PATH)
       CONCEPT: save schedule alongside model — inference needs the same schedule.
       Print: f"Trajectory diffuser saved to {SAVE_PATH}"

    7. Demonstrate one sample:
       test_conditioning = torch.tensor([[1/GRID_WIDTH, 1/GRID_HEIGHT,
                                          (GRID_WIDTH-2)/GRID_WIDTH,
                                          (GRID_HEIGHT-2)/GRID_HEIGHT]])
       generated_path = sample_trajectory(denoiser, test_conditioning, schedule)
       Print: f"Sample path start: {generated_path[0]}, end: {generated_path[-1]}"
    """
    trajectory_data = collect_astar_trajectories(NUM_TRAINING_PATHS)

    dataset     = TrajectoryDataset(trajectory_data)
    data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    print(f"Dataset: {len(dataset)} trajectories")

    schedule  = build_diffusion_schedule()
    denoiser  = TrajectoryDenoiser()
    optimiser = torch.optim.Adam(denoiser.parameters(), lr=LEARNING_RATE)

    for epoch in range(NUM_EPOCHS):
        train_loss = train_one_epoch(denoiser, data_loader, optimiser, schedule)
        print(f"Epoch {epoch + 1:3d}/{NUM_EPOCHS} | Loss: {train_loss:.6f}")

    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
    torch.save({
        'model_state': denoiser.state_dict(),
        'schedule':    {k: v.tolist() for k, v in schedule.items()},
    }, SAVE_PATH)
    print(f"Trajectory diffuser saved to {SAVE_PATH}")

    test_conditioning = torch.tensor([[
        1.0 / GRID_WIDTH,  1.0 / GRID_HEIGHT,
        (GRID_WIDTH - 2) / GRID_WIDTH, (GRID_HEIGHT - 2) / GRID_HEIGHT,
    ]])
    generated_path = sample_trajectory(denoiser, test_conditioning, schedule)
    print(f"Sample path start: {generated_path[0]}, end: {generated_path[-1]}")


if __name__ == '__main__':
    main()
