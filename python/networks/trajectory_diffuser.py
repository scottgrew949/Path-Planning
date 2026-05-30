# python/networks/trajectory_diffuser.py
#
# PURPOSE: Generate complete navigation paths by denoising Gaussian noise —
#          treating path planning as a trajectory generation problem.
#
# CORE CONCEPT — Why generative models for planning?
#   All previous methods (A*, DQN, PPO, DT) produce ONE path given start and goal.
#   A generative model learns the DISTRIBUTION over valid paths: p(τ | start, goal).
#   This enables:
#     1. DIVERSE paths: sample multiple different routes, pick the safest one.
#     2. MULTIMODAL planning: a maze with three valid corridors → model samples all three.
#     3. CONSTRAINED generation: condition the denoiser to avoid certain regions.
#   This is the frontier of research: Janner et al. 2022 "Planning with Diffusion"
#   and Chi et al. 2023 "Diffusion Policy" both apply this to robotics trajectories.
#
# CORE CONCEPT — Denoising Diffusion Probabilistic Models (DDPM)
#   Original paper: Ho et al. 2020 "Denoising Diffusion Probabilistic Models"
#   Applied to images: start from Gaussian noise → denoise 1000 steps → realistic image.
#   Applied to trajectories: start from Gaussian noise → denoise T steps → valid path.
#
#   FORWARD PROCESS (training, not learned):
#     q(τ_t | τ_{t-1}) = N(τ_t; sqrt(1-β_t) * τ_{t-1}, β_t * I)
#     Each step adds a small amount of Gaussian noise. After T=100 steps, τ_T ≈ N(0, I).
#     We can sample τ_t AT ANY STEP DIRECTLY:
#       τ_t = sqrt(ᾱ_t) * τ_0 + sqrt(1 - ᾱ_t) * ε,   where ε ~ N(0, I)
#     This is the "reparameterisation trick" — critical for efficient training.
#     ᾱ_t = product of (1 - β_s) for s = 0..t. Precomputed, not learned.
#
#   REVERSE PROCESS (inference, learned):
#     p_θ(τ_{t-1} | τ_t) = N(τ_{t-1}; μ_θ(τ_t, t), σ_t^2 * I)
#     The denoiser network predicts the NOISE that was added (ε_θ), not the clean trajectory.
#     Then μ_θ is computed from τ_t and ε_θ analytically.
#
#   TRAINING LOSS:
#     L = E_{τ_0, t, ε}[||ε - ε_θ(τ_t, t, conditioning)||^2]
#     Predict the noise that was added — MSE between true and predicted noise.
#     Conditioning: start position + goal position (what we're planning toward).
#
# CORE CONCEPT — Trajectory representation
#   A path from A* = list of (x, y) positions, length varies per maze.
#   We normalize to fixed length: truncate or pad with final position.
#   Normalise coordinates: (x/W, y/H) to map all paths to [0, 1]^2.
#   Final shape: (MAX_PATH_LENGTH, 2) — a sequence of normalised waypoints.
#
# CORE CONCEPT — Conditioning (classifier-free guidance)
#   The denoiser takes start_pos + goal_pos as conditioning input.
#   WITHOUT conditioning: denoiser learns average trajectory across all (start, goal) pairs.
#   WITH conditioning: denoiser learns which trajectories are appropriate for THESE endpoints.
#   We concatenate conditioning as extra tokens or add it to the timestep embedding.
#   At inference: always conditioned — we always have a specific start and goal.
#
# CONCEPT — Sinusoidal timestep embedding
#   The denoiser sees the noisy trajectory AND the diffusion step t (0..T-1).
#   It must behave DIFFERENTLY at t=0 (slightly noisy, fine adjustments)
#   vs t=99 (pure noise, rough structure).
#   Sinusoidal embedding maps integer t to a continuous vector that encodes
#   both magnitude and periodicity — identical to Transformer positional encodings.
#   Formula: emb[2i] = sin(t / 10000^(2i/d)), emb[2i+1] = cos(t / 10000^(2i/d))

import torch
import torch.nn as nn
import math


MAX_PATH_LENGTH        = 80     # trajectories padded/truncated to this length
WAYPOINT_DIM           = 2      # (x/W, y/H) normalised coordinates
CONDITIONING_DIM       = 4      # [start_x/W, start_y/H, goal_x/W, goal_y/H]
DIFFUSION_TIMESTEPS    = 100    # T: forward process steps
HIDDEN_DIM             = 128    # Transformer hidden size
NUM_ATTENTION_HEADS    = 4
NUM_TRANSFORMER_LAYERS = 4      # deeper than policy — trajectory is a long sequence
FFN_EXPANSION_FACTOR   = 4
DROPOUT_RATE           = 0.1

# DDPM noise schedule: linear from β_start to β_end over T steps.
# Small β → slow, controlled corruption. Chosen from Ho et al. 2020.
BETA_START = 0.0001
BETA_END   = 0.02


def build_diffusion_schedule(
    num_timesteps: int = DIFFUSION_TIMESTEPS,
    beta_start:    float = BETA_START,
    beta_end:      float = BETA_END
) -> dict:
    """
    CONCEPT — Precomputing the diffusion schedule:
    All coefficients (betas, alphas, cumulative alphas) are FIXED — not learned.
    They define the noise schedule for the forward process.
    Precomputing them once avoids recalculating inside the training loop.

    Returns a dict with:
      betas:       (T,) — noise variance added at each step
      alphas:      (T,) — 1 - beta per step
      alpha_bars:  (T,) — cumulative product of alphas (ᾱ_t)
      sqrt_alpha_bars:       (T,) — for sampling forward noise: sqrt(ᾱ_t)
      sqrt_one_minus_alpha_bars: (T,) — for sampling forward noise: sqrt(1 - ᾱ_t)

    Implement:
    1. betas      = torch.linspace(beta_start, beta_end, num_timesteps)
    2. alphas     = 1.0 - betas
    3. alpha_bars = torch.cumprod(alphas, dim=0)
    4. sqrt_alpha_bars           = torch.sqrt(alpha_bars)
    5. sqrt_one_minus_alpha_bars = torch.sqrt(1.0 - alpha_bars)
    6. return dict with all five tensors above
    """
    betas      = torch.linspace(beta_start, beta_end, num_timesteps)
    alphas     = 1.0 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)

    return {
        'betas':                       betas,
        'alphas':                      alphas,
        'alpha_bars':                  alpha_bars,
        'sqrt_alpha_bars':             torch.sqrt(alpha_bars),
        'sqrt_one_minus_alpha_bars':   torch.sqrt(1.0 - alpha_bars),
    }


def sinusoidal_timestep_embedding(timesteps: torch.Tensor, embedding_dim: int) -> torch.Tensor:
    """
    CONCEPT — Encoding the diffusion step t as a continuous vector:
    The denoiser network needs to know HOW NOISY the input is.
    A raw integer t is uninformative — the network cannot easily reason
    about "t=50 is halfway through the diffusion process."
    Sinusoidal embedding maps t to a vector where:
      - Nearby timesteps have similar (but not identical) embeddings
      - The embedding has both low and high frequency components
      - The network can extract "is this early or late in diffusion?" easily
    This is identical to Transformer positional encodings and to the original DDPM paper.

    Formula for embedding dimension d:
      emb[2i]   = sin(t / 10000^(2i / embedding_dim))
      emb[2i+1] = cos(t / 10000^(2i / embedding_dim))

    Input:  timesteps of shape (batch_size,) — integer diffusion step indices
    Output: embeddings of shape (batch_size, embedding_dim)

    Implement:
    1. half_dim = embedding_dim // 2
    2. frequencies = torch.exp(
           -math.log(10000) * torch.arange(half_dim, device=timesteps.device) / half_dim
       )
    3. arguments = timesteps.float().unsqueeze(1) * frequencies.unsqueeze(0)
       Shape: (batch_size, half_dim)
    4. embedding = torch.cat([torch.sin(arguments), torch.cos(arguments)], dim=-1)
       Shape: (batch_size, embedding_dim)
    5. return embedding
    """
    half_dim   = embedding_dim // 2
    frequencies = torch.exp(
        -math.log(10000) * torch.arange(half_dim, device=timesteps.device) / half_dim
    )
    arguments = timesteps.float().unsqueeze(1) * frequencies.unsqueeze(0)
    return torch.cat([torch.sin(arguments), torch.cos(arguments)], dim=-1)


class TrajectoryDenoiser(nn.Module):
    """
    CONCEPT — The denoiser network ε_θ(τ_t, t, conditioning):
    Given a noisy trajectory τ_t and the diffusion step t and goal conditioning,
    predict the noise ε that was added to produce τ_t from τ_0.
    If we subtract ε from τ_t, we recover a cleaner trajectory τ_{t-1}.

    Architecture: Transformer over the trajectory sequence.
    Each waypoint (x/W, y/H) becomes one token.
    Conditioning (start + goal) is injected via a separate conditioning token
    prepended to the sequence — the waypoints attend to it freely.

    Input:
      noisy_trajectory:   (batch_size, MAX_PATH_LENGTH, 2)     — noisy waypoints
      diffusion_timestep: (batch_size,)                         — integer t
      conditioning:       (batch_size, CONDITIONING_DIM)        — [sx/W, sy/H, gx/W, gy/H]

    Output:
      predicted_noise:    (batch_size, MAX_PATH_LENGTH, 2)      — predicted ε
    """

    def __init__(
        self,
        max_path_length:        int   = MAX_PATH_LENGTH,
        waypoint_dim:           int   = WAYPOINT_DIM,
        conditioning_dim:       int   = CONDITIONING_DIM,
        hidden_dim:             int   = HIDDEN_DIM,
        num_attention_heads:    int   = NUM_ATTENTION_HEADS,
        num_transformer_layers: int   = NUM_TRANSFORMER_LAYERS,
        ffn_expansion_factor:   int   = FFN_EXPANSION_FACTOR,
        dropout_rate:           float = DROPOUT_RATE
    ):
        super().__init__()
        self.max_path_length = max_path_length
        self.hidden_dim      = hidden_dim

        # Project noisy waypoints (2D) to hidden_dim.
        self.waypoint_projection = nn.Linear(waypoint_dim, hidden_dim)

        # CONCEPT — Timestep MLP:
        # The sinusoidal embedding is a fixed function of t.
        # We pass it through a small MLP (two Linear + SiLU layers)
        # to let the model learn task-specific transformations of the timestep.
        # SiLU (sigmoid linear unit) = x * sigmoid(x) — smoother than ReLU for diffusion.
        timestep_embedding_dim = hidden_dim * 4
        self.timestep_mlp = nn.Sequential(
            nn.Linear(hidden_dim, timestep_embedding_dim),
            nn.SiLU(),
            nn.Linear(timestep_embedding_dim, hidden_dim)
        )

        # Project conditioning vector (4D start+goal) to hidden_dim.
        self.conditioning_projection = nn.Linear(conditioning_dim, hidden_dim)

        # Positional encoding for the trajectory sequence.
        self.positional_embedding = nn.Parameter(
            torch.zeros(1, max_path_length + 1, hidden_dim))  # +1 for conditioning token
        nn.init.normal_(self.positional_embedding, std=0.02)

        # Transformer encoder: waypoints attend to each other AND to the conditioning token.
        transformer_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_attention_heads,
            dim_feedforward=hidden_dim * ffn_expansion_factor,
            dropout=dropout_rate,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer=transformer_layer,
            num_layers=num_transformer_layers
        )

        # Project hidden_dim back to waypoint_dim to predict noise in the same space.
        self.output_projection = nn.Linear(hidden_dim, waypoint_dim)

    def forward(
        self,
        noisy_trajectory:   torch.Tensor,
        diffusion_timestep: torch.Tensor,
        conditioning:       torch.Tensor
    ) -> torch.Tensor:
        """
        CONCEPT — Conditioning token injection:
        We prepend ONE conditioning token to the trajectory sequence.
        All waypoint tokens can attend to it freely (no causal mask here —
        denoising is NOT autoregressive, we denoise the whole trajectory at once).
        This is simpler and more effective than concatenating conditioning to each token.

        CONCEPT — Timestep injection via addition:
        The timestep embedding (hidden_dim) is added to EVERY token in the sequence.
        This is called "adaptive bias injection" — the timestep shifts all activations
        globally, telling the model "be aggressive (early step) or conservative (late step)."

        Input:
          noisy_trajectory:   (batch_size, MAX_PATH_LENGTH, 2)
          diffusion_timestep: (batch_size,) — integer step t
          conditioning:       (batch_size, 4) — [sx/W, sy/H, gx/W, gy/H]

        Output:
          predicted_noise: (batch_size, MAX_PATH_LENGTH, 2)

        Implement:
        1. Project waypoints:
               waypoint_tokens = self.waypoint_projection(noisy_trajectory)
           Shape: (batch_size, MAX_PATH_LENGTH, hidden_dim)

        2. Compute timestep embedding and project through MLP:
               timestep_sinusoidal = sinusoidal_timestep_embedding(diffusion_timestep, self.hidden_dim)
               timestep_token      = self.timestep_mlp(timestep_sinusoidal)
           Shape: (batch_size, hidden_dim)

        3. Add timestep bias to all waypoint tokens (broadcast over sequence):
               waypoint_tokens = waypoint_tokens + timestep_token.unsqueeze(1)

        4. Project conditioning and create conditioning token:
               conditioning_token = self.conditioning_projection(conditioning).unsqueeze(1)
           Shape: (batch_size, 1, hidden_dim)

        5. Prepend conditioning token to waypoint sequence:
               full_sequence = torch.cat([conditioning_token, waypoint_tokens], dim=1)
           Shape: (batch_size, MAX_PATH_LENGTH + 1, hidden_dim)

        6. Add positional embeddings:
               full_sequence = full_sequence + self.positional_embedding

        7. Run Transformer (no causal mask — denoising attends to full trajectory):
               transformer_output = self.transformer(full_sequence)

        8. Extract waypoint tokens (drop conditioning token at index 0):
               waypoint_outputs = transformer_output[:, 1:, :]
           Shape: (batch_size, MAX_PATH_LENGTH, hidden_dim)

        9. Project to noise prediction:
               predicted_noise = self.output_projection(waypoint_outputs)
        10. return predicted_noise
        """
        waypoint_tokens = self.waypoint_projection(noisy_trajectory)

        timestep_sinusoidal = sinusoidal_timestep_embedding(diffusion_timestep, self.hidden_dim)
        timestep_token      = self.timestep_mlp(timestep_sinusoidal)
        waypoint_tokens     = waypoint_tokens + timestep_token.unsqueeze(1)

        conditioning_token = self.conditioning_projection(conditioning).unsqueeze(1)

        full_sequence = torch.cat([conditioning_token, waypoint_tokens], dim=1)
        full_sequence = full_sequence + self.positional_embedding

        transformer_output = self.transformer(full_sequence)

        waypoint_outputs = transformer_output[:, 1:, :]
        return self.output_projection(waypoint_outputs)
