# python/networks/decision_transformer.py
#
# PURPOSE: Offline reinforcement learning as supervised sequence prediction.
#          Given a context window of (return_to_go, state, action) triplets,
#          predict the next action that achieves the target return. No Bellman.
#
# CORE CONCEPT — Why this is fundamentally different from DQN and PPO
#   DQN and PPO are ONLINE, BELLMAN-BASED methods:
#     Q(s, a) = r + γ * max_a' Q(s', a')   (DQN — bootstraps from next-state estimate)
#     V(s)    = E[r + γ * V(s')]             (PPO critic — same bootstrapping idea)
#   Both require iterative interaction with the environment, careful hyperparameter
#   tuning (learning rate, γ, clip ratio), and are sensitive to reward scale.
#
#   Decision Transformer is OFFLINE and SUPERVISED:
#     Given: a fixed dataset of recorded episodes (from A* or a trained PPO agent)
#     Learn:  P(action_t | RTG_t, state_t, action_{t-1}, state_{t-1}, ...)
#   Training is just cross-entropy on action predictions — exactly like language modeling.
#   The model implicitly learns the value function from context, without computing it.
#
# CORE CONCEPT — Return-to-go (RTG) as the control signal
#   RTG_t = sum of all rewards from timestep t to end of episode.
#   Example for our maze (goal reward = +100, step penalty = -1, max path = 40 steps):
#     At t=0 (start):  RTG_0 ≈ 100 - 40 = 60 (for a near-optimal episode)
#     At t=10:         RTG_10 = RTG_0 - (sum of rewards for steps 0..9)
#     At goal:         RTG_T = 100 (the final reward)
#
#   At INFERENCE: set RTG_0 to a high value (e.g., 80.0) — asking for good performance.
#   After each environment step: RTG_{t+1} = RTG_t - reward_t
#   The model learned "when the desired return is high and the state is X, act like A*."
#
# CORE CONCEPT — Causal (autoregressive) attention
#   At timestep t, the model predicts action_t. It must NOT see action_t+1 or state_t+1.
#   We apply a causal mask: an upper-triangular matrix of -inf values.
#   After softmax, positions corresponding to future tokens get zero attention weight.
#   This is exactly how GPT works for text generation — we adapt it to trajectories.
#
# CORE CONCEPT — Token interleaving
#   Each episode timestep t contributes THREE tokens to the sequence:
#     Token 3t:    embedded RTG_t         (scalar → hidden_dim)
#     Token 3t+1:  embedded state_t       (state features → hidden_dim)
#     Token 3t+2:  embedded action_t      (int → embedding lookup)
#   For context K=20: total sequence length = 60 tokens.
#   The model predicts action_t FROM state token 3t+1 (not from the RTG or prev action).
#   Source: Chen et al. 2021 "Decision Transformer: Reinforcement Learning via Sequence Modeling"
#
# CORE CONCEPT — State representation choice
#   We use an 8-dimensional flat state: [x/w, y/h, gx/w, gy/h, wall_up, wall_down, wall_left, wall_right]
#   Why not the full grid tensor?
#     Storing K=20 full (3, 41, 41) grid snapshots per context window costs
#     20 × 3 × 41 × 41 × 4 bytes = ~400KB per training sample. Infeasible for large batches.
#   The 8-dim state is compact and rich enough: agent + goal positions + immediate walls.
#   The Transformer's attention over K=20 timesteps provides the "memory" to reason
#   about patterns not visible in the immediate state.

import torch
import torch.nn as nn


CONTEXT_LENGTH         = 20    # K: timesteps of history the model sees per prediction
STATE_DIM              = 8     # [x/w, y/h, gx/w, gy/h, wall_up, wall_down, wall_left, wall_right]
NUM_ACTIONS            = 4     # UP DOWN LEFT RIGHT
HIDDEN_DIM             = 128   # Transformer d_model — larger than Phase 9 MLP, GPT-scale
NUM_ATTENTION_HEADS    = 4     # must divide HIDDEN_DIM evenly — 128 / 4 = 32 per head
NUM_TRANSFORMER_LAYERS = 3     # depth: 3 layers, each refines action prediction
FFN_EXPANSION_FACTOR   = 4     # inner FFN hidden dim = HIDDEN_DIM * 4 = 512
DROPOUT_RATE           = 0.1
MAX_EPISODE_TIMESTEP   = 4000  # upper bound for timestep positional encoding


class DecisionTransformer(nn.Module):
    """
    CONCEPT — Full model architecture:

    Input per timestep t (batch_size omitted for clarity):
      return_to_go_t:  (1,)     scalar return → Linear(1, 128)
      state_t:         (8,)     flat features → Linear(8, 128)
      action_t:        (1,)     int index     → Embedding(4, 128)

    After embedding K timesteps and interleaving:
      sequence: (3*K, 128) = (60, 128) tokens
      Each of 3 token positions per timestep shares a timestep positional embedding.

    GPT-style Transformer (causal mask prevents future token access):
      Output: (3*K, 128) refined token representations

    Prediction: from each state token position (indices 1, 4, 7, ..., 3*K-2):
      Linear(128, 4) → action logit
    """

    def __init__(
        self,
        state_dim:              int   = STATE_DIM,
        num_actions:            int   = NUM_ACTIONS,
        hidden_dim:             int   = HIDDEN_DIM,
        num_attention_heads:    int   = NUM_ATTENTION_HEADS,
        num_transformer_layers: int   = NUM_TRANSFORMER_LAYERS,
        context_length:         int   = CONTEXT_LENGTH,
        dropout_rate:           float = DROPOUT_RATE,
        max_episode_timestep:   int   = MAX_EPISODE_TIMESTEP
    ):
        super().__init__()
        self.context_length = context_length
        self.hidden_dim     = hidden_dim

        # CONCEPT — Separate embedding for each token type:
        # RTG and state are continuous vectors → Linear layers project them to hidden_dim.
        # Actions are discrete indices → nn.Embedding is the standard "word embedding" lookup.
        # All three end up as hidden_dim-dimensional vectors — the common currency
        # the Transformer can mix across types in its attention layers.
        self.return_to_go_embedding = nn.Linear(1, hidden_dim)
        self.state_embedding        = nn.Linear(state_dim, hidden_dim)
        self.action_embedding       = nn.Embedding(num_actions, hidden_dim)

        # CONCEPT — Timestep positional encoding (episode time, not sequence position):
        # Standard positional encodings mark sequence position (1st token, 2nd token...).
        # Here we want the model to know HOW LONG INTO the episode each triple belongs —
        # early in the episode vs. near the goal call for different reasoning.
        # One embedding per episode timestep, shared across all three tokens at that step.
        self.timestep_embedding = nn.Embedding(max_episode_timestep, hidden_dim)

        # Layer norm + dropout applied to each embedded token before Transformer.
        # Layer norm stabilises training when three different modalities (continuous + discrete)
        # are mixed: normalises each token to zero mean, unit variance.
        self.embedding_layer_norm = nn.LayerNorm(hidden_dim)
        self.embedding_dropout    = nn.Dropout(dropout_rate)

        # CONCEPT — GPT-style Transformer (encoder-only with causal masking):
        # We use nn.TransformerEncoder + a causal mask (not nn.TransformerDecoder,
        # which expects encoder input). The causal mask enforces autoregressive order.
        transformer_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_attention_heads,
            dim_feedforward=hidden_dim * FFN_EXPANSION_FACTOR,
            dropout=dropout_rate,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer=transformer_layer,
            num_layers=num_transformer_layers
        )

        # CONCEPT — Action prediction head at state token positions only:
        # We only supervise and predict from state tokens (positions 1, 4, 7, ...).
        # The model learns to predict the action GIVEN the current state + context.
        # RTG tokens and action tokens provide context but are not prediction targets.
        self.action_prediction_head = nn.Linear(hidden_dim, num_actions)

        # CONCEPT — Causal mask (pre-built, stored as non-parameter buffer):
        # Upper-triangular -inf matrix of shape (3*K, 3*K).
        # Masks out future token positions — softmax drives their weights to ~0.
        # register_buffer: saved in state_dict, moved to GPU with model.to(device),
        # but NOT updated by the optimiser (it's a fixed structural constraint).
        total_sequence_length = 3 * context_length
        causal_mask = torch.triu(
            torch.ones(total_sequence_length, total_sequence_length) * float('-inf'),
            diagonal=1
        )
        self.register_buffer('causal_attention_mask', causal_mask)

    def forward(
        self,
        return_to_go_sequence: torch.Tensor,
        state_sequence:        torch.Tensor,
        action_sequence:       torch.Tensor,
        timestep_sequence:     torch.Tensor
    ) -> torch.Tensor:
        """
        CONCEPT — Interleaved sequence construction:
        We interleave the three embedded sequences into ONE sequence:
          [RTG_0, s_0, a_0, RTG_1, s_1, a_1, ..., RTG_{K-1}, s_{K-1}, a_{K-1}]
        Positions: 0=RTG_0, 1=s_0, 2=a_0, 3=RTG_1, 4=s_1, 5=a_1, ...
        The causal mask ensures token at position t sees only tokens at positions 0..t-1.

        Input shapes (all batch-first):
          return_to_go_sequence: (batch_size, context_length, 1)         — RTG scalars
          state_sequence:        (batch_size, context_length, state_dim) — flat state vectors
          action_sequence:       (batch_size, context_length)            — action int indices
          timestep_sequence:     (batch_size, context_length)            — episode step indices

        Output:
          action_logits: (batch_size, context_length, num_actions)
          One logit vector per timestep, predicted from the state token at that step.
          Apply cross-entropy loss against the true actions from the episode.

        Implement:
        1. timestep_emb = self.timestep_embedding(timestep_sequence)
           Shape: (batch_size, context_length, hidden_dim)

        2. Embed each modality and ADD the timestep embedding to each:
               rtg_embedded    = self.return_to_go_embedding(return_to_go_sequence) + timestep_emb
               state_embedded  = self.state_embedding(state_sequence)               + timestep_emb
               action_embedded = self.action_embedding(action_sequence)             + timestep_emb
           Shapes: all (batch_size, context_length, hidden_dim)

        3. Build interleaved sequence:
               batch_size = return_to_go_sequence.shape[0]
               interleaved = torch.zeros(
                   batch_size, 3 * self.context_length, self.hidden_dim,
                   device=return_to_go_sequence.device
               )
               interleaved[:, 0::3, :] = rtg_embedded     # positions 0, 3, 6, ...
               interleaved[:, 1::3, :] = state_embedded    # positions 1, 4, 7, ...
               interleaved[:, 2::3, :] = action_embedded   # positions 2, 5, 8, ...

        4. Apply layer norm and dropout:
               interleaved = self.embedding_dropout(self.embedding_layer_norm(interleaved))

        5. Run Transformer with causal mask:
               transformer_output = self.transformer(
                   interleaved,
                   mask=self.causal_attention_mask
               )
           Shape: (batch_size, 3 * context_length, hidden_dim)

        6. Extract state token positions (predict actions FROM state representations):
               state_token_outputs = transformer_output[:, 1::3, :]
           Shape: (batch_size, context_length, hidden_dim)

        7. action_logits = self.action_prediction_head(state_token_outputs)
        8. return action_logits
        """
        timestep_emb = self.timestep_embedding(timestep_sequence)

        rtg_embedded    = self.return_to_go_embedding(return_to_go_sequence) + timestep_emb
        state_embedded  = self.state_embedding(state_sequence)               + timestep_emb
        action_embedded = self.action_embedding(action_sequence)             + timestep_emb

        batch_size  = return_to_go_sequence.shape[0]
        interleaved = torch.zeros(
            batch_size, 3 * self.context_length, self.hidden_dim,
            device=return_to_go_sequence.device
        )
        interleaved[:, 0::3, :] = rtg_embedded
        interleaved[:, 1::3, :] = state_embedded
        interleaved[:, 2::3, :] = action_embedded

        interleaved = self.embedding_dropout(self.embedding_layer_norm(interleaved))

        transformer_output  = self.transformer(interleaved, mask=self.causal_attention_mask)
        state_token_outputs = transformer_output[:, 1::3, :]

        return self.action_prediction_head(state_token_outputs)
