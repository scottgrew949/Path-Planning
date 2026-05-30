# python/networks/transformer_policy.py
#
# PURPOSE: Actor-critic network backed by a Transformer encoder rather than an MLP.
#          Direct replacement for ActorCriticNetwork in train_ppo.py — same interface,
#          richer spatial reasoning.
#
# CORE CONCEPT — Why Transformer beats MLP for grid navigation
#   The Phase 5 MLP actor-critic sees [x/w, y/h, wall_up, wall_down, wall_left, wall_right].
#   It reacts to immediate walls but has no spatial reasoning beyond 1 step.
#   The critical limitation: it cannot answer "is the goal reachable in this direction?"
#
#   Self-attention changes the fundamental computation:
#     MLP:         output_i = f(linear_combination_of_ALL_inputs)
#                  — every output neuron implicitly looks at everything at once
#     Transformer: each token QUERIES other tokens for relevant information
#                  — the goal patch actively broadcasts its location to every other patch
#
#   A maze cell containing a wall naturally develops a key that says "I am blocked."
#   The agent patch's query learns to ask "which direction leads toward the goal?"
#   The goal patch's key learns to say "I am the destination."
#   These relationships emerge from training — not hardcoded.
#
# CORE CONCEPT — Query, Key, Value mechanics
#   For each token i, multi-head self-attention computes:
#     Attention(Q_i, K, V) = softmax(Q_i * K^T / sqrt(d_head)) * V
#   Q_i (query):  what is token i looking for?
#   K (keys):     what does each token offer?
#   V (values):   what information does each token contribute if selected?
#   sqrt(d_head): scaling prevents softmax from saturating in high dimensions.
#
#   "Multi-head" means we run H independent attention computations in parallel,
#   each learning a different relationship type (e.g., "distance to wall",
#   "direction to goal", "presence of corridor"). Their outputs are concatenated.
#
# CORE CONCEPT — This is a DROP-IN REPLACEMENT for ActorCriticNetwork
#   train_ppo.py calls: action_probs, value = network(state_tensor)
#   TransformerActorCritic keeps this exact interface.
#   The only change: state_tensor is now a (batch, 3, H, W) grid tensor
#   from GridEncoder.build_grid_tensor() instead of a 6-element float vector.
#
# Self-driving analog:
#   Tesla FSD's "HydraNet" processes multiple camera feeds through a shared
#   Transformer backbone and produces simultaneously: detection, lane lines, depth.
#   Our TransformerActorCritic does the same: one backbone, two heads (policy + value).

import torch
import torch.nn as nn
from networks.grid_encoder import GridEncoder, EMBEDDING_DIM


NUM_ATTENTION_HEADS    = 4     # must divide EMBEDDING_DIM evenly — 64 / 4 = 16 per head
NUM_TRANSFORMER_LAYERS = 2     # 2 layers: first finds local patterns, second integrates globally
FFN_HIDDEN_EXPANSION   = 4     # feedforward sublayer hidden size = EMBEDDING_DIM * 4 = 256
DROPOUT_RATE           = 0.1   # applied inside attention weights and FFN for regularisation
NUM_ACTIONS            = 4     # UP DOWN LEFT RIGHT


class TransformerActorCritic(nn.Module):
    """
    CONCEPT — Architecture:

      Input: (batch_size, 3, 41, 41) grid tensor
        ↓ GridEncoder: patch into tokens, prepend CLS, add positional embeddings
      Encoded sequence: (batch_size, total_patches + 1, 64)
        ↓ nn.TransformerEncoder (2 layers, 4 heads, FFN hidden=256)
      Transformer output: (batch_size, total_patches + 1, 64)
        ↓ extract CLS token at index 0
      CLS representation: (batch_size, 64)
        ├─ policy head: Linear(64→4) → softmax → action probabilities
        └─ value head:  Linear(64→1)            → state value V(s)

    Why CLS at index 0?
      After the Transformer, the CLS token has attended to ALL patch tokens.
      It accumulates global information about the grid state — the ideal input
      for a policy that needs to make a decision about the whole situation.
      Alternatives (mean-pool all patches) work but are noisier.
    """

    def __init__(
        self,
        grid_height:           int   = 41,
        grid_width:            int   = 41,
        embedding_dim:         int   = EMBEDDING_DIM,
        num_attention_heads:   int   = NUM_ATTENTION_HEADS,
        num_transformer_layers: int  = NUM_TRANSFORMER_LAYERS,
        ffn_hidden_expansion:  int   = FFN_HIDDEN_EXPANSION,
        dropout_rate:          float = DROPOUT_RATE,
        num_actions:           int   = NUM_ACTIONS
    ):
        super().__init__()

        self.grid_encoder = GridEncoder(
            grid_height=grid_height,
            grid_width=grid_width,
            embedding_dim=embedding_dim
        )

        # CONCEPT — nn.TransformerEncoderLayer anatomy:
        #   d_model:         token embedding dimension (64)
        #   nhead:           number of parallel attention heads (4)
        #   dim_feedforward: inner dimension of the two-layer FFN sublayer
        #                    (64 × 4 = 256 — standard 4× expansion from "Attention Is All You Need")
        #   dropout:         applied to attention weights and FFN intermediate activations
        #   batch_first:     True means input shape is (batch, sequence, features)
        #                    (older PyTorch used (sequence, batch, features) — batch_first is modern)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_attention_heads,
            dim_feedforward=embedding_dim * ffn_hidden_expansion,
            dropout=dropout_rate,
            batch_first=True
        )

        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_transformer_layers
        )

        # CONCEPT — Independent heads from the same representation:
        # Both heads read from the same CLS embedding.
        # The Transformer learns to pack both "what should I do?" (policy)
        # and "how good is this state?" (value) into the CLS token simultaneously.
        # Shared backbone reduces total parameters vs two separate networks.
        self.policy_head = nn.Linear(embedding_dim, num_actions)
        self.value_head  = nn.Linear(embedding_dim, 1)

    def forward(self, grid_tensor: torch.Tensor) -> tuple:
        """
        CONCEPT — Full forward pass:
        The grid tensor goes through three transformations:
          1. Spatial → sequential: patches become tokens via GridEncoder
          2. Sequential → contextual: each token reads from all others via Transformer
          3. Contextual → decisions: CLS token drives policy and value predictions

        After step 2, the CLS token "knows" the entire grid layout — it has attended
        to every patch, learning which ones are relevant for the current decision.

        Input:  grid_tensor of shape (batch_size, 3, grid_height, grid_width)
        Output: tuple of:
                  action_probs: (batch_size, num_actions)  — sum to 1.0 per row
                  state_value:  (batch_size, 1)             — estimated V(s)

        Implement:
        1. encoded_sequence = self.grid_encoder(grid_tensor)
           Shape: (batch_size, total_patches + 1, embedding_dim)

        2. transformer_output = self.transformer_encoder(encoded_sequence)
           Shape: (batch_size, total_patches + 1, embedding_dim)
           Note: input and output shapes are IDENTICAL — the Transformer refines tokens,
           it does not change their count or dimension.

        3. cls_representation = transformer_output[:, 0, :]
           Shape: (batch_size, embedding_dim)
           Explanation: index 0 along the sequence dimension = CLS token.

        4. action_logits = self.policy_head(cls_representation)
           action_probs  = torch.softmax(action_logits, dim=-1)

        5. state_value = self.value_head(cls_representation)

        6. return (action_probs, state_value)
        """
        encoded_sequence    = self.grid_encoder(grid_tensor)
        transformer_output  = self.transformer_encoder(encoded_sequence)
        cls_representation  = transformer_output[:, 0, :]

        action_logits = self.policy_head(cls_representation)
        action_probs  = torch.softmax(action_logits, dim=-1)
        state_value   = self.value_head(cls_representation)

        return (action_probs, state_value)
