# python/networks/grid_encoder.py
#
# PURPOSE: Convert a raw 41×41 grid into a sequence of patch embeddings
#          that Transformer-based policies process with self-attention.
#
# CORE CONCEPT — Why a grid encoder at all
#   The Phase 5 MLP sees state as [x/w, y/h, wall_up, wall_down, wall_left, wall_right].
#   Six numbers. It cannot see that the goal is three corridors over, or that a dead
#   end blocks the direct route. The MLP must rediscover spatial relationships
#   from scratch through millions of training steps.
#   A Transformer needs a SEQUENCE of tokens to apply self-attention.
#   The grid encoder converts the 2D spatial grid into that sequence — one token
#   per patch of cells. This is the "perception" module.
#
# CORE CONCEPT — Patch embedding (ViT-style)
#   Divide the grid into non-overlapping 4×4 patches.
#   Each patch becomes one "token" in the Transformer's input sequence.
#   A Linear projection maps the flattened patch features (48 values) to a
#   d_model-dimensional vector. This is identical to how the Vision Transformer
#   (ViT, Dosovitskiy et al. 2020) handles images — we are applying the same idea
#   to a binary grid rather than an RGB image.
#
# CORE CONCEPT — Three input channels
#   Channel 0: obstacle map    (1.0 = wall, 0.0 = free space)
#   Channel 1: agent position  (1.0 at agent cell, 0.0 everywhere else)
#   Channel 2: goal position   (1.0 at goal cell, 0.0 everywhere else)
#   Using full spatial maps instead of coordinate pairs gives the Transformer
#   GLOBAL context — every patch can see where the agent and goal are relative
#   to itself through the attention mechanism, not just through explicit distance features.
#
# CORE CONCEPT — Positional encoding
#   After patch embedding, every token is a 64-dim vector with no memory of where
#   in the grid it came from. We add learned positional embeddings (one per sequence
#   position) so the model can reason about spatial relationships.
#   Learned embeddings (what we use) outperform fixed sinusoidal encodings on
#   fixed-size grids because the model can adapt which positional signals matter.

import torch
import torch.nn as nn


PATCH_SIZE    = 4    # divide grid into non-overlapping 4×4 patches
NUM_CHANNELS  = 3    # obstacle map, agent map, goal map
EMBEDDING_DIM = 64   # output embedding dimension per patch token


class PatchEmbedding(nn.Module):
    """
    CONCEPT — Tokenisation via linear projection:
    Each 4×4 patch contains PATCH_SIZE * PATCH_SIZE * NUM_CHANNELS = 48 values.
    A single Linear layer maps these 48 values to EMBEDDING_DIM (64) — the patch token.
    This is the "tokenisation" step: after this, the grid is no longer a 2D image —
    it is a flat sequence of vectors, ready for attention.

    We do NOT use a CNN here. A CNN would need conv layers, pooling, and a final
    projection anyway. For small 4×4 patches, a direct linear projection is simpler,
    more efficient, and gives the Transformer full control over what to extract.
    """

    def __init__(
        self,
        grid_height:   int = 41,
        grid_width:    int = 41,
        patch_size:    int = PATCH_SIZE,
        num_channels:  int = NUM_CHANNELS,
        embedding_dim: int = EMBEDDING_DIM
    ):
        super().__init__()
        self.patch_size = patch_size
        self.num_channels = num_channels

        # Number of complete patches that fit in each spatial dimension.
        # Integer division — trailing rows/cols that don't fill a patch are cropped.
        self.num_patches_vertical   = grid_height // patch_size
        self.num_patches_horizontal = grid_width  // patch_size
        self.total_patches          = self.num_patches_vertical * self.num_patches_horizontal

        # Raw values per patch: all pixels across all channels flattened.
        self.patch_feature_count = patch_size * patch_size * num_channels

        # Single linear layer maps flattened patch → embedding vector.
        self.projection = nn.Linear(self.patch_feature_count, embedding_dim)

    def forward(self, grid_tensor: torch.Tensor) -> torch.Tensor:
        """
        CONCEPT — Extracting all patches at once via tensor reshaping:
        We avoid a Python loop over patches (slow) by using view/reshape to
        reorganise the spatial dimensions into the patch layout. This processes
        all patches in a single operation — critical for batched training.

        Input:  grid_tensor of shape (batch_size, num_channels, grid_height, grid_width)
        Output: patch embeddings of shape (batch_size, total_patches, embedding_dim)

        Implement:
        1. Crop the grid to an exact multiple of patch_size in both dimensions:
               batch_size, channels, height, width = grid_tensor.shape
               crop_height = (height // self.patch_size) * self.patch_size
               crop_width  = (width  // self.patch_size) * self.patch_size
               cropped = grid_tensor[:, :, :crop_height, :crop_width]

        2. Reshape into patch blocks:
               patches = cropped.view(
                   batch_size,
                   channels,
                   self.num_patches_vertical,   self.patch_size,
                   self.num_patches_horizontal, self.patch_size
               )
           This groups the spatial dimensions into (patch_row, within_row,
                                                    patch_col, within_col).

        3. Permute to bring patch indices together:
               patches = patches.permute(0, 2, 4, 1, 3, 5)
           Shape: (batch, patches_v, patches_h, channels, patch_size, patch_size)

        4. Flatten the last three dims (channels, H, W within patch) into one:
               patches = patches.reshape(
                   batch_size,
                   self.total_patches,
                   self.patch_feature_count
               )

        5. Apply linear projection to each patch:
               return self.projection(patches)
           Shape: (batch_size, total_patches, embedding_dim)
        """
        batch_size, channels, height, width = grid_tensor.shape

        crop_height = (height // self.patch_size) * self.patch_size
        crop_width  = (width  // self.patch_size) * self.patch_size
        cropped = grid_tensor[:, :, :crop_height, :crop_width]

        patches = cropped.view(
            batch_size,
            channels,
            self.num_patches_vertical,   self.patch_size,
            self.num_patches_horizontal, self.patch_size
        )
        patches = patches.permute(0, 2, 4, 1, 3, 5)
        patches = patches.reshape(batch_size, self.total_patches, self.patch_feature_count)

        return self.projection(patches)


class GridEncoder(nn.Module):
    """
    CONCEPT — Full grid encoder pipeline:
      PatchEmbedding → prepend CLS token → add positional embeddings → output sequence

    The CLS (classification) token is a learnable vector prepended to the patch sequence.
    After the Transformer encoder, the CLS output position holds an aggregate
    representation of the entire grid — the policy and value heads use this.
    This pattern comes from BERT (Devlin et al. 2018) where CLS is used for
    sentence-level classification from a sequence model.
    """

    def __init__(
        self,
        grid_height:   int = 41,
        grid_width:    int = 41,
        patch_size:    int = PATCH_SIZE,
        num_channels:  int = NUM_CHANNELS,
        embedding_dim: int = EMBEDDING_DIM
    ):
        super().__init__()

        self.patch_embedding = PatchEmbedding(
            grid_height, grid_width, patch_size, num_channels, embedding_dim)

        total_sequence_length = self.patch_embedding.total_patches + 1  # +1 for CLS

        # CONCEPT — Learnable CLS token:
        # Shape (1, 1, embedding_dim) — the first two dims are broadcast across batches.
        # Initialised near zero so it starts as a blank slate, not biasing any direction.
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embedding_dim))

        # CONCEPT — Learned positional embeddings:
        # One embedding per sequence position (CLS + all patches = total_sequence_length).
        # Added element-wise to the token embeddings BEFORE the Transformer sees them.
        # Without this, the model cannot distinguish "patch in top-left corner"
        # from "patch in bottom-right corner" — both would look identical after projection.
        self.positional_embedding = nn.Parameter(
            torch.zeros(1, total_sequence_length, embedding_dim))

        # Small random initialisation breaks symmetry so each position can specialise.
        nn.init.normal_(self.cls_token,           std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.02)

    def forward(self, grid_tensor: torch.Tensor) -> torch.Tensor:
        """
        CONCEPT — CLS prepend and positional encoding:
        The final output is a token sequence where:
          position 0:     CLS token  — aggregate "what the whole grid means"
          positions 1..N: patch tokens — local information per grid region

        After the Transformer processes this sequence, index 0 is used by
        downstream policy/value heads. Indices 1..N are discarded (or can be
        used for auxiliary tasks).

        Input:  grid_tensor of shape (batch_size, num_channels, grid_height, grid_width)
        Output: sequence of shape (batch_size, total_patches + 1, embedding_dim)

        Implement:
        1. batch_size = grid_tensor.shape[0]
        2. patch_embeddings = self.patch_embedding(grid_tensor)
           Shape: (batch_size, total_patches, embedding_dim)

        3. Expand CLS token to match batch:
               cls_expanded = self.cls_token.expand(batch_size, -1, -1)
           Shape: (batch_size, 1, embedding_dim)

        4. Concatenate CLS + patches along sequence dimension:
               full_sequence = torch.cat([cls_expanded, patch_embeddings], dim=1)
           Shape: (batch_size, total_patches + 1, embedding_dim)

        5. Add positional embeddings (broadcast over batch):
               return full_sequence + self.positional_embedding
        """
        batch_size = grid_tensor.shape[0]

        patch_embeddings = self.patch_embedding(grid_tensor)

        cls_expanded  = self.cls_token.expand(batch_size, -1, -1)
        full_sequence = torch.cat([cls_expanded, patch_embeddings], dim=1)

        return full_sequence + self.positional_embedding

    @staticmethod
    def build_grid_tensor(
        environment,          # pathplanning.GridEnvironment — pybind11 object
        agent_x:  int,
        agent_y:  int,
        grid_height: int = 41,
        grid_width:  int = 41
    ) -> torch.Tensor:
        """
        CONCEPT — Building the 3-channel input from pybind11 data:
        The C++ environment stores the grid as a bitset and cell vector.
        Python accesses it via env.isObstacle(x, y). This helper converts
        that into the (1, 3, H, W) float32 tensor the encoder expects.

        Channel layout:
          [0, :, y, x] = 1.0 if env.isObstacle(x, y) else 0.0
          [1, :, y, x] = 1.0 if (x, y) == agent position else 0.0
          [2, :, y, x] = 1.0 if (x, y) == goal position else 0.0

        Note on indexing: the tensor layout is [batch, channel, row(=y), col(=x)].
        Height indexes rows (y), width indexes columns (x).

        Implement:
        1. Create: grid = torch.zeros(1, 3, grid_height, grid_width, dtype=torch.float32)
        2. Fill obstacle channel:
               for y in range(grid_height):
                   for x in range(grid_width):
                       if environment.isObstacle(x, y):
                           grid[0, 0, y, x] = 1.0
        3. Fill agent channel:   grid[0, 1, agent_y, agent_x] = 1.0
        4. Fill goal channel:
               goal = environment.getGoal()   # returns [gx, gy]
               grid[0, 2, goal[1], goal[0]] = 1.0
        5. return grid
        """
        grid = torch.zeros(1, 3, grid_height, grid_width, dtype=torch.float32)

        for y in range(grid_height):
            for x in range(grid_width):
                if environment.isObstacle(x, y):
                    grid[0, 0, y, x] = 1.0

        grid[0, 1, agent_y, agent_x] = 1.0

        goal = environment.getGoal()
        grid[0, 2, goal[1], goal[0]] = 1.0

        return grid
