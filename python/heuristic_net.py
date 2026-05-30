# python/heuristic_net.py
#
# PURPOSE: Small MLP that approximates h*(current_position, goal).
#
# CORE CONCEPT — Universal Approximation Theorem
#   A neural network with at least one hidden layer and a non-linear activation
#   can approximate ANY continuous function to arbitrary precision, given enough
#   neurons and training data. Here, h* is a continuous function of position —
#   small position changes produce small h* changes — so an MLP can learn it.
#   This is WHY neural networks work as function approximators.
#
# CORE CONCEPT — What a "layer" actually is
#   Each layer computes: output = activation(W @ input + b)
#   W is a weight matrix (out_size x in_size), b is a bias vector (out_size).
#   This is an AFFINE TRANSFORMATION followed by a NON-LINEARITY.
#   Without the non-linearity, stacking layers is useless — any chain of
#   affine transforms collapses to a single affine transform (matrix multiply).
#   The activation function is what gives depth its expressive power.
#
# CORE CONCEPT — ReLU activation
#   ReLU(x) = max(0, x). Sounds trivial. But it creates PIECEWISE LINEAR regions.
#   Each neuron either "fires" (positive region) or "silences" (negative region).
#   The network partitions input space into regions, fitting a different linear
#   function in each. More neurons = more regions = finer-grained approximation.
#   ReLU is preferred over sigmoid/tanh because it does not saturate for large
#   positive inputs — gradients do not vanish during backpropagation.
#
# CORE CONCEPT — Why this architecture (4 → 64 → 64 → 1)?
#   Input: 4 features (positions, normalised). Small.
#   Two hidden layers of 64: enough to capture the non-linear structure of h*
#   in a maze (walls create discontinuities in the cost landscape).
#   Output: 1 scalar (predicted cost). Regression, not classification.
#   Deeper networks are harder to train (gradient flow issues) for little gain here.
#
# BINARY EXPORT CONCEPT:
#   After training, weights are exported in a custom binary format so C++ can
#   load them without any Python or JSON dependency. The contract between this
#   file (producer) and HeuristicNetwork.cpp (consumer) is the binary layout.
#   See export_weights() below for the exact format.

import struct
import numpy as np
import torch
import torch.nn as nn


INPUT_SIZE  = 4   # [curr_x/W, curr_y/H, goal_x/W, goal_y/H]
HIDDEN_SIZE = 64
OUTPUT_SIZE = 1   # scalar h* estimate

# Binary format magic number — "NAST" in ASCII (Neural A* SeTa).
# Purpose: if C++ reads this file and the first 4 bytes are NOT 0x4E415354,
# something is wrong (wrong file, corrupted, wrong endian). Fail fast.
MAGIC_NUMBER = 0x4E415354


class HeuristicNetwork(nn.Module):
    """
    CONCEPT — Module composition:
    PyTorch's nn.Module is a tree of sub-modules. Each nn.Linear IS a layer —
    it owns its weight tensor and bias tensor. forward() chains them together.
    This mirrors the mathematical composition: f3(f2(f1(x))).
    """

    def __init__(self, input_size: int = INPUT_SIZE, hidden_size: int = HIDDEN_SIZE):
        """
        CONCEPT — Weight initialisation:
        PyTorch initialises Linear weights with Kaiming uniform by default.
        This is designed for ReLU layers — it scales variance to prevent
        signal from shrinking or exploding as it propagates through layers.
        You do NOT need to manually initialise; understanding WHY the default
        is good matters more than the mechanics.
        """
        super().__init__()
        self.layer_one    = nn.Linear(input_size,  hidden_size)
        self.layer_two    = nn.Linear(hidden_size, hidden_size)
        self.output_layer = nn.Linear(hidden_size, OUTPUT_SIZE)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        CONCEPT — Forward pass is just function composition:
        h_hat = W3 * ReLU(W2 * ReLU(W1 * x + b1) + b2) + b3
        PyTorch tracks this computation graph automatically for backpropagation.
        The ONLY job of forward() is to describe WHAT the function computes.
        HOW gradients flow back through it is handled by autograd.

        Note: the final layer has NO activation. We want a raw scalar output
        that can be any non-negative value, not squashed to [0,1].

        Implement:
        1. features = relu(layer_one(input_tensor))
        2. features = relu(layer_two(features))
        3. return output_layer(features)   ← no activation on final layer
        """
        features = torch.relu(self.layer_one(input_tensor))
        features = torch.relu(self.layer_two(features))
        return self.output_layer(features)


def export_weights(model: HeuristicNetwork, output_path: str) -> None:
    """
    CONCEPT — Producer/consumer binary contract:
    Python produces the weights; C++ consumes them. They share no runtime,
    so the only contract is the BINARY LAYOUT of the file.
    We write a fixed-format header followed by layer data in a known order.
    The C++ reader must parse this EXACT layout — any mismatch = garbage inference.

    Binary layout (little-endian, matching x86/ARM default):
      Bytes 0-3:   magic uint32 = 0x4E415354
      Bytes 4:     version uint8 = 1
      Bytes 5-8:   num_layers uint32 (= 3 for this network)
      For each layer in order [layer_one, layer_two, output_layer]:
        uint32 rows          = output neurons (weight matrix rows)
        uint32 cols          = input  neurons (weight matrix cols)
        rows*cols float64s   = weight matrix, row-major order
        rows      float64s   = bias vector

    CONCEPT — Row-major vs column-major:
    NumPy stores arrays in row-major order (C order) by default.
    PyTorch Linear stores weights as (out_features, in_features) — rows are output neurons.
    When we write weight.numpy() directly, we get row-major doubles.
    C++ reads them the same way — row index i means "weights for output neuron i".
    This is the implicit contract. If you use Fortran-order (column-major), C++ reads garbage.

    Implement:
    1. Open output_path in binary write mode ('wb')
    2. Write magic number: struct.pack('<I', MAGIC_NUMBER)  — '<' = little-endian, 'I' = uint32
    3. Write version:      struct.pack('<B', 1)              — 'B' = uint8
    4. Write num_layers:   struct.pack('<I', 3)
    5. For each of the three nn.Linear layers:
       a. Get weight as numpy: W = layer.weight.detach().numpy()   shape: (rows, cols)
       b. Get bias  as numpy: b = layer.bias.detach().numpy()      shape: (rows,)
       c. Write struct.pack('<II', rows, cols)
       d. Write W.astype(np.float64).tobytes()   ← ensures 8-byte doubles, not 4-byte floats
       e. Write b.astype(np.float64).tobytes()
    """
    layers = [model.layer_one, model.layer_two, model.output_layer]
    with open(output_path, 'wb') as binary_file:
        binary_file.write(struct.pack('<I', MAGIC_NUMBER))
        binary_file.write(struct.pack('<B', 1))
        binary_file.write(struct.pack('<I', len(layers)))
        for layer in layers:
            weight_matrix = layer.weight.detach().numpy()
            bias_vector   = layer.bias.detach().numpy()
            rows, cols    = weight_matrix.shape
            binary_file.write(struct.pack('<II', rows, cols))
            binary_file.write(weight_matrix.astype(np.float64).tobytes())
            binary_file.write(bias_vector.astype(np.float64).tobytes())
