// planning/hybrid/HeuristicNetwork.h
//
// PURPOSE: Load trained MLP weights from binary file and run forward inference
//          entirely in C++, with zero Python/pybind11 dependency at runtime.
//
// CORE CONCEPT — Why pure C++ inference?
//   The heuristic is called once per node expansion in A*.
//   For a 41x41 maze, findPath() may expand thousands of nodes.
//   Crossing the Python/C++ boundary (GIL acquire/release) costs ~1-5 microseconds.
//   5000 expansions * 5 us = 25ms overhead — makes Neural A* slower than plain A*.
//   By doing inference in C++, we pay only the actual compute cost: ~2000
//   multiply-adds per call, which runs in nanoseconds.
//   This is why production ML inference engines (TensorRT, ONNX Runtime, CoreML)
//   exist — they sever the Python runtime from the inference hot path.
//
// CORE CONCEPT — A neural network as a list of matrix operations
//   Mathematically, a 3-layer MLP is just:
//       h1 = ReLU(W1 * x  + b1)
//       h2 = ReLU(W2 * h1 + b2)
//       y  = W3 * h2 + b3
//   W1, W2, W3 are weight matrices. b1, b2, b3 are bias vectors.
//   The "* " operations are matrix-vector multiplications.
//   No magic. The "intelligence" is encoded in the VALUES of W and b,
//   not in the structure of the computation.
//
// CORE CONCEPT — Layer struct as the fundamental unit
//   Each layer owns exactly two things: weights (W) and biases (b).
//   W has shape [out_neurons][in_neurons] — one row per output neuron.
//   b has shape [out_neurons] — one offset per output neuron.
//   The forward computation for one layer: for each output neuron i,
//       output[i] = dot(W[i], input) + b[i]
//   This is the inner product you learned in linear algebra.
//   You are literally doing linear algebra on the trained weight tensors.
//
// BINARY FORMAT CONTRACT (must match python/heuristic_net.py export_weights()):
//   Bytes 0-3:   uint32  magic = 0x4E415354  ("NAST")
//   Byte  4:     uint8   version = 1
//   Bytes 5-8:   uint32  num_layers (3 for our network)
//   For each layer:
//     uint32  rows   (= output neurons)
//     uint32  cols   (= input  neurons)
//     rows*cols float64s  (weight matrix, row-major)
//     rows      float64s  (bias vector)

#ifndef HEURISTIC_NETWORK_H
#define HEURISTIC_NETWORK_H

#include <vector>
#include <string>

// ---- Layer ------------------------------------------------------------------
// Fundamental data unit. Owns one weight matrix and one bias vector.
// weights[i][j] = weight from input neuron j to output neuron i.
// bias[i]       = offset added to output neuron i before activation.
struct Layer
{
    std::vector<std::vector<double>> weights;  // shape: [out_neurons][in_neurons]
    std::vector<double>              bias;     // shape: [out_neurons]

    // Convenience accessor: number of output neurons this layer produces.
    std::size_t outputSize() const;

    // Convenience accessor: number of input neurons this layer expects.
    std::size_t inputSize() const;
};

// ---- HeuristicNetwork -------------------------------------------------------
class HeuristicNetwork
{
public:
    // CONCEPT — Construction loads weights from disk once, at startup.
    //   All subsequent calls to predict() are pure computation — no I/O.
    //   If the file cannot be opened or the magic number is wrong, isLoaded()
    //   returns false and predict() returns the Manhattan fallback.
    explicit HeuristicNetwork(const std::string& weightsFilePath);

    // Returns true if weights were loaded successfully from the binary file.
    bool isLoaded() const;

    // CONCEPT — Feature normalisation at the call site:
    //   We pass raw grid coordinates and dimensions. HeuristicNetwork normalises
    //   them internally: currentX/gridWidth, currentY/gridHeight, etc.
    //   This way the caller never needs to know the normalisation convention.
    //   The network was trained on normalised inputs — if we skip normalisation,
    //   we feed it inputs from a completely different distribution → garbage output.
    double predict(
        int currentX, int currentY,
        int goalX,    int goalY,
        int gridWidth, int gridHeight
    ) const;

private:
    bool                loaded_ = false;
    std::vector<Layer>  layers_;

    // CONCEPT — Deserialization: reading structured binary data.
    //   We open the file in binary mode (std::ios::binary) and use
    //   std::ifstream::read() to copy raw bytes into typed variables.
    //   The types MUST match the Python writer:
    //     Python struct.pack('<I', x) → uint32_t in C++ → 4 bytes, little-endian.
    //     Python np.float64           → double    in C++ → 8 bytes, IEEE 754.
    //   If the magic number does not match EXPECTED_MAGIC, return false.
    bool loadFromFile(const std::string& filePath);

    // CONCEPT — Forward pass: pure linear algebra.
    //   This is THE key function. Understand it and you understand all MLP inference.
    //   input is a vector of feature values (4 elements for our network).
    //   Each layer transforms the vector: new_vector = ReLU(W * old_vector + b).
    //   The final layer skips ReLU (linear output for regression).
    //   Returns a vector of length 1 — the raw output (predicted h*, normalised).
    std::vector<double> forward(const std::vector<double>& inputFeatures) const;

    // CONCEPT — ReLU applied element-wise after each hidden layer.
    //   ReLU(x) = max(0, x). Simple. But without it, stacking linear layers
    //   is mathematically equivalent to ONE linear layer — they collapse.
    //   The non-linearity is what makes depth useful.
    static void applyReLU(std::vector<double>& activations);

    // CONCEPT — Matrix-vector multiply: the core computation.
    //   output[i] = sum_j(layer.weights[i][j] * input[j]) + layer.bias[i]
    //   This is O(rows * cols) — for a 64x4 layer that's 256 multiplications.
    //   BLAS/Eigen do this with SIMD instructions, ~8 multiplications per clock cycle.
    //   Your loop will do one per clock cycle. The difference shows why libraries exist.
    static std::vector<double> linearTransform(
        const Layer&               layer,
        const std::vector<double>& inputVector
    );

    static constexpr uint32_t EXPECTED_MAGIC   = 0x4E415354;
    static constexpr uint8_t  EXPECTED_VERSION = 1;
};

#endif  // HEURISTIC_NETWORK_H
