// planning/hybrid/HeuristicNetwork.cpp
#include "HeuristicNetwork.h"
#include <fstream>
#include <cmath>
#include <stdexcept>
#include <cstring>

// ---- Layer ------------------------------------------------------------------

std::size_t Layer::outputSize() const
{
    // TODO: return weights.size()
    // weights has one row per output neuron
    return 0;
}

std::size_t Layer::inputSize() const
{
    // TODO: return weights.empty() ? 0 : weights[0].size()
    // each row has one weight per input neuron
    return 0;
}

// ---- HeuristicNetwork -------------------------------------------------------

HeuristicNetwork::HeuristicNetwork(const std::string& weightsFilePath)
{
    // TODO: call loadFromFile(weightsFilePath), assign result to loaded_
    // If weightsFilePath is empty: loaded_ = false (Manhattan fallback will be used)
}

bool HeuristicNetwork::isLoaded() const
{
    return loaded_;
}

bool HeuristicNetwork::loadFromFile(const std::string& filePath)
{
    // CONCEPT — Binary deserialization:
    // We open in std::ios::binary to prevent the OS from translating line endings
    // (on Windows, '\n' bytes would otherwise be transformed — corrupting raw doubles).
    // std::ifstream::read(char* buffer, n) reads exactly n bytes into buffer.
    // We cast our typed variables to char* to use them as byte buffers.
    // This is SAFE here because we are reading into fundamental C++ types (uint32_t,
    // double) whose size is fixed by the standard (4 and 8 bytes respectively).

    // TODO implement:
    // 1. Open filePath with std::ifstream in binary mode.
    //    If it fails to open: return false.
    //
    // 2. Read magic number (4 bytes into uint32_t):
    //      uint32_t magic = 0;
    //      file.read(reinterpret_cast<char*>(&magic), sizeof(uint32_t));
    //    If magic != EXPECTED_MAGIC: return false.
    //
    // 3. Read version (1 byte into uint8_t). If version != EXPECTED_VERSION: return false.
    //
    // 4. Read num_layers (4 bytes into uint32_t).
    //
    // 5. For each layer (loop num_layers times):
    //      a. Read rows (uint32_t) and cols (uint32_t).
    //      b. Create Layer with weights resized to [rows][cols], bias to [rows].
    //      c. Read rows*cols doubles into weights row by row:
    //           for (uint32_t row = 0; row < rows; ++row)
    //               file.read(reinterpret_cast<char*>(layer.weights[row].data()),
    //                         cols * sizeof(double));
    //         CONCEPT: .data() gives a raw pointer to the vector's contiguous buffer.
    //         std::vector<double> IS contiguous in memory — the standard guarantees this.
    //         So reading directly into .data() is safe and avoids per-element overhead.
    //      d. Read rows doubles into bias:
    //           file.read(reinterpret_cast<char*>(layer.bias.data()),
    //                     rows * sizeof(double));
    //      e. Push layer into layers_.
    //
    // 6. Return true if file.good() (no read errors occurred).
    return false;
}

double HeuristicNetwork::predict(
    int currentX, int currentY,
    int goalX,    int goalY,
    int gridWidth, int gridHeight
) const
{
    // CONCEPT — Normalisation at inference time:
    // The network was trained on inputs in [0,1].
    // If we feed it raw pixel coordinates, we are out-of-distribution.
    // The network's internal weight values were optimised for normalised inputs —
    // raw coordinates would produce activations in completely wrong ranges.
    // Always normalise IDENTICALLY to how training data was normalised.

    // TODO implement:
    // 1. Build features vector:
    //      std::vector<double> features = {
    //          static_cast<double>(currentX) / gridWidth,
    //          static_cast<double>(currentY) / gridHeight,
    //          static_cast<double>(goalX)    / gridWidth,
    //          static_cast<double>(goalY)    / gridHeight
    //      };
    //
    // 2. std::vector<double> output = forward(features);
    //
    // 3. Denormalise: raw_h_star = output[0] * (gridWidth + gridHeight - 2)
    //    CONCEPT: The network output is in [0,1] because we normalised h* during training.
    //    Multiplying back by (W+H-2) converts to raw step count.
    //
    // 4. Return std::max(0.0, raw_h_star)
    //    Clamp to non-negative — h* is always >= 0, network may predict slightly negative.
    return 0.0;
}

std::vector<double> HeuristicNetwork::forward(const std::vector<double>& inputFeatures) const
{
    // CONCEPT — Forward pass as sequential transformation:
    // The input vector flows through each layer in order.
    // After each HIDDEN layer: apply ReLU to every element.
    // After the FINAL layer: no activation (raw linear output for regression).
    //
    // The intermediate result after each layer is called an "activation vector"
    // or just "activations". Its size changes at each layer boundary:
    //   Input:          4 elements
    //   After layer 1:  64 elements  (W1 is 64x4)
    //   After layer 2:  64 elements  (W2 is 64x64)
    //   After layer 3:  1 element    (W3 is 1x64)

    // TODO implement:
    // 1. std::vector<double> activations = inputFeatures;
    //
    // 2. Loop over layers_ with index i:
    //      activations = linearTransform(layers_[i], activations);
    //      if (i < layers_.size() - 1): applyReLU(activations);
    //    The condition skips ReLU on the last layer (output layer is linear).
    //
    // 3. Return activations  (length 1 — the predicted h*)
    return {};
}

void HeuristicNetwork::applyReLU(std::vector<double>& activations)
{
    // CONCEPT — Element-wise activation:
    // ReLU is applied independently to EACH element of the activation vector.
    // There is no interaction between neurons here — pure element-wise clamping.
    // max(0.0, x) means negative activations are "silenced" (zeroed out).
    // This creates sparse activations — typically half the neurons fire.
    // Sparse activations are actually beneficial: they create cleaner gradient signals.

    // TODO: for each element in activations: element = std::max(0.0, element)
}

std::vector<double> HeuristicNetwork::linearTransform(
    const Layer&               layer,
    const std::vector<double>& inputVector
)
{
    // CONCEPT — Matrix-vector multiplication: dot product per output neuron.
    // For output neuron i:
    //   output[i] = b[i] + sum over j of (W[i][j] * input[j])
    //
    // This is the INNER PRODUCT of the i-th row of W with the input vector,
    // plus the bias b[i].
    //
    // CONCEPT — Why bias matters:
    // Without bias, every neuron's activation is forced through the origin.
    // The bias shifts the activation threshold — like the y-intercept in y=mx+b.
    // Without it, the network cannot represent functions that are non-zero
    // when the input is zero.
    //
    // CONCEPT — Computational complexity:
    // rows * cols multiplications + rows additions.
    // For a 64x64 layer: 4096 multiplications. Fast on modern hardware.
    // For comparison, a single A* path in this project does ~5000 expansions,
    // each calling this function. That's 20M multiplications — still < 1ms.

    // TODO implement:
    // 1. std::size_t rows = layer.outputSize();
    //    std::size_t cols = layer.inputSize();
    // 2. std::vector<double> outputVector(rows, 0.0);
    // 3. For each i in [0, rows):
    //      outputVector[i] = layer.bias[i];
    //      For each j in [0, cols):
    //          outputVector[i] += layer.weights[i][j] * inputVector[j];
    // 4. return outputVector;
    return {};
}
