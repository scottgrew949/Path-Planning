// planning/hybrid/HeuristicNetwork.cpp
#include "HeuristicNetwork.h"
#include <fstream>
#include <cstring>

// ---- Layer ------------------------------------------------------------------

std::size_t Layer::outputSize() const
{
    // weights has one row per output neuron
    return weights.size();
}

std::size_t Layer::inputSize() const
{
    // each row has one weight per input neuron
    return weights.empty() ? 0 : weights[0].size();
}

// ---- HeuristicNetwork -------------------------------------------------------

HeuristicNetwork::HeuristicNetwork(const std::string& weightsFilePath)
{
    if (weightsFilePath.empty())
    {
        loaded_ = false;
        return;
    }
    loaded_ = loadFromFile(weightsFilePath);
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

    std::ifstream file(filePath, std::ios::binary);
    if (!file.is_open())
        return false;

    uint32_t magic = 0;
    file.read(reinterpret_cast<char*>(&magic), sizeof(uint32_t));
    if (magic != EXPECTED_MAGIC)
        return false;

    uint8_t version = 0;
    file.read(reinterpret_cast<char*>(&version), sizeof(uint8_t));
    if (version != EXPECTED_VERSION)
        return false;

    uint32_t numLayers = 0;
    file.read(reinterpret_cast<char*>(&numLayers), sizeof(uint32_t));

    for (uint32_t layerIndex = 0; layerIndex < numLayers; ++layerIndex)
    {
        uint32_t rows = 0;
        uint32_t cols = 0;
        file.read(reinterpret_cast<char*>(&rows), sizeof(uint32_t));
        file.read(reinterpret_cast<char*>(&cols), sizeof(uint32_t));

        Layer layer;
        layer.weights.resize(rows, std::vector<double>(cols));
        layer.bias.resize(rows);

        for (uint32_t row = 0; row < rows; ++row)
            file.read(reinterpret_cast<char*>(layer.weights[row].data()),
                      cols * sizeof(double));

        // CONCEPT: .data() gives a raw pointer to the vector's contiguous buffer.
        // std::vector<double> IS contiguous in memory — the standard guarantees this.
        // So reading directly into .data() is safe and avoids per-element overhead.
        file.read(reinterpret_cast<char*>(layer.bias.data()),
                  rows * sizeof(double));

        layers_.push_back(layer);
    }

    return file.good();
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

    std::vector<double> features = {
        static_cast<double>(currentX) / gridWidth,
        static_cast<double>(currentY) / gridHeight,
        static_cast<double>(goalX)    / gridWidth,
        static_cast<double>(goalY)    / gridHeight
    };

    std::vector<double> output = forward(features);

    // CONCEPT: The network output is in [0,1] because we normalised h* during training.
    // Multiplying back by (W+H-2) converts to raw step count.
    double rawHStar = output[0] * (gridWidth + gridHeight - 2);

    return std::max(0.0, rawHStar);
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

    std::vector<double> activations = inputFeatures;

    for (std::size_t i = 0; i < layers_.size(); ++i)
    {
        activations = linearTransform(layers_[i], activations);
        if (i < layers_.size() - 1)
            applyReLU(activations);
    }

    return activations;
}

void HeuristicNetwork::applyReLU(std::vector<double>& activations)
{
    // CONCEPT — Element-wise activation:
    // ReLU is applied independently to EACH element of the activation vector.
    // There is no interaction between neurons here — pure element-wise clamping.
    // max(0.0, x) means negative activations are "silenced" (zeroed out).
    // This creates sparse activations — typically half the neurons fire.
    // Sparse activations are actually beneficial: they create cleaner gradient signals.

    for (double& element : activations)
        element = std::max(0.0, element);
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

    std::size_t rows = layer.outputSize();
    std::size_t cols = layer.inputSize();

    std::vector<double> outputVector(rows, 0.0);

    for (std::size_t i = 0; i < rows; ++i)
    {
        outputVector[i] = layer.bias[i];
        for (std::size_t j = 0; j < cols; ++j)
            outputVector[i] += layer.weights[i][j] * inputVector[j];
    }

    return outputVector;
}
