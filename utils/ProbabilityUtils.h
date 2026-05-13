// utils/ProbabilityUtils.h
// Probability and statistics utilities for uncertain obstacle modeling.
//
// These functions model the same mathematics used in real autonomous vehicle
// sensor fusion pipelines:
//   - Bayesian updates: combine prior obstacle belief with new sensor evidence
//   - Expected value:   compute mean cost across candidate paths
//   - Entropy:          measure uncertainty in a probability distribution
//
// All methods are static. The class is non-instantiable.
#ifndef PROBABILITY_UTILS_H
#define PROBABILITY_UTILS_H

#include <vector>
#include <stdexcept>
#include <numeric>
#include <cmath>

class ProbabilityUtils
{
public:
    ProbabilityUtils()  = delete;
    ~ProbabilityUtils() = delete;

    // -------------------------------------------------------------------------
    // Bayesian obstacle belief update
    // -------------------------------------------------------------------------

    // Full Bayes theorem:
    //   P(obstacle | evidence) = P(evidence | obstacle) * P(obstacle) / P(evidence)
    //
    // prior:      current belief that a cell is blocked, in [0, 1]
    // likelihood: P(sensor reading | obstacle is actually there)
    // marginal:   total probability of the sensor reading (normalisation constant)
    //             = likelihood * prior + P(reading | free) * (1 - prior)
    //
    // Self-driving car analog: every LiDAR / radar return triggers this update
    // on the occupancy grid cell the beam hit.
    static double bayesUpdate(double prior, double likelihood, double marginal);

    // Convenience wrapper that computes the marginal internally from sensor rates.
    //
    // truePositiveRate:  P(sensor fires | obstacle present)  — e.g. 0.9
    // falsePositiveRate: P(sensor fires | cell is free)      — e.g. 0.1
    // sensorFired:       true if the sensor reported an obstacle
    static double bayesUpdateSensor(double prior,
                                    double truePositiveRate,
                                    double falsePositiveRate,
                                    bool   sensorFired);

    // -------------------------------------------------------------------------
    // Expected value
    // -------------------------------------------------------------------------

    // E[X] = sum_i( values[i] * probs[i] )
    // Requires probs to be a valid PMF (non-negative, sum to 1.0).
    // Used to compute mean path cost across multiple route candidates.
    static double expectedValue(const std::vector<double>& values,
                                const std::vector<double>& probs);

    // Template overload — accepts any numeric value type (int, float, etc.)
    // implicitly convertible to double.
    // Example: expectedValueT<int>({5, 10, 15}, {0.5, 0.3, 0.2}) → 9.0
    template <typename T>
    static double expectedValueT(const std::vector<T>&      values,
                                 const std::vector<double>& probs);

    // -------------------------------------------------------------------------
    // Entropy
    // -------------------------------------------------------------------------

    // Shannon entropy: H = -sum( p * log2(p) )  — measured in bits.
    // A uniform distribution over N outcomes has entropy log2(N).
    // Used to quantify route uncertainty or sensor information gain.
    static double entropy(const std::vector<double>& probs);

    // -------------------------------------------------------------------------
    // Helpers
    // -------------------------------------------------------------------------

    // Returns true iff all elements >= 0 and sum is within tolerance of 1.0.
    static bool isValidPMF(const std::vector<double>& probs,
                           double tolerance = 1e-9);

    // Normalise a non-negative weight vector in-place so it sums to 1.0.
    // Throws std::invalid_argument if all weights are zero.
    static void normalise(std::vector<double>& weights);
};

// ---- Template implementation (must live in header) --------------------------

template <typename T>
double ProbabilityUtils::expectedValueT(const std::vector<T>&      values,
                                        const std::vector<double>& probs)
{
    if (values.size() != probs.size())
        throw std::invalid_argument("size mismatch");
    if (!isValidPMF(probs))
        throw std::invalid_argument("not a valid PMF");

    double sum = 0.0;
    for (size_t i = 0; i < values.size(); ++i)
        sum += static_cast<double>(values[i]) * probs[i];

    return sum;
}

#endif  // PROBABILITY_UTILS_H
