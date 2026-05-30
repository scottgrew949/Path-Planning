// utils/ProbabilityUtils.cpp
#include "ProbabilityUtils.h"

using namespace std;

// ---- Bayesian updates -------------------------------------------------------

double ProbabilityUtils::bayesUpdate(double priorObstacleProbability,
                                     double sensorReadingProbability,
                                     double totalSensorReadingProbability)
{
    if (priorObstacleProbability > 1.0 || priorObstacleProbability < 0.0)
        throw std::invalid_argument("Invalid priorObstacleProbability");
    else if (sensorReadingProbability > 1.0 || sensorReadingProbability < 0.0)
        throw std::invalid_argument("Invalid sensorReadingProbability");
    else if (totalSensorReadingProbability > 1.0 || totalSensorReadingProbability < 0.0)
        throw std::invalid_argument("Invalid totalSensorReadingProbability");

    if (totalSensorReadingProbability == 0)
        throw std::invalid_argument("totalSensorReadingProbability is zero. Division by zero invalid.");

    return (sensorReadingProbability * priorObstacleProbability) / totalSensorReadingProbability;
}

double ProbabilityUtils::bayesUpdateSensor(double priorObstacleProbability,
                                           double truePositiveRate,
                                           double falsePositiveRate,
                                           bool   sensorFired)
{
    if (priorObstacleProbability > 1.0 || priorObstacleProbability < 0.0)
        throw std::invalid_argument("Invalid priorObstacleProbability");
    else if (truePositiveRate > 1.0 || truePositiveRate < 0.0)
        throw std::invalid_argument("Invalid TruePositiveRate");
    else if (falsePositiveRate > 1.0 || falsePositiveRate < 0.0)
        throw std::invalid_argument("Invalid FalsePositiveRate");

    double sensorReadingProbability = 0.0;
    double totalSensorReadingProbability   = 0.0;
    if (sensorFired)
    {
        sensorReadingProbability = truePositiveRate;
        totalSensorReadingProbability  = truePositiveRate * priorObstacleProbability + falsePositiveRate * 
                    (1.0 - priorObstacleProbability);
    }
    else
    {
        sensorReadingProbability = 1.0 - truePositiveRate;          // (sensor did NOT fire)
        totalSensorReadingProbability   = (1.0 - truePositiveRate) * priorObstacleProbability + 
                    (1.0 - falsePositiveRate) * (1.0 - priorObstacleProbability);
    }

    return bayesUpdate(priorObstacleProbability, sensorReadingProbability, totalSensorReadingProbability);
    // Self-driving car context:
    //   A LiDAR return at a cell (sensorFired=true) with 90% TPR and 10% FPR
    //   on a cell with 20% prior → posterior ≈ 0.69.  One hit nearly triples
    //   the belief. Multiple consistent hits converge to near-certainty.
}

// ---- Log-odds occupancy grid ------------------------------------------------

double ProbabilityUtils::logOddsUpdate(double currentLogOdds,
                                       double truePositiveRate,
                                       double falsePositiveRate,
                                       bool   sensorFired,
                                       double logOddsMin,
                                       double logOddsMax)
{
    // CONCEPT — Sensor log-likelihood ratio:
    //   When sensor fires:   update = log(P(z=1|occ) / P(z=1|free)) = log(TPR / FPR)
    //   When sensor silent:  update = log(P(z=0|occ) / P(z=0|free)) = log((1-TPR)/(1-FPR))
    //   Adding these to the current log-odds is equivalent to the full Bayesian update
    //   but numerically stable and O(1) — no division, no product of small floats.
    double inverseLogLikelihood;
    if (sensorFired)
        inverseLogLikelihood = std::log(truePositiveRate  / falsePositiveRate);
    else
        inverseLogLikelihood = std::log((1.0 - truePositiveRate) / (1.0 - falsePositiveRate));

    double updated = currentLogOdds + inverseLogLikelihood;

    if (updated < logOddsMin) return logOddsMin;
    if (updated > logOddsMax) return logOddsMax;
    return updated;
}

double ProbabilityUtils::logOddsToProb(double logOdds)
{
    return 1.0 / (1.0 + std::exp(-logOdds));
}

double ProbabilityUtils::probToLogOdds(double probability)
{
    // Clamp away from 0 and 1 to keep log finite.
    const double epsilon = 1e-9;
    if (probability < epsilon)        probability = epsilon;
    if (probability > 1.0 - epsilon)  probability = 1.0 - epsilon;
    return std::log(probability / (1.0 - probability));
}

// ---- Expected value ---------------------------------------------------------

double ProbabilityUtils::expectedValue(const vector<double>& possibleOutcomes,
                                       const vector<double>& outcomesProbabilities)
{
    if (outcomesProbabilities.size() != possibleOutcomes.size())
        throw std::invalid_argument("Values and possibleOutcome mismatch.");
    if (!isValidPMF(outcomesProbabilities)) 
        throw std::invalid_argument("not a valid PMF");
        
    return std::inner_product(possibleOutcomes.begin(), possibleOutcomes.end(), outcomesProbabilities.begin(), 0.0);
    //       inner_product computes sum(values[i] * probs[i]) in one pass.
}

// ---- Entropy ----------------------------------------------------------------

double ProbabilityUtils::entropy(const vector<double>& outcomesProbabilities)
{
    if (!isValidPMF(outcomesProbabilities)) 
        throw std::invalid_argument("Not a valid PMF");
    
    double H = 0.0;
    for (double p : outcomesProbabilities)
    {
        if (p > 0.0) 
        {
            H -= p * log2(p);
        }
    }

    return H;
}

// ---- Helpers ----------------------------------------------------------------

bool ProbabilityUtils::isValidPMF(const vector<double>& outcomesProbabilities, double tolerance)
{
    for (const double& p : outcomesProbabilities)
    {
        if (p < 0.0)
            return false;
    }
    
    double sum = accumulate(outcomesProbabilities.begin(), outcomesProbabilities.end(), 0.0);
    return std::abs(sum - 1.0) <= tolerance;
}

void ProbabilityUtils::normalise(vector<double>& weights)
{
    double total = accumulate(weights.begin(), weights.end(), 0.0);
   
    if (total == 0.0)
        throw std::invalid_argument("All weights are zero.");

    for (double& w : weights)
        w /= total;
}
