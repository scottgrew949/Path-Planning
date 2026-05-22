// environment/SensorModel.h
// SensorModel simulates a range-limited, optionally noisy sensor.
//
// Self-driving car analog:
//   A LiDAR or stereo camera has a finite range. Cells beyond that range are
//   unknown — the planner must treat them as free or keep a prior belief.
//   Noise models real sensor error: false positives create phantom obstacles;
//   false negatives hide real ones. Setting both rates to 0.0 gives a perfect
//   sensor within range — a good baseline before introducing noise.
//
// STL highlights:
//   uniform_real_distribution<double>  — noise probability sampling
//   mt19937                            — fast Mersenne-Twister PRNG
//
// CONCEPT — Why mutable for the RNG?
//   observe() is logically const — it doesn't change what the sensor *is*.
//   But noise generation mutates the internal PRNG state. mutable allows
//   a const function to mutate implementation detail state (the random engine)
//   without exposing that mutability to callers.
#ifndef SENSOR_MODEL_H
#define SENSOR_MODEL_H

#include "../core/Position.h"
#include "Environment.h"
#include <vector>
#include <random>

class SensorModel
{
public:
    // sensorRange       — Manhattan distance limit. Cells beyond are not observed.
    // falsePositiveRate — P(report obstacle | cell is clear). Must be in [0, 1].
    // falseNegativeRate — P(report clear    | cell is obstacle). Must be in [0, 1].
    // seed              — PRNG seed for reproducible noise sequences.
    // Throws std::invalid_argument if any parameter is out of range.
    explicit SensorModel(int      sensorRange,
                         double   falsePositiveRate = 0.0,
                         double   falseNegativeRate = 0.0,
                         unsigned seed              = 42);

    // One observation for a single cell within sensor range.
    struct Observation
    {
        Position cell;
        bool     reportedAsObstacle;  // may differ from ground truth when noise > 0
    };

    // Scan all in-bounds cells within sensorRange of agentPos (Manhattan distance).
    // Returns one Observation per scanned cell, with optional noise applied.
    std::vector<Observation> observe(const Position&    agentPos,
                                     const Environment& env) const;

    // True if target is within sensorRange of agentPos.
    bool isInRange(const Position& agentPos, const Position& target) const;

    int    getSensorRange()       const;
    double getFalsePositiveRate() const;
    double getFalseNegativeRate() const;

private:
    int    sensorRange_;
    double falsePositiveRate_;
    double falseNegativeRate_;

    mutable std::mt19937                          rng_;
    mutable std::uniform_real_distribution<double> uniform_;
};

#endif  // SENSOR_MODEL_H
