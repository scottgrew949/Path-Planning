// environment/SensorModel.cpp
#include "SensorModel.h"
#include <stdexcept>
#include <algorithm>
#include <cstdlib>

SensorModel::SensorModel(int sensorRange, double falsePositiveRate, double falseNegativeRate,
                         unsigned seed)
    : sensorRange_(sensorRange),
      falsePositiveRate_(falsePositiveRate),
      falseNegativeRate_(falseNegativeRate),
      rng_(seed),
      uniform_(0.0, 1.0)
{
    if (sensorRange < 1)
        throw std::invalid_argument("sensorRange must be >= 1");
    if (falsePositiveRate < 0.0 || falsePositiveRate > 1.0)
        throw std::invalid_argument("falsePositiveRate must be in [0, 1]");
    if (falseNegativeRate < 0.0 || falseNegativeRate > 1.0)
        throw std::invalid_argument("falseNegativeRate must be in [0, 1]");
}

std::vector<SensorModel::Observation> SensorModel::observe(const Position&    agentPos,
                                                            const Environment& env) const
{
    std::vector<Observation> observations;

    // Iterate over Manhattan diamond rows — guarantees every cell visited is
    // within sensorRange without a secondary isInRange() filter.
    for (int deltaY = -sensorRange_; deltaY <= sensorRange_; ++deltaY)
    {
        int y = agentPos.y + deltaY;
        if (y < 0 || y >= env.getHeight()) continue;

        int xRadius = sensorRange_ - std::abs(deltaY);
        int xMin    = std::max(0, agentPos.x - xRadius);
        int xMax    = std::min(env.getWidth() - 1, agentPos.x + xRadius);

        for (int x = xMin; x <= xMax; ++x)
        {
            Position cell(x, y);

            bool groundTruth = env.isObstacle(cell);
            bool reported    = groundTruth;

            if (groundTruth && uniform_(rng_) < falseNegativeRate_)
                reported = false;
            else if (!groundTruth && uniform_(rng_) < falsePositiveRate_)
                reported = true;

            observations.push_back({ cell, reported });
        }
    }

    return observations;
}

bool SensorModel::isInRange(const Position& agentPos, const Position& target) const
{
    return (std::abs(agentPos.x - target.x) + std::abs(agentPos.y - target.y)) <= sensorRange_;
}

int SensorModel::getSensorRange() const
{
    return sensorRange_;
}

double SensorModel::getFalsePositiveRate() const
{
    return falsePositiveRate_;
}

double SensorModel::getFalseNegativeRate() const
{
    return falseNegativeRate_;
}
