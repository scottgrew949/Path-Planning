// environment/DynamicEnvironment.cpp
#include "DynamicEnvironment.h"
#include <stdexcept>

DynamicEnvironment::DynamicEnvironment(int width, int height)
    : Environment(width, height)
{}

void DynamicEnvironment::addDynamicObstacle(const std::vector<Position>& trajectory,
                                             int ticksPerStep)
{
    if (trajectory.size() < 2)
        throw std::invalid_argument("Trajectory must have at least 2 positions");
    if (ticksPerStep < 1)
        throw std::invalid_argument("ticksPerStep must be >= 1");

    for (const Position& waypoint : trajectory)
    {
        if (!inBounds(waypoint))
            throw std::invalid_argument("Trajectory position out of bounds");
    }

    setObstacle(trajectory[0]);

    DynamicObstacle obstacle;
    obstacle.trajectory         = trajectory;
    obstacle.currentIndex       = 0;
    obstacle.ticksPerStep       = ticksPerStep;
    obstacle.ticksSinceLastMove = 0;
    dynamicObstacles_.push_back(obstacle);
}

std::vector<Position> DynamicEnvironment::tick()
{
    std::vector<Position> changedCells;
    ++tickCount_;

    for (DynamicObstacle& obstacle : dynamicObstacles_)
    {
        ++obstacle.ticksSinceLastMove;
        if (obstacle.ticksSinceLastMove < obstacle.ticksPerStep)
            continue;

        obstacle.ticksSinceLastMove = 0;

        Position vacatedCell = obstacle.trajectory[obstacle.currentIndex];
        obstacle.currentIndex = (obstacle.currentIndex + 1)
                                % static_cast<int>(obstacle.trajectory.size());
        Position occupiedCell = obstacle.trajectory[obstacle.currentIndex];

        clearObstacle(vacatedCell);
        setObstacle(occupiedCell);

        changedCells.push_back(vacatedCell);
        changedCells.push_back(occupiedCell);
    }

    return changedCells;
}

int DynamicEnvironment::getDynamicObstacleCount() const
{
    return static_cast<int>(dynamicObstacles_.size());
}

int DynamicEnvironment::getTickCount() const
{
    return tickCount_;
}

std::vector<Position> DynamicEnvironment::getCurrentObstaclePositions() const
{
    std::vector<Position> positions;
    positions.reserve(dynamicObstacles_.size());
    for (const DynamicObstacle& obstacle : dynamicObstacles_)
        positions.push_back(obstacle.trajectory[obstacle.currentIndex]);
    return positions;
}
