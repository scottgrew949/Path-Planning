// environment/DynamicEnvironment.h
// DynamicEnvironment extends Environment with obstacles that follow
// cyclic trajectories, simulating moving agents (pedestrians, vehicles).
//
// Self-driving car analog:
//   Each DynamicObstacle is a pedestrian or vehicle with a repeating path.
//   tick() advances the simulation one step. After each tick, call findPath()
//   on your planner to replan around the new obstacle positions.
//
// STL highlights:
//   vector<DynamicObstacle>  — registry of all moving obstacles
//   vector<Position>         — each obstacle's cyclic waypoint list
//
// CONCEPT — Why cyclic trajectories?
//   Real pedestrians and vehicles don't move randomly — they follow predictable
//   patterns (crosswalks, patrol routes, parking loops). Cyclic trajectories
//   let us model repeating behaviour without complex agent logic.
#ifndef DYNAMIC_ENVIRONMENT_H
#define DYNAMIC_ENVIRONMENT_H

#include "Environment.h"
#include <vector>

class DynamicEnvironment : public Environment
{
public:
    explicit DynamicEnvironment(int width, int height);

    // Register a moving obstacle that cycles through trajectory in order.
    // The first position in trajectory is placed as an obstacle immediately.
    // ticksPerStep: how many tick() calls elapse between each waypoint advance.
    // Throws std::invalid_argument if trajectory has fewer than 2 positions,
    // ticksPerStep < 1, or any position is out of bounds.
    void addDynamicObstacle(const std::vector<Position>& trajectory, int ticksPerStep = 1);

    // Advance the simulation by one tick.
    // For each obstacle whose ticksPerStep interval has elapsed, clears the
    // old cell and sets the next cell in the trajectory.
    // Returns all cells whose obstacle state changed — pass each to your
    // planner's update/replan logic.
    std::vector<Position> tick();

    int getDynamicObstacleCount() const;
    int getTickCount()            const;

private:
    struct DynamicObstacle
    {
        std::vector<Position> trajectory;
        int                   currentIndex;
        int                   ticksPerStep;
        int                   ticksSinceLastMove;
    };

    std::vector<DynamicObstacle> dynamicObstacles_;
    int                          tickCount_ = 0;
};

#endif  // DYNAMIC_ENVIRONMENT_H
