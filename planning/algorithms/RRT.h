// planning/algorithms/RRT.h
// Rapidly-exploring Random Tree — probabilistic path planning algorithm.
// Builds a tree by randomly sampling the space and extending toward samples.
// Unlike grid-based algorithms, RRT works in continuous space and naturally
// handles high-dimensional configuration spaces.
//
// Self-driving car analog: used for motion planning in continuous space —
// parking maneuvers, lane changes, arm movements in robotic systems.
//
// Key concepts:
//   - Sample a random position in the grid
//   - Find the nearest node in the tree
//   - Extend toward the sample by stepSize
//   - If the extension is collision-free, add it to the tree
//   - Repeat until goal is within stepSize of a tree node
//
// Limitation: paths are not optimal — RRT finds any valid path, not the
// shortest. RRT* (a future extension) adds rewiring to approach optimality.
//
// STL highlights:
//   vector<RRTNode>  — the tree stored as flat list
//   mt19937          — random position sampling
#ifndef RRT_H
#define RRT_H

#include <vector>
#include <string>
#include <random>
#include <limits>
#include "../../core/Position.h"
#include "../../core/Types.h"
#include "../../environment/Environment.h"
#include "../IPathfinder.h"

// ---- RRTNode ----------------------------------------------------------------
struct RRTNode
{
    Position pos;
    int      parentIndex;   // index into the tree vector, -1 for root
};

// ---- RRT --------------------------------------------------------------------
class RRT : public IPathfinder
{
public:
    RRT() = default;

    std::vector<Position> findPath(
        const Environment& env,
        const Position&    start,
        const Position&    goal
    ) override;

    std::string   getName()          const override;
    AlgorithmType getType()          const override;
    int           getNodesExplored() const override;

private:
    std::vector<RRTNode> tree_;
    int                  nodesExplored_ = 0;

    // Maximum distance to extend toward a sample per iteration
    static constexpr int stepSize_ = 2;

    // Maximum iterations before giving up
    static constexpr int maxIterations_ = 10000;

    void     clearState();

    // Find index of node in tree_ closest to target
    int      nearestNodeIndex(const Position& target)                  const;

    // Step from nearest toward target by stepSize_
    Position steer(const Position& nearest, const Position& target)    const;

    // Check if the straight line from a to b is obstacle-free
    bool     isCollisionFree(const Environment& env,
                              const Position&    a,
                              const Position&    b)                    const;

    // Walk parentIndex chain from goalNode back to root, return path
    std::vector<Position> extractPath(int goalNodeIndex)               const;
};

#endif  // RRT_H
