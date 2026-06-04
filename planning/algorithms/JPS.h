// planning/algorithms/JPS.h
// Jump Point Search — optimized A* for uniform-cost grids.
// Prunes symmetrical paths by identifying "jump points" — cells where
// the optimal path must pass through. Skips all intermediate cells,
// dramatically reducing nodes explored on open maps.
//
// Key insight: on a uniform grid, many paths between two points are
// symmetric (same cost). JPS exploits this to skip redundant exploration.
//
// Limitation: most effective on open grids. In dense mazes the benefit
// shrinks because jump points are closer together.
#ifndef JPS_H
#define JPS_H

#include <vector>
#include <queue>
#include <unordered_map>
#include <unordered_set>
#include <string>
#include <limits>
#include "../../core/Position.h"
#include "../../core/Types.h"
#include "../../environment/Environment.h"
#include "../IPathfinder.h"

struct JPSNode
{
    Position pos;
    Position previous;
    double   costFromStart;
    double   totalEstimatedCost;
};

struct JPSComparator
{
    bool operator()(const JPSNode& a, const JPSNode& b) const;
};

class JPS : public IPathfinder
{
public:
    JPS() = default;

    std::vector<Position> findPath(
        const Environment& env,
        const Position&    start,
        const Position&    goal
    ) override;

    std::string   getName()          const override;
    AlgorithmType getType()          const override;
    int           getNodesExplored() const override;

private:
    std::unordered_map<Position, double,   PositionHash> costFromStart_;
    std::unordered_map<Position, Position, PositionHash> arrivedFrom_;
    std::unordered_set<Position,           PositionHash> finalized_;

    int nodesExplored_ = 0;

    void   clearState();
    double heuristicDistance(const Position& a, const Position& b) const;
    double costFromStartTo(const Position& p)                       const;

    // Identify natural and forced neighbours in the direction of travel
    std::vector<Position> identifySuccessors(const Environment& env,
                                              const Position&    current,
                                              const Position&    goal)  const;

    // Recursively jump in a direction until a jump point or obstacle is found
    // Returns the jump point position, or an invalid Position if none found
    Position jump(const Environment& env,
                  const Position&    current,
                  const Position&    direction,
                  const Position&    goal)                              const;
};

#endif
