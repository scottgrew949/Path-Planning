// planning/algorithms/AStar.h
// A* search — optimal when heuristic is admissible (never overestimates).
#ifndef ASTAR_H
#define ASTAR_H

#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <string>
#include "../../core/Position.h"
#include "../../core/Types.h"
#include "../../environment/Environment.h"
#include "../IPathfinder.h"

// Lightweight value type pushed onto the priority_queue.
struct AStarNode
{
    Position    pos;
    Position    previous;
    double      currentCost;            // cost from start to this node
    double      totalEstimatedCost;     // gCost + h(pos, goal); drives heap ordering
};

// Makes priority_queue a min-heap on fCost (lower fCost = higher priority).
struct NodeComparator
{
    // Returns true when 'a' should be popped AFTER 'b' (i.e. a has lower priority).
    bool operator()(const AStarNode& a, const AStarNode& b) const;
};

class AStar : public IPathfinder
{
public:
    // epsilon = 1.0 → standard A* (default, backward compatible).
    // epsilon > 1.0 → weighted A*. Throws if epsilon < 1.0.
    explicit AStar(double epsilon = 1.0);

    std::vector<Position> findPath(
        const Environment& env,
        const Position&    start,
        const Position&    goal
    ) override;

    std::string   getName() const override;
    AlgorithmType getType() const override;
    int           getNodesExplored() const override;
    double        getEpsilon() const;

private:
    double epsilon_       = 1.0;
    int    nodesExplored_ = 0;
    // Per-call state — cleared before each findPath() invocation.
    std::unordered_map<Position, double,   PositionHash> costFromStart_;
    std::unordered_map<Position, Position, PositionHash> arrivedFrom_;
    std::unordered_set<Position,           PositionHash> finalized_;

    // Manhattan distance heuristic — admissible for cardinal-only movement.
    double heuristicDistance(const Position& a, const Position& b) const;

    // Reset all per-call maps so findPath() is safe to call multiple times.
    void clearState();

    // Helper: return gScore_[p] or +∞ if p has not been discovered yet.
    double costFromStartTo(const Position& p) const;
};

#endif
