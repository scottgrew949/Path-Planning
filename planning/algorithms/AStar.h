// planning/algorithms/AStar.h
// A* search — optimal when heuristic is admissible (never overestimates).
//
// STL highlights:
//   priority_queue<AStarNode, vector<AStarNode>, NodeComparator>
//       – min-heap ordered by fCost = gCost + h(pos, goal)
//   unordered_map<Position, double,   PositionHash>  – O(1) gScore lookup
//   unordered_map<Position, Position, PositionHash>  – O(1) parent tracking
//
// Algorithm sketch (for implementation reference):
//   1. Push {start, g=0, f=h(start,goal)} onto openSet.
//   2. gScore[start] = 0; all others implicitly +∞.
//   3. Loop until openSet empty:
//        current = openSet.top(); openSet.pop()
//        if current.pos == goal: return reconstructPath(goal)
//        for each neighbour n of current.pos:
//          tentative_g = gScore[current.pos] + moveCost(current, n)
//          if tentative_g < gScore[n] (or n not yet in gScore):
//            cameFrom[n]  = current.pos
//            gScore[n]    = tentative_g
//            fScore[n]    = tentative_g + heuristic(n, goal)
//            push {n, gScore[n], fScore[n]} onto openSet
//   4. Return {} (no path).
#ifndef ASTAR_H
#define ASTAR_H

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

// ---- AStarNode --------------------------------------------------------------
// Lightweight value type pushed onto the priority_queue.
// Stores a copy of pos to avoid pointer invalidation.
struct AStarNode
{
    Position    pos;
    Position    previous;
    double      currentCost;            // cost from start to this node
    double      totalEstimatedCost;     // gCost + h(pos, goal); drives heap ordering
};

// ---- NodeComparator ---------------------------------------------------------
// Makes priority_queue a min-heap on fCost (lower fCost = higher priority).
// Passed as the third template argument to std::priority_queue.
struct NodeComparator
{
    // Returns true when 'a' should be popped AFTER 'b' (i.e. a has lower priority).
    bool operator()(const AStarNode& a, const AStarNode& b) const;
};

// ---- AStar ------------------------------------------------------------------
class AStar : public IPathfinder
{
public:
    AStar() = default;

    std::vector<Position> findPath(
        const Environment& env,
        const Position&    start,
        const Position&    goal
    ) override;

    std::string   getName() const override;
    AlgorithmType getType() const override;
    int getNodesExplored() const override;

private:
    int nodesExplored_ = 0;
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

#endif  // ASTAR_H
