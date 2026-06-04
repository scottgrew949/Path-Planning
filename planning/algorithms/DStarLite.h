// planning/algorithms/DStarLite.h
// D* Lite — dynamic replanning algorithm. Finds the optimal path, then
// efficiently repairs it when obstacles appear or disappear without
// restarting from scratch.
//
// Key concepts:
//   g(s)   — current best cost estimate from s to goal
//   rhs(s) — one-step lookahead cost (more stable than g)
//   A node is locally consistent when g(s) == rhs(s)
//   Inconsistent nodes are queued for processing
#ifndef DSTAR_LITE_H
#define DSTAR_LITE_H

#include <vector>
#include <queue>
#include <unordered_map>
#include <string>
#include <limits>
#include <utility>
#include "../../core/Position.h"
#include "../../core/Types.h"
#include "../../environment/Environment.h"
#include "../IPathfinder.h"

// Priority key is a pair — primary sort on first, secondary on second.
struct DStarKey
{
    double primary;
    double secondary;

    bool operator>(const DStarKey& other) const;
};

struct DStarNode
{
    Position pos;
    DStarKey key;
};

struct DStarComparator
{
    bool operator()(const DStarNode& a, const DStarNode& b) const;
};

class DStarLite : public IPathfinder
{
public:
    DStarLite() = default;

    std::vector<Position> findPath(
        const Environment& env,
        const Position&    start,
        const Position&    goal
    ) override;

    std::string   getName()          const override;
    AlgorithmType getType()          const override;
    int           getNodesExplored() const override;

    // Call when an obstacle changes — triggers incremental replan
    void updateObstacle(const Environment& env,
                        const Position&    changedCell,
                        const Position&    start);

private:
    std::unordered_map<Position, double, PositionHash> g_;
    std::unordered_map<Position, double, PositionHash> rhs_;

    int nodesExplored_ = 0;

    void   clearState();
    double heuristicDistance(const Position& a, const Position& b) const;
    DStarKey calculateKey(const Position& s, const Position& start) const;
    void   updateVertex(const Position& s, const Environment& env, const Position& start);
    void   computeShortestPath(const Environment& env, const Position& start, const Position& goal);

    std::priority_queue<DStarNode,
                        std::vector<DStarNode>,
                        DStarComparator> openList_;
};

#endif
