// planning/algorithms/Dijkstra.h
#ifndef DIJKSTRA_H
#define DIJKSTRA_H

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

struct DijkstraNode
{
    Position    pos;
    Position    previous;
    double      costFromStart;   // cumulative cost from start
};

// Min-heap on costFromStart —  always expands the closest unsettled node.
struct DijkstraComparator
{
    bool operator()(const DijkstraNode& a, const DijkstraNode& b) const;
};

class Dijkstra : public IPathfinder
{
public:
    Dijkstra() = default;

    std::vector<Position> findPath(
        const Environment& env,
        const Position&    start,
        const Position&    goal
    ) override;

    std::string   getName() const override;
    AlgorithmType getType() const override;
    int getNodesExplored() const override;

private:
    std::unordered_map<Position, double,   PositionHash> costFromStart_;
    std::unordered_map<Position, Position, PositionHash> arrivedFrom_;
    std::unordered_set<Position,           PositionHash> finalized_;

    int nodesExplored_ = 0;

    void   clearState();
    double costFromStartTo(const Position& p) const;   // costFromStart_[p] or +∞
};

#endif
