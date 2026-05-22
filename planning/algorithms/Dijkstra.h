// planning/algorithms/Dijkstra.h
// Dijkstra's algorithm — optimal, explores by cumulative path cost.
//
// Difference from A*: no heuristic. Every reachable cell is eventually
// settled in non-decreasing distance order, guaranteeing optimality but
// exploring more nodes than A* on grid problems.
//
// STL highlights:
//   priority_queue<DijkstraNode, vector<DijkstraNode>, DijkstraComparator>
//       – min-heap on accumulated distance
//   unordered_map<Position, double,   PositionHash>  – distance table
//   unordered_map<Position, Position, PositionHash>  – predecessor table
//   unordered_set<Position,           PositionHash>  – settled set (O(1) check)
//
// Algorithm sketch:
//   1. costFromStart[start] = 0; all others +∞.
//   2. Push {start, 0} onto heap.
//   3. Loop until heap empty:
//        {u, cost} = heap.top(); pop
//        if u already finalized: skip (stale entry)
//        mark u finalized
//        if u == goal: return reconstructPath(goal, start)
//        for each neighbour v of u:
//          alt = costFromStart[u] + moveCost(u, v)
//          if alt < costFromStart[v]:
//            costFromStart[v] = alt; arrivedFrom[v] = u
//            push {v, alt} onto heap
//   4. Return {}.
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

// ---- DijkstraNode -----------------------------------------------------------
struct DijkstraNode
{
    Position    pos;
    Position    previous;
    double      costFromStart;   // cumulative cost from start
};

// ---- DijkstraComparator -----------------------------------------------------
// Min-heap on costFromStart — Dijkstra always expands the closest unsettled node.
struct DijkstraComparator
{
    bool operator()(const DijkstraNode& a, const DijkstraNode& b) const;
};

// ---- Dijkstra ---------------------------------------------------------------
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

#endif  // DIJKSTRA_H
