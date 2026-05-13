// planning/algorithms/BFS.h
// Breadth-First Search — guarantees shortest path in hop count (unweighted graph).
//
// BFS expands nodes in FIFO order, so the first time the goal is reached it
// must have been reached via the fewest hops (not necessarily lowest cost).
// On a uniform-cost grid BFS and Dijkstra return identical paths, but BFS
// is faster because it avoids the heap overhead.
//
// STL highlights:
//   queue<Position>                              – FIFO expansion order
//   unordered_set<Position,   PositionHash>      – O(1) visited check
//   unordered_map<Position, Position, PositionHash> – parent tracking
//
// Algorithm sketch:
//   1. Enqueue start; mark start visited.
//   2. Loop until queue empty:
//        current = front(); pop()
//        if current == goal: return reconstructPath(goal, start)
//        for each neighbour n of current:
//          if not visited[n]:
//            mark n visited; parent[n] = current; enqueue n
//   3. Return {}.
#ifndef BFS_H
#define BFS_H

#include <vector>
#include <queue>
#include <unordered_map>
#include <unordered_set>
#include <string>
#include "../../core/Position.h"
#include "../../core/Types.h"
#include "../../environment/Environment.h"
#include "../IPathfinder.h"

class BFS : public IPathfinder
{
public:
    BFS() = default;

    std::vector<Position> findPath(
        const Environment& env,
        const Position&    start,
        const Position&    goal
    ) override;

    std::string   getName() const override;
    AlgorithmType getType() const override;
    int getNodesExplored() const override;

private:
    std::unordered_map<Position, Position, PositionHash> arrivedFrom_;
    std::unordered_set<Position, PositionHash>           visited_;

    int nodesExplored_;

    void clearState();
};

#endif  // BFS_
