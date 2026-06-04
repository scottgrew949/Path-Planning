// planning/algorithms/BFS.h
// Breadth-First Search — guarantees shortest path in hop count (unweighted graph).
// On a uniform-cost grid BFS and Dijkstra return identical paths, but BFS
// is faster because it avoids the heap overhead.
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

    int nodesExplored_ = 0;

    void clearState();
};

#endif 
