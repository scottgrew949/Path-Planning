// planning/IPathfinder.h
// Abstract interface for all pathfinding algorithms.
//
// Design rationale:
//   All three algorithms (A*, Dijkstra, BFS) share the same external contract:
//     given an Environment plus start/goal positions, return an ordered path.
//   The IPathfinder interface enforces this contract so that main.cpp can
//   iterate over a vector<unique_ptr<IPathfinder>> to benchmark all three
//   without algorithm-specific branching.
//
//   reconstructPath is shared logic — all algorithms trace arrivedFrom_
//   backwards from goal to start identically, so it lives here once.
//
// STL used here:
//   vector<Position>          – canonical path return type
//   vector<unique_ptr<T>>     – polymorphic algorithm store (see main.cpp)
#ifndef IPATHFINDER_H
#define IPATHFINDER_H

#include <vector>
#include <string>
#include <unordered_map>
#include <algorithm>
#include "../core/Position.h"
#include "../core/Types.h"
#include "../environment/Environment.h"

class IPathfinder
{
public:
    virtual ~IPathfinder() = default;

    // Core search routine.
    // Returns an ordered vector<Position> from start to goal (inclusive).
    // Returns an empty vector if no path exists.
    virtual std::vector<Position> findPath(
        const Environment& env,
        const Position&    start,
        const Position&    goal
    ) = 0;

    // Human-readable algorithm name for display and map keying.
    virtual std::string getName() const = 0;

    virtual int getNodesExplored() const = 0;

    // Algorithm category tag.
    virtual AlgorithmType getType() const = 0;

protected:
    // Shared path reconstruction — used by A*, Dijkstra, and BFS.
    // Walks arrivedFrom backwards from goal to start, then reverses.
    static std::vector<Position> reconstructPath(
                        const Position& goal,
                        const Position& start,
                        const std::unordered_map<Position, Position, PositionHash>& arrivedFrom)
    {
        std::vector<Position> path;
        Position current = goal;
        while (current != start)
        {
            path.push_back(current);
            auto it = arrivedFrom.find(current);
            if (it == arrivedFrom.end()) return {};
            current = it->second;
        }
        path.push_back(start);
        std::reverse(path.begin(), path.end());
        return path;
    }
};

#endif  // IPATHFINDER_H
