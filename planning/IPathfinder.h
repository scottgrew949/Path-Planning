// planning/IPathfinder.h
// Abstract interface for all pathfinding algorithms.
// All algorithms trace arrivedFrom_
//   backwards from goal to start identically, so it lives here once.
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

    // Returns an ordered vector<Position> from start to goal (inclusive).
    // Returns an empty vector if no path exists.
    virtual std::vector<Position> findPath(
        const Environment& env,
        const Position&    start,
        const Position&    goal
    ) = 0;

    virtual std::string getName() const = 0;

    virtual int getNodesExplored() const = 0;

    virtual AlgorithmType getType() const = 0;

protected:
    // Shared path reconstruction.
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

#endif 
