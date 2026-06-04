// planning/algorithms/ThetaStar.h
// Theta* — any-angle path planning. Extension of A* that allows paths to
//      travel in any direction rather than being constrained to grid edges.
//
// Key difference from A*: when relaxing a neighbour, checks if there is a
// direct line of sight from the neighbour's grandparent. If so, shortcuts
// the path by connecting directly, producing smoother, shorter paths.
#ifndef THETA_STAR_H
#define THETA_STAR_H

#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <string>
#include "../../core/Position.h"
#include "../../core/Types.h"
#include "../../environment/Environment.h"
#include "../IPathfinder.h"

struct ThetaStarNode
{
    Position pos;
    Position previous;
    double   costFromStart;
    double   totalEstimatedCost;
};

struct ThetaStarComparator
{
    bool operator()(const ThetaStarNode& nodeA, const ThetaStarNode& nodeB) const;
};

class ThetaStar : public IPathfinder
{
public:
    ThetaStar() = default;

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
    double heuristicDistance(const Position& from, const Position& to)      const;
    double euclideanDistance(const Position& from, const Position& to)      const;
    bool   hasLineOfSight(const Environment& env,
                          const Position&    from,
                          const Position&    to)                            const;
    double costFromStartTo(const Position& position)                        const;
};

#endif
