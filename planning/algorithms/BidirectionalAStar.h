// planning/algorithms/BidirectionalAStar.h
// Bidirectional A* — runs two simultaneous A* searches, one forward from start,
// one backward from goal. Terminates when the two frontiers meet.
//
// Advantage over standard A*: explores roughly half the nodes by attacking
// the problem from both ends simultaneously.
//
// STL highlights:
//   Two priority_queue<BidirectionalNode, vector<BidirectionalNode>, BidirectionalComparator>
//       — forward and backward open sets
//   Two unordered_map<Position, double,   PositionHash>  — cost tables
//   Two unordered_map<Position, Position, PositionHash>  — predecessor tables
//   Two unordered_set<Position,           PositionHash>  — finalized sets
#ifndef BIDIRECTIONAL_ASTAR_H
#define BIDIRECTIONAL_ASTAR_H

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

// ---- BidirectionalNode ------------------------------------------------------
struct BidirectionalNode
{
    Position pos;
    Position previous;
    double   costFromStart;
    double   totalEstimatedCost;
};

// ---- BidirectionalComparator ------------------------------------------------
struct BidirectionalComparator
{
    bool operator()(const BidirectionalNode& a, const BidirectionalNode& b) const;
};

// ---- BidirectionalAStar -----------------------------------------------------
class BidirectionalAStar : public IPathfinder
{
public:
    BidirectionalAStar() = default;

    std::vector<Position> findPath(
        const Environment& env,
        const Position&    start,
        const Position&    goal
    ) override;

    std::string   getName()          const override;
    AlgorithmType getType()          const override;
    int           getNodesExplored() const override;

private:
    // Forward search state
    std::unordered_map<Position, double,   PositionHash> forwardCost_;
    std::unordered_map<Position, Position, PositionHash> forwardArrivedFrom_;
    std::unordered_set<Position,           PositionHash> forwardFinalized_;

    // Backward search state
    std::unordered_map<Position, double,   PositionHash> backwardCost_;
    std::unordered_map<Position, Position, PositionHash> backwardArrivedFrom_;
    std::unordered_set<Position,           PositionHash> backwardFinalized_;

    int nodesExplored_ = 0;

    void   clearState();
    double heuristicDistance(const Position& a, const Position& b) const;

    // Merge forward and backward paths at meeting point
    std::vector<Position> mergePaths(const Position& meetingPoint,
                                     const Position& start,
                                     const Position& goal) const;
};

#endif  // BIDIRECTIONAL_ASTAR_H
