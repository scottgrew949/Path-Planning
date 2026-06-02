// planning/algorithms/RRT.h
// Rapidly-exploring Random Tree — probabilistic path planning.
// Default mode: RRT* (rewiring enabled), which is asymptotically optimal.
// Set useRewiring=false in constructor to get classic RRT behaviour.
//
// CONCEPT — RRT vs RRT*
//   RRT (Lavalle 1998): finds ANY collision-free path by randomly sampling
//   the space and growing a tree. Fast but not optimal — the found path
//   depends entirely on which random samples were drawn.
//
//   RRT* (Karaman & Frazzoli 2011): adds two steps after each new node:
//     1. Best-parent selection: among all nodes within radius r, pick the
//        parent that gives the new node the lowest total cost from start.
//     2. Rewiring: for each near node, check if routing THROUGH the new node
//        gives it a lower cost. If yes, redirect its parent pointer.
//   As the tree grows, path quality improves toward optimal.
//   Asymptotic optimality: as iterations → ∞, found path → shortest path.
//
// CONCEPT — Rewire radius
//   r = 3 * stepSize_ — a practical fixed radius for grid environments.
//   The theoretically optimal radius shrinks as log(n)/n, but the fixed
//   constant works well in practice and avoids expensive log computation.
//
// Self-driving car analog: motion planning for parking and lane changes —
// RRT finds a valid maneuver, RRT* finds the smoothest valid maneuver.
//
// STL highlights:
//   vector<RRTNode>  — flat tree, index = node ID
//   mt19937          — random position sampling
#ifndef RRT_H
#define RRT_H

#include <vector>
#include <string>
#include "../../core/Position.h"
#include "../../core/Types.h"
#include "../../environment/Environment.h"
#include "../IPathfinder.h"

// ---- RRTNode ----------------------------------------------------------------
struct RRTNode
{
    Position pos;
    int      parentIndex;    // index into tree_, -1 for root
    double   costFromStart;  // Euclidean distance along tree from root to this node
};

// ---- RRT --------------------------------------------------------------------
class RRT : public IPathfinder
{
public:
    // useRewiring = true  → RRT* (asymptotically optimal, default)
    // useRewiring = false → classic RRT (faster per iteration, not optimal)
    explicit RRT(bool useRewiring = true);

    std::vector<Position> findPath(
        const Environment& env,
        const Position&    start,
        const Position&    goal
    ) override;

    std::string   getName()          const override;
    AlgorithmType getType()          const override;
    int           getNodesExplored() const override;

private:
    bool                 useRewiring_;
    std::vector<RRTNode> tree_;
    int                  nodesExplored_ = 0;

    static constexpr int    stepSize_     = 2;
    static constexpr int    maxIterations_ = 10000;
    static constexpr double rewireRadius_ = 3.0 * stepSize_;

    void     clearState();
    int      nearestNodeIndex(const Position& target)                    const;
    Position steer(const Position& nearest, const Position& target)      const;
    bool     isCollisionFree(const Environment& env,
                              const Position&    a,
                              const Position&    b)                      const;
    std::vector<Position> extractPath(int goalNodeIndex)                 const;

    // Euclidean distance between two grid positions.
    double euclideanDistance(const Position& a, const Position& b)       const;

    // Return indices of all tree nodes within rewireRadius_ of center.
    std::vector<int> findNearNodes(const Position& center)               const;

    // For each near node, redirect its parent to newNodeIndex if routing through
    // the new node gives it a lower costFromStart. Updates costFromStart in place,
    // then propagates the cost delta down through all descendants of the rewired node.
    void rewire(int newNodeIndex, const std::vector<int>& nearNodeIndices,
                const Environment& env);

    // BFS from rewiredNodeIndex downward: update every descendant's costFromStart
    // by the same delta applied to the rewired node. O(n) worst case but necessary
    // for RRT* correctness — without propagation future best-parent decisions use
    // stale costs and asymptotic optimality degrades.
    void propagateSubtreeCosts(int rewiredNodeIndex, double costDelta);
};

#endif  // RRT_H
