// planning/hybrid/NeuralAStar.h
//
// PURPOSE: A* search with a learned heuristic (HeuristicNetwork) instead of
//          Manhattan distance, using the ε-weighted A* variant.
//
// CORE CONCEPT — Template Method pattern
//   Classic A* and Neural A* share the EXACT SAME algorithm skeleton:
//     open set → expand → update costs → push neighbours → repeat.
//   The ONLY difference is how heuristicDistance() is computed.
//   This is the Template Method design pattern: a base algorithm with one
//   "hook" (the heuristic) that subclasses can vary.
//   Here we do NOT use C++ inheritance for this — NeuralAStar is standalone
//   and duplicates the minimal A* loop, with heuristic call swapped out.
//   In a larger system you would extract a templated A* base class.
//
// CORE CONCEPT — Heuristic quality spectrum
//   There are three regimes for a heuristic h(n):
//
//   1. h(n) = 0 (trivial):    A* degrades to Dijkstra. Explores the whole graph.
//                              Optimal, but slow — no guidance toward the goal.
//
//   2. h(n) = Manhattan:      Admissible (never overestimates on cardinal grids).
//                              Explores far fewer nodes than Dijkstra.
//                              But loose — ignores maze topology.
//
//   3. h(n) = h*(n) (perfect):Expands ONLY the nodes on the optimal path.
//                              Zero wasted expansions. Theoretical ideal.
//
//   Our learned h_hat sits between regimes 2 and 3 — tighter than Manhattan
//   (because it learned maze structure), imperfect (it's an approximation).
//   The empirical question: HOW MUCH closer to h* are we? The benchmark tells us.
//
// CORE CONCEPT — Weighted A* and the ε-suboptimality bound
//   Standard A*:  f(n) = g(n) + h(n)
//   Weighted A*:  f(n) = g(n) + ε * h(n)    where ε >= 1
//
//   Increasing ε makes the heuristic more aggressive — the search is MORE
//   biased toward the goal, expanding fewer nodes, but possibly finding
//   a suboptimal path.
//
//   The key theorem: if h(n) ≤ h*(n) (admissible), then weighted A* with ε
//   finds a path of cost ≤ ε * optimal_cost.
//
//   Our network may overestimate h* occasionally (inadmissible).
//   By choosing ε = 1.5, we claim: path cost ≤ 1.5 * optimal.
//   We VERIFY this empirically by measuring actual path costs vs A* on test mazes.
//
//   Real-world analog: ARA* (Anytime Repairing A*) starts with large ε for
//   speed, then tightens ε iteratively when time allows. Same idea.
//
// CONCEPT — Why getType() matters
//   main.cpp uses PathResult.algorithm (AlgorithmType enum) to key result maps.
//   NEURAL_ASTAR must be added to AlgorithmType in core/Types.h before this compiles.
//   Enums are the right tool here: compile-time exhaustiveness checking —
//   a switch() on AlgorithmType will warn if NEURAL_ASTAR is unhandled.

#ifndef NEURAL_ASTAR_H
#define NEURAL_ASTAR_H

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
#include "HeuristicNetwork.h"

// Reuse AStarNode and NodeComparator — identical to AStar.h.
// In a production codebase these would live in a shared header.
// Here we re-declare to keep hybrid/ self-contained.
struct NeuralAStarNode
{
    Position pos;
    Position previous;
    double   currentCost;          // g(n): true cost from start
    double   totalEstimatedCost;   // g(n) + ε * h_hat(n): drives heap order
};

struct NeuralNodeComparator
{
    bool operator()(const NeuralAStarNode& firstNode, const NeuralAStarNode& secondNode) const;
};

// ---- NeuralAStar ------------------------------------------------------------
class NeuralAStar : public IPathfinder
{
public:
    // CONCEPT — Dependency injection for the heuristic:
    // We pass the weights file path at construction time.
    // HeuristicNetwork is built once and reused across all findPath() calls.
    // This is the Dependency Injection principle: callers supply the heuristic
    // source, NeuralAStar does not hardcode where weights come from.
    // If weightsFilePath is empty or file missing, network_.isLoaded() = false
    // and we fall back to Manhattan distance (graceful degradation).
    explicit NeuralAStar(
        const std::string& weightsFilePath,
        double             weightEpsilon = 1.5
    );

    std::vector<Position> findPath(
        const Environment& environment,
        const Position&    startPosition,
        const Position&    goalPosition
    ) override;

    std::string   getName()         const override;
    AlgorithmType getType()         const override;
    int           getNodesExplored() const override;

    // Expose ε so main.cpp can print it in the benchmark table.
    double getEpsilon() const;

private:
    HeuristicNetwork network_;
    double           weightEpsilon_;
    int              nodesExplored_ = 0;

    std::unordered_map<Position, double,   PositionHash> costFromStart_;
    std::unordered_map<Position, Position, PositionHash> arrivedFrom_;
    std::unordered_set<Position,           PositionHash> finalized_;

    // CONCEPT — Heuristic dispatch:
    // If network is loaded: return ε * network_.predict(...)
    // Otherwise:            return ε * Manhattan distance (admissible fallback)
    // The ε multiplier goes HERE, not in findPath(), so the caller cannot forget it.
    double heuristicDistance(
        const Position& currentPosition,
        const Position& goalPosition,
        int             gridWidth,
        int             gridHeight
    ) const;

    double costFromStartTo(const Position& position) const;
    void   clearState();
};

#endif  // NEURAL_ASTAR_H
