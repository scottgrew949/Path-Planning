// planning/algorithms/MCTS.h
//
// PURPOSE: Monte Carlo Tree Search for grid pathfinding via UCT
//          (Upper Confidence Trees) with a transposition table.
//
// CONCEPT — Why MCTS / UCT?
//   A* requires an admissible heuristic and full graph knowledge up front.
//   UCT builds its own value estimates from simulation — no heuristic needed.
//   Each position accumulates statistics across thousands of simulated walks:
//   how often does exploring from here lead to the goal?
//   This makes UCT a planning algorithm that works from experience alone,
//   connecting classical search to the Monte Carlo methods used in AlphaGo.
//
// CONCEPT — Why transposition table, not a tree?
//   A naïve tree creates a new node for every (path, position) pair.
//   From (0,0) with 4 moves and a 28-step path: 4^28 possible nodes.
//   A transposition table stores statistics PER GRID POSITION instead:
//     Q[pos][action] — total accumulated value for (position, action)
//     N[pos][action] — visit count for (position, action)
//     N[pos]         — total visits to position (UCB denominator)
//   State space caps at width × height regardless of path length.
//
// CONCEPT — UCB1 action selection
//   At each position, pick the action maximising:
//     UCB(pos, a) = Q(pos,a)/N(pos,a) + C * sqrt(ln(N(pos)) / N(pos,a))
//   Exploitation term Q/N: prefer actions that historically led to goal.
//   Exploration term C*sqrt(...): prefer rarely-tried actions.
//   C = sqrt(2) is the theoretically optimal constant.
//
// CONCEPT — AlphaGo connection
//   AlphaGo replaces the random rollout with a neural network value estimate.
//   The same UCT loop applies; only the value signal changes.
//   This project's NeuralAStar learned heuristic could substitute here,
//   making this an AlphaZero-style planner for the grid.

#ifndef MCTS_H
#define MCTS_H

#include <vector>
#include <string>
#include <random>
#include "../../core/Position.h"
#include "../../core/Types.h"
#include "../../environment/Environment.h"
#include "../IPathfinder.h"


class MCTS : public IPathfinder
{
public:
    // numSimulations:      simulated walks per findPath call
    // explorationConstant: C in UCB1 — sqrt(2) is the canonical default
    // maxRolloutDepth:     steps per simulation and path extraction cap
    explicit MCTS(int    numSimulations      = 2000,
                  double explorationConstant  = 1.414,
                  int    maxRolloutDepth      = 300);

    std::vector<Position> findPath(const Environment& env,
                                   const Position&    start,
                                   const Position&    goal) override;

    std::string   getName()          const override;
    AlgorithmType getType()          const override;
    int           getNodesExplored() const override;

private:
    int    numSimulations_;
    double explorationConstant_;
    int    maxRolloutDepth_;
    mutable int       nodesExplored_;
    mutable std::mt19937 randomEngine_;
};

#endif // MCTS_H
