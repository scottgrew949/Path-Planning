// planning/hybrid/NeuralAStar.cpp
#include "NeuralAStar.h"
#include <cmath>
#include <limits>

using namespace std;

// ---- NeuralNodeComparator ---------------------------------------------------

bool NeuralNodeComparator::operator()(
    const NeuralAStarNode& firstNode,
    const NeuralAStarNode& secondNode
) const
{
    // CONCEPT — Min-heap ordering:
    // priority_queue is a MAX-heap by default.
    // Returning (a > b) flips it to a MIN-heap — lowest f-cost node is popped first.
    // This is correct for A*: we always want to expand the most promising node.

    // TODO: return firstNode.totalEstimatedCost > secondNode.totalEstimatedCost
    return false;
}

// ---- NeuralAStar ------------------------------------------------------------

NeuralAStar::NeuralAStar(const std::string& weightsFilePath, double weightEpsilon)
    : network_(weightsFilePath), weightEpsilon_(weightEpsilon)
{
    // HeuristicNetwork constructor does the file loading.
    // Nothing else needed here — member initialiser list handles it.
}

string NeuralAStar::getName() const
{
    // TODO: return a name that includes ε, e.g. "Neural A* (ε=1.5)"
    // Use std::to_string(weightEpsilon_) for the value.
    return "Neural A*";
}

AlgorithmType NeuralAStar::getType() const
{
    return AlgorithmType::NEURAL_ASTAR;
}

int NeuralAStar::getNodesExplored() const
{
    return nodesExplored_;
}

double NeuralAStar::getEpsilon() const
{
    return weightEpsilon_;
}

void NeuralAStar::clearState()
{
    // TODO: call .clear() on costFromStart_, arrivedFrom_, finalized_
    // Same pattern as AStar::clearState()
}

double NeuralAStar::costFromStartTo(const Position& position) const
{
    // CONCEPT — Implicit infinity:
    // Undiscovered nodes have cost "infinity" conceptually.
    // We represent this by returning numeric_limits<double>::infinity()
    // when the position is not yet in the map.
    // This avoids storing +inf for every cell upfront (wastes memory).

    // TODO: look up position in costFromStart_
    // If found: return its value. If not: return numeric_limits<double>::infinity()
    return numeric_limits<double>::infinity();
}

double NeuralAStar::heuristicDistance(
    const Position& currentPosition,
    const Position& goalPosition,
    int             gridWidth,
    int             gridHeight
) const
{
    // CONCEPT — Heuristic dispatch with graceful degradation:
    // If the network loaded successfully, use the learned heuristic.
    // If not (file missing, corrupt, wrong format), fall back to Manhattan.
    // The ε multiplier is applied HERE in both branches — the caller sees
    // only "heuristic value" and never needs to know about ε directly.
    //
    // CONCEPT — Why multiply h by ε here and not in findPath()?
    // findPath() computes f = g + h_epsilon where h_epsilon = ε * h.
    // Encapsulating ε in heuristicDistance() means findPath() looks identical
    // to standard A* — only the heuristic call differs. Easier to read and verify.

    // TODO implement:
    // if (network_.isLoaded()):
    //     double rawPrediction = network_.predict(currentX, currentY, goalX, goalY, width, height)
    //     return weightEpsilon_ * rawPrediction
    // else:
    //     double manhattan = abs(dx) + abs(dy)
    //     return weightEpsilon_ * manhattan
    return 0.0;
}

vector<Position> NeuralAStar::findPath(
    const Environment& environment,
    const Position&    startPosition,
    const Position&    goalPosition
)
{
    // CONCEPT — This is IDENTICAL to AStar::findPath() except heuristicDistance()
    // is the only function call that differs. That's the point — the same proven
    // algorithm, a different information source for the heuristic.
    //
    // Reviewing this side-by-side with AStar.cpp is valuable: it makes the
    // "one hook" pattern concrete. Everything else is boilerplate.
    //
    // CONCEPT — Why does this still work if h_hat is inadmissible?
    // Weighted A* with ε still terminates correctly — it always finds A path.
    // The suboptimality guarantee: cost(found path) ≤ ε * cost(optimal path).
    // We verify this claim empirically in the benchmark.

    // TODO implement — identical structure to AStar::findPath():
    // 1. clearState()
    // 2. Initialize priority_queue<NeuralAStarNode, vector<...>, NeuralNodeComparator>
    // 3. costFromStart_[startPosition] = 0.0
    // 4. Push {startPosition, startPosition, 0.0, heuristicDistance(start, goal, w, h)}
    // 5. nodesExplored_ = 0
    // 6. While openSet not empty:
    //      NeuralAStarNode current = openSet.top(); openSet.pop()
    //      if finalized_.count(current.pos): continue
    //      finalized_.insert(current.pos)
    //      ++nodesExplored_
    //      if current.pos == goalPosition: return reconstructPath(goal, start, arrivedFrom_)
    //      for each neighbour in environment.getNeighbors(current.pos):
    //          double newCost = costFromStartTo(current.pos)
    //                         + environment.moveCost(current.previous, current.pos, neighbour)
    //          if newCost < costFromStartTo(neighbour):
    //              arrivedFrom_[neighbour]   = current.pos
    //              costFromStart_[neighbour] = newCost
    //              double h = heuristicDistance(neighbour, goalPosition,
    //                                           environment.getWidth(), environment.getHeight())
    //              openSet.push({neighbour, current.pos, newCost, newCost + h})
    // 7. Return {} (no path found)
    return {};
}
