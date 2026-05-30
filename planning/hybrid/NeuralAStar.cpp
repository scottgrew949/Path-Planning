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

    return firstNode.totalEstimatedCost > secondNode.totalEstimatedCost;
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
    return "Neural A* (e=" + std::to_string(weightEpsilon_) + ")";
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

int NeuralAStar::getHeuristicCallCount() const
{
    return heuristicCallCount_;
}

void NeuralAStar::clearState()
{
    costFromStart_.clear();
    arrivedFrom_.clear();
    finalized_.clear();
    nodesExplored_      = 0;
    heuristicCallCount_ = 0;
}

double NeuralAStar::costFromStartTo(const Position& position) const
{
    // CONCEPT — Implicit infinity:
    // Undiscovered nodes have cost "infinity" conceptually.
    // We represent this by returning numeric_limits<double>::infinity()
    // when the position is not yet in the map.
    // This avoids storing +inf for every cell upfront (wastes memory).

    auto iterator = costFromStart_.find(position);
    if (iterator != costFromStart_.end())
        return iterator->second;
    return numeric_limits<double>::infinity();
}

double NeuralAStar::heuristicDistance(
    const Position& currentPosition,
    const Position& goalPosition,
    int             gridWidth,
    int             gridHeight
) const
{
    // Mechanical counter — scaffolding, not algorithm logic.
    ++heuristicCallCount_;

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

    if (network_.isLoaded())
    {
        double rawPrediction = network_.predict(
            currentPosition.x, currentPosition.y,
            goalPosition.x,    goalPosition.y,
            gridWidth,         gridHeight
        );
        return weightEpsilon_ * rawPrediction;
    }

    double manhattan = abs(currentPosition.x - goalPosition.x)
                     + abs(currentPosition.y - goalPosition.y);
    return weightEpsilon_ * manhattan;
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

    clearState();

    priority_queue<NeuralAStarNode, vector<NeuralAStarNode>, NeuralNodeComparator> openSet;

    costFromStart_[startPosition] = 0.0;
    openSet.push({
        startPosition,
        startPosition,
        0.0,
        heuristicDistance(startPosition, goalPosition, environment.getWidth(), environment.getHeight())
    });

    while (!openSet.empty())
    {
        NeuralAStarNode current = openSet.top();
        openSet.pop();

        if (finalized_.count(current.pos)) continue;
        finalized_.insert(current.pos);
        ++nodesExplored_;

        if (current.pos == goalPosition)
            return reconstructPath(goalPosition, startPosition, arrivedFrom_);

        for (const Position& neighbour : environment.getNeighbors(current.pos))
        {
            double newCost = costFromStartTo(current.pos)
                           + environment.moveCost(current.previous, current.pos, neighbour);

            if (newCost < costFromStartTo(neighbour))
            {
                arrivedFrom_[neighbour]   = current.pos;
                costFromStart_[neighbour] = newCost;
                double heuristic = heuristicDistance(
                    neighbour, goalPosition,
                    environment.getWidth(), environment.getHeight()
                );
                openSet.push({neighbour, current.pos, newCost, newCost + heuristic});
            }
        }
    }

    return {};
}
