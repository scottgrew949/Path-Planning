// planning/algorithms/AStar.cpp
#include "AStar.h"
#include <cmath>
#include <limits>

using namespace std;

// ---- NodeComparator ---------------------------------------------------------

bool NodeComparator::operator()(const AStarNode& a, const AStarNode& b) const
{
    return a.totalEstimatedCost > b.totalEstimatedCost;
}

// ---- AStar ------------------------------------------------------------------

AStar::AStar(double epsilon)
    : epsilon_(epsilon)
{
    if (epsilon < 1.0)
        throw std::invalid_argument("AStar: epsilon must be >= 1.0 (values below 1 break admissibility)");
}

string AStar::getName() const
{
    if (epsilon_ == 1.0) return "A*";
    return "A* (e=" + std::to_string(epsilon_).substr(0, 4) + ")";
}

int AStar::getNodesExplored() const
{
    return nodesExplored_;
}

double AStar::getEpsilon() const
{
    return epsilon_;
}

AlgorithmType AStar::getType() const
{
    return AlgorithmType::ASTAR;
}

double AStar::heuristicDistance(const Position& a, const Position& b) const
{
    // epsilon_ = 1.0 → standard admissible A*.
    // epsilon_ > 1.0 → weighted A*: heuristic is inflated, search is greedier.
    // Path quality guarantee: found_cost ≤ epsilon_ × optimal_cost.
    return epsilon_ * (abs(a.x - b.x) + abs(a.y - b.y));
}

double AStar::costFromStartTo(const Position& p) const
{
    if (auto it = costFromStart_.find(p); it != costFromStart_.end())
        return it->second;
    return numeric_limits<double>::infinity();
}

void AStar::clearState()
{
    costFromStart_.clear();
    arrivedFrom_.clear();
    finalized_.clear();
}

vector<Position> AStar::findPath(const Environment& env,
                                  const Position&    start,
                                  const Position&    goal)
{
    clearState();
    priority_queue<AStarNode, vector<AStarNode>, NodeComparator> openSet;

    costFromStart_[start] = 0.0;
    openSet.push({start, start, 0.0, heuristicDistance(start, goal)});

    nodesExplored_ = 0;
    while (!openSet.empty())
    {
        AStarNode current = openSet.top();
        openSet.pop();

        if (finalized_.count(current.pos)) continue;
        finalized_.insert(current.pos);
        ++nodesExplored_;

        if (current.pos == goal)
            return reconstructPath(goal, start, arrivedFrom_);

        for (const Position& neighbour : env.getNeighbors(current.pos))
        {
            double newCost = costFromStartTo(current.pos) + env.moveCost(current.previous, current.pos, neighbour);

            if (newCost < costFromStartTo(neighbour))
            {
                arrivedFrom_[neighbour]   = current.pos;
                costFromStart_[neighbour] = newCost;
                openSet.push({neighbour, current.pos, newCost, newCost + heuristicDistance(neighbour, goal)});
            }
        }
    }

    return {};
}