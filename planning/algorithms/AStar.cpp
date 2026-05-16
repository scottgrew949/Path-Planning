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

string AStar::getName() const
{
    return "A*";
}

int AStar::getNodesExplored() const
{
    return nodesExplored_;
}

AlgorithmType AStar::getType() const
{
    return AlgorithmType::ASTAR;
}

double AStar::heuristicDistance(const Position& a, const Position& b) const
{
    return abs(a.x - b.x) + abs(a.y - b.y);
}

double AStar::costFromStartTo(const Position& p) const
{
    unordered_map<Position, double, PositionHash>::const_iterator iterCostTable = costFromStart_.find(p);
    if (iterCostTable != costFromStart_ .end())
        return iterCostTable->second;
    else
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