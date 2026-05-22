// planning/algorithms/Dijkstra.cpp
#include "Dijkstra.h"
#include <limits>

using namespace std;

// ---- DijkstraComparator -----------------------------------------------------

bool DijkstraComparator::operator()(const DijkstraNode& a,
                                    const DijkstraNode& b) const
{
    return a.costFromStart > b.costFromStart;
}

// ---- Dijkstra ---------------------------------------------------------------

string Dijkstra::getName() const
{
    return "Dijkstra";
}

int Dijkstra::getNodesExplored() const
{
    return nodesExplored_;
}

AlgorithmType Dijkstra::getType() const
{
    return AlgorithmType::DIJKSTRA;
}

double Dijkstra::costFromStartTo(const Position& p) const
{
    auto it = costFromStart_.find(p);
    return (it != costFromStart_.end()) ? it->second : numeric_limits<double>::infinity();
}

void Dijkstra::clearState()
{
    costFromStart_.clear(); 
    arrivedFrom_.clear(); 
    finalized_.clear();
    nodesExplored_ = 0;
}

vector<Position> Dijkstra::findPath(const Environment& env,
                                     const Position&    start,
                                     const Position&    goal)
{
    // wipe data from any previous run
    clearState();

    // min-heap — always gives cheapest unfinalized node next
    priority_queue<DijkstraNode,
                   vector<DijkstraNode>,
                   DijkstraComparator> unexploredNodes;

    // cost to reach start from start is zero
    costFromStart_[start] = 0.0;

    // seed the heap with the start node
    unexploredNodes.push({start, start, 0.0});

    while (!unexploredNodes.empty())
    {
        // grab the cheapest node
        DijkstraNode current = unexploredNodes.top();
        unexploredNodes.pop();

        if (finalized_.count(current.pos))
            continue;

        finalized_.insert(current.pos);
        ++nodesExplored_;

        // reached the goal via cheapest path — reconstruct and return
        if (current.pos == goal)
            return reconstructPath(goal, start, arrivedFrom_);

        for (const Position& neighbour : env.getNeighbors(current.pos))
        {
            // already have optimal cost for this neighbour — skip
            if (finalized_.count(neighbour))
                continue;

            // cost to reach neighbour through current node
            double alt = costFromStartTo(current.pos) + env.moveCost(current.previous, current.pos, neighbour);

            // if cheaper than what we already know, update everything
            if (alt < costFromStartTo(neighbour))
            {
                costFromStart_[neighbour] = alt;        // record new cheaper cost
                arrivedFrom_[neighbour]   = current.pos; // record how we got here
                unexploredNodes.push({neighbour, current.pos, alt}); // push updated entry onto heap
            }
        }
    }

    // exhausted all nodes — no path exists
    return {};
}
