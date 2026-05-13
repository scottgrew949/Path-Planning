// planning/algorithms/BFS.cpp
#include "BFS.h"

using namespace std;

string BFS::getName() const
{
    return "BFS";
}

AlgorithmType BFS::getType() const
{
    return AlgorithmType::BFS;
}

void BFS::clearState()
{
    arrivedFrom_.clear(); 
    visited_.clear();
}

int BFS::getNodesExplored() const
{
    return nodesExplored_;
}

vector<Position> BFS::findPath(const Environment& env,
                                const Position&    start,
                                const Position&    goal)
{
    clearState();

    queue<Position> searchQueue;
    searchQueue.push(start);
    visited_.insert(start);

    while (!searchQueue.empty())
    {
        Position current = searchQueue.front(); 
        searchQueue.pop();
        ++nodesExplored_;

        if (current == goal)
            return reconstructPath(goal, start, arrivedFrom_);

        for (Position reachableCell : env.getNeighbors(current))
        {
            if (!visited_.count(reachableCell))
            {
                visited_.insert(reachableCell);
                arrivedFrom_[reachableCell] = current;
                searchQueue.push(reachableCell);
            }
        }
    }
    
    return {};
}
