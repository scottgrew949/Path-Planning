// planning/algorithms/BidirectionalAStar.cpp
#include "BidirectionalAStar.h"
#include <cmath>
#include <limits>
#include <algorithm>

using namespace std;

// ---- BidirectionalComparator ------------------------------------------------

bool BidirectionalComparator::operator()(const BidirectionalNode& a,
                                         const BidirectionalNode& b) const
{
    return a.totalEstimatedCost > b.totalEstimatedCost;
}

// ---- BidirectionalAStar -----------------------------------------------------

string BidirectionalAStar::getName() const
{
    return "Bidirectional A*";
}

AlgorithmType BidirectionalAStar::getType() const
{
    return AlgorithmType::BIDIRECTIONAL_ASTAR;
}

int BidirectionalAStar::getNodesExplored() const
{
    return nodesExplored_;
}

double BidirectionalAStar::heuristicDistance(const Position& a, const Position& b) const
{
    return abs(a.x - b.x) + abs(a.y - b.y);
}

void BidirectionalAStar::clearState()
{
    forwardCost_.clear();
    forwardArrivedFrom_.clear();
    forwardFinalized_.clear();
    backwardCost_.clear();
    backwardArrivedFrom_.clear();
    backwardFinalized_.clear();
    nodesExplored_ = 0;
}

vector<Position> BidirectionalAStar::mergePaths(const Position& meetingPoint,
                                                 const Position& start,
                                                 const Position& goal) const
{
    vector<Position> forwardPath;
    Position current = meetingPoint;
    while(current != start)
    {
        forwardPath.push_back(current);
        current = forwardArrivedFrom_.at(current);
    }
    forwardPath.push_back(start);
    reverse(forwardPath.begin(), forwardPath.end());    

    vector<Position> backwardPath;
    if (meetingPoint != goal)
    {
        current = backwardArrivedFrom_.at(meetingPoint);
        while (current != goal)
        {
            backwardPath.push_back(current);
            current = backwardArrivedFrom_.at(current);
        }
        backwardPath.push_back(goal);
    }

    forwardPath.insert(forwardPath.end(), backwardPath.begin(), backwardPath.end());
    return forwardPath;
}

vector<Position> BidirectionalAStar::findPath(const Environment& env,
                                               const Position&    start,
                                               const Position&    goal)
{
    clearState();
    
    double startHeuristic = heuristicDistance(start, goal);
    priority_queue<BidirectionalNode, vector<BidirectionalNode>, BidirectionalComparator> openSetForward;
    priority_queue<BidirectionalNode, vector<BidirectionalNode>, BidirectionalComparator> openSetBackward;
    openSetForward.push({start, start, 0.0, startHeuristic});
    openSetBackward.push({goal, goal, 0.0, startHeuristic});
    forwardCost_[start] = 0.0;
    backwardCost_[goal] = 0.0;


    while(!openSetForward.empty() && !openSetBackward.empty())
    {
        //Forward Movement
        BidirectionalNode currentForward = openSetForward.top();
        openSetForward.pop();
        ++nodesExplored_;
        forwardFinalized_.insert(currentForward.pos);
        if (backwardFinalized_.count(currentForward.pos))         // Meeting point check
            return mergePaths(currentForward.pos, start, goal);

        //Backward Movement
        BidirectionalNode currentBackward = openSetBackward.top();
        openSetBackward.pop();
        ++nodesExplored_;
        backwardFinalized_.insert(currentBackward.pos);
        if (forwardFinalized_.count(currentBackward.pos))         // Meeting point check
            return mergePaths(currentBackward.pos, start, goal);

        // Neighbor Checking forward
        for (const Position& neighbor : env.getNeighbors(currentForward.pos))
        {
            if (forwardFinalized_.count(neighbor))
                continue;  
            
             double newCost = forwardCost_[currentForward.pos] + env.moveCost(currentForward.previous, currentForward.pos, neighbor);
            

            if (forwardCost_.find(neighbor) == forwardCost_.end() || newCost < forwardCost_[neighbor])
            {
                forwardCost_[neighbor] = newCost;
                forwardArrivedFrom_[neighbor] = currentForward.pos;
                double estimated = newCost + heuristicDistance(neighbor, goal);
                openSetForward.push({neighbor, currentForward.pos, newCost, estimated});
            }
        }

        // Neighbor Checking backward
        for (const Position& neighbor : env.getNeighbors(currentBackward.pos))
        {
            if (backwardFinalized_.count(neighbor))
                continue;  
            
             double newCost = backwardCost_[currentBackward.pos] + env.moveCost(currentBackward.previous, currentBackward.pos, neighbor);
            

            if (backwardCost_.find(neighbor) == backwardCost_.end() || newCost < backwardCost_[neighbor])
            {
                backwardCost_[neighbor] = newCost;
                backwardArrivedFrom_[neighbor] = currentBackward.pos;
                double estimated = newCost + heuristicDistance(neighbor, start);
                openSetBackward.push({neighbor, currentBackward.pos, newCost, estimated});
            }
        }

    }

    return {};
}
