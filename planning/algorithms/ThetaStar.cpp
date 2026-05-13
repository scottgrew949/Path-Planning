// planning/algorithms/ThetaStar.cpp
#include "ThetaStar.h"
#include <cmath>
#include <limits>

using namespace std;

// ---- ThetaStarComparator ----------------------------------------------------

bool ThetaStarComparator::operator()(const ThetaStarNode& a,
                                      const ThetaStarNode& b) const
{
    return a.totalEstimatedCost > b.totalEstimatedCost;
}

// ---- ThetaStar --------------------------------------------------------------

string ThetaStar::getName() const
{
    return "Theta *";
}

AlgorithmType ThetaStar::getType() const
{
    return AlgorithmType::THETA_STAR;
}

int ThetaStar::getNodesExplored() const
{
    return nodesExplored_;
}

void ThetaStar::clearState()
{
    costFromStart_.clear();
    arrivedFrom_.clear();
    finalized_.clear();
    nodesExplored_ = 0;
}

double ThetaStar::heuristicDistance(const Position& current, const Position& goal) const
{
    return euclideanDistance(current, goal);
}

double ThetaStar::euclideanDistance(const Position& a, const Position& b) const
{
    return sqrt(pow(a.x - b.x, 2) + pow(a.y - b.y, 2));
}

bool ThetaStar::hasLineOfSight(const Environment& env,
                                const Position&    from,
                                const Position&    to) const
{
    // TODO: Bresenham's line algorithm — step from a to b checking each cell
    //       return false if any cell along the line is an obstacle
    //       return true if clear
    // DONE
    int stepX = (from.x < to.x) ? 1 : -1,
        stepY = (from.y < to.y) ? 1 : -1,
        dy = abs(from.y - to.y),
        dx = abs(from.x - to.x),
        error = dx - dy,
        x = from.x,
        y = from.y,
        e2 = 0;

    while(true)
    {
        if (env.isObstacle(Position(x, y))) return false;
        if (Position(x, y) == to) return true;
    
        e2 = 2 * error;
        if (e2 > -dy)
        {
            error -= dy;
            x += stepX;
        }
        if (e2 < dx)
        {
            error += dx;
            y += stepY;
        }
    }
    return false;
}

double ThetaStar::costFromStartTo(const Position& p) const
{
    unordered_map<Position, double, PositionHash>::const_iterator it = costFromStart_.find(p);
    return (it != costFromStart_.end()) ? it->second : INFINITY;
}

vector<Position> ThetaStar::findPath(const Environment& env,
                                      const Position&    start,
                                      const Position&    goal)
{
    clearState();
    priority_queue<ThetaStarNode, vector<ThetaStarNode>, ThetaStarComparator> openSet;

    costFromStart_[start] = 0.0;
    openSet.push({start, start, 0.0, heuristicDistance(start, goal)});

    nodesExplored_ = 0;
    while (!openSet.empty())
    {
        ThetaStarNode current = openSet.top();
        openSet.pop();
        ++nodesExplored_;

        if (current.pos == goal)
            return reconstructPath(goal, start, arrivedFrom_);

        for (Position neighbour : env.getNeighbors(current.pos))
        {
            if (hasLineOfSight(env, current.previous, neighbour))
            {
                double shortcutCost = costFromStartTo(current.previous) + euclideanDistance(current.previous, neighbour);
                // current.previous is the same as arrivedFrom_[current.pos]. 
                // both return the parent, or previous node, of the current node.
                if (shortcutCost < costFromStartTo(neighbour))
                {
                    arrivedFrom_[neighbour]  = current.previous;
                    costFromStart_[neighbour] = shortcutCost;
                    openSet.push({neighbour, current.previous, shortcutCost, shortcutCost + heuristicDistance(neighbour, goal)});
                }
            }
            else
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

    }
}
