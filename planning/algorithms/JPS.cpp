// planning/algorithms/JPS.cpp
#include "JPS.h"
#include <limits>
#include <queue>

using namespace std;

static int sign(int n) { return (n > 0) ? 1 : (n < 0) ? -1 : 0; }

bool JPSComparator::operator()(const JPSNode& a, const JPSNode& b) const
{
    return a.totalEstimatedCost > b.totalEstimatedCost;
}

string JPS::getName() const
{
    return "JPS";
}

AlgorithmType JPS::getType() const
{
    return AlgorithmType::JPS;
}

int JPS::getNodesExplored() const
{
    return nodesExplored_;
}

void JPS::clearState()
{
    costFromStart_.clear();
    arrivedFrom_.clear();
    finalized_.clear();
    nodesExplored_ = 0;
}

double JPS::heuristicDistance(const Position& a, const Position& b) const
{
    return abs(a.x - b.x) + abs(a.y - b.y);
}

double JPS::costFromStartTo(const Position& p) const
{
    unordered_map<Position, double, PositionHash>::const_iterator it = costFromStart_.find(p);
    if (it != costFromStart_.end())
        return it->second;
    return numeric_limits<double>::infinity();
}

Position JPS::jump(const Environment& env,
                   const Position&    current,
                   const Position&    direction,
                   const Position&    goal) const
{
    Position next(current.x + direction.x, current.y + direction.y);

    if (!env.isValid(next))
        return Position(-1, -1);

    if (next == goal)
        return next;

    // Forced neighbor: perpendicular cell at current is blocked AND at next is open.
    // Without both conditions, nearly every open cell becomes a jump point.
    if (direction.x != 0)
    {
        bool forcedAbove = !env.isValid(Position(current.x, next.y + 1)) && env.isValid(Position(next.x, next.y + 1));
        bool forcedBelow = !env.isValid(Position(current.x, next.y - 1)) && env.isValid(Position(next.x, next.y - 1));
        if (forcedAbove || forcedBelow)
            return next;
    }
    else
    {
        bool forcedRight = !env.isValid(Position(next.x + 1, current.y)) && env.isValid(Position(next.x + 1, next.y));
        bool forcedLeft  = !env.isValid(Position(next.x - 1, current.y)) && env.isValid(Position(next.x - 1, next.y));
        if (forcedRight || forcedLeft)
            return next;
    }

    return jump(env, next, direction, goal);
}

vector<Position> JPS::identifySuccessors(const Environment& env,
                                          const Position&    current,
                                          const Position&    goal) const
{
    vector<Position> successors;

    // Directions to scan from current node (stored as Position for jump() signature)
    vector<Position> dirs;

    auto it = arrivedFrom_.find(current);
    if (it == arrivedFrom_.end())
    {
        // Start node — try all 4 cardinal directions
        dirs = { Position(1,0), Position(-1,0), Position(0,1), Position(0,-1) };
    }
    else
    {
        const Position& parent = it->second;
        Position d( sign(current.x - parent.x), sign(current.y - parent.y) );

        dirs.push_back(d);

        if (d.x != 0)
        {
            dirs.push_back(Position(0,  1));
            dirs.push_back(Position(0, -1));
        }
        else
        {
            dirs.push_back(Position( 1, 0));
            dirs.push_back(Position(-1, 0));
        }
    }

    for (const Position& d : dirs)
    {
        Position jp = jump(env, current, d, goal);
        if (jp.x >= 0)
            successors.push_back(jp);
    }

    return successors;
}

vector<Position> JPS::findPath(const Environment& env,
                                const Position&    start,
                                const Position&    goal)
{
    clearState();

    priority_queue<JPSNode, vector<JPSNode>, JPSComparator> openSet;

    costFromStart_[start] = 0.0;
    openSet.push({start, start, 0.0, heuristicDistance(start, goal)});

    while (!openSet.empty())
    {
        JPSNode current = openSet.top();
        openSet.pop();

        ++nodesExplored_;

        if (finalized_.count(current.pos)) continue;
        finalized_.insert(current.pos);

        if (current.pos == goal)
            return reconstructPath(goal, start, arrivedFrom_);

        vector<Position> successors = identifySuccessors(env, current.pos, goal);

        for (const Position& s : successors)
        {
            if (finalized_.count(s)) continue;

            double newCost = costFromStart_[current.pos]
                           + abs(s.x - current.pos.x)
                           + abs(s.y - current.pos.y);

            if (newCost < costFromStartTo(s))
            {
                costFromStart_[s] = newCost;
                arrivedFrom_[s]   = current.pos;
                openSet.push({s, current.pos, newCost, newCost + heuristicDistance(s, goal)});
            }
        }
    }

    return {};
}
