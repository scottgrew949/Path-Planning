// planning/algorithms/RRT.cpp
#include "RRT.h"
#include <cmath>
#include <limits>
#include <algorithm>

using namespace std;

// ---- RRT --------------------------------------------------------------------

string RRT::getName() const
{
    return "RRT";
}

AlgorithmType RRT::getType() const
{
    return AlgorithmType::RRT;
}

int RRT::getNodesExplored() const
{
    return nodesExplored_;
}

void RRT::clearState()
{
    tree_.clear();
    nodesExplored_ = 0;
}

int RRT::nearestNodeIndex(const Position& target) const
{
    int    bestIdx  = 0;
    double bestDist = numeric_limits<double>::max();

    for (int i = 0; i < (int)tree_.size(); ++i)
    {
        double dx   = tree_[i].pos.x - target.x;
        double dy   = tree_[i].pos.y - target.y;
        double dist = dx * dx + dy * dy;
        if (dist < bestDist)
        {
            bestDist = dist;
            bestIdx  = i;
        }
    }
    return bestIdx;
}

Position RRT::steer(const Position& nearest, const Position& target) const
{
    double dx   = target.x - nearest.x;
    double dy   = target.y - nearest.y;
    double dist = sqrt(dx * dx + dy * dy);

    if (dist <= stepSize_)
        return target;

    double scale = stepSize_ / dist;
    return Position(
        (int)round(nearest.x + scale * dx),
        (int)round(nearest.y + scale * dy)
    );
}

bool RRT::isCollisionFree(const Environment& env,
                           const Position&    a,
                           const Position&    b) const
{
    int stepX = (a.x < b.x) ? 1 : -1,
        stepY = (a.y < b.y) ? 1 : -1,
        dy    = abs(a.y - b.y),
        dx    = abs(a.x - b.x),
        error = dx - dy,
        x     = a.x,
        y     = a.y,
        e2    = 0;

    while (x != b.x || y != b.y)
    {
        Position cur(x, y);
        if (!env.inBounds(cur) || env.isObstacle(cur)) return false;
        e2 = 2 * error;
        if (e2 > -dy) { error -= dy; x += stepX; }
        if (e2 <  dx) { error += dx; y += stepY; }
    }

    return env.inBounds(b) && !env.isObstacle(b);
}

vector<Position> RRT::extractPath(int goalNodeIndex) const
{
    vector<Position> path;
    int idx = goalNodeIndex;

    while (idx != -1)
    {
        path.push_back(tree_[idx].pos);
        idx = tree_[idx].parentIndex;
    }

    reverse(path.begin(), path.end());
    return path;
}

vector<Position> RRT::findPath(const Environment& env,
                                const Position&    start,
                                const Position&    goal)
{
    clearState();
    tree_.push_back({start, -1});

    random_device                      rd;
    mt19937                            rng(rd());
    uniform_int_distribution<int>      distX(0, env.getWidth()  - 1);
    uniform_int_distribution<int>      distY(0, env.getHeight() - 1);
    uniform_real_distribution<double>  distBias(0.0, 1.0);

    for (int iter = 0; iter < maxIterations_; ++iter)
    {
        Position sample = (distBias(rng) < 0.10)
                          ? goal
                          : Position(distX(rng), distY(rng));

        int      nearIdx = nearestNodeIndex(sample);
        Position nearest = tree_[nearIdx].pos;
        Position newPos  = steer(nearest, sample);

        if (!env.isValid(newPos))          continue;
        if (!isCollisionFree(env, nearest, newPos)) continue;

        int newIdx = (int)tree_.size();
        tree_.push_back({newPos, nearIdx});
        ++nodesExplored_;

        double dx   = newPos.x - goal.x;
        double dy   = newPos.y - goal.y;
        double dist = sqrt(dx * dx + dy * dy);

        if (dist <= stepSize_ && isCollisionFree(env, newPos, goal))
        {
            tree_.push_back({goal, newIdx});
            return extractPath((int)tree_.size() - 1);
        }
    }

    return {};
}
