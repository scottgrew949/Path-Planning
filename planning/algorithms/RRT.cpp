// planning/algorithms/RRT.cpp
#include "RRT.h"
#include <cmath>
#include <random>
#include <algorithm>

using namespace std;

// ---- Constructor ------------------------------------------------------------

RRT::RRT(bool useRewiring)
    : useRewiring_(useRewiring)
{}

// ---- IPathfinder interface --------------------------------------------------

string RRT::getName() const
{
    return useRewiring_ ? "RRT*" : "RRT";
}

AlgorithmType RRT::getType() const
{
    return AlgorithmType::RRT;
}

int RRT::getNodesExplored() const
{
    return nodesExplored_;
}

// ---- Tree helpers -----------------------------------------------------------

void RRT::clearState()
{
    tree_.clear();
    nodesExplored_ = 0;
}

int RRT::nearestNodeIndex(const Position& target) const
{
    auto nearest = std::min_element(tree_.begin(), tree_.end(),
        [&target](const RRTNode& a, const RRTNode& b) {
            double dxa = a.pos.x - target.x,  dya = a.pos.y - target.y;
            double dxb = b.pos.x - target.x,  dyb = b.pos.y - target.y;
            return (dxa * dxa + dya * dya) < (dxb * dxb + dyb * dyb);
        });
    return static_cast<int>(nearest - tree_.begin());
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
    int index = goalNodeIndex;

    while (index != -1)
    {
        path.push_back(tree_[index].pos);
        index = tree_[index].parentIndex;
    }

    reverse(path.begin(), path.end());
    return path;
}

double RRT::euclideanDistance(const Position& a, const Position& b) const
{
    double dx = a.x - b.x;
    double dy = a.y - b.y;
    return sqrt(dx * dx + dy * dy);
}

vector<int> RRT::findNearNodes(const Position& center) const
{
    vector<int> nearNodeIndices;
    for (int index = 0; index < (int)tree_.size(); ++index)
    {
        if (euclideanDistance(tree_[index].pos, center) <= rewireRadius_)
            nearNodeIndices.push_back(index);
    }
    return nearNodeIndices;
}

void RRT::rewire(int newNodeIndex, const vector<int>& nearNodeIndices,
                  const Environment& env)
{
    // CONCEPT — Rewiring:
    //   For each near node, check if routing through the new node gives it a
    //   lower total cost from start. If yes and the direct edge is obstacle-free,
    //   redirect its parent pointer to the new node and update its cost.
    //   After updating the near node's cost, propagate the delta down through
    //   its entire subtree — without propagation, descendants retain stale costs
    //   and future best-parent selections use wrong numbers, breaking optimality.
    const RRTNode& newNode = tree_[newNodeIndex];

    for (int nearIndex : nearNodeIndices)
    {
        if (nearIndex == newNode.parentIndex) continue;

        double edgeCost       = euclideanDistance(newNode.pos, tree_[nearIndex].pos);
        double costThroughNew = newNode.costFromStart + edgeCost;

        if (costThroughNew < tree_[nearIndex].costFromStart
            && isCollisionFree(env, newNode.pos, tree_[nearIndex].pos))
        {
            double costDelta                = costThroughNew - tree_[nearIndex].costFromStart;
            tree_[nearIndex].parentIndex    = newNodeIndex;
            tree_[nearIndex].costFromStart  = costThroughNew;
            propagateSubtreeCosts(nearIndex, costDelta);
        }
    }
}

void RRT::propagateSubtreeCosts(int rewiredNodeIndex, double costDelta)
{
    // CONCEPT — O(n) subtree propagation via children map:
    //   Naive BFS scans all of tree_ for each queued node → O(n²).
    //   Building a children adjacency list first (one O(n) pass) lets BFS
    //   walk only actual children → O(n) total regardless of subtree size.
    vector<vector<int>> children(tree_.size());
    for (int index = 0; index < (int)tree_.size(); ++index)
    {
        int parentIndex = tree_[index].parentIndex;
        if (parentIndex >= 0 && parentIndex != index)
            children[parentIndex].push_back(index);
    }

    vector<int> workQueue = {rewiredNodeIndex};
    while (!workQueue.empty())
    {
        int currentIndex = workQueue.back();
        workQueue.pop_back();

        for (int childIndex : children[currentIndex])
        {
            tree_[childIndex].costFromStart += costDelta;
            workQueue.push_back(childIndex);
        }
    }
}

// ---- findPath ---------------------------------------------------------------

vector<Position> RRT::findPath(const Environment& env,
                                const Position&    start,
                                const Position&    goal)
{
    clearState();
    tree_.push_back({start, -1, 0.0});

    random_device                     rd;
    mt19937                           rng(rd());
    uniform_int_distribution<int>     distX(0, env.getWidth()  - 1);
    uniform_int_distribution<int>     distY(0, env.getHeight() - 1);
    uniform_real_distribution<double> distBias(0.0, 1.0);

    for (int iter = 0; iter < maxIterations_; ++iter)
    {
        Position sample = (distBias(rng) < 0.10)
                          ? goal
                          : Position(distX(rng), distY(rng));

        int      nearestIndex = nearestNodeIndex(sample);
        Position nearest      = tree_[nearestIndex].pos;
        Position newPos       = steer(nearest, sample);

        if (!env.isValid(newPos))                      continue;
        if (!isCollisionFree(env, nearest, newPos))    continue;

        int newNodeIndex = (int)tree_.size();

        if (useRewiring_)
        {
            // CONCEPT — Best-parent selection:
            //   Before adding the new node, check all near nodes to see if any
            //   give a lower cost-from-start than the nearest node.
            //   This alone (without rewiring) already improves path quality.
            vector<int> nearNodeIndices = findNearNodes(newPos);

            int    bestParentIndex = nearestIndex;
            double bestCost        = tree_[nearestIndex].costFromStart
                                     + euclideanDistance(nearest, newPos);

            for (int candidateIndex : nearNodeIndices)
            {
                double candidateCost = tree_[candidateIndex].costFromStart
                                       + euclideanDistance(tree_[candidateIndex].pos, newPos);
                if (candidateCost < bestCost
                    && isCollisionFree(env, tree_[candidateIndex].pos, newPos))
                {
                    bestCost        = candidateCost;
                    bestParentIndex = candidateIndex;
                }
            }

            tree_.push_back({newPos, bestParentIndex, bestCost});
            rewire(newNodeIndex, nearNodeIndices, env);
        }
        else
        {
            double costFromStart = tree_[nearestIndex].costFromStart
                                   + euclideanDistance(nearest, newPos);
            tree_.push_back({newPos, nearestIndex, costFromStart});
        }

        ++nodesExplored_;

        double distToGoal = euclideanDistance(newPos, goal);
        if (distToGoal <= stepSize_ && isCollisionFree(env, newPos, goal))
        {
            double goalCost = tree_[newNodeIndex].costFromStart + distToGoal;
            tree_.push_back({goal, newNodeIndex, goalCost});
            return extractPath((int)tree_.size() - 1);
        }
    }

    return {};
}
