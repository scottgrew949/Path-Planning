// planning/algorithms/DStarLite.cpp
#include "DStarLite.h"
#include <cmath>
#include <algorithm>
#include <unordered_map>

using namespace std;

static bool keyEqual(const DStarKey& a, const DStarKey& b)
{
    return a.primary == b.primary && a.secondary == b.secondary;
}

bool DStarKey::operator>(const DStarKey& other) const
{
    return primary > other.primary || (primary == other.primary && secondary > other.secondary);
}

bool DStarComparator::operator()(const DStarNode& a, const DStarNode& b) const
{
    return a.key > b.key;
}

string DStarLite::getName() const
{
    return "D* Lite";
}

AlgorithmType DStarLite::getType() const
{
    return AlgorithmType::DSTAR_LITE;
}

int DStarLite::getNodesExplored() const
{
    return nodesExplored_;
}

void DStarLite::clearState()
{
    g_.clear();
    rhs_.clear();
    while (!openList_.empty()) openList_.pop();
    nodesExplored_ = 0;
}

double DStarLite::heuristicDistance(const Position& a, const Position& b) const
{
    return abs(a.x - b.x) + abs(a.y - b.y);
}

DStarKey DStarLite::calculateKey(const Position& s, const Position& start) const
{
    double gs   = g_.count(s)   ? g_.at(s)   : INFINITY;
    double rhs  = rhs_.count(s) ? rhs_.at(s) : INFINITY;
    double m    = min(gs, rhs);
    return { m + heuristicDistance(start, s), m };
}

// No-op stubs — logic lives inline in findPath via lambdas
void DStarLite::updateVertex(const Position&, const Environment&, const Position&) {}
void DStarLite::computeShortestPath(const Environment&, const Position&, const Position&) {}
void DStarLite::updateObstacle(const Environment&, const Position&, const Position&) {}

vector<Position> DStarLite::findPath(const Environment& env,
                                      const Position&    start,
                                      const Position&    goal)
{
    clearState();

    // Lazy-deletion tracker: maps pos -> key currently valid in openList_
    unordered_map<Position, DStarKey, PositionHash> inOpen;

    auto gVal = [&](const Position& s) -> double {
        return g_.count(s) ? g_.at(s) : INFINITY;
    };
    auto rhsVal = [&](const Position& s) -> double {
        return rhs_.count(s) ? rhs_.at(s) : INFINITY;
    };

    auto calcKey = [&](const Position& s) -> DStarKey {
        double m = min(gVal(s), rhsVal(s));
        return { m + heuristicDistance(start, s), m };
    };

    auto insertOpen = [&](const Position& s) {
        DStarKey k = calcKey(s);
        inOpen[s] = k;
        openList_.push(DStarNode{s, k});
    };

    auto removeOpen = [&](const Position& s) {
        inOpen.erase(s);
        // stale entry left in openList_; lazy deletion handles it on pop
    };

    auto updateVtx = [&](const Position& s) {
        if (s != goal) {
            double best = INFINITY;
            for (const Position& n : env.getNeighbors(s))
                best = min(best, 1.0 + gVal(n));
            rhs_[s] = best;
        }
        removeOpen(s);
        if (gVal(s) != rhsVal(s))
            insertOpen(s);
    };

    // Initialization: rhs[goal] = 0, everything else stays infinity
    rhs_[goal] = 0.0;
    insertOpen(goal);

    // computeShortestPath loop
    while (!openList_.empty())
    {
        // Termination: start is locally consistent AND top key >= key(start)
        if (gVal(start) == rhsVal(start) && inOpen.count(start) == 0) {
            if (openList_.empty()) break;
            DStarKey topKey = openList_.top().key;
            DStarKey startKey = calcKey(start);
            if (!(topKey > startKey)) {
                // top key <= start key: must keep processing
            } else {
                break;
            }
        }

        if (openList_.empty()) break;

        DStarNode u = openList_.top();
        openList_.pop();

        // Lazy deletion: skip stale entries
        if (!inOpen.count(u.pos) || !keyEqual(inOpen.at(u.pos), u.key))
            continue;

        ++nodesExplored_;
        inOpen.erase(u.pos);

        if (gVal(u.pos) > rhsVal(u.pos)) {
            g_[u.pos] = rhs_[u.pos];
        } else {
            g_[u.pos] = INFINITY;
            updateVtx(u.pos);
        }

        for (const Position& s : env.getNeighbors(u.pos))
            updateVtx(s);
    }

    // No path exists
    if (gVal(start) == INFINITY) return {};

    // Reconstruct path by greedily following minimum g values from start to goal
    vector<Position> path;
    Position cur = start;
    path.push_back(cur);

    size_t limit = static_cast<size_t>(env.getWidth() * env.getHeight()) + 1;
    while (cur != goal)
    {
        Position best = cur;
        double   bestCost = INFINITY;
        for (const Position& n : env.getNeighbors(cur)) {
            double c = 1.0 + gVal(n);
            if (c < bestCost) { bestCost = c; best = n; }
        }
        if (best == cur) return {};   // stuck — no valid neighbor
        path.push_back(best);
        cur = best;
        if (path.size() > limit) return {};  // cycle guard
    }

    return path;
}
