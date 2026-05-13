// planning/algorithms/DStarLite.cpp
#include "DStarLite.h"
#include <cmath>
#include <limits>

using namespace std;

// ---- DStarKey ---------------------------------------------------------------

bool DStarKey::operator>(const DStarKey& other) const
{
    // TODO: return primary > other.primary, break ties with secondary
}

// ---- DStarComparator --------------------------------------------------------

bool DStarComparator::operator()(const DStarNode& a, const DStarNode& b) const
{
    // TODO: min-heap on key — lower key = higher priority
}

// ---- DStarLite --------------------------------------------------------------

string DStarLite::getName() const
{
    // TODO
}

AlgorithmType DStarLite::getType() const
{
    // TODO
}

int DStarLite::getNodesExplored() const
{
    // TODO
}

void DStarLite::clearState()
{
    // TODO: clear g_, rhs_, openList_, reset nodesExplored_
}

double DStarLite::heuristicDistance(const Position& a, const Position& b) const
{
    // TODO: Manhattan distance
}

DStarKey DStarLite::calculateKey(const Position& s, const Position& start) const
{
    // TODO: primary   = min(g[s], rhs[s]) + heuristic(start, s)
    //       secondary = min(g[s], rhs[s])
    //       return {primary, secondary}
}

void DStarLite::updateVertex(const Position& s, const Environment& env, const Position& start)
{
    // TODO: if s != goal: rhs[s] = min over neighbours n of (moveCost(s,n) + g[n])
    //       if s is in openList_: remove it
    //       if g[s] != rhs[s]: insert s with calculateKey(s)
}

void DStarLite::computeShortestPath(const Environment& env,
                                     const Position&    start,
                                     const Position&    goal)
{
    // TODO: while openList_ top key < calculateKey(start) OR rhs[start] != g[start]:
    //         pop u from openList_
    //         ++nodesExplored_
    //         if g[u] > rhs[u]: g[u] = rhs[u]
    //         else: g[u] = infinity; updateVertex(u)
    //         for each neighbour s of u: updateVertex(s)
}

void DStarLite::updateObstacle(const Environment& env,
                                const Position&    changedCell,
                                const Position&    start)
{
    // TODO: update edge costs around changedCell
    //       call updateVertex on affected neighbours
    //       call computeShortestPath to repair the plan
}

vector<Position> DStarLite::findPath(const Environment& env,
                                      const Position&    start,
                                      const Position&    goal)
{
    // TODO: clearState()
    // TODO: rhs[goal] = 0; g[s] = infinity for all s; insert goal into openList_
    // TODO: computeShortestPath(env, start, goal)
    // TODO: walk from start to goal following minimum g values to reconstruct path
    // TODO: return path, or {} if no path exists
}
