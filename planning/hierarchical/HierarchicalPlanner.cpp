// planning/hierarchical/HierarchicalPlanner.cpp
#include "HierarchicalPlanner.h"
#include <cmath>
#include <stdexcept>

using namespace std;

// ---- Internal node type for both A* phases ----------------------------------
// Kept file-local; not exposed in the header because only this translation unit
// needs it. Both phases use identical heap ordering so one struct suffices.
namespace
{
    struct SearchNode
    {
        Position pos;
        double   gCost;               // cost from start to this node
        double   totalEstimatedCost;  // gCost + Manhattan heuristic to goal
    };

    struct SearchNodeComparator
    {
        bool operator()(const SearchNode& nodeA, const SearchNode& nodeB) const
        {
            return nodeA.totalEstimatedCost > nodeB.totalEstimatedCost;
        }
    };

    int manhattanDistance(const Position& positionA, const Position& positionB)
    {
        return abs(positionA.x - positionB.x) + abs(positionA.y - positionB.y);
    }
}

// ---- HierarchicalPlanner ----------------------------------------------------

HierarchicalPlanner::HierarchicalPlanner(int tileSize)
    : tileSize_(tileSize)
{
    if (tileSize < 1)
        throw invalid_argument("HierarchicalPlanner: tileSize must be >= 1");
}

void HierarchicalPlanner::setKalmanTracker(const KalmanTracker* tracker,
                                           double predictionHorizon)
{
    kalmanTracker_     = tracker;
    predictionHorizon_ = predictionHorizon;
}

string HierarchicalPlanner::getName() const
{
    return "HierarchicalA*";
}

AlgorithmType HierarchicalPlanner::getType() const
{
    return AlgorithmType::HIERARCHICAL_ASTAR;
}

int HierarchicalPlanner::getNodesExplored() const
{
    return nodesExplored_;
}

vector<Position> HierarchicalPlanner::findPath(
    const Environment& env,
    const Position&    start,
    const Position&    goal)
{
    nodesExplored_ = 0;

    AbstractMap abstractMap(env, tileSize_);

    // If a Kalman tracker is attached, mark predicted obstacle tiles as blocked
    // before the abstract search runs — the route avoids them proactively.
    if (kalmanTracker_ != nullptr)
        abstractMap.updateWithPredictions(
            kalmanTracker_->getPredictedObstaclePositions(predictionHorizon_));

    Position startTile = abstractMap.tileOf(start);
    Position goalTile  = abstractMap.tileOf(goal);

    vector<Position> abstractPath = abstractSearch(abstractMap, startTile, goalTile);

    if (abstractPath.empty())
        return {};

    // Build corridor: abstract path tiles + their 4-cardinal neighbors (1-tile padding).
    // A labyrinth's passages wind through cells that may sit in tiles adjacent to the
    // abstract path. Without padding, the local search gets cut off at tile boundaries
    // and returns no path even though one exists.
    unordered_set<Position, PositionHash> corridorTiles;
    corridorTiles.reserve(abstractPath.size() * 5);

    vector<pair<int,int>> padOffsets = {{0,0},{1,0},{-1,0},{0,1},{0,-1}};
    for (const Position& tile : abstractPath)
    {
        for (const auto& [dx, dy] : padOffsets)
        {
            Position paddedTile(tile.x + dx, tile.y + dy);
            if (paddedTile.x >= 0 && paddedTile.x < abstractMap.getTileWidth() &&
                paddedTile.y >= 0 && paddedTile.y < abstractMap.getTileHeight())
                corridorTiles.insert(paddedTile);
        }
    }

    return localSearch(env, start, goal, corridorTiles, abstractMap);
}

vector<Position> HierarchicalPlanner::abstractSearch(
    const AbstractMap& abstractMap,
    const Position&    startTile,
    const Position&    goalTile)
{
    unordered_map<Position, double,   PositionHash> gScore;
    unordered_map<Position, Position, PositionHash> arrivedFrom;
    unordered_set<Position,           PositionHash> finalized;

    priority_queue<SearchNode, vector<SearchNode>, SearchNodeComparator> openSet;

    gScore[startTile] = 0.0;
    openSet.push({startTile, 0.0,
                  static_cast<double>(manhattanDistance(startTile, goalTile))});

    while (!openSet.empty())
    {
        SearchNode current = openSet.top();
        openSet.pop();

        if (finalized.count(current.pos))
            continue;
        finalized.insert(current.pos);
        ++nodesExplored_;

        if (current.pos == goalTile)
            return reconstructPath(goalTile, startTile, arrivedFrom);

        for (const Position& neighborTile : abstractMap.getAbstractNeighbors(current.pos))
        {
            // Each tile-to-tile step costs 1.0 in the abstract graph.
            double tentativeGCost = gScore[current.pos] + 1.0;

            if (gScore.find(neighborTile) == gScore.end()
             || tentativeGCost < gScore[neighborTile])
            {
                arrivedFrom[neighborTile] = current.pos;
                gScore[neighborTile]      = tentativeGCost;
                double estimatedTotal     = tentativeGCost
                    + static_cast<double>(manhattanDistance(neighborTile, goalTile));
                openSet.push({neighborTile, tentativeGCost, estimatedTotal});
            }
        }
    }

    return {};
}

vector<Position> HierarchicalPlanner::localSearch(
    const Environment&                                env,
    const Position&                                   start,
    const Position&                                   goal,
    const unordered_set<Position, PositionHash>&      corridorTiles,
    const AbstractMap&                                abstractMap)
{
    unordered_map<Position, double,   PositionHash> gScore;
    unordered_map<Position, Position, PositionHash> arrivedFrom;
    unordered_set<Position,           PositionHash> finalized;

    priority_queue<SearchNode, vector<SearchNode>, SearchNodeComparator> openSet;

    gScore[start] = 0.0;
    openSet.push({start, 0.0,
                  static_cast<double>(manhattanDistance(start, goal))});

    while (!openSet.empty())
    {
        SearchNode current = openSet.top();
        openSet.pop();

        if (finalized.count(current.pos))
            continue;
        finalized.insert(current.pos);
        ++nodesExplored_;

        if (current.pos == goal)
            return reconstructPath(goal, start, arrivedFrom);

        for (const Position& neighbor : env.getNeighbors(current.pos))
        {
            // Corridor constraint: only expand cells whose tile was on the abstract path.
            // This is the key pruning step that makes the local search fast.
            if (corridorTiles.count(abstractMap.tileOf(neighbor)) == 0)
                continue;

            double tentativeGCost = gScore[current.pos] + 1.0;

            if (gScore.find(neighbor) == gScore.end()
             || tentativeGCost < gScore[neighbor])
            {
                arrivedFrom[neighbor] = current.pos;
                gScore[neighbor]      = tentativeGCost;
                double estimatedTotal = tentativeGCost
                    + static_cast<double>(manhattanDistance(neighbor, goal));
                openSet.push({neighbor, tentativeGCost, estimatedTotal});
            }
        }
    }

    return {};
}
