// planning/hierarchical/HierarchicalPlanner.h
// Two-phase hierarchical pathfinder (HPA*-lite) implementing IPathfinder.
//
// CONCEPT — Two-phase hierarchical search (HPA*-lite):
//   Phase 1 (abstract): Run A* on a coarse tile graph. Each tile groups N×N cells.
//   This finds a corridor of tiles from start to goal — fast, low node count.
//   Phase 2 (local):    Run fine-grained A* constrained to corridor tiles only.
//   This finds the precise cell-level path without exploring the whole grid.
//
//   Self-driving analog: global planner selects a highway route (abstract tiles),
//   local planner navigates lane-by-lane within that route (cell-level A*).
//   Standard in ROS nav stack: move_base global_planner + local_planner.
//
// STL highlights:
//   priority_queue  – min-heap for both abstract and local A* open sets
//   unordered_map   – gScore and arrivedFrom tables (O(1) lookup)
//   unordered_set   – corridor membership test (O(1) per neighbor check)
#ifndef HIERARCHICAL_PLANNER_H
#define HIERARCHICAL_PLANNER_H

#include <vector>
#include <unordered_set>
#include <string>
#include "../../core/Position.h"
#include "../../core/Types.h"
#include "../../environment/Environment.h"
#include "../../environment/KalmanTracker.h"
#include "../IPathfinder.h"
#include "AbstractMap.h"

class HierarchicalPlanner : public IPathfinder
{
public:
    // tileSize: tile side length in cells.
    // Throws std::invalid_argument if tileSize < 1.
    explicit HierarchicalPlanner(int tileSize = 4);

    // Optional: attach a Kalman tracker so findPath marks predicted obstacle
    // tiles as blocked before the abstract search.
    // predictionHorizon: how many ticks ahead to predict (default 2).
    // Pass nullptr to detach.
    void setKalmanTracker(const KalmanTracker* tracker, double predictionHorizon = 2.0);

    std::vector<Position> findPath(
        const Environment& env,
        const Position&    start,
        const Position&    goal
    ) override;

    std::string   getName()          const override;
    AlgorithmType getType()          const override;
    int           getNodesExplored() const override;

private:
    int                  tileSize_;
    int                  nodesExplored_      = 0;
    const KalmanTracker* kalmanTracker_      = nullptr;
    double               predictionHorizon_  = 2.0;

    // Phase 1: Manhattan-heuristic A* on the abstract tile graph.
    // Returns ordered tile positions from startTile to goalTile, or {} if unreachable.
    std::vector<Position> abstractSearch(
        const AbstractMap& abstractMap,
        const Position&    startTile,
        const Position&    goalTile
    );

    // Phase 2: A* constrained to cells whose tile is in corridorTiles.
    // Neighbors outside the corridor are skipped.
    // Returns ordered cell path from start to goal, or {} if not found.
    std::vector<Position> localSearch(
        const Environment&                                env,
        const Position&                                   start,
        const Position&                                   goal,
        const std::unordered_set<Position, PositionHash>& corridorTiles,
        const AbstractMap&                                abstractMap
    );
};

#endif  // HIERARCHICAL_PLANNER_H
