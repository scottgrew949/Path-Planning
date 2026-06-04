// planning/hierarchical/AbstractMap.h
// Divides the Environment grid into N×N tiles for coarse abstract search.
// A tile is passable if it contains at least one non-obstacle cell.
//
// STL highlights:
//   vector<vector<bool>>  – precomputed passability table indexed [tileX][tileY]
//   vector<Position>      – neighbor and cell-list return types
#ifndef ABSTRACT_MAP_H
#define ABSTRACT_MAP_H

#include <vector>
#include "../../core/Position.h"
#include "../../environment/Environment.h"

class AbstractMap
{
public:
    // tileSize: side length (in grid cells) of each square tile.
    // Throws std::invalid_argument if tileSize < 1 or either env dimension is 0.
    AbstractMap(const Environment& env, int tileSize);

    // Map a cell coordinate to its tile coordinate.
    // tile.x = cell.x / tileSize_,  tile.y = cell.y / tileSize_
    Position tileOf(const Position& cell) const;

    // Returns abstract neighbors: adjacent tiles (4-cardinal) that are passable.
    std::vector<Position> getAbstractNeighbors(const Position& tile) const;

    // True if the tile at (tileX, tileY) has at least one non-obstacle cell.
    bool isTilePassable(int tileX, int tileY) const;

    // All in-bounds cell positions that belong to tile (tileX, tileY).
    std::vector<Position> cellsInTile(const Position& tile) const;

    int getTileWidth()  const;  // number of tiles horizontally
    int getTileHeight() const;  // number of tiles vertically
    int getTileSize()   const;

    // Integration seam: future Kalman predictions will call this to mark tiles
    // as probabilistically blocked. Stub for now — does nothing.
    void updateWithPredictions(const std::vector<Position>& predictedObstacles);

private:
    const Environment& env_;
    int tileSize_;
    int tileWidth_;   // = ceil(envWidth  / tileSize_)
    int tileHeight_;  // = ceil(envHeight / tileSize_)

    // Precomputed: tilePassable_[tx][ty] = true if tile contains any walkable cell.
    std::vector<std::vector<bool>> tilePassable_;

    void computePassability();
};

#endif  // ABSTRACT_MAP_H
