// planning/hierarchical/AbstractMap.cpp
#include "AbstractMap.h"
#include <algorithm>
#include <stdexcept>

using namespace std;

AbstractMap::AbstractMap(const Environment& env, int tileSize)
    : env_(env), tileSize_(tileSize)
{
    if (tileSize < 1)
        throw invalid_argument("AbstractMap: tileSize must be >= 1");
    if (env.getWidth() <= 0 || env.getHeight() <= 0)
        throw invalid_argument("AbstractMap: environment dimensions must be > 0");

    // Integer ceiling division: e.g. a 10-cell wide grid with tileSize=3
    // produces 4 tiles (tiles 0,1,2 span 3 cells each; tile 3 spans 1 cell).
    tileWidth_  = (env.getWidth()  + tileSize_ - 1) / tileSize_;
    tileHeight_ = (env.getHeight() + tileSize_ - 1) / tileSize_;

    tilePassable_.assign(tileWidth_, vector<bool>(tileHeight_, false));
    computePassability();
}

void AbstractMap::computePassability()
{
    for (int tileX = 0; tileX < tileWidth_; ++tileX)
    {
        for (int tileY = 0; tileY < tileHeight_; ++tileY)
        {
            // Cell range covered by this tile, clamped to env dimensions.
            // Edge tiles may be smaller than tileSize_ if the grid doesn't divide evenly.
            int cellXStart = tileX * tileSize_;
            int cellYStart = tileY * tileSize_;
            int cellXEnd   = min(cellXStart + tileSize_, env_.getWidth());
            int cellYEnd   = min(cellYStart + tileSize_, env_.getHeight());

            for (int cellX = cellXStart; cellX < cellXEnd && !tilePassable_[tileX][tileY]; ++cellX)
            {
                for (int cellY = cellYStart; cellY < cellYEnd; ++cellY)
                {
                    if (!env_.isObstacle(Position(cellX, cellY)))
                    {
                        tilePassable_[tileX][tileY] = true;
                        break;
                    }
                }
            }
        }
    }
}

Position AbstractMap::tileOf(const Position& cell) const
{
    return Position(cell.x / tileSize_, cell.y / tileSize_);
}

bool AbstractMap::isTilePassable(int tileX, int tileY) const
{
    if (tileX < 0 || tileX >= tileWidth_ || tileY < 0 || tileY >= tileHeight_)
        return false;
    return tilePassable_[tileX][tileY];
}

vector<Position> AbstractMap::getAbstractNeighbors(const Position& tile) const
{
    const int cardinalOffsetX[4] = { 0,  0, -1,  1};
    const int cardinalOffsetY[4] = {-1,  1,  0,  0};

    vector<Position> neighbors;
    neighbors.reserve(4);

    for (int direction = 0; direction < 4; ++direction)
    {
        int neighborTileX = tile.x + cardinalOffsetX[direction];
        int neighborTileY = tile.y + cardinalOffsetY[direction];

        if (neighborTileX >= 0 && neighborTileX < tileWidth_
         && neighborTileY >= 0 && neighborTileY < tileHeight_
         && tilePassable_[neighborTileX][neighborTileY])
        {
            neighbors.push_back(Position(neighborTileX, neighborTileY));
        }
    }

    return neighbors;
}

vector<Position> AbstractMap::cellsInTile(const Position& tile) const
{
    int cellXStart = tile.x * tileSize_;
    int cellYStart = tile.y * tileSize_;
    int cellXEnd   = min(cellXStart + tileSize_, env_.getWidth());
    int cellYEnd   = min(cellYStart + tileSize_, env_.getHeight());

    vector<Position> cells;
    cells.reserve((cellXEnd - cellXStart) * (cellYEnd - cellYStart));

    for (int cellX = cellXStart; cellX < cellXEnd; ++cellX)
        for (int cellY = cellYStart; cellY < cellYEnd; ++cellY)
            cells.push_back(Position(cellX, cellY));

    return cells;
}

int AbstractMap::getTileWidth()  const { return tileWidth_;  }
int AbstractMap::getTileHeight() const { return tileHeight_; }
int AbstractMap::getTileSize()   const { return tileSize_;   }

void AbstractMap::updateWithPredictions(const vector<Position>& predictedObstacles)
{
    // Mark the tile containing each predicted obstacle position as impassable.
    // The abstract search then routes around those tiles — the planner avoids
    // predicted obstacle locations before they actually arrive.
    for (const Position& predictedPosition : predictedObstacles)
    {
        if (!env_.inBounds(predictedPosition)) continue;
        Position tile = tileOf(predictedPosition);
        if (tile.x >= 0 && tile.x < tileWidth_ &&
            tile.y >= 0 && tile.y < tileHeight_)
            tilePassable_[tile.x][tile.y] = false;
    }
}
