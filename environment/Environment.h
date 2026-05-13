// environment/Environment.h
// High-level grid environment for robot path planning.
//
// STL highlights used here:
//   vector<vector<Cell>>              – 2D grid storage
//   unordered_set<Position,PositionHash> – O(1) visited lookup
//   bitset<MAX_CELLS>                 – compact, O(1) obstacle membership
#ifndef ENVIRONMENT_H
#define ENVIRONMENT_H

#include <vector>
#include <unordered_set>
#include <bitset>
#include <string>
#include "../core/Position.h"
#include "../core/Types.h"
#include "Cell.h"

// Maximum supported grid size (width * height must not exceed this).
// Sized for 100×100; increase for larger experiments.
constexpr int MAX_CELLS = 10000;

class Environment
{
public:
    // ---- Construction & initialisation ------------------------------------

    // width = number of columns (x-axis), height = number of rows (y-axis)
    Environment(int width, int height);

    // Populate from a plain-text file.
    // Format: each row is a string of chars, using Cell::toChar() convention:
    //   '.' = EMPTY, '#' = OBSTACLE, 'S' = START, 'G' = GOAL
    void loadFromFile(const std::string& filename);

    // Fill obstacles randomly; obstacleDensity in [0.0, 1.0].
    // Uses std::mt19937 seeded from std::random_device.
    void generateRandom(double obstacleDensity);
    void generateLabyrinth(double loopDensity);

    // ---- Mutation ---------------------------------------------------------

    void setObstacle(const Position& p);
    void clearObstacle(const Position& p);
    void setStart(const Position& p);
    void setGoal(const Position& p);
    void setTurnPenalty(double penalty);

    // Mark a cell as visited (called by algorithms during search).
    void markVisited(const Position& p);

    // Overlay PATH type onto cells listed in path (excludes start/goal).
    void markPath(const std::vector<Position>& path);

    // Reset visited markings and PATH overlays; preserve obstacles/start/goal.
    void reset();

    // ---- Queries ----------------------------------------------------------

    bool isObstacle(const Position& p) const;   // O(1) via bitset
    bool isVisited(const Position& p)  const;   // O(1) via unordered_set
    bool isValid(const Position& p)    const;   // in-bounds AND not obstacle
    bool inBounds(const Position& p)   const;
    bool inBounds(int x, int y)        const;   // raw index overload

    // Returns up to 4 cardinal neighbours that are in-bounds and not obstacles.
    std::vector<Position> getNeighbors(const Position& p) const;

    // Cost grid: turn cost > straight cost
    double moveCost(const Position& previous, const Position& current, const Position& next) const;

    // ---- Accessors --------------------------------------------------------

    Position                              getStart()  const;
    Position                              getGoal()   const;
    int                                   getWidth()  const;
    int                                   getHeight() const;
    const std::vector<std::vector<Cell>>& getGrid()   const;

    // Count of cells currently marked as VISITED.
    int visitedCount() const;

    // Count of obstacles currently set.
    int obstacleCount() const;

private:
    int width_;                     // number of columns (x-axis)
    int height_;                    // number of rows    (y-axis)
    double turnPenalty_ = 0.5;      // cost of moves total path

    // Primary grid representation — grid_[x][y] gives the Cell at column x, row y.
    std::vector<std::vector<Cell>> grid_;

    // Fast visited set — algorithms call markVisited() to register expansions.
    std::unordered_set<Position, PositionHash> visited_;

    // Bit-parallel obstacle index — mirrors grid_ obstacle state.
    // Bit i = 1 means the cell at toIndex(x,y) is an obstacle.
    std::bitset<MAX_CELLS> obstacleBits_;

    Position startPos_;
    Position goalPos_;

    // Flatten (x, y) → single index for obstacleBits_.
    int  toIndex(int x, int y)        const;
    int  toIndex(const Position& p)   const;

    // Internal helpers that keep grid_ and obstacleBits_ in sync.
    void syncObstacleBit(const Position& p, bool set);
};

#endif  // ENVIRONMENT_H
