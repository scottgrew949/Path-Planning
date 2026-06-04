// environment/Environment.h
// High-level grid environment for robot path planning.
// STL Components:
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

// Maximum supported grid size: 100×100
constexpr int MAX_CELLS = 10000;

class Environment
{
public:
    // ---- Construction & initialisation ------------------------------------
    Environment(int width, int height);

    // Fill obstacles randomly; obstacleDensity in [0.0, 1.0].
    void generateRandom(double obstacleDensity);
    void generateRandom(double obstacleDensity, unsigned seed);   // seeded overload for reproducible mazes
    void generateLabyrinth(double loopDensity, unsigned seed = 0);

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

    // ---- Log-odds occupancy grid ---------------------------------------------
    //
    // CONCEPT — Production sensor fusion standard:
    //   Each cell stores log-odds l = log(p/(1-p)) instead of raw probability.
    //   Updates are additive (no multiply/divide), numerically stable, and clamped
    //   to [LOG_ODDS_MIN, LOG_ODDS_MAX] to prevent saturation lock-in.
    //   getBeliefAt() converts back to probability for external callers and
    //   visualization — internal storage always in log-odds space.
    
    static constexpr double LOG_ODDS_MIN   = -20.0;   // clamp floor (~p ≈ 2e-9)
    static constexpr double LOG_ODDS_MAX   =  20.0;   // clamp ceiling (~p ≈ 1 - 2e-9)
    static constexpr double LOG_ODDS_PRIOR = -2.197;  // log(0.1/0.9) — 10% prior

    // Apply one sensor reading to cell (x, y). Uses log-odds additive update.
    void updateBelief(int x, int y, bool sensorFired,
                      double truePositiveRate, double falsePositiveRate);

    // Return P(occupied) in [0,1] for cell (x, y). Converts from log-odds.
    double getBeliefAt(int x, int y) const;

    // Return raw log-odds for cell (x, y). Returns LOG_ODDS_MIN if out of bounds.
    double getLogOddsAt(int x, int y) const;

    // Reset all cells to LOG_ODDS_PRIOR (or supplied log-odds value).
    void resetBeliefs(double logOddsPrior = LOG_ODDS_PRIOR);

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

    int visitedCount() const;
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
    std::bitset<MAX_CELLS> obstacleBits_;

    // Log-odds occupancy grid — one double per cell in [LOG_ODDS_MIN, LOG_ODDS_MAX].
    // Stores log(p/(1-p)) for each cell. Additive updates, clamp-bounded.
    std::vector<double> cellLogOdds_;

    Position startPos_;
    Position goalPos_;

    // Flatten (x, y) -> single index for obstacleBits_.
    int  toIndex(int x, int y)        const;
    int  toIndex(const Position& p)   const;

    // Keeps grid_ and obstacleBits_ in sync.
    void syncObstacleBit(const Position& p, bool set);
};

#endif 
