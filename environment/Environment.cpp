// environment/Environment.cpp
#include "Environment.h"
#include <fstream>
#include <random>
#include <stdexcept>

using namespace std;

// ---- Construction -----------------------------------------------------------

Environment::Environment(int width, int height)
    : width_(width), height_(height),
      grid_(width, vector<Cell>(height)),
      startPos_(0, 0), goalPos_(width - 1, height - 1)
{
    if (width <= 0 || height <= 0 || width * height > MAX_CELLS)
        throw invalid_argument("Grid dimensions out of range");

    grid_[startPos_.x][startPos_.y].setCellType(CellType::START);
    grid_[goalPos_.x][goalPos_.y].setCellType(CellType::GOAL);
}

// ---- Mutation ---------------------------------------------------------------

void Environment::setObstacle(const Position& p)
{
    if (inBounds(p) && p != startPos_ && p != goalPos_)
    {
        grid_[p.x][p.y].setCellType(CellType::OBSTACLE);
        syncObstacleBit(p, true);
    }
}

void Environment::clearObstacle(const Position& p)
{
    if (inBounds(p))
    {
        grid_[p.x][p.y].setCellType(CellType::EMPTY);
        syncObstacleBit(p, false);
    }
}

void Environment::setStart(const Position& p)
{
    if (inBounds(p) && !isObstacle(p))
    {
        grid_[startPos_.x][startPos_.y].setCellType(CellType::EMPTY);
        grid_[p.x][p.y].setCellType(CellType::START);
        startPos_ = p;
    }
}

void Environment::setGoal(const Position& p)
{
    if (inBounds(p) && !isObstacle(p))
    {
        grid_[goalPos_.x][goalPos_.y].setCellType(CellType::EMPTY);
        grid_[p.x][p.y].setCellType(CellType::GOAL);
        goalPos_ = p;
    }
}

void Environment::markVisited(const Position& p)
{
    if (!inBounds(p)) return;
    visited_.insert(p);
    if (grid_[p.x][p.y].getCellType() == CellType::EMPTY)
        grid_[p.x][p.y].setCellType(CellType::VISITED);
}

void Environment::markPath(const vector<Position>& path)
{
    for (int i = 1; i < (int)path.size() - 1; i++)
        grid_[path[i].x][path[i].y].setCellType(CellType::PATH);
}

void Environment::reset()
{
    visited_.clear();
    for (int x = 0; x < width_; x++)
        for (int y = 0; y < height_; y++)
            if (grid_[x][y].getCellType() == CellType::VISITED ||
                grid_[x][y].getCellType() == CellType::PATH)
                grid_[x][y].setCellType(CellType::EMPTY);
}

// ---- File I/O ---------------------------------------------------------------

// Wait until later to implement saved map loading
void Environment::loadFromFile(const string& filename)
{
    // TODO: open filename with ifstream; throw runtime_error if open fails
    // TODO: read line by line; each char maps to a CellType:
    //         '.' -> EMPTY, '#' -> OBSTACLE, 'S' -> START, 'G' -> GOAL
    // TODO: call setObstacle / setStart / setGoal for each special cell
    // TODO: validate parsed dimensions match width_ and height_
}

void Environment::generateLabyrinth(double loopDensity)
{
    for (int x = 0; x < width_; x++)
        for (int y = 0; y < height_; y++)
                setObstacle(Position(x, y));

    mt19937 randomEngine(random_device{}());
    function<void(Position)> carvePassages = [&](Position current)
    {
        vector<Position> neighborsOfCurrent;

        vector<pair<int,int>> offsets = {{2,0},{-2,0},{0,2},{0,-2}};
        for (const pair<int,int>& offset : offsets)
        {
            Position candidate(current.x + offset.first, current.y + offset.second);
            if (inBounds(candidate))
                neighborsOfCurrent.push_back(candidate);
        }

        shuffle(neighborsOfCurrent.begin(), neighborsOfCurrent.end(), randomEngine);

        for(Position n : neighborsOfCurrent)
        {
            if(!isVisited(n))
            {
                clearObstacle(Position((current.x + n.x) / 2, (current.y + n.y) / 2));
                clearObstacle(n);
                markVisited(n);
                carvePassages(n);
            }
        }
    };
    clearObstacle(startPos_);
    markVisited(startPos_);
    carvePassages(startPos_);

    uniform_real_distribution<double> distribution(0.0, 1.0);
    for (int x = 1; x < width_-1; x++)
        for (int y = 1; y < height_-1; y++)
        {
            Position candidate(x, y);
            if (isObstacle(candidate) && distribution(randomEngine) < loopDensity)
                clearObstacle(candidate);
        }

    setStart(startPos_);
    setGoal(goalPos_);
}

// Simpler less random sprinkling of obstacles
// Status: UNUSED
void Environment::generateRandom(double obstacleDensity)
{
    if (obstacleDensity < 0.0 || obstacleDensity > 1.0)
        throw invalid_argument("obstacleDensity must be in [0.0, 1.0]");

    mt19937 gen(random_device{}());
    uniform_real_distribution<double> dist(0.0, 1.0);

    for (int x = 0; x < width_; x++)
        for (int y = 0; y < height_; y++)
        {
            Position pos(x, y);
            if (pos != startPos_ && pos != goalPos_)
                if (dist(gen) < obstacleDensity)
                    setObstacle(pos);
        }
}

// ---- Queries ----------------------------------------------------------------

bool Environment::isObstacle(const Position& p) const
{
    if (!inBounds(p)) return false;
    return obstacleBits_.test(toIndex(p));
}

bool Environment::isVisited(const Position& p) const
{
    return visited_.count(p) > 0;
}

bool Environment::isValid(const Position& p) const
{
    return inBounds(p) && !isObstacle(p);
}

bool Environment::inBounds(const Position& p) const
{
    return inBounds(p.x, p.y);
}

bool Environment::inBounds(int x, int y) const
{
    return (x >= 0 && x < width_) && (y >= 0 && y < height_);
}

vector<Position> Environment::getNeighbors(const Position& p) const
{
    vector<Position> result;

    vector<pair<int,int>> offsets = {{1,0},{-1,0},{0,1},{0,-1}};

    for (auto& [dx, dy] : offsets)
    {
        Position candidate(p.x + dx, p.y + dy);
        if (isValid(candidate))
            result.push_back(candidate);
    }

    return result;
}

double Environment::moveCost(const Position& previous, const Position& current, const Position& next) const
{
    int directionX1 = current.x - previous.x;
    int directionY1 = current.y - previous.y;
    int directionX2 = next.x - current.x;
    int directionY2 = next.y - current.y;

    bool isTurn = (directionX1 != directionX2) || (directionY1 != directionY2);
    return isTurn ? 1.0 + turnPenalty_ : 1.0;
}

// ---- Accessors --------------------------------------------------------------

Position Environment::getStart()  const { return startPos_; }
Position Environment::getGoal()   const { return goalPos_;  }
int      Environment::getWidth()  const { return width_;    }
int      Environment::getHeight() const { return height_;   }

const vector<vector<Cell>>& Environment::getGrid() const
{
    return grid_;
}

int Environment::visitedCount() const
{
    return static_cast<int>(visited_.size());
}

int Environment::obstacleCount() const
{
    return static_cast<int>(obstacleBits_.count());
}

// ---- Private helpers --------------------------------------------------------

int Environment::toIndex(int x, int y) const
{
    return x * height_ + y;
}

int Environment::toIndex(const Position& p) const
{
    return toIndex(p.x, p.y);
}

void Environment::syncObstacleBit(const Position& p, bool set)
{
    int index = toIndex(p);
    if (set)
        obstacleBits_.set(index);
    else
        obstacleBits_.reset(index);
}
