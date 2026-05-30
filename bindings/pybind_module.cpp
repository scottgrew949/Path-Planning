// bindings/pybind_module.cpp
// pybind11 bridge — exposes the C++ RL environment to Python.
//
// CONCEPT — What is pybind11?
//   A header-only C++ library that generates Python extension modules.
//   You describe your C++ classes and functions once in this file.
//   pybind11 generates all the glue code that lets Python call them.
//   The result is a .so shared library Python imports like any module:
//     import pathplanning
//     env = pathplanning.GridEnvironment(41, 41)
//
// CONCEPT — Why not rewrite the environment in Python?
//   The C++ environment is fast — critical when running 100,000+ training
//   steps per second for DQN. Python would be 10-100x slower for the same
//   logic. pybind11 gives us Python's ML ecosystem (PyTorch) with C++ speed
//   for the inner loop.
//
// CONCEPT — GridEnvironment wrapper
//   Python cannot manage C++ reference lifetimes. RLEnvironment holds
//   Environment& — if Python created them separately, the Environment
//   could be garbage collected while RLEnvironment still holds a reference.
//   GridEnvironment owns both internally, solving the lifetime problem.
//   Python only ever sees one object.
//
// Self-driving analog:
//   This file is the CAN bus — the bridge between the high-level
//   planning software (Python/PyTorch) and the low-level hardware
//   simulation (C++ environment).

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "../core/Position.h"
#include "../core/Types.h"
#include "../environment/Environment.h"
#include "../environment/DynamicEnvironment.h"
#include "../rl/RLEnvironment.h"
#include "../planning/algorithms/AStar.h"
#include "../planning/algorithms/DStarLite.h"
#include "../planning/hybrid/NeuralAStar.h"

namespace py = pybind11;

// ---- GridEnvironment --------------------------------------------------------
// Owns both Environment and RLEnvironment so Python manages one object.
// Exposes the full gym-style interface Python needs for DQN training.

class GridEnvironment
{
public:
    GridEnvironment(int      width,
                    int      height,
                    int      startX,
                    int      startY,
                    int      goalX,
                    int      goalY,
                    double   labyrinthDensity,
                    unsigned seed = 0)
        : environment_(width, height),
          rlEnvironment_(environment_)
    {
        environment_.setStart(Position(startX, startY));
        environment_.setGoal(Position(goalX, goalY));
        environment_.generateLabyrinth(labyrinthDensity, seed);
    }

    // Gym interface — forwarded to RLEnvironment
    std::vector<int> reset()
    {
        Position startPosition = rlEnvironment_.reset();
        return { startPosition.x, startPosition.y };
    }

    // Returns [newX, newY, reward, done] — simple list Python unpacks
    std::vector<double> step(int actionIndex)
    {
        Action action = static_cast<Action>(actionIndex);
        StepResult result = rlEnvironment_.step(action);
        return {
            static_cast<double>(result.newPosition.x),
            static_cast<double>(result.newPosition.y),
            result.reward,
            result.done ? 1.0 : 0.0
        };
    }

    int getWidth()  const { return rlEnvironment_.getWidth();  }
    int getHeight() const { return rlEnvironment_.getHeight(); }

    std::vector<int> getGoal() const
    {
        Position goal = rlEnvironment_.getGoal();
        return { goal.x, goal.y };
    }

    // Returns [up, down, left, right] — normalized distance to nearest wall in each direction.
    // 0.0 = wall immediately adjacent, 1.0 = clear to grid boundary.
    std::vector<float> getLineOfSight(int x, int y) const
    {
        int maxD = std::max(environment_.getWidth(), environment_.getHeight());
        auto rayDist = [&](int dx, int dy) -> float {
            for (int i = 1; i <= maxD; i++) {
                int nx = x + dx * i, ny = y + dy * i;
                if (!environment_.inBounds(nx, ny) || environment_.isObstacle(Position(nx, ny)))
                    return (float)i / maxD;
            }
            return 1.0f;
        };
        return { rayDist(0, -1), rayDist(0, 1), rayDist(-1, 0), rayDist(1, 0) };
    }

    // Returns optimal action index (0-3) for agent at (x,y) to reach goal.
    // UP=0 (dy=-1), DOWN=1 (dy=+1), LEFT=2 (dx=-1), RIGHT=3 (dx=+1).
    // Returns -1 if no path exists from (x,y) to goal.
    int getExpertAction(int x, int y) const
    {
        AStar astar;
        std::vector<Position> path = astar.findPath(
            environment_, Position(x, y), environment_.getGoal());
        if (path.size() < 2) return -1;
        int dx = path[1].x - path[0].x;
        int dy = path[1].y - path[0].y;
        if (dx ==  0 && dy == -1) return 0;
        if (dx ==  0 && dy ==  1) return 1;
        if (dx == -1 && dy ==  0) return 2;
        if (dx ==  1 && dy ==  0) return 3;
        return -1;
    }

    // Runs A* from start to goal, returns flat list [x0,y0,a0, x1,y1,a1, ...].
    // Each triple is (position, expert action at that position).
    // Length = 3 * (path_length - 1). Empty if no path.
    std::vector<int> getExpertTrajectory() const
    {
        AStar astar;
        std::vector<Position> path = astar.findPath(
            environment_, environment_.getStart(), environment_.getGoal());
        std::vector<int> result;
        result.reserve((path.size() > 0 ? path.size() - 1 : 0) * 3);
        for (size_t i = 0; i + 1 < path.size(); ++i) {
            int dx = path[i+1].x - path[i].x;
            int dy = path[i+1].y - path[i].y;
            int action = -1;
            if (dx ==  0 && dy == -1) action = 0;
            else if (dx ==  0 && dy ==  1) action = 1;
            else if (dx == -1 && dy ==  0) action = 2;
            else if (dx ==  1 && dy ==  0) action = 3;
            result.push_back(path[i].x);
            result.push_back(path[i].y);
            result.push_back(action);
        }
        return result;
    }

    // Returns true if cell (x,y) contains an obstacle.
    bool isObstacle(int x, int y) const
    {
        return environment_.isObstacle(Position(x, y));
    }

    // Returns true if (x,y) is in-bounds AND not an obstacle.
    bool isValid(int x, int y) const
    {
        return environment_.isValid(Position(x, y));
    }

    // CONCEPT — Bridge method: why here and not as a pybind lambda?
    //   NeuralAStar::findPath() takes `const Environment&` — a C++ internal type
    //   that Python must never hold directly. If we exposed `environment_` via an
    //   accessor and let Python pass it around, Python's GC could theoretically
    //   collect GridEnvironment while NeuralAStar still held the reference.
    //   Keeping the bridge INSIDE GridEnvironment is the right call: this class
    //   already IS the ownership boundary. It owns environment_, so it is the only
    //   safe site to hand environment_ into other C++ objects.
    //   Python sees one call, one object — no internal references leak outward.
    //
    // Return type: vector<vector<int>> where each inner vector is [x, y].
    //   NeuralAStar returns vector<Position>, which pybind11 cannot convert
    //   automatically (Position is not registered as a Python type).
    //   Converting here — at the boundary — keeps the rest of C++ clean.
    std::vector<std::vector<int>> runNeuralAStar(
        NeuralAStar& algorithm,
        int          startX,
        int          startY,
        int          goalX,
        int          goalY) const
    {
        std::vector<Position> path = algorithm.findPath(
            environment_,
            Position(startX, startY),
            Position(goalX,  goalY));

        std::vector<std::vector<int>> result;
        result.reserve(path.size());
        for (const Position& position : path)
            result.push_back({ position.x, position.y });
        return result;
    }

private:
    Environment    environment_;   // owned — lives as long as this object
    RLEnvironment  rlEnvironment_; // holds reference to environment_ above
};

// ---- DynamicGridEnvironment -------------------------------------------------
// Owns DynamicEnvironment so Python manages one object with no raw references.
// Mirrors the GridEnvironment pattern — same lifetime safety, same API style.
//
// CONCEPT — Why a separate wrapper instead of inheriting GridEnvironment?
//   GridEnvironment owns Environment + RLEnvironment (holds Environment&).
//   DynamicEnvironment IS-A Environment but RLEnvironment does not know about
//   dynamic obstacles. Keeping wrappers separate avoids a tangled inheritance
//   chain for something Python only ever sees as a single opaque object.

class DynamicGridEnvironment
{
public:
    DynamicGridEnvironment(int width, int height)
        : environment_(width, height)
    {}

    void setStart(int x, int y) { environment_.setStart(Position(x, y)); }
    void setGoal (int x, int y) { environment_.setGoal (Position(x, y)); }

    void generateLabyrinth(double loopDensity, unsigned seed = 0)
    {
        environment_.generateLabyrinth(loopDensity, seed);
    }

    // trajectory: list of [x, y] pairs — converted here at the boundary.
    void addDynamicObstacle(const std::vector<std::vector<int>>& trajectory,
                            int ticksPerStep = 1)
    {
        std::vector<Position> positions;
        positions.reserve(trajectory.size());
        for (const std::vector<int>& point : trajectory)
            positions.emplace_back(point[0], point[1]);
        environment_.addDynamicObstacle(positions, ticksPerStep);
    }

    // Advance simulation one tick. Returns [[x,y], ...] of all changed cells.
    std::vector<std::vector<int>> tick()
    {
        std::vector<Position> changed = environment_.tick();
        std::vector<std::vector<int>> result;
        result.reserve(changed.size());
        for (const Position& position : changed)
            result.push_back({ position.x, position.y });
        return result;
    }

    // Returns 2D grid of ints: 1 = obstacle, 0 = free.
    // Python can feed this directly into matplotlib imshow.
    std::vector<std::vector<int>> getObstacleGrid() const
    {
        int width  = environment_.getWidth();
        int height = environment_.getHeight();
        std::vector<std::vector<int>> grid(height, std::vector<int>(width, 0));
        for (int x = 0; x < width; ++x)
            for (int y = 0; y < height; ++y)
                if (environment_.isObstacle(Position(x, y)))
                    grid[y][x] = 1;
        return grid;
    }

    // Runs A* from (startX,startY) to (goalX,goalY) through current obstacle state.
    // Returns [[x,y], ...] path. Empty list if no path exists.
    std::vector<std::vector<int>> findPath(int startX, int startY, int goalX, int goalY) const
    {
        AStar astar;
        std::vector<Position> path = astar.findPath(
            environment_, Position(startX, startY), Position(goalX, goalY));
        std::vector<std::vector<int>> result;
        result.reserve(path.size());
        for (const Position& position : path)
            result.push_back({ position.x, position.y });
        return result;
    }

    // Place or remove a static obstacle. Used by Python to build the grid layout.
    void setObstacle  (int x, int y) { environment_.setObstacle  (Position(x, y)); }
    void clearObstacle(int x, int y) { environment_.clearObstacle(Position(x, y)); }

    // Returns [[x,y], ...] of each dynamic obstacle's current position.
    std::vector<std::vector<int>> getDynamicObstaclePositions() const
    {
        std::vector<Position> positions = environment_.getCurrentObstaclePositions();
        std::vector<std::vector<int>> result;
        result.reserve(positions.size());
        for (const Position& position : positions)
            result.push_back({ position.x, position.y });
        return result;
    }

    // D* Lite variant — incremental replanner designed for dynamic environments.
    // Same interface as findPath but tracks nodes explored for stats comparison.
    std::vector<std::vector<int>> findPathDStar(int startX, int startY, int goalX, int goalY)
    {
        DStarLite dstar;
        std::vector<Position> path = dstar.findPath(
            environment_, Position(startX, startY), Position(goalX, goalY));
        nodesExploredLastReplan_ = dstar.getNodesExplored();
        std::vector<std::vector<int>> result;
        result.reserve(path.size());
        for (const Position& position : path)
            result.push_back({ position.x, position.y });
        return result;
    }

    int getNodesExploredLastReplan() const { return nodesExploredLastReplan_; }

    bool isObstacle(int x, int y) const { return environment_.isObstacle(Position(x, y)); }
    bool isValid   (int x, int y) const { return environment_.isValid   (Position(x, y)); }

    std::vector<int> getStart() const { Position p = environment_.getStart(); return { p.x, p.y }; }
    std::vector<int> getGoal () const { Position p = environment_.getGoal (); return { p.x, p.y }; }

    int getWidth ()               const { return environment_.getWidth();               }
    int getHeight()               const { return environment_.getHeight();              }
    int getDynamicObstacleCount() const { return environment_.getDynamicObstacleCount(); }
    int getTickCount()            const { return environment_.getTickCount();            }

private:
    DynamicEnvironment environment_;
    int                nodesExploredLastReplan_ = 0;
};

// ---- Module registration ----------------------------------------------------

PYBIND11_MODULE(pathplanning, module)
{
    module.doc() = "C++ path planning and RL environment bridge";

    py::class_<GridEnvironment>(module, "GridEnvironment")
        .def(py::init<int, int, int, int, int, int, double, unsigned>(),
             py::arg("width"),
             py::arg("height"),
             py::arg("startX"),
             py::arg("startY"),
             py::arg("goalX"),
             py::arg("goalY"),
             py::arg("labyrinthDensity"),
             py::arg("seed") = 0)
        .def("reset",           &GridEnvironment::reset)
        .def("step",            &GridEnvironment::step)
        .def("getWidth",        &GridEnvironment::getWidth)
        .def("getHeight",       &GridEnvironment::getHeight)
        .def("getGoal",         &GridEnvironment::getGoal)
        .def("getLineOfSight",  &GridEnvironment::getLineOfSight)
        .def("getExpertAction",    &GridEnvironment::getExpertAction)
        .def("getExpertTrajectory",&GridEnvironment::getExpertTrajectory)
        .def("isObstacle",         &GridEnvironment::isObstacle)
        .def("isValid",            &GridEnvironment::isValid)

        // CONCEPT — Bridge method exposed as a plain .def():
        //   Python calls env.runNeuralAStar(algo, sx, sy, gx, gy).
        //   GridEnvironment handles the Environment& hand-off internally.
        //   From Python's perspective this is just a method call — it never
        //   touches or stores a raw Environment reference.
        .def("runNeuralAStar",     &GridEnvironment::runNeuralAStar,
             py::arg("algorithm"),
             py::arg("startX"),
             py::arg("startY"),
             py::arg("goalX"),
             py::arg("goalY"));

    // ---- DynamicGridEnvironment ---------------------------------------------
    py::class_<DynamicGridEnvironment>(module, "DynamicGridEnvironment")
        .def(py::init<int, int>(),
             py::arg("width"),
             py::arg("height"))
        .def("setStart",              &DynamicGridEnvironment::setStart)
        .def("setGoal",               &DynamicGridEnvironment::setGoal)
        .def("generateLabyrinth",     &DynamicGridEnvironment::generateLabyrinth,
             py::arg("loopDensity"),
             py::arg("seed") = 0)
        .def("addDynamicObstacle",    &DynamicGridEnvironment::addDynamicObstacle,
             py::arg("trajectory"),
             py::arg("ticksPerStep") = 1)
        .def("tick",                  &DynamicGridEnvironment::tick)
        .def("getObstacleGrid",       &DynamicGridEnvironment::getObstacleGrid)
        .def("findPath",                  &DynamicGridEnvironment::findPath)
        .def("findPathDStar",             &DynamicGridEnvironment::findPathDStar)
        .def("getNodesExploredLastReplan",&DynamicGridEnvironment::getNodesExploredLastReplan)
        .def("setObstacle",           &DynamicGridEnvironment::setObstacle)
        .def("clearObstacle",         &DynamicGridEnvironment::clearObstacle)
        .def("getDynamicObstaclePositions", &DynamicGridEnvironment::getDynamicObstaclePositions)
        .def("isObstacle",            &DynamicGridEnvironment::isObstacle)
        .def("isValid",               &DynamicGridEnvironment::isValid)
        .def("getStart",              &DynamicGridEnvironment::getStart)
        .def("getGoal",               &DynamicGridEnvironment::getGoal)
        .def("getWidth",              &DynamicGridEnvironment::getWidth)
        .def("getHeight",             &DynamicGridEnvironment::getHeight)
        .def("getDynamicObstacleCount", &DynamicGridEnvironment::getDynamicObstacleCount)
        .def("getTickCount",          &DynamicGridEnvironment::getTickCount);

    // ---- NeuralAStar --------------------------------------------------------
    // CONCEPT — Registering NeuralAStar as a first-class Python object:
    //   Python can construct it, check whether weights loaded, and pass it
    //   into env.runNeuralAStar(). The algorithm object is stateful (it tracks
    //   nodesExplored_ across calls), so Python retains it across benchmarks.
    //
    // CONCEPT — py::arg("weightEpsilon") = 1.5:
    //   Default argument in Python binding — identical semantics to C++ default.
    //   Python callers can override: NeuralAStar("weights.bin", weightEpsilon=1.0)
    //   to test strict admissibility vs the relaxed ε=1.5 mode.
    py::class_<NeuralAStar>(module, "NeuralAStar")
        .def(py::init<std::string, double>(),
             py::arg("weightsFilePath"),
             py::arg("weightEpsilon") = 1.5)
        .def("getNodesExplored",     &NeuralAStar::getNodesExplored)
        .def("getEpsilon",           &NeuralAStar::getEpsilon)

        // CONCEPT — getHeuristicCallCount():
        //   Exposes the internal call counter so the benchmark can report
        //   "Neural A* called heuristic N times vs A*'s M calls."
        //   Fewer calls = better guidance = tighter learned h*.
        .def("getHeuristicCallCount", &NeuralAStar::getHeuristicCallCount);
}
