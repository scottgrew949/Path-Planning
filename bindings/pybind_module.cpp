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
#include "../rl/RLEnvironment.h"

namespace py = pybind11;

// ---- GridEnvironment --------------------------------------------------------
// Owns both Environment and RLEnvironment so Python manages one object.
// Exposes the full gym-style interface Python needs for DQN training.

class GridEnvironment
{
public:
    GridEnvironment(int    width,
                    int    height,
                    int    startX,
                    int    startY,
                    int    goalX,
                    int    goalY,
                    double labyrinthDensity)
        : environment_(width, height),
          rlEnvironment_(environment_)
    {
        environment_.setStart(Position(startX, startY));
        environment_.setGoal(Position(goalX, goalY));
        environment_.generateLabyrinth(labyrinthDensity);
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

private:
    Environment    environment_;   // owned — lives as long as this object
    RLEnvironment  rlEnvironment_; // holds reference to environment_ above
};

// ---- Module registration ----------------------------------------------------

PYBIND11_MODULE(pathplanning, module)
{
    module.doc() = "C++ path planning and RL environment bridge";

    py::class_<GridEnvironment>(module, "GridEnvironment")
        .def(py::init<int, int, int, int, int, int, double>(),
             py::arg("width"),
             py::arg("height"),
             py::arg("startX"),
             py::arg("startY"),
             py::arg("goalX"),
             py::arg("goalY"),
             py::arg("labyrinthDensity"))
        .def("reset",           &GridEnvironment::reset)
        .def("step",            &GridEnvironment::step)
        .def("getWidth",        &GridEnvironment::getWidth)
        .def("getHeight",       &GridEnvironment::getHeight)
        .def("getGoal",         &GridEnvironment::getGoal)
        .def("getLineOfSight",  &GridEnvironment::getLineOfSight);
}
