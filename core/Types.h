// core/Types.h
// Project-wide enumerations and lightweight value types.
#ifndef TYPES_H
#define TYPES_H

#include <string>
#include <vector>
#include "Position.h"

// ---- CellType ---------------------------------------------------------------
enum class CellType
{
    EMPTY,
    OBSTACLE,
    START,
    GOAL,
    PATH,       // cell lies on the found path
    VISITED,     // cell was explored but is not on final path

};

// ---- AlgorithmType ----------------------------------------------------------
// Used to select or label which pathfinder is running.
enum class AlgorithmType
{
    ASTAR,
    DIJKSTRA,
    BFS,
    BIDIRECTIONAL_ASTAR,
    DSTAR_LITE,
    THETA_STAR,
    JPS,
    RRT,
    NEURAL_ASTAR, // learned heuristic — planning/hybrid/NeuralAStar
    CBS           // Conflict-Based Search — multi-agent pathfinding
};

// ---- PathResult -------------------------------------------------------------
// Returned by each algorithm run; bundles outcome with diagnostics.
// All data members are value types — safe to copy / store in STL containers.
struct PathResult
{
    AlgorithmType   algorithm;
    std::string     algorithmName;
    std::vector<Position> path;     // ordered start → goal; empty = no path found
    double          pathCost;       // sum of edge weights along path
    double          elapsedMs;      // wall-clock time for findPath() call
    int             nodesExplored;  // cells popped from the open/queue structure

    PathResult();  // zero-initialise all numeric fields
};

// ---- Multi-agent types (CBS) ------------------------------------------------

// A single agent defined by its start and goal positions.
struct Agent
{
    Position start;
    Position goal;
    Agent(Position start, Position goal) : start(start), goal(goal) {}
};

// A vertex constraint: agent agentIndex must NOT be at position at timestep.
struct Constraint
{
    int      agentIndex;
    Position position;
    int      timestep;

    bool operator==(const Constraint& other) const
    {
        return agentIndex == other.agentIndex
            && position   == other.position
            && timestep   == other.timestep;
    }
};

// An edge constraint: agent agentIndex must not traverse from posFrom to posTo
// at timestep (prevents swap conflicts between agents).
struct EdgeConstraint
{
    int      agentIndex = 0;
    Position posFrom    = Position(0, 0);
    Position posTo      = Position(0, 0);
    int      timestep   = 0;
};

// A conflict between two agents detected in the CBS high-level search.
struct Conflict
{
    enum class Type { VERTEX, EDGE };
    Type     type      = Type::VERTEX;
    int      agent1    = 0;
    int      agent2    = 0;
    Position position  = Position(0, 0);  // vertex conflict location (or posFrom for edge)
    Position position2 = Position(0, 0); // posTo for edge conflict (unused for vertex)
    int      timestep  = 0;
};

// Result of a CBS solve: one time-stamped path per agent.
// Index: paths[agentIndex][timestep] = Position.
// Shorter paths are padded with the goal position (agent waits at goal).
using MultiAgentPaths = std::vector<std::vector<Position>>;

// ---- Action -----------------------------------------------------------------
// The four moves the RL agent can take from any grid cell.
// Stored as int values so they can index directly into the QTable's
// array<double,4> — one slot per action per cell.
//
// Self-driving analog: these are the robot's available controls —
// move forward, back, turn left, turn right.
enum class Action
{
    UP    = 0,
    DOWN  = 1,
    LEFT  = 2,
    RIGHT = 3
};

// ---- StepResult -------------------------------------------------------------
// Returned by RLEnvironment::step() after the agent takes one action.
// Bundles everything the agent needs to learn from that single move:
//   - where it ended up
//   - how good or bad that move was (reward)
//   - whether the episode is over (done)
//
// This mirrors the OpenAI Gym interface used in Python RL research —
// same concept, C++ implementation.
struct StepResult
{
    Position newPosition;   // cell the agent is now in after the action
    double   reward;        // immediate reward signal for this move
    bool     done;          // true when episode ends (goal reached)

    StepResult();           // zero-initialise numeric fields
    StepResult(Position newPosition, double reward, bool done);
};

#endif  // TYPES_H
