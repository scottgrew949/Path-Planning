// planning/algorithms/CBS.h
//
// PURPOSE: Conflict-Based Search (CBS) for optimal multi-agent pathfinding.
//
// CORE CONCEPT — The multi-agent pathfinding problem (MAPF):
//   N agents must each reach a goal without colliding. Single-agent planners
//   ignore other agents and produce paths that intersect. MAPF solvers find
//   a set of collision-free paths that jointly minimise total path cost.
//   This matters directly for autonomous vehicle fleets and robot warehouses.
//
// CORE CONCEPT — Why CBS instead of a joint state space search?
//   Joint state space has N agents × 4 actions = 4^N states per step.
//   For 10 agents on a 41×41 grid: 4^10 ≈ 1M actions per step — intractable.
//   CBS decomposes the problem into N single-agent searches connected by a
//   shared constraint system. It finds the OPTIMAL solution while only ever
//   running O(conflicts * N) single-agent A* calls. Sharon et al., 2012.
//
// CORE CONCEPT — Two-level architecture:
//
//   HIGH LEVEL — Constraint Tree (CT):
//     A binary search tree. Each node stores:
//       - A set of constraints (agent i must not be at position p at time t)
//       - One path per agent (individually optimal given those constraints)
//       - Total cost = sum of path lengths
//     The root has no constraints. Each conflict produces two child nodes:
//     one constraining agent A, one constraining agent B at the conflict site.
//     The CT is searched by best-first on total cost (priority queue).
//
//   LOW LEVEL — Space-Time A* (single agent):
//     State = (x, y, timestep). The agent can WAIT in place.
//     Constraints are enforced by pruning states that violate them.
//     Returns the shortest path for ONE agent given its constraint set.
//
// CORE CONCEPT — Conflict types:
//   VERTEX conflict: agents A and B both at position p at time t.
//     Resolution: constrain A or B to leave p at t.
//   EDGE conflict: A moves p1→p2 while B moves p2→p1 at the same time t.
//     (Agents swap positions — physically impossible.)
//     Resolution: constrain the swap direction for one agent.
//
// CORE CONCEPT — Completeness and optimality:
//   CBS is complete (always finds a solution if one exists) and optimal
//   (finds minimum sum-of-costs solution). Trade-off: the CT can grow
//   exponentially in the worst case, but empirically performs well when
//   conflicts are sparse — which is the typical case for real navigation.
//
// USAGE:
//   CBS cbs;
//   std::vector<Agent> agents = { {start1, goal1}, {start2, goal2} };
//   MultiAgentPaths paths = cbs.findPaths(env, agents);
//   // paths[0] = path for agent 0, paths[1] = path for agent 1
//   // Each path is padded to the same length with the goal (agent waits).

#ifndef CBS_H
#define CBS_H

#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <queue>
#include <set>
#include "../../core/Position.h"
#include "../../core/Types.h"
#include "../../environment/Environment.h"

// ---- SpaceTimeState ---------------------------------------------------------
// State for the low-level Space-Time A* planner.
// Position + timestep uniquely identifies a node in the search graph.
struct SpaceTimeState
{
    Position position;
    int      timestep;

    bool operator==(const SpaceTimeState& other) const
    {
        return position  == other.position
            && timestep  == other.timestep;
    }

    bool operator<(const SpaceTimeState& other) const
    {
        if (timestep != other.timestep) return timestep < other.timestep;
        return position < other.position;
    }
};

struct SpaceTimeStateHash
{
    std::size_t operator()(const SpaceTimeState& state) const
    {
        PositionHash posHash;
        std::size_t h1 = posHash(state.position);
        std::size_t h2 = std::hash<int>{}(state.timestep);
        return h1 ^ (h2 * 2654435761ULL);
    }
};

// ---- CBSNode ----------------------------------------------------------------
// One node in the Constraint Tree (high level).
// Stores a complete assignment of paths for all agents plus the constraint set
// that produced those paths.
struct CBSNode
{
    std::vector<Constraint>    constraints;    // all constraints for all agents
    std::vector<EdgeConstraint> edgeConstraints;
    MultiAgentPaths             paths;         // one path per agent
    int                         totalCost;     // sum of path lengths

    bool operator>(const CBSNode& other) const { return totalCost > other.totalCost; }
};

// ---- CBS --------------------------------------------------------------------
class CBS
{
public:
    // CONCEPT — maxTimesteps caps the Space-Time A* search horizon.
    // A value of width*height*3 is generous enough for all practical cases
    // while preventing infinite loops when no solution exists.
    // maxTimesteps: caps Space-Time A* horizon. 0 = auto (3 * width * height).
    // maxCTNodes:   caps Constraint Tree expansions to prevent exponential blowup
    //               on adversarial inputs. 0 = no cap (use for small grids only).
    explicit CBS(int maxTimesteps = 0, int maxCTNodes = 10000);

    // Solve MAPF: find collision-free optimal paths for all agents.
    // Returns empty MultiAgentPaths if no solution found within the search budget.
    MultiAgentPaths findPaths(const Environment&         environment,
                              const std::vector<Agent>&  agents);

    // Number of CT nodes expanded in the last findPaths() call.
    int getNodesExpanded() const;

private:
    int maxTimesteps_;
    int maxCTNodes_;    // 0 = no cap; >0 = hard limit on CT expansions
    int nodesExpanded_;

    // ---- Low-level planner --------------------------------------------------

    // CONCEPT — Space-Time A*:
    //   Standard A* extended to 3D state space (x, y, t).
    //   The wait action expands (x, y, t) → (x, y, t+1) with cost 1.
    //   Constraints are enforced during expansion: a state (x, y, t) is pruned
    //   if any constraint in agentConstraints forbids agent agentIndex being at
    //   position (x,y) at time t. Similarly, edge constraints forbid specific
    //   (from, to, t) transitions.
    //   Heuristic: Manhattan distance to goal (ignores time — admissible).
    //
    // Returns the shortest-cost path. Empty if unreachable within maxTimesteps_.
    std::vector<Position> spaceTimeAStar(
        const Environment&               environment,
        const Agent&                     agent,
        int                              agentIndex,
        const std::vector<Constraint>&   constraints,
        const std::vector<EdgeConstraint>& edgeConstraints,
        int                              maxTime) const;

    // ---- High-level helpers -------------------------------------------------

    // CONCEPT — Conflict detection:
    //   Scan the paths array timestep by timestep.
    //   For each pair (i, j): check vertex conflict at each t, then edge conflict
    //   between t and t+1. Return the FIRST conflict found (earliest timestep,
    //   lowest agent indices) — CBS branches on the first conflict only.
    //   Returns an empty optional (hasConflict=false) if all paths are valid.
    struct ConflictResult { bool hasConflict; Conflict conflict; };
    ConflictResult findFirstConflict(const MultiAgentPaths& paths) const;

    // Pad path to targetLength by repeating the last position (agent waits at goal).
    static void padPath(std::vector<Position>& path, int targetLength);

    // Total cost = sum of path lengths (before padding is subtracted).
    static int computeTotalCost(const MultiAgentPaths& paths);
};

#endif  // CBS_H
