// planning/algorithms/CBS.cpp
#include "CBS.h"
#include <algorithm>
#include <limits>
#include <cmath>
#include <optional>

using namespace std;

// ---- CBS constructor --------------------------------------------------------

CBS::CBS(int maxTimesteps, int maxCTNodes)
    : maxTimesteps_(maxTimesteps), maxCTNodes_(maxCTNodes), nodesExpanded_(0)
{}

int CBS::getNodesExpanded() const
{
    return nodesExpanded_;
}

// ---- Low-level: Space-Time A* ----------------------------------------------

vector<Position> CBS::spaceTimeAStar(
    const Environment&               environment,
    const Agent&                     agent,
    int                              agentIndex,
    const vector<Constraint>&        constraints,
    const vector<EdgeConstraint>&    edgeConstraints,
    int                              maxTime) const
{
    // CONCEPT — Space-Time A* search:
    //   State = (Position, timestep). Goal = reach agent.goal at any timestep.
    //   The open set is a min-heap on f = g + h where:
    //     g = timestep (each move or wait costs 1)
    //     h = Manhattan distance to goal (admissible — ignores obstacles)
    //
    //   Constraints are checked at expansion time:
    //     Vertex constraint (agentIndex, pos, t): skip state (pos, t) if matching.
    //     Edge constraint (agentIndex, from, to, t): skip the transition from→to at t.
    //
    //   The wait action: expand (pos, t) → (pos, t+1). This allows an agent to
    //   pause at its current cell to let another agent pass — the key mechanism
    //   that makes CBS resolutions possible without drastically rerouting.

    // Build fast-lookup sets for this agent's constraints.
    // Vertex constraints: (position, timestep) pairs to reject.
    unordered_set<SpaceTimeState, SpaceTimeStateHash> vertexBlocked;
    for (const Constraint& constraint : constraints)
    {
        if (constraint.agentIndex == agentIndex)
            vertexBlocked.insert({constraint.position, constraint.timestep});
    }

    // Edge constraints: (posFrom, posTo, timestep) triples to reject.
    // Stored as map from (timestep, posFrom) → set of blocked posTo.
    // We use the SpaceTimeState hash for the from-state key.
    unordered_map<SpaceTimeState,
                  vector<Position>,
                  SpaceTimeStateHash> edgeBlocked;
    for (const EdgeConstraint& ec : edgeConstraints)
    {
        if (ec.agentIndex == agentIndex)
            edgeBlocked[{ec.posFrom, ec.timestep}].push_back(ec.posTo);
    }

    // A* open set: (f_cost, g_cost, state).
    // Using tuple so the heap orders by f first, then g (prefer shallower paths).
    using OpenEntry = tuple<int, int, SpaceTimeState>;
    priority_queue<OpenEntry, vector<OpenEntry>, greater<OpenEntry>> openSet;

    // Cost and parent maps.
    unordered_map<SpaceTimeState, int,      SpaceTimeStateHash> gCost;
    unordered_map<SpaceTimeState,
                  SpaceTimeState,           SpaceTimeStateHash> cameFrom;
    unordered_set<SpaceTimeState,           SpaceTimeStateHash> closed;

    SpaceTimeState startState = {agent.start, 0};
    gCost[startState] = 0;

    int hStart = abs(agent.start.x - agent.goal.x)
               + abs(agent.start.y - agent.goal.y);
    openSet.push({hStart, 0, startState});

    while (!openSet.empty())
    {
        auto [fVal, gVal, current] = openSet.top();
        openSet.pop();

        if (closed.count(current)) continue;
        closed.insert(current);

        // Goal check: reached goal position at any timestep.
        if (current.position == agent.goal)
        {
            // Reconstruct path.
            vector<Position> path;
            SpaceTimeState   node = current;
            while (!(node == startState))
            {
                path.push_back(node.position);
                node = cameFrom.at(node);
            }
            path.push_back(agent.start);
            reverse(path.begin(), path.end());
            return path;
        }

        if (current.timestep >= maxTime) continue;

        int nextTimestep = current.timestep + 1;

        // ---- Expand neighbours (4 cardinal moves + wait) --------------------
        vector<Position> candidates = environment.getNeighbors(current.position);
        candidates.push_back(current.position);   // wait action

        for (const Position& nextPos : candidates)
        {
            SpaceTimeState nextState = {nextPos, nextTimestep};

            // Check vertex constraint.
            if (vertexBlocked.count(nextState)) continue;

            // Check edge constraint (from current.position → nextPos at current.timestep).
            auto edgeIt = edgeBlocked.find({current.position, current.timestep});
            if (edgeIt != edgeBlocked.end())
            {
                const vector<Position>& blocked = edgeIt->second;
                if (find(blocked.begin(), blocked.end(), nextPos) != blocked.end())
                    continue;
            }

            int newG = gVal + 1;
            auto costIt = gCost.find(nextState);
            if (costIt != gCost.end() && costIt->second <= newG) continue;

            gCost[nextState]   = newG;
            cameFrom[nextState] = current;

            int h = abs(nextPos.x - agent.goal.x)
                  + abs(nextPos.y - agent.goal.y);
            openSet.push({newG + h, newG, nextState});
        }
    }

    return {};  // no path found within maxTime
}

// ---- High-level: conflict detection ----------------------------------------

CBS::ConflictResult CBS::findFirstConflict(const MultiAgentPaths& paths) const
{
    // CONCEPT — Conflict detection sweep:
    //   For every timestep t and every pair (i, j) with i < j:
    //   1. VERTEX: if paths[i][t] == paths[j][t] → vertex conflict.
    //   2. EDGE: if paths[i][t]==paths[j][t+1] && paths[j][t]==paths[i][t+1]
    //            → swap/edge conflict.
    //   Return the first (lowest t, lowest i) conflict found.
    //
    //   Paths are padded to the same length (agents wait at goal), so indexing
    //   with t is safe across all agents.

    if (paths.empty()) return {false, Conflict{}};

    int numAgents = static_cast<int>(paths.size());
    int maxLen    = 0;
    for (const auto& path : paths)
        maxLen = max(maxLen, static_cast<int>(path.size()));

    // Helper to get position at timestep t, clamped to path end (wait at goal).
    auto posAt = [&](int agentIdx, int t) -> Position {
        const vector<Position>& path = paths[agentIdx];
        int index = min(t, static_cast<int>(path.size()) - 1);
        return path[index];
    };

    for (int t = 0; t < maxLen; ++t)
    {
        for (int i = 0; i < numAgents - 1; ++i)
        {
            for (int j = i + 1; j < numAgents; ++j)
            {
                // Vertex conflict.
                Position pi = posAt(i, t);
                Position pj = posAt(j, t);
                if (pi == pj)
                {
                    Conflict conflict;
                    conflict.type      = Conflict::Type::VERTEX;
                    conflict.agent1    = i;
                    conflict.agent2    = j;
                    conflict.position  = pi;
                    conflict.timestep  = t;
                    return {true, conflict};
                }

                // Edge conflict: i goes pi→pi_next while j goes pj→pj_next = pi.
                if (t + 1 < maxLen)
                {
                    Position pi_next = posAt(i, t + 1);
                    Position pj_next = posAt(j, t + 1);
                    if (pi == pj_next && pj == pi_next)
                    {
                        Conflict conflict;
                        conflict.type      = Conflict::Type::EDGE;
                        conflict.agent1    = i;
                        conflict.agent2    = j;
                        conflict.position  = pi;
                        conflict.position2 = pj;
                        conflict.timestep  = t;
                        return {true, conflict};
                    }
                }
            }
        }
    }

    return {false, {}};
}

// ---- Helpers ----------------------------------------------------------------

void CBS::padPath(vector<Position>& path, int targetLength)
{
    if (path.empty() || static_cast<int>(path.size()) >= targetLength) return;
    Position goal = path.back();
    while (static_cast<int>(path.size()) < targetLength)
        path.push_back(goal);
}

int CBS::computeTotalCost(const MultiAgentPaths& paths)
{
    int total = 0;
    for (const auto& path : paths)
        total += static_cast<int>(path.size());
    return total;
}

// ---- High-level: CBS main search -------------------------------------------

MultiAgentPaths CBS::findPaths(const Environment&        environment,
                                const vector<Agent>&      agents)
{
    // CONCEPT — CBS high-level search:
    //   1. Build root CT node: no constraints, one individually-optimal path
    //      per agent (from Space-Time A* with no constraints).
    //   2. Check for conflicts. If none → return paths (optimal solution).
    //   3. Branch on the first conflict: create two child nodes.
    //      Child A: add vertex constraint (agent1, conflict.position, conflict.timestep)
    //      Child B: add vertex constraint (agent2, conflict.position, conflict.timestep)
    //      For edge conflicts: add edge constraints on both directions.
    //   4. For each child: replan only the constrained agent (others unchanged).
    //      If replanning fails → prune this child (no valid solution under these constraints).
    //   5. Push both valid children onto the priority queue (ordered by total cost).
    //   6. Pop minimum-cost node and repeat from step 2.
    //
    //   The CT is best-first on total cost, guaranteeing CBS returns the
    //   sum-of-costs-optimal solution (Sharon et al. Theorem 1).

    nodesExpanded_ = 0;

    int numAgents = static_cast<int>(agents.size());
    if (numAgents == 0) return {};

    int gridMax    = environment.getWidth() * environment.getHeight();
    int maxTime    = (maxTimesteps_ > 0) ? maxTimesteps_ : gridMax * 3;

    // ---- Build root node ----------------------------------------------------
    CBSNode root;
    root.paths.resize(numAgents);

    for (int agentIdx = 0; agentIdx < numAgents; ++agentIdx)
    {
        root.paths[agentIdx] = spaceTimeAStar(
            environment, agents[agentIdx], agentIdx,
            root.constraints, root.edgeConstraints, maxTime);

        if (root.paths[agentIdx].empty())
            return {};  // agent has no path even without constraints
    }
    root.totalCost = computeTotalCost(root.paths);

    // ---- CT priority queue (min-heap on total cost) -------------------------
    priority_queue<CBSNode, vector<CBSNode>, greater<CBSNode>> openCT;
    openCT.push(root);

    while (!openCT.empty())
    {
        CBSNode current = openCT.top();
        openCT.pop();
        ++nodesExpanded_;

        // Pad all paths to the same length so conflict detection can index uniformly.
        int maxPathLen = 0;
        for (const auto& path : current.paths)
            maxPathLen = max(maxPathLen, static_cast<int>(path.size()));
        MultiAgentPaths paddedPaths = current.paths;
        for (auto& path : paddedPaths)
            padPath(path, maxPathLen);

        // Check for conflicts in the padded paths.
        ConflictResult result = findFirstConflict(paddedPaths);
        if (!result.hasConflict)
            return paddedPaths;  // all paths valid — optimal solution found

        const Conflict& conflict = result.conflict;

        // ---- Branch: create two child nodes ---------------------------------
        // For each of the two involved agents, add the appropriate constraint
        // and replan that agent. Leave all other agents' paths unchanged.

        auto makeChild = [&](int constrainedAgent) -> optional<CBSNode>
        {
            CBSNode child;
            child.constraints     = current.constraints;
            child.edgeConstraints = current.edgeConstraints;
            child.paths           = current.paths;

            if (conflict.type == Conflict::Type::VERTEX)
            {
                child.constraints.push_back(
                    {constrainedAgent, conflict.position, conflict.timestep});
            }
            else  // EDGE conflict
            {
                // Constrain the swap: agent cannot make the transition that causes the swap.
                Position from = (constrainedAgent == conflict.agent1)
                                ? conflict.position : conflict.position2;
                Position to   = (constrainedAgent == conflict.agent1)
                                ? conflict.position2 : conflict.position;
                child.edgeConstraints.push_back(
                    {constrainedAgent, from, to, conflict.timestep});
            }

            // Replan only the constrained agent.
            child.paths[constrainedAgent] = spaceTimeAStar(
                environment, agents[constrainedAgent], constrainedAgent,
                child.constraints, child.edgeConstraints, maxTime);

            if (child.paths[constrainedAgent].empty())
                return {};  // no solution under this constraint — prune

            child.totalCost = computeTotalCost(child.paths);
            return child;
        };

        auto childA = makeChild(conflict.agent1);
        auto childB = makeChild(conflict.agent2);

        if (childA) openCT.push(*childA);
        if (childB) openCT.push(*childB);
    }

    return {};  // CT exhausted — no solution found
}
