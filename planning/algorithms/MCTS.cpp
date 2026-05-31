// planning/algorithms/MCTS.cpp
#include "MCTS.h"
#include <cmath>
#include <limits>
#include <unordered_map>
#include <unordered_set>
#include <array>

using namespace std;

// ---- Internal types ---------------------------------------------------------

struct StateActionStats
{
    int    visitCount = 0;
    double totalValue = 0.0;
};

// Per-position: one stats entry per action (4 actions = indices 0–3).
using PositionStats = array<StateActionStats, 4>;

static constexpr Action ALL_ACTIONS[] = {
    Action::UP, Action::DOWN, Action::LEFT, Action::RIGHT
};

static int actionIndex(Action action)
{
    return static_cast<int>(action);
}

static Position applyAction(const Position& position, Action action)
{
    switch (action)
    {
        case Action::UP:    return Position(position.x,     position.y - 1);
        case Action::DOWN:  return Position(position.x,     position.y + 1);
        case Action::LEFT:  return Position(position.x - 1, position.y    );
        case Action::RIGHT: return Position(position.x + 1, position.y    );
    }
    return position;
}

// ---- MCTS -------------------------------------------------------------------

MCTS::MCTS(int numSimulations, double explorationConstant, int maxRolloutDepth)
    : numSimulations_(numSimulations),
      explorationConstant_(explorationConstant),
      maxRolloutDepth_(maxRolloutDepth),
      nodesExplored_(0),
      randomEngine_(random_device{}())
{}

string MCTS::getName() const
{
    return "MCTS (n=" + to_string(numSimulations_) + ")";
}

AlgorithmType MCTS::getType()          const { return AlgorithmType::MCTS; }
int           MCTS::getNodesExplored() const { return nodesExplored_; }

// ---- findPath ---------------------------------------------------------------

vector<Position> MCTS::findPath(const Environment& env,
                                 const Position&    start,
                                 const Position&    goal)
{
    nodesExplored_ = 0;
    if (start == goal)                             return {start};
    if (!env.isValid(start) || !env.isValid(goal)) return {};

    // Transposition table: per-position action statistics.
    unordered_map<Position, PositionStats, PositionHash> table;
    unordered_map<Position, int,           PositionHash> stateVisits;

    int maxDist = env.getWidth() + env.getHeight() - 2;

    // UCB action selection at a position.
    auto selectAction = [&](const Position& pos) -> Action
    {
        int    parentVisits = stateVisits.count(pos) ? stateVisits.at(pos) : 0;
        Action bestAction   = ALL_ACTIONS[0];
        double bestScore    = -numeric_limits<double>::infinity();

        for (Action action : ALL_ACTIONS)
        {
            if (!env.isValid(applyAction(pos, action))) continue;

            const StateActionStats& stats = table[pos][actionIndex(action)];
            double score;
            if (stats.visitCount == 0 || parentVisits == 0)
            {
                score = numeric_limits<double>::infinity();
            }
            else
            {
                double exploitation = stats.totalValue / stats.visitCount;
                double exploration  = explorationConstant_
                                      * sqrt(log(parentVisits) / stats.visitCount);
                score = exploitation + exploration;
            }

            if (score > bestScore) { bestScore = score; bestAction = action; }
        }
        return bestAction;
    };

    // Run simulations.
    for (int sim = 0; sim < numSimulations_; ++sim)
    {
        vector<pair<Position, Action>> trajectory;
        Position current = start;
        bool     reached = false;

        for (int depth = 0; depth < maxRolloutDepth_; ++depth)
        {
            if (current == goal) { reached = true; break; }

            Action   chosen = selectAction(current);
            Position next   = applyAction(current, chosen);
            if (!env.isValid(next)) break;

            trajectory.emplace_back(current, chosen);
            ++stateVisits[current];
            current = next;
            ++nodesExplored_;
        }

        if (current == goal) reached = true;

        // Value: 1.0 if goal reached, else normalised distance improvement.
        int    endDist = abs(current.x - goal.x) + abs(current.y - goal.y);
        double value   = reached ? 1.0
                                 : max(0.0, 1.0 - static_cast<double>(endDist) / maxDist);

        for (auto& [pos, action] : trajectory)
        {
            StateActionStats& stats = table[pos][actionIndex(action)];
            ++stats.visitCount;
            stats.totalValue += value;
        }
    }

    // Extract greedy path: follow highest-visit-count action at each position.
    vector<Position>                      path;
    unordered_set<Position, PositionHash> seen;
    Position current = start;
    path.push_back(current);
    seen.insert(current);

    while (current != goal
           && static_cast<int>(path.size()) < maxRolloutDepth_)
    {
        Action bestAction = ALL_ACTIONS[0];
        int    bestCount  = -1;

        for (Action action : ALL_ACTIONS)
        {
            Position next = applyAction(current, action);
            if (!env.isValid(next) || seen.count(next)) continue;
            int count = table.count(current)
                        ? table.at(current)[actionIndex(action)].visitCount : 0;
            if (count > bestCount) { bestCount = count; bestAction = action; }
        }

        if (bestCount <= 0) break;

        Position next = applyAction(current, bestAction);
        path.push_back(next);
        seen.insert(next);
        current = next;
    }

    return (current == goal) ? path : vector<Position>{};
}
