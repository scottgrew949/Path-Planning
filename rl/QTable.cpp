// rl/QTable.cpp
#include "QTable.h"
#include <algorithm>

// ---- Constructor ------------------------------------------------------------

QTable::QTable(int /*width*/, int /*height*/)
{}
    // table_ is default-constructed as an empty unordered_map.
    // Q-values are implicitly 0.0 — getValue returns 0.0 for any unseen key.
    // No pre-allocation needed; entries are created lazily on first setValue call.
    //
    // CONCEPT — Why lazy initialisation?
    //   The agent only visits reachable cells. Obstacle cells will never appear
    //   as keys in the table. Lazy init wastes no memory on unreachable states.

// ---- Core Q-value access ----------------------------------------------------

double QTable::getValue(const Position& position, Action action) const
{
    // Unseen states return 0.0 — optimistic initialisation would use a positive value,
    // but 0.0 is the standard starting point.
    std::unordered_map<Position, std::array<double, 4>, PositionHash>::const_iterator it = table_.find(position);
    if (it == table_.end()) return 0.0;
    return it->second[static_cast<int>(action)];
}

void QTable::setValue(const Position& position, Action action, double value)
{
    table_[position][static_cast<int>(action)] = value;
}

// ---- Policy helpers ---------------------------------------------------------

Action QTable::getBestAction(const Position& position) const
{
    std::unordered_map<Position, std::array<double, 4>, PositionHash>::const_iterator it = table_.find(position);
    if (it == table_.end()) return Action::UP;

    int bestIndex = 0;
    double bestValue = it->second[0];
    for(int k = 1; k <= 3; ++k)
    {
        if(bestValue < it->second[k])
        {
            bestValue = it->second[k];
            bestIndex = k;
        }
    }
    return static_cast<Action>(bestIndex); 

    // CONCEPT — Ties
    //   When two actions have equal Q-values we return the lowest enum index.
    //   This is deterministic — same state always produces same action.
    //   Non-deterministic tie-breaking (random) is valid but adds noise to
    //   training curves without meaningfully improving final policy quality.
}

double QTable::getMaxValue(const Position& position) const
{
    std::unordered_map<Position, std::array<double, 4>, PositionHash>::const_iterator it = table_.find(position);
    if (it == table_.end()) return 0.0;
    
    return *std::max_element(it->second.begin(), it->second.end());
    
    // CONCEPT — This value is the "future reward estimate" in Bellman:
    //   gamma * getMaxValue(nextPosition) — discounted best possible future outcome
    //   from the state the agent just landed in.
}

// ---- Diagnostics ------------------------------------------------------------

void QTable::reset()
{
    table_.clear();
    //       On the next getValue call, unseen entries will return 0.0 again.
    //       Effectively wipes all learned knowledge — agent starts from scratch.
}
