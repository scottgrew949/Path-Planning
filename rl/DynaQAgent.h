// rl/DynaQAgent.h
// Dyna-Q agent — model-based RL that combines direct RL with planning.
//
// CONCEPT — What is Dyna-Q?
//   Dyna-Q (Sutton 1990) augments Q-learning with a world model.
//   After each real environment step it does two things:
//     1. Direct RL update  — same Bellman update as Q-learning (from real experience)
//     2. Planning sweep     — n additional Bellman updates replayed from the model
//
//   The model stores: given (state, action) I previously saw -> (reward, next_state).
//   The planning sweep randomly samples n past transitions and re-applies the update.
//   This is equivalent to the agent "imagining" past experiences in its head.
//
// CONCEPT — Why is this more sample-efficient than Q-learning?
//   Each real step produces (n+1) Q-value updates instead of 1.
//   With n=10 the agent learns ~10x faster in terms of environment interactions.
//   This matters when real interactions are expensive (physical robots, slow sims).
//
// CONCEPT — model_ and modelKeys_
//   model_ maps (Position, Action) -> (reward, next_Position).
//   modelKeys_ is a parallel vector of all keys ever inserted, used for O(1)
//   random sampling during planning. std::unordered_map has no random access;
//   without this mirror we would need std::advance (O(n) per planning step).
//
// CONCEPT — planningSteps_ = 0
//   A valid configuration — the planning sweep is skipped entirely and DynaQ
//   degenerates to pure Q-learning. No-op guard: if (planningSteps_ > 0 && ...)
#ifndef DYNA_Q_AGENT_H
#define DYNA_Q_AGENT_H

#include <unordered_map>
#include <vector>
#include <utility>
#include "RLAgent.h"

// ---- Model types ------------------------------------------------------------

using ModelKey   = std::pair<Position, Action>;
using ModelValue = std::pair<double, Position>;  // {reward, nextPos}

struct ModelKeyHash
{
    std::size_t operator()(const ModelKey& k) const noexcept
    {
        std::size_t h1 = PositionHash{}(k.first);
        std::size_t h2 = std::hash<int>{}(static_cast<int>(k.second));
        return h1 ^ (h2 * 2654435761ULL);
    }
};

// ---- DynaQAgent -------------------------------------------------------------

class DynaQAgent : public RLAgent
{
public:
    // planningSteps: number of model-based updates per real step.
    //   0 = no planning (equivalent to Q-learning).
    //   Typical values: 5, 10, 50.
    DynaQAgent(RLEnvironment& environment,
               double         learningRate,
               double         discountFactor,
               double         epsilonStart,
               double         epsilonMin,
               double         epsilonDecay,
               int            planningSteps);

    // Run one episode with Dyna-Q: real step + model update + planning sweep.
    TrainingResult runEpisode(int episodeNumber, int maxStepsPerEpisode) override;

private:
    // World model: (state, action) -> (reward, next_state)
    std::unordered_map<ModelKey, ModelValue, ModelKeyHash> model_;

    // Mirror of model_ keys for O(1) random sampling during planning.
    // Grows in sync with model_ — keys are pushed when first inserted.
    std::vector<ModelKey> modelKeys_;

    int planningSteps_;

    // Cached distribution for sampling modelKeys_ — range updated lazily.
    // Avoids reconstructing a new distribution object on every planningUpdate() call.
    mutable std::uniform_int_distribution<int> keyDist_;

    // Perform planningSteps_ Bellman updates using randomly-sampled model entries.
    // No-op if planningSteps_ == 0 or model_ is empty.
    void planningUpdate();
};

#endif  // DYNA_Q_AGENT_H
