// rl/DynaQAgent.h
// Dyna-Q agent — model-based RL with prioritized sweeping.
//
// CONCEPT — What is Dyna-Q?
//   Dyna-Q (Sutton 1990) augments Q-learning with a world model.
//   After each real environment step it does two things:
//     1. Direct RL update  — same Bellman update as Q-learning (from real experience)
//     2. Planning sweep     — n additional Bellman updates replayed from the model
//
//   The model stores: given (state, action) I previously saw -> (reward, next_state).
//   This is equivalent to the agent "imagining" past experiences in its head.
//
// CONCEPT — Why is this more sample-efficient than Q-learning?
//   Each real step produces (n+1) Q-value updates instead of 1.
//   With n=10 the agent learns ~10x faster in terms of environment interactions.
//   This matters when real interactions are expensive (physical robots, slow sims).
//
// CONCEPT — Prioritized sweeping (Moore & Atkeson, 1993)
//   Original Dyna-Q samples random (state, action) pairs during planning.
//   Prioritized sweeping replaces random selection with a max-heap ordered by
//   |ΔQ| — the magnitude of the Bellman error. Transitions where the agent was
//   most wrong get replayed first. Same planning budget, dramatically faster
//   value propagation because the most informative updates happen first.
//
//   After each real transition:
//     1. Compute |ΔQ| = |reward + γ·maxQ(s') - Q(s,a)|
//     2. If |ΔQ| > priorityThreshold_: push (|ΔQ|, s, a) onto the heap
//   In planningUpdate():
//     1. Pop the highest-priority (s, a) from the heap
//     2. Apply Bellman update — this may change Q(s,a) significantly
//     3. For each predecessor (s_prev, a_prev) that leads to s: recompute
//        their |ΔQ| and re-enqueue if significant (cascades value backwards)
//
// CONCEPT — planningSteps_ = 0
//   A valid configuration — the planning sweep is skipped entirely and DynaQ
//   degenerates to pure Q-learning.
#ifndef DYNA_Q_AGENT_H
#define DYNA_Q_AGENT_H

#include <unordered_map>
#include <queue>
#include <utility>
#include "RLAgent.h"

// ---- Model types ------------------------------------------------------------

using ModelKey = std::pair<Position, Action>;

// CONCEPT — Why a struct instead of pair<double, Position>?
//   The original pair stored only {reward, nextPos} with no way to distinguish
//   terminal transitions. Planning replays that land on the goal incorrectly
//   bootstrapped γ·maxQ(goal) instead of 0. Adding terminalTransition fixes
//   this and makes the model semantically complete.
struct ModelTransition
{
    double   reward;
    Position nextPosition;
    bool     terminalTransition;
};

struct ModelKeyHash
{
    std::size_t operator()(const ModelKey& key) const noexcept
    {
        std::size_t h1 = PositionHash{}(key.first);
        std::size_t h2 = std::hash<int>{}(static_cast<int>(key.second));
        return h1 ^ (h2 * 2654435761ULL);
    }
};

// ---- PrioritizedTransition --------------------------------------------------
//
// CONCEPT — Named struct over std::pair:
//   std::priority_queue<std::pair<double, ModelKey>> would work, but
//   pair's first/second naming makes the code harder to read. A named struct
//   with descriptive fields (bellmanError, key) documents intent at every use site.
struct PrioritizedTransition
{
    double   bellmanError;  // |ΔQ| — larger = replayed sooner
    ModelKey key;           // (state, action) to replay

    bool operator<(const PrioritizedTransition& other) const
    {
        return bellmanError < other.bellmanError;  // max-heap: largest error first
    }
};

// ---- DynaQAgent -------------------------------------------------------------

class DynaQAgent : public RLAgent
{
public:
    // planningSteps: number of model-based updates per real step (0 = no planning).
    // priorityThreshold: minimum |ΔQ| to enqueue a transition (default 0.01).
    DynaQAgent(RLEnvironment& environment,
               double         learningRate,
               double         discountFactor,
               double         epsilonStart,
               double         epsilonMin,
               double         epsilonDecay,
               int            planningSteps,
               double         priorityThreshold = 0.01);

    // Run one episode with Dyna-Q + prioritized sweeping.
    TrainingResult runEpisode(int episodeNumber, int maxStepsPerEpisode) override;

private:
    // World model: (state, action) -> transition
    // Iterated directly for predecessor scanning — no mirror vector needed.
    std::unordered_map<ModelKey, ModelTransition, ModelKeyHash> model_;

    int    planningSteps_;
    double priorityThreshold_;

    // Max-heap of transitions ordered by Bellman error magnitude.
    std::priority_queue<PrioritizedTransition> prioritizedTransitions_;

    // Compute |ΔQ| for (position, action) without writing to the Q-table.
    // done mirrors updateQValue semantics: zeroes future-value term for terminal transitions.
    [[nodiscard]] double computeBellmanError(const Position& position,
                               Action          action,
                               double          reward,
                               const Position& nextPosition,
                               bool            done) const;

    // Push (position, action) onto the heap if its Bellman error exceeds threshold.
    void enqueueIfSignificant(const Position& position,
                              Action          action,
                              double          reward,
                              const Position& nextPosition,
                              bool            done);

    // Pop from the priority heap and perform up to planningSteps_ updates.
    // Cascades: after each update, re-enqueues predecessors whose |ΔQ| rises.
    void planningUpdate();
};

#endif  // DYNA_Q_AGENT_H
