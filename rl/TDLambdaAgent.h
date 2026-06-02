// rl/TDLambdaAgent.h
// Q(λ) agent — Q-learning with eligibility traces.
//
// CONCEPT — What are eligibility traces?
//   Standard Q-learning does 1-step TD: only the last (state, action) pair gets
//   a Bellman update per step. If the reward is delayed — like a maze where +100
//   only arrives at the goal — credit has to flow backwards one step at a time,
//   which is slow.
//
//   Eligibility traces keep a decaying memory of every (state, action) pair
//   visited in the current episode. Each time a pair is visited, its trace
//   increments. After every step, ALL traced pairs get a Q-update weighted by
//   their trace value. Then every trace decays by γλ.
//
//   Effect: credit for a reward flows backwards through the entire episode in a
//   single update pass, not one step per episode. A path of 40 steps gets credit
//   propagated to step 1 in 40 real steps instead of 40,000 episodes.
//
// CONCEPT — The λ parameter
//   λ = 0: traces decay immediately → equivalent to standard 1-step Q-learning.
//   λ = 1: traces decay only by γ → close to Monte Carlo (full episode credit).
//   λ = 0.9: traces decay slowly, credit flows back ~10 steps on average.
//   In practice λ ∈ [0.8, 0.95] works well for maze-like environments.
//
// CONCEPT — Accumulating vs replacing traces
//   Accumulating: e(s,a) += 1 each visit.
//   Replacing:    e(s,a)  = 1 each visit (caps the trace).
//   This implementation uses accumulating traces — standard for Q(λ).
//
// Self-driving car analog:
//   Standard Q-learning blames only the last steering input for the crash.
//   TD-λ distributes blame back through the last N steering decisions.
//   Longer λ = longer memory = credit reaches further back in the episode.
#ifndef TD_LAMBDA_AGENT_H
#define TD_LAMBDA_AGENT_H

#include <unordered_map>
#include <array>
#include "RLAgent.h"
#include "../core/Position.h"
#include "../core/Types.h"

class TDLambdaAgent : public RLAgent
{
public:
    // lambda: trace decay parameter in [0, 1]. Throws std::invalid_argument if out of range.
    // All other parameters delegate to RLAgent.
    TDLambdaAgent(RLEnvironment& environment,
                  double         learningRate,
                  double         discountFactor,
                  double         epsilonStart,
                  double         epsilonMin,
                  double         epsilonDecay,
                  double         lambda);

    // Run one episode with Q(λ): eligibility traces propagate credit backwards.
    TrainingResult runEpisode(int episodeNumber, int maxStepsPerEpisode) override;

    double getLambda() const;

private:
    double lambda_;

    // Eligibility traces: (state, action) → trace value.
    // Mirrors QTable::table_ structure — array index = static_cast<int>(Action).
    // Entries below TRACE_PRUNE_THRESHOLD are erased to prevent map growth.
    std::unordered_map<Position, std::array<double, 4>, PositionHash> eligibilityTraces_;

    // Traces below this value have negligible effect — pruned to keep map small.
    static constexpr double TRACE_PRUNE_THRESHOLD = 1e-9;

    // Increment the trace for (position, action) by 1.0 (accumulating trace).
    void incrementTrace(const Position& position, Action action);

    // Apply α·δ·e(s,a) to every Q-value in the table, then decay each trace by γλ.
    // Prunes entries that fall below TRACE_PRUNE_THRESHOLD after decay.
    void applyTracesAndDecay(double tdError);

    // Zero all traces at the start of each episode.
    void clearTraces();
};

#endif  // TD_LAMBDA_AGENT_H
