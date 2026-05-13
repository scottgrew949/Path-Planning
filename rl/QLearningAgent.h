// rl/QLearningAgent.h
// Tabular Q-learning agent with ε-greedy exploration policy.
//
// CONCEPT — What does this agent do?
//   It runs episodes in RLEnvironment, choosing actions and updating QTable
//   after every step. Over thousands of episodes it learns which action is
//   best from every reachable cell — a complete navigation policy.
//
// CONCEPT — ε-greedy policy
//   Every step the agent flips a biased coin:
//     With probability ε   → pick a RANDOM action (explore)
//     With probability 1-ε → pick the BEST known action (exploit)
//
//   Early training: ε is high (e.g. 0.9) — mostly random, agent discovers the grid.
//   Late training:  ε is low (e.g. 0.05) — mostly greedy, agent refines known routes.
//   ε is decayed each episode: ε = max(ε_min, ε * ε_decay)
//
//   Self-driving analog: a new driver explores unfamiliar routes (high ε).
//   An experienced driver exploits known fast routes (low ε).
//
// CONCEPT — Bellman update (applied after every step)
//   Q(s,a) += alpha * [ reward + gamma * maxQ(s') - Q(s,a) ]
//     s      = cell before the action
//     a      = action taken
//     reward = immediate reward signal from RLEnvironment::step()
//     s'     = cell after the action
//     alpha  = learning rate    — how fast to incorporate new experience (0 < α ≤ 1)
//     gamma  = discount factor  — how much to value future vs immediate reward (0 < γ ≤ 1)
//
//   Intuition: "Was this move better or worse than I expected?
//               Shift my estimate by alpha in that direction."
//
// CONCEPT — Hyperparameters
//   alpha (learning rate) — small = slow but stable, large = fast but noisy
//   gamma (discount)      — near 1.0 = cares about long-term reward (good for navigation)
//                           near 0.0 = only cares about immediate reward (shortsighted)
//   epsilon               — exploration rate, decays toward epsilon_min over training
//
// CONCEPT — Convergence
//   With enough episodes and correct hyperparameters, Q-values converge to the
//   true optimal policy (proven by Q-learning convergence theorem).
//   For this 41×41 grid, ~5000 episodes is typically sufficient.
#ifndef QLEARNING_AGENT_H
#define QLEARNING_AGENT_H

#include <vector>
#include <random>
#include "QTable.h"
#include "RLEnvironment.h"
#include "../core/Types.h"
#include "../core/Position.h"

// ---- TrainingResult ---------------------------------------------------------
// Diagnostic snapshot of a single training episode.
// Returned by runEpisode() so callers can track learning progress.
struct TrainingResult
{
    int    episodeNumber;    // which episode this was (1-indexed)
    double totalReward;      // sum of all rewards received in this episode
    int    stepsToGoal;      // number of steps taken before done == true
    bool   goalReached;      // true if the agent found the goal
    double epsilonAtEnd;     // exploration rate at the end of this episode

    TrainingResult();
    TrainingResult(int episodeNumber, double totalReward, int stepsToGoal, bool goalReached, double epsilonAtEnd);
};

// ---- QLearningAgent ---------------------------------------------------------

class QLearningAgent
{
public:
    // Constructs the agent with all hyperparameters.
    // RLEnvironment is held by reference — not owned, must outlive agent.
    QLearningAgent(RLEnvironment& environment,
                   double         learningRate,
                   double         discountFactor,
                   double         epsilonStart,
                   double         epsilonMin,
                   double         epsilonDecay);

    // ---- Training -----------------------------------------------------------

    // Run a single episode: reset env, loop step/update until done or maxSteps.
    // Returns a TrainingResult snapshot for this episode.
    // Called in a loop by train().
    TrainingResult runEpisode(int episodeNumber, int maxStepsPerEpisode);

    // Run numEpisodes episodes, collecting TrainingResult for each.
    // Returns the full training history — one entry per episode.
    // Use the returned vector to plot learning curves.
    std::vector<TrainingResult> train(int numEpisodes, int maxStepsPerEpisode);

    // ---- Policy extraction --------------------------------------------------

    // Extract the greedy policy as a path from start to goal.
    // Follows getBestAction() at each cell until goal or maxSteps exceeded.
    // Returns an empty vector if the goal is not reached within maxSteps.
    // Called after training to visualize the learned route.
    std::vector<Position> extractGreedyPath(int maxSteps) const;

    // ---- Accessors ----------------------------------------------------------

    const QTable& getQTable()  const;
    double        getEpsilon() const;

private:
    RLEnvironment& env_;
    QTable         qTable_;

    // Hyperparameters
    double learningRate_;    // alpha
    double discountFactor_;  // gamma
    double epsilon_;         // current exploration rate — decays each episode
    double epsilonMin_;      // floor: epsilon never drops below this
    double epsilonDecay_;    // multiplier applied to epsilon each episode (e.g. 0.995)

    // Random number generation for ε-greedy action selection.
    // mt19937 is the Mersenne Twister — high quality, fast PRNG.
    mutable std::mt19937                            randomEngine_;
    mutable std::uniform_real_distribution<double>  realDistribution_;   // [0.0, 1.0)
    mutable std::uniform_int_distribution<int>      actionDistribution_; // [0, 3]

    // Choose action by ε-greedy policy at the given position.
    // Rolls realDistribution_: if roll < epsilon_ → random action, else best action.
    Action selectAction(const Position& position) const;

    // Apply one Bellman update to Q(currentPosition, actionTaken).
    // Called immediately after every RLEnvironment::step() call.
    void updateQValue(const Position& currentPosition,
                      Action          actionTaken,
                      double          reward,
                      const Position& nextPosition);

    // Decay epsilon by one step: epsilon_ = max(epsilonMin_, epsilon_ * epsilonDecay_)
    void decayEpsilon();
};

#endif  // QLEARNING_AGENT_H
