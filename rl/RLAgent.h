// rl/RLAgent.h
// Abstract base class for tabular reinforcement learning agents.
//
// CONCEPT — Why a base class?
//   QLearningAgent and DynaQAgent share the same hyperparameters, Q-table,
//   PRNG state, and helper methods (selectAction, updateQValue, decayEpsilon,
//   train, extractGreedyPath). The only difference between them is runEpisode():
//   Q-learning updates Q after each real step; Dyna-Q also does planning sweeps
//   after each real step using a learned model. The base class owns everything
//   common; each derived class overrides only runEpisode().
//
// CONCEPT — Ownership rule (I-1)
//   RLEnvironment is held by reference, never owned. The caller allocates it;
//   the agent only borrows it. This mirrors the OpenAI Gym convention where the
//   environment is external to the agent.
#ifndef RL_AGENT_H
#define RL_AGENT_H

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
    TrainingResult(int episodeNumber, double totalReward, int stepsToGoal,
                   bool goalReached, double epsilonAtEnd);
};

// ---- RLAgent ----------------------------------------------------------------

class RLAgent
{
public:
    // Constructs the base with all hyperparameters.
    // RLEnvironment is held by reference — not owned, must outlive agent.
    // Throws std::invalid_argument if any hyperparameter is out of range.
    RLAgent(RLEnvironment& environment,
            double         learningRate,
            double         discountFactor,
            double         epsilonStart,
            double         epsilonMin,
            double         epsilonDecay);

    // Virtual destructor — required for correct polymorphic deletion.
    virtual ~RLAgent() = default;

    // ---- Training -----------------------------------------------------------

    // Run a single episode: reset env, loop step/update until done or maxSteps.
    // Returns a TrainingResult snapshot for this episode.
    // Pure virtual — each derived agent implements its own episode logic.
    virtual TrainingResult runEpisode(int episodeNumber, int maxStepsPerEpisode) = 0;

    // Run numEpisodes episodes, collecting TrainingResult for each.
    // Returns the full training history — one entry per episode.
    std::vector<TrainingResult> train(int numEpisodes, int maxStepsPerEpisode);

    // ---- Policy extraction --------------------------------------------------

    // Extract the greedy policy as a path from start to goal.
    // Follows getBestAction() at each cell — never calls env_.step().
    // Returns empty vector if the goal is not reached within maxSteps.
    std::vector<Position> extractGreedyPath(int maxSteps) const;

    // ---- Accessors ----------------------------------------------------------

    const QTable& getQTable()  const;
    double        getEpsilon() const;

protected:
    RLEnvironment& env_;     // borrowed reference — not owned (I-1)
    QTable         qTable_;  // owned by base class as value member (I-4)

    // Hyperparameters
    double learningRate_;    // alpha
    double discountFactor_;  // gamma
    double epsilon_;         // current exploration rate — decays each episode
    double epsilonMin_;      // floor: epsilon never drops below this
    double epsilonDecay_;    // multiplier applied to epsilon each episode

    // Random number generation for epsilon-greedy action selection.
    // mt19937 is the Mersenne Twister — high quality, fast PRNG.
    mutable std::mt19937                           randomEngine_;
    mutable std::uniform_real_distribution<double> realDistribution_;   // [0.0, 1.0)
    mutable std::uniform_int_distribution<int>     actionDistribution_; // [0, 3]

    // Choose action by epsilon-greedy policy at the given position.
    Action selectAction(const Position& position) const;

    // Apply one Bellman update to Q(currentPosition, actionTaken).
    // done: true when nextPosition is the terminal goal state — zeroes future Q.
    void updateQValue(const Position& currentPosition,
                      Action          actionTaken,
                      double          reward,
                      const Position& nextPosition,
                      bool            done);

    // Decay epsilon by one step: epsilon_ = max(epsilonMin_, epsilon_ * epsilonDecay_)
    void decayEpsilon();
};

#endif  // RL_AGENT_H
