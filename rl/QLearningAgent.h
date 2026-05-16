// rl/QLearningAgent.h
// Tabular Q-learning agent — concrete RLAgent that learns via the Bellman update.
//
// CONCEPT — How does Q-learning work?
//   After every real environment step the agent applies one Bellman update:
//     Q(s,a) += alpha * [ reward + gamma * maxQ(s') - Q(s,a) ]
//   Over thousands of episodes the Q-values converge to the optimal policy.
//   No model of the environment is needed — updates come from real experience only.
//
// CONCEPT — How does this differ from Dyna-Q?
//   Q-learning uses real transitions exclusively.
//   Dyna-Q adds a "planning" phase: after each real step it also replays n
//   previously-seen transitions from an internal model, getting n extra updates
//   for free. This makes Dyna-Q dramatically more sample-efficient.
//
// CONCEPT — epsilon-greedy policy
//   Every step the agent flips a biased coin:
//     With probability epsilon   -> pick a RANDOM action (explore)
//     With probability 1-epsilon -> pick the BEST known action (exploit)
//   epsilon decays each episode toward epsilon_min.
#ifndef QLEARNING_AGENT_H
#define QLEARNING_AGENT_H

#include "RLAgent.h"

// ---- QLearningAgent ---------------------------------------------------------
// Only adds runEpisode() — all other logic lives in RLAgent.

class QLearningAgent : public RLAgent
{
public:
    // Delegates entirely to RLAgent constructor.
    QLearningAgent(RLEnvironment& environment,
                   double         learningRate,
                   double         discountFactor,
                   double         epsilonStart,
                   double         epsilonMin,
                   double         epsilonDecay);

    // Run one episode: reset env (I-2), step until done or maxSteps, update Q.
    TrainingResult runEpisode(int episodeNumber, int maxStepsPerEpisode) override;
};

#endif  // QLEARNING_AGENT_H
