// rl/QLearningAgent.cpp
#include "QLearningAgent.h"

// ---- QLearningAgent constructor --------------------------------------------

QLearningAgent::QLearningAgent(RLEnvironment& environment,
                               double         learningRate,
                               double         discountFactor,
                               double         epsilonStart,
                               double         epsilonMin,
                               double         epsilonDecay)
    : RLAgent(environment, learningRate, discountFactor,
              epsilonStart, epsilonMin, epsilonDecay)
{
}

// ---- runEpisode -------------------------------------------------------------

TrainingResult QLearningAgent::runEpisode(int episodeNumber, int maxStepsPerEpisode)
{
    // I-2: env_.reset() called exactly once per episode at start of runEpisode().
    Position currentPosition = env_.reset();
    double totalReward = 0.0;
    int    stepCount   = 0;
    bool   goalReached = false;

    for (int step = 0; step < maxStepsPerEpisode; ++step)
    {
        Action     action     = selectAction(currentPosition);
        StepResult stepResult = env_.step(action);

        updateQValue(currentPosition, action, stepResult.reward, stepResult.newPosition);

        totalReward      += stepResult.reward;
        ++stepCount;
        currentPosition   = stepResult.newPosition;

        if (stepResult.done)
        {
            goalReached = true;
            break;
        }
    }

    decayEpsilon();
    return TrainingResult(episodeNumber, totalReward, stepCount, goalReached, epsilon_);

    // CONCEPT — Why cap steps per episode?
    //   Without a cap, early episodes (random policy) may wander indefinitely.
    //   The cap acts as a timeout: "you had N chances, episode over."
}
