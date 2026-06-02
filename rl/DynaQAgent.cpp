// rl/DynaQAgent.cpp
#include "DynaQAgent.h"
#include <stdexcept>
#include <cmath>

// ---- DynaQAgent constructor -------------------------------------------------

DynaQAgent::DynaQAgent(RLEnvironment& environment,
                       double         learningRate,
                       double         discountFactor,
                       double         epsilonStart,
                       double         epsilonMin,
                       double         epsilonDecay,
                       int            planningSteps,
                       double         priorityThreshold)
    : RLAgent(environment, learningRate, discountFactor,
              epsilonStart, epsilonMin, epsilonDecay),
      planningSteps_(planningSteps),
      priorityThreshold_(priorityThreshold)
{
    if (planningSteps < 0)
        throw std::invalid_argument("DynaQAgent: planningSteps must be >= 0");
    if (priorityThreshold < 0.0)
        throw std::invalid_argument("DynaQAgent: priorityThreshold must be >= 0");
}

// ---- runEpisode -------------------------------------------------------------

TrainingResult DynaQAgent::runEpisode(int episodeNumber, int maxStepsPerEpisode)
{
    Position currentPosition = env_.reset();
    double   totalReward     = 0.0;
    int      stepCount       = 0;
    bool     goalReached     = false;

    for (int step = 0; step < maxStepsPerEpisode; ++step)
    {
        // 1. Choose action by epsilon-greedy policy.
        Action     action     = selectAction(currentPosition);
        StepResult stepResult = env_.step(action);

        // 2. Compute Bellman error BEFORE the update so the priority reflects
        //    true pre-update surprise — the property that makes prioritized
        //    sweeping work. Enqueue first, then apply the update.
        enqueueIfSignificant(currentPosition, action,
                             stepResult.reward, stepResult.newPosition,
                             stepResult.done);

        // 3. Direct RL update.
        updateQValue(currentPosition, action, stepResult.reward,
                     stepResult.newPosition, stepResult.done);

        totalReward += stepResult.reward;
        ++stepCount;

        // 4. Update world model with this real transition — now includes done flag.
        ModelKey modelKey{currentPosition, action};
        model_[modelKey] = ModelTransition{stepResult.reward,
                                           stepResult.newPosition,
                                           stepResult.done};

        currentPosition = stepResult.newPosition;

        // 5. Prioritized planning sweep.
        planningUpdate();

        if (stepResult.done)
        {
            goalReached = true;
            break;
        }
    }

    decayEpsilon();
    return TrainingResult(episodeNumber, totalReward, stepCount, goalReached, epsilon_);
}

// ---- Private helpers --------------------------------------------------------

double DynaQAgent::computeBellmanError(const Position& position,
                                        Action          action,
                                        double          reward,
                                        const Position& nextPosition,
                                        bool            done) const
{
    // CONCEPT — Bellman error without writing:
    //   Mirrors updateQValue arithmetic exactly — including the done flag which
    //   zeroes the future-value term for terminal transitions. Without done,
    //   goal transitions would be over-prioritised by γ·maxQ(goal).
    double currentQValue  = qTable_.getValue(position, action);
    double bestNextQValue = done ? 0.0 : qTable_.getMaxValue(nextPosition);
    double bellmanTarget  = reward + discountFactor_ * bestNextQValue;
    return std::abs(bellmanTarget - currentQValue);
}

void DynaQAgent::enqueueIfSignificant(const Position& position,
                                       Action          action,
                                       double          reward,
                                       const Position& nextPosition,
                                       bool            done)
{
    double bellmanError = computeBellmanError(position, action, reward, nextPosition, done);
    if (bellmanError > priorityThreshold_)
        prioritizedTransitions_.push({bellmanError, {position, action}});
}

void DynaQAgent::planningUpdate()
{
    // CONCEPT — Cascading value propagation:
    //   After updating Q(s, a), predecessors of s may now have significant
    //   Bellman error. We scan modelKeys_ for predecessors and re-enqueue them.
    //   This cascades value backwards through the model automatically.

    if (planningSteps_ <= 0 || prioritizedTransitions_.empty()) return;

    for (int sweep = 0; sweep < planningSteps_ && !prioritizedTransitions_.empty(); ++sweep)
    {
        PrioritizedTransition top = prioritizedTransitions_.top();
        prioritizedTransitions_.pop();

        if (auto modelIterator = model_.find(top.key); modelIterator == model_.end()) continue;
        else
        {
            const Position&        sampledPosition = top.key.first;
            Action                 sampledAction   = top.key.second;
            const ModelTransition& transition      = modelIterator->second;

            updateQValue(sampledPosition, sampledAction,
                         transition.reward, transition.nextPosition,
                         transition.terminalTransition);

            for (const auto& [predecessorKey, predecessorTransition] : model_)
            {
                if (!(predecessorTransition.nextPosition == sampledPosition)) continue;
                double predecessorError = computeBellmanError(
                    predecessorKey.first,
                    predecessorKey.second,
                    predecessorTransition.reward,
                    predecessorTransition.nextPosition,
                    predecessorTransition.terminalTransition
                );
                if (predecessorError > priorityThreshold_)
                    prioritizedTransitions_.push({predecessorError, predecessorKey});
            }
        }
    }
}
