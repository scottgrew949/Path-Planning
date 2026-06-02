// rl/TDLambdaAgent.cpp
#include "TDLambdaAgent.h"
#include <stdexcept>
#include <cmath>

// ---- Constructor ------------------------------------------------------------

TDLambdaAgent::TDLambdaAgent(RLEnvironment& environment,
                               double         learningRate,
                               double         discountFactor,
                               double         epsilonStart,
                               double         epsilonMin,
                               double         epsilonDecay,
                               double         lambda)
    : RLAgent(environment, learningRate, discountFactor,
              epsilonStart, epsilonMin, epsilonDecay),
      lambda_(lambda)
{
    if (lambda < 0.0 || lambda > 1.0)
        throw std::invalid_argument("TDLambdaAgent: lambda must be in [0, 1]");
}

// ---- Accessors --------------------------------------------------------------

double TDLambdaAgent::getLambda() const
{
    return lambda_;
}

// ---- Episode ----------------------------------------------------------------

TrainingResult TDLambdaAgent::runEpisode(int episodeNumber, int maxStepsPerEpisode)
{
    // CONCEPT — Trace lifecycle:
    //   Traces are episode-scoped. Clear at start so no credit leaks from the
    //   previous episode. New episode = fresh slate of visited (state, action) pairs.
    clearTraces();

    Position currentPosition = env_.reset();
    double   totalReward     = 0.0;
    int      stepCount       = 0;
    bool     goalReached     = false;

    for (int step = 0; step < maxStepsPerEpisode; ++step)
    {
        Action     action     = selectAction(currentPosition);
        StepResult stepResult = env_.step(action);

        // CONCEPT — TD error (δ):
        //   δ = reward + γ·maxQ(s') - Q(s, a)
        //   This is the "surprise" signal — how wrong was our current Q estimate?
        //   Positive δ: reward was better than expected → increase Q for all traced pairs.
        //   Negative δ: reward was worse than expected → decrease Q for all traced pairs.
        //   δ is computed BEFORE any Q-update so it reflects the current estimate.
        double currentQValue  = qTable_.getValue(currentPosition, action);
        double bestNextQValue = stepResult.done
                                ? 0.0
                                : qTable_.getMaxValue(stepResult.newPosition);
        double tdError = stepResult.reward + discountFactor_ * bestNextQValue - currentQValue;

        // Increment trace for the visited (state, action) pair, then propagate.
        incrementTrace(currentPosition, action);
        applyTracesAndDecay(tdError);

        totalReward     += stepResult.reward;
        ++stepCount;
        currentPosition  = stepResult.newPosition;

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

void TDLambdaAgent::incrementTrace(const Position& position, Action action)
{
    eligibilityTraces_[position][static_cast<int>(action)] += 1.0;
}

void TDLambdaAgent::applyTracesAndDecay(double tdError)
{
    // CONCEPT — Simultaneous update and decay:
    //   For every (s, a) pair with a non-zero trace:
    //     Q(s,a) += α · δ · e(s,a)    ← Q-value update weighted by trace
    //     e(s,a) *= γ · λ              ← trace decays toward zero
    //   The decay factor γλ controls how fast traces fade:
    //     γ  discounts future rewards (standard RL discount)
    //     λ  is the additional trace decay — λ=0 kills the trace immediately,
    //        λ=1 decays only by the discount factor (very long memory)

    double decayFactor = discountFactor_ * lambda_;

    auto traceIterator = eligibilityTraces_.begin();
    while (traceIterator != eligibilityTraces_.end())
    {
        const Position& tracedPosition = traceIterator->first;
        std::array<double, 4>& traces  = traceIterator->second;

        bool anyAboveThreshold = false;
        for (int actionIndex = 0; actionIndex < 4; ++actionIndex)
        {
            if (std::abs(traces[actionIndex]) < TRACE_PRUNE_THRESHOLD) continue;

            Action tracedAction = static_cast<Action>(actionIndex);
            double currentValue = qTable_.getValue(tracedPosition, tracedAction);
            qTable_.setValue(tracedPosition, tracedAction,
                             currentValue + learningRate_ * tdError * traces[actionIndex]);

            traces[actionIndex] *= decayFactor;
            if (std::abs(traces[actionIndex]) >= TRACE_PRUNE_THRESHOLD)
                anyAboveThreshold = true;
        }

        if (!anyAboveThreshold)
            traceIterator = eligibilityTraces_.erase(traceIterator);
        else
            ++traceIterator;
    }
}

void TDLambdaAgent::clearTraces()
{
    eligibilityTraces_.clear();
}
