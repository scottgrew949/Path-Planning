// rl/DynaQAgent.cpp
#include "DynaQAgent.h"

// ---- DynaQAgent constructor -------------------------------------------------

DynaQAgent::DynaQAgent(RLEnvironment& environment,
                       double         learningRate,
                       double         discountFactor,
                       double         epsilonStart,
                       double         epsilonMin,
                       double         epsilonDecay,
                       int            planningSteps)
    : RLAgent(environment, learningRate, discountFactor,
              epsilonStart, epsilonMin, epsilonDecay),
      planningSteps_(planningSteps),
      keyDist_(0, 0)
{
    // model_ and modelKeys_ start empty — populated on first real transitions.
    // No reserve/pre-allocation per spec: let them grow naturally.
    // keyDist_ range is updated lazily in planningUpdate() as model grows.
}

// ---- runEpisode -------------------------------------------------------------

TrainingResult DynaQAgent::runEpisode(int episodeNumber, int maxStepsPerEpisode)
{
    // I-2: env_.reset() called exactly once per episode at start of runEpisode().
    Position currentPosition = env_.reset();
    double totalReward = 0.0;
    int    stepCount   = 0;
    bool   goalReached = false;

    for (int step = 0; step < maxStepsPerEpisode; ++step)
    {
        // 1. Choose action by epsilon-greedy policy.
        Action     action     = selectAction(currentPosition);
        StepResult stepResult = env_.step(action);

        // 2. Direct RL update — same Bellman update as Q-learning.
        updateQValue(currentPosition, action, stepResult.reward, stepResult.newPosition, stepResult.done);

        totalReward     += stepResult.reward;
        ++stepCount;

        // 3. Update world model with this real transition.
        //    If this (state, action) pair is new, push key to mirror vector.
        ModelKey key{currentPosition, action};
        bool isNewKey = (model_.find(key) == model_.end());
        model_[key] = ModelValue{stepResult.reward, stepResult.newPosition};
        if (isNewKey)
        {
            modelKeys_.push_back(key);
        }

        currentPosition = stepResult.newPosition;

        // 4. Planning sweep — n imagined Bellman updates from model.
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

// ---- planningUpdate ---------------------------------------------------------

void DynaQAgent::planningUpdate()
{
    // Guard: skip entirely when no planning or model is empty (spec requirement).
    if (planningSteps_ <= 0 || modelKeys_.empty()) return;

    // Update cached distribution range to match current model size.
    keyDist_.param(std::uniform_int_distribution<int>::param_type(
        0, static_cast<int>(modelKeys_.size()) - 1));

    for (int i = 0; i < planningSteps_; ++i)
    {
        // Sample a random previously-seen (state, action) pair.
        int idx = keyDist_(randomEngine_);
        const ModelKey&   key   = modelKeys_[idx];
        const ModelValue& value = model_.at(key);

        const Position& sampledPos    = key.first;
        Action          sampledAction = key.second;
        double          sampledReward = value.first;
        const Position& sampledNext   = value.second;

        // Apply Bellman update using the imagined transition.
        // Planning replays never carry terminal status — model doesn't store done.
        updateQValue(sampledPos, sampledAction, sampledReward, sampledNext, false);
    }

    // CONCEPT — Why does this work?
    //   The model stores deterministic transitions (last observed outcome for each
    //   state-action pair). Replaying them is equivalent to re-experiencing those
    //   moments — the Q-values converge faster because each real step triggers
    //   planningSteps_ additional updates, not just one.
}
