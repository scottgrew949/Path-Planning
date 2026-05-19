// rl/RLAgent.cpp
#include "RLAgent.h"
#include <algorithm>
#include <stdexcept>
#include <unordered_set>
#include <array>
#include <utility>

// ---- TrainingResult constructors --------------------------------------------

TrainingResult::TrainingResult()
    : episodeNumber(0),
      totalReward(0.0),
      stepsToGoal(0),
      goalReached(false),
      epsilonAtEnd(0.0)
{
}

TrainingResult::TrainingResult(int episodeNumber, double totalReward,
                               int stepsToGoal, bool goalReached,
                               double epsilonAtEnd)
    : episodeNumber(episodeNumber),
      totalReward(totalReward),
      stepsToGoal(stepsToGoal),
      goalReached(goalReached),
      epsilonAtEnd(epsilonAtEnd)
{
}

// ---- RLAgent constructor ----------------------------------------------------

RLAgent::RLAgent(RLEnvironment& environment,
                 double         learningRate,
                 double         discountFactor,
                 double         epsilonStart,
                 double         epsilonMin,
                 double         epsilonDecay)
    : env_(environment),
      qTable_(environment.getWidth(), environment.getHeight()),
      learningRate_(learningRate),
      discountFactor_(discountFactor),
      epsilon_(epsilonStart),
      epsilonMin_(epsilonMin),
      epsilonDecay_(epsilonDecay),
      randomEngine_(std::random_device{}()),
      realDistribution_(0.0, 1.0),
      actionDistribution_(0, 3)
{
    // Validate hyperparameters (I-5) — bad values produce silent failures.
    if (learningRate_   <= 0.0 || learningRate_   >  1.0) throw std::invalid_argument("learningRate must be in (0, 1]");
    if (discountFactor_ <  0.0 || discountFactor_ >  1.0) throw std::invalid_argument("discountFactor must be in [0, 1]");
    if (epsilonStart    <  0.0 || epsilonStart    >  1.0) throw std::invalid_argument("epsilonStart must be in [0, 1]");
    if (epsilonMin_     <  0.0 || epsilonMin_     >  1.0) throw std::invalid_argument("epsilonMin must be in [0, 1]");
    if (epsilonDecay_   <= 0.0 || epsilonDecay_   >  1.0) throw std::invalid_argument("epsilonDecay must be in (0, 1]");
}

// ---- Training ---------------------------------------------------------------

std::vector<TrainingResult> RLAgent::train(int numEpisodes, int maxStepsPerEpisode)
{
    std::vector<TrainingResult> history;
    history.reserve(numEpisodes);

    for (int round = 1; round <= numEpisodes; ++round)
    {
        history.push_back(runEpisode(round, maxStepsPerEpisode));
    }

    return history;
    // CONCEPT — Why collect every episode?
    //   The history vector is the training log. Plotting totalReward per episode
    //   produces the "learning curve" — the primary diagnostic in RL.
}

// ---- Policy extraction ------------------------------------------------------

// Helper: compute next position one step in the given direction.
static Position stepInDirection(const Position& fromPosition, Action action)
{
    int nextX = fromPosition.x;
    int nextY = fromPosition.y;
    switch (action) {
        case Action::UP:    nextY -= 1; break;
        case Action::DOWN:  nextY += 1; break;
        case Action::LEFT:  nextX -= 1; break;
        case Action::RIGHT: nextX += 1; break;
    }
    return Position{nextX, nextY};
}

std::vector<Position> RLAgent::extractGreedyPath(int maxSteps) const
{
    Position currentPosition = env_.getCurrentPosition();
    Position goal             = env_.getGoal();
    std::vector<Position> greedyPath;
    greedyPath.push_back(currentPosition);

    // Track visited cells to detect cycles — a cycle means the policy looped.
    std::unordered_set<Position, PositionHash> visited;
    visited.insert(currentPosition);

    for (int stepIndex = 1; stepIndex <= maxSteps; ++stepIndex)
    {
        if (currentPosition == goal) return greedyPath;

        // Rank all 4 actions by their Q-value at this cell, descending.
        // With curriculum learning the Q-table contains values from multiple
        // maze layouts. The top-ranked valid action is the best the policy knows.
        // Falling back to lower-ranked actions handles stale values that point
        // at walls that exist in the current maze but not in an earlier stage.
        std::array<std::pair<double, Action>, 4> rankedActions = {{
            {qTable_.getValue(currentPosition, Action::UP),    Action::UP},
            {qTable_.getValue(currentPosition, Action::DOWN),  Action::DOWN},
            {qTable_.getValue(currentPosition, Action::LEFT),  Action::LEFT},
            {qTable_.getValue(currentPosition, Action::RIGHT), Action::RIGHT}
        }};
        std::sort(rankedActions.begin(), rankedActions.end(),
                  [](const std::pair<double, Action>& firstPair,
                     const std::pair<double, Action>& secondPair) {
                      return firstPair.first > secondPair.first;
                  });

        bool movedThisStep = false;
        for (const std::pair<double, Action>& rankedAction : rankedActions)
        {
            Position candidatePosition = stepInDirection(currentPosition, rankedAction.second);

            if (candidatePosition == goal)
            {
                greedyPath.push_back(candidatePosition);
                return greedyPath;
            }

            if (env_.isValid(candidatePosition) && visited.count(candidatePosition) == 0)
            {
                visited.insert(candidatePosition);
                greedyPath.push_back(candidatePosition);
                currentPosition = candidatePosition;
                movedThisStep   = true;
                break;
            }
        }

        if (!movedThisStep) break; // surrounded or all neighbours visited — no path
    }

    return {};
}

// ---- Accessors --------------------------------------------------------------

const QTable& RLAgent::getQTable() const
{
    return qTable_;
}

double RLAgent::getEpsilon() const
{
    return epsilon_;
}

// ---- Protected helpers ------------------------------------------------------

Action RLAgent::selectAction(const Position& position) const
{
    double roll = realDistribution_(randomEngine_);

    if (roll < epsilon_)
    {
        return static_cast<Action>(actionDistribution_(randomEngine_));
    }

    return qTable_.getBestAction(position);
}

void RLAgent::updateQValue(const Position& currentPosition,
                            Action          actionTaken,
                            double          reward,
                            const Position& nextPosition)
{
    double currentQValue  = qTable_.getValue(currentPosition, actionTaken);
    double bestNextQValue = qTable_.getMaxValue(nextPosition);
    double bellmanTarget  = reward + discountFactor_ * bestNextQValue;
    double newQValue      = currentQValue + learningRate_ * (bellmanTarget - currentQValue);
    qTable_.setValue(currentPosition, actionTaken, newQValue);
}

void RLAgent::decayEpsilon()
{
    epsilon_ = std::max(epsilonMin_, epsilon_ * epsilonDecay_);
}
