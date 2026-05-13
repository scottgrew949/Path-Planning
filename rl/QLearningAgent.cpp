// rl/QLearningAgent.cpp
#include "QLearningAgent.h"
#include <algorithm>
#include <stdexcept>

// ---- TrainingResult constructor --------------------------------------------

TrainingResult::TrainingResult()
    : episodeNumber(0),
      totalReward(0.0),
      stepsToGoal(0),
      goalReached(false),
      epsilonAtEnd(0.0)
{
}

TrainingResult::TrainingResult(int episodeNumber, double totalReward, int stepsToGoal, bool goalReached, double epsilonAtEnd)
    : episodeNumber(episodeNumber),
      totalReward(totalReward),
      stepsToGoal(stepsToGoal),
      goalReached(goalReached),
      epsilonAtEnd(epsilonAtEnd)
{}

// ---- QLearningAgent constructor --------------------------------------------

QLearningAgent::QLearningAgent(RLEnvironment& environment,
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
    // Validate hyperparameters — bad values produce silent training failures
    // that are extremely hard to diagnose after the fact.
    if (learningRate_  <= 0.0 || learningRate_  >  1.0) throw std::invalid_argument("learningRate must be in (0, 1]");
    if (discountFactor_ < 0.0 || discountFactor_ > 1.0) throw std::invalid_argument("discountFactor must be in [0, 1]");
    if (epsilonStart    < 0.0 || epsilonStart    > 1.0) throw std::invalid_argument("epsilonStart must be in [0, 1]");
    if (epsilonMin_    <  0.0 || epsilonMin_     > 1.0) throw std::invalid_argument("epsilonMin must be in [0, 1]");
    if (epsilonDecay_  <= 0.0 || epsilonDecay_   > 1.0) throw std::invalid_argument("epsilonDecay must be in (0, 1]");
}

// ---- Training ---------------------------------------------------------------

TrainingResult QLearningAgent::runEpisode(int episodeNumber, int maxStepsPerEpisode)
{
    Position currentPosition = env_.reset();
    double totalReward = 0.0;
    int    stepCount   = 0;
    bool   goalReached = false;

    int step = 0; 
    while(step < maxStepsPerEpisode)
    {
        Action greedyAction = selectAction(currentPosition);
        StepResult stepResult = env_.step(greedyAction);
        updateQValue(currentPosition, greedyAction, stepResult.reward, stepResult.newPosition);
        totalReward += stepResult.reward;
        ++stepCount;
        currentPosition = stepResult.newPosition;
        if (stepResult.done)
        {
            goalReached = true;
            break;
        }
        ++step;
    }

    decayEpsilon();
    return TrainingResult(episodeNumber, totalReward, stepCount, goalReached, epsilon_);
    // CONCEPT — Why cap steps per episode?
    //   Without a cap, early episodes (random policy) may wander indefinitely.
    //   The cap acts as a timeout: "you had N chances, episode over."
    //   Typical value: width * height * 4  (enough to visit every cell multiple times)
}

std::vector<TrainingResult> QLearningAgent::train(int numEpisodes, int maxStepsPerEpisode)
{
    std::vector<TrainingResult> history;
    history.reserve(numEpisodes);

    for(int round = 1; round <= numEpisodes; ++round)
    {
        history.push_back(runEpisode(round, maxStepsPerEpisode));
    }

    return history;
    // CONCEPT — Why collect every episode?
    //   The history vector is the training log. Plotting totalReward per episode
    //   produces the "learning curve" — the most important diagnostic in RL.
    //   A rising reward curve confirms the agent is improving.
    //   A flat curve means the hyperparameters need tuning.
}

// ---- Policy extraction ------------------------------------------------------

std::vector<Position> QLearningAgent::extractGreedyPath(int maxSteps) const
{
    Position currentPosition = env_.getCurrentPosition();
    Position goal = env_.getGoal();
    std::vector<Position> greedyPath;
    greedyPath.push_back(currentPosition);
    
    for(int i = 1; i <= maxSteps; ++i)
    {
        if(currentPosition == goal) return greedyPath;

        Action bestAction = qTable_.getBestAction(currentPosition);

        int nextX = currentPosition.x;
        int nextY = currentPosition.y;

        switch (bestAction) {
            case Action::UP:    nextY -= 1; break;
            case Action::DOWN:  nextY += 1; break;
            case Action::LEFT:  nextX -= 1; break;
            case Action::RIGHT: nextX += 1; break;
        }

        Position nextPosition{nextX, nextY};

        if(nextPosition == env_.getGoal())
        {
            greedyPath.push_back(nextPosition);
            return greedyPath;
        }
        else if(!env_.isValid(nextPosition))
        {
            break;
        }
        else
        {
            greedyPath.push_back(nextPosition);
            currentPosition = nextPosition;
        }
    }
    
    return (greedyPath.back() == env_.getGoal()) ? greedyPath : std::vector<Position>{};
    // CONCEPT — Why not use env_.step() here?
    //   extractGreedyPath is const — it must not modify the environment state.
    //   step() would move the agent and change currentPosition_ inside env_.
    //   We read-only follow the policy instead.
}

// ---- Accessors --------------------------------------------------------------

const QTable& QLearningAgent::getQTable() const
{
    return qTable_;
}

double QLearningAgent::getEpsilon() const
{
    return epsilon_;
}

// ---- Private helpers --------------------------------------------------------

Action QLearningAgent::selectAction(const Position& position) const
{
    double roll = realDistribution_(randomEngine_);
    
    if(roll < epsilon_)
    {
        return static_cast<Action>(actionDistribution_(randomEngine_));
    }
    
    return qTable_.getBestAction(position);
    // TODO: if roll < epsilon_:
    //   return static_cast<Action>(actionDistribution_(randomEngine_))
    //   — random action: explore
    // TODO: else:
    //   return qTable_.getBestAction(position)
    //   — greedy action: exploit best known move
    
    // CONCEPT — The explore/exploit tradeoff is the central tension of RL.
    //   Too much exploration: agent never converges on a good policy.
    //   Too much exploitation: agent gets stuck in a locally good but globally bad policy.
    //   ε decay balances both: explore early, exploit late.
}

void QLearningAgent::updateQValue(const Position& currentPosition,
                                  Action          actionTaken,
                                  double          reward,
                                  const Position& nextPosition)
{
    double currentQValue = qTable_.getValue(currentPosition, actionTaken);
    double bestNextQValue = qTable_.getMaxValue(nextPosition);
    double bellmanTarget = reward + discountFactor_ * bestNextQValue;
    double newQValue = currentQValue + learningRate_ * (bellmanTarget - currentQValue);
    qTable_.setValue(currentPosition, actionTaken, newQValue);

    // CONCEPT — The Bellman equation in plain english:
    //   "The quality of (position, action) equals:
    //    the immediate reward I got, plus
    //    how good the best move from where I landed is (discounted by gamma),
    //    blended toward my old estimate at rate alpha."
    //
    //   Each update nudges the estimate a little — not a full replacement.
    //   Over thousands of episodes the estimates stabilise at the true optimal values.
}

void QLearningAgent::decayEpsilon()
{
    epsilon_ = std::max(epsilonMin_, epsilon_ * epsilonDecay_);
    
    // CONCEPT — Why max with epsilonMin_?
    //   Without a floor, epsilon_ would eventually reach 0.0 and the agent
    //   would stop exploring entirely. A small floor (e.g. 0.05) keeps 5%
    //   of moves random — the agent can still discover better routes found
    //   late in training, and handles any environment changes gracefully.
}
