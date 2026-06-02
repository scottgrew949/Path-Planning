// rl/RLEnvironment.cpp
#include "RLEnvironment.h"
#include <stdexcept>
#include <cmath>

// ---- Constructor ------------------------------------------------------------

RLEnvironment::RLEnvironment(Environment& environment)
    : env_(environment),
      currentPosition_(environment.getStart())
{}

// ---- Episode control --------------------------------------------------------

Position RLEnvironment::reset()
{
    currentPosition_ = env_.getStart();
    env_.reset();
    return currentPosition_;
    
    // CONCEPT — Why reset the grid overlays?
    //   Each episode should start with a clean visual slate.
    //   PATH and VISITED markings from the previous run would mislead
    //   the visualizer and corrupt any overlay-based state.
    //   env_.reset() clears those overlays without touching OBSTACLE/START/GOAL cells.
}

StepResult RLEnvironment::step(Action action)
{
    Position candidatePosition = applyAction(currentPosition_, action);
    
    if(env_.inBounds(candidatePosition) && !env_.isObstacle(candidatePosition))
    {
        Position previousPosition = currentPosition_;
        currentPosition_ = candidatePosition;
        if (currentPosition_ == env_.getGoal())
            return { currentPosition_, REWARD_GOAL, true };
        double bonus = shapingBonus(previousPosition, currentPosition_);
        return { currentPosition_, REWARD_STEP + bonus, false };
    }

    double wallBonus = shapingBonus(currentPosition_, currentPosition_);
    return { currentPosition_, REWARD_WALL + wallBonus, false };
    // CONCEPT — Why stay put on a wall hit?
    //   In a real robot, colliding with a wall is not "teleporting back to start."
    //   The robot stays where it is but loses time and energy.
    //   The -10.0 reward teaches the agent that wall-bumping is costly.
}

// ---- Accessors --------------------------------------------------------------

Position RLEnvironment::getCurrentPosition() const
{
    return currentPosition_;
}

int RLEnvironment::getWidth() const
{
    return env_.getWidth();
}

int RLEnvironment::getHeight() const
{
    return env_.getHeight();
}

Position RLEnvironment::getGoal() const
{
    return env_.getGoal();
}
bool RLEnvironment::isValid(const Position& position) const
{
    return env_.isValid(position);
}

// ---- Shaping ----------------------------------------------------------------

void RLEnvironment::enableShaping(double discountFactor)
{
    if (discountFactor < 0.0 || discountFactor > 1.0)
        throw std::invalid_argument("RLEnvironment::enableShaping — discountFactor must be in [0, 1]");
    shapingEnabled_  = true;
    shapingDiscount_ = discountFactor;
}

void RLEnvironment::disableShaping()
{
    shapingEnabled_ = false;
}

bool RLEnvironment::isShapingEnabled() const
{
    return shapingEnabled_;
}

double RLEnvironment::shapingBonus(const Position& fromPosition, const Position& toPosition) const
{
    if (!shapingEnabled_) return 0.0;

    // No movement (wall hit) → no shaping. Potential-based shaping F(s,s') is
    // defined for transitions; applying it to a no-op gives dist*(1-γ) > 0,
    // which would partially cancel the wall penalty and distort learning.
    if (fromPosition == toPosition) return 0.0;

    Position goal = env_.getGoal();
    double distFrom = std::abs(fromPosition.x - goal.x) + std::abs(fromPosition.y - goal.y);
    double distTo   = std::abs(toPosition.x   - goal.x) + std::abs(toPosition.y   - goal.y);

    // F(s, s') = γ·Φ(s') - Φ(s) = γ·(-distTo) - (-distFrom) = distFrom - γ·distTo
    return distFrom - shapingDiscount_ * distTo;
}

// ---- Private helpers --------------------------------------------------------

Position RLEnvironment::applyAction(const Position& fromPosition, Action action) const
{
    switch(action)
    {
        case Action::UP: return Position(fromPosition.x,     fromPosition.y - 1);
        case Action::DOWN: return Position(fromPosition.x,     fromPosition.y + 1);
        case Action::LEFT: return Position(fromPosition.x - 1, fromPosition.y    );
        case Action::RIGHT: return Position(fromPosition.x + 1, fromPosition.y    );
        default: throw std::invalid_argument("RLEnvironment::applyAction — unknown Action value");
    }
}
