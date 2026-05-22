// rl/RLEnvironment.cpp
#include "RLEnvironment.h"
#include <stdexcept>

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
        currentPosition_ = candidatePosition;
        if (currentPosition_ == env_.getGoal())
            return { currentPosition_, REWARD_GOAL, true  };
        return { currentPosition_, REWARD_STEP, false };
    }

    return { currentPosition_, REWARD_WALL, false };
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
