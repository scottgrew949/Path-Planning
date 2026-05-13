// core/Types.cpp
#include "Types.h"
#include "Position.h"

// ---- PathResult -------------------------------------------------------------

PathResult::PathResult()
    : algorithm(AlgorithmType::ASTAR),
      algorithmName(""),
      pathCost(0.0),
      elapsedMs(0.0),
      nodesExplored(0)
{
    // path is default-constructed as an empty vector<Position>
}

// ---- StepResult -------------------------------------------------------------

StepResult::StepResult()
    : newPosition(0, 0),
      reward(0.0),
      done(false)
{
}

StepResult::StepResult(Position newPosition, double reward, bool done)
    : newPosition(newPosition), reward(reward), done(done)
{}