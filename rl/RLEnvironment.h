// rl/RLEnvironment.h
// Gym-style wrapper around Environment for reinforcement learning.
//
// CONCEPT — What is a "gym-style" interface?
//   In RL research, every environment exposes two functions:
//     reset() — start a new episode, return the initial state
//     step(action) — apply one action, return (new state, reward, done)
//   This mirrors the OpenAI Gym standard used in Python RL.
//   Our RLEnvironment wraps the existing grid Environment with this interface.
//
// CONCEPT — What is an episode?
//   One full run from start to goal (or until the agent gives up).
//   The agent may take thousands of episodes to learn a good policy.
//   Each episode resets the agent to start — like restarting a video game level.
//
// CONCEPT — Reward signal
//   The only feedback the agent gets. Design:
//     +100.0  reaching the goal  — strong positive signal, this is the objective
//      -1.0   each valid step    — penalises long paths, agent learns to be efficient
//     -10.0   hitting a wall     — penalises wasted moves into obstacles
//
// CONCEPT — Potential-based reward shaping (Ng et al., 1999)
//   Sparse rewards (+100 only at goal) make early training slow — the agent
//   wanders randomly until it accidentally reaches the goal.
//   Shaping adds a small directional bonus F(s, s') = γ·Φ(s') - Φ(s)
//   where Φ(s) = -manhattan_distance(s, goal) is the potential function.
//   Moving closer to the goal gives a small positive bonus.
//   Moving away gives a small penalty.
//   Ng et al. proved that any potential-based shaping term leaves the
//   OPTIMAL POLICY UNCHANGED — the agent will still find the shortest path.
//   It just finds it faster because the reward landscape is less flat.
//
// Self-driving car analog:
//   RLEnvironment = the simulator. The car (agent) sends steering commands,
//   the simulator moves the car and reports speed, position, crash status.
//
// Design note:
//   RLEnvironment does NOT inherit from IPathfinder — it is a completely
//   different interface. Classical planners compute a full path at once.
//   RL environments process one step at a time.
#ifndef RL_ENVIRONMENT_H
#define RL_ENVIRONMENT_H

#include "../core/Position.h"
#include "../core/Types.h"
#include "../environment/Environment.h"

class RLEnvironment
{
public:
    // Binds to an existing Environment — does not copy the grid.
    // The Environment must outlive this RLEnvironment.
    explicit RLEnvironment(Environment& environment);

    // ---- Episode control ----------------------------------------------------

    // Start a new episode: clear visited overlays, return start position.
    // Must be called before the first step of every episode.
    Position reset();

    // Apply one action from the current position.
    //   Valid move  → move there, reward = REWARD_STEP (or REWARD_GOAL if done)
    //   Wall/bounds → stay put,   reward = REWARD_WALL
    // Returns {newPosition, reward, done}.
    StepResult step(Action action);

    // ---- Accessors ----------------------------------------------------------

    Position getCurrentPosition() const;
    int      getWidth()           const;
    int      getHeight()          const;
    Position getGoal()             const;
    bool     isValid(const Position& position) const;

    // ---- Reward shaping -----------------------------------------------------

    // Enable potential-based shaping using Φ(s) = -manhattan_distance(s, goal).
    // discountFactor must match the agent's γ — see CONCEPT above for why.
    // Throws std::invalid_argument if discountFactor is not in [0, 1].
    void enableShaping(double discountFactor);

    // Disable shaping — step() returns base rewards only.
    void disableShaping();

    bool isShapingEnabled() const;

private:
    Environment& env_;             // grid — not owned, must outlive this object
    Position     currentPosition_; // agent's current cell in the active episode

    bool   shapingEnabled_  = false;
    double shapingDiscount_ = 0.0;  // γ used in F(s,s') = γ·Φ(s') - Φ(s)

    // Reward constants — see CONCEPT note above
    static constexpr double REWARD_GOAL = 100.0;
    static constexpr double REWARD_STEP =  -1.0;
    static constexpr double REWARD_WALL = -10.0;

    // Translate an Action into a candidate Position one step away.
    // Does not validate whether the position is in-bounds or obstacle-free.
    Position applyAction(const Position& fromPosition, Action action) const;

    // Compute shaping bonus F(s, s') = dist(s,goal) - γ·dist(s',goal).
    // Returns 0.0 if shaping is disabled.
    double shapingBonus(const Position& fromPosition, const Position& toPosition) const;
};

#endif  // RL_ENVIRONMENT_H
