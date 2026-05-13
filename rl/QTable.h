// rl/QTable.h
// Tabular Q-learning state-action value store.
//
// CONCEPT — What is a Q-table?
//   A lookup table: every (state, action) pair maps to a quality score.
//   "State" here is a grid cell (Position).
//   "Action" is one of the four moves (UP, DOWN, LEFT, RIGHT — enum values 0-3).
//   Q(s, a) answers: "if I am at cell s and take action a, how much
//   total future reward should I expect?"
//
//   At the start of training every Q-value is 0.0 — the agent knows nothing.
//   After thousands of episodes the values converge:
//     cells near the goal get high Q-values for the action that points toward it.
//     cells near walls get low (negative) Q-values for the action that hits them.
//
// CONCEPT — Why array<double,4> per cell?
//   Action is an enum class with underlying int values 0-3.
//   Storing one slot per action per cell lets us index directly:
//     table[position][static_cast<int>(Action::UP)]
//   This is O(1) lookup — no searching, no hashing on the action dimension.
//
// CONCEPT — Bellman update (called from QLearningAgent, not here)
//   Q(s,a) += alpha * [ reward + gamma * max_a'(Q(s',a')) - Q(s,a) ]
//     alpha  — learning rate:   how much each new experience overwrites old knowledge
//     gamma  — discount factor: how much future rewards are worth vs immediate ones
//     max_a' — greedy action:   best known move from the next state
//   QTable exposes the raw read/write needed for this equation.
//
// Self-driving car analog:
//   The Q-table is the car's "experience memory." After millions of miles
//   it knows: "at this intersection, turning right toward the highway is best."
//
// Storage complexity:
//   width × height × 4 doubles.
//   For a 41×41 grid: 41 × 41 × 4 × 8 bytes ≈ 54 KB — trivially small.
#ifndef QTABLE_H
#define QTABLE_H

#include <array>
#include <unordered_map>
#include "../core/Position.h"
#include "../core/Types.h"

class QTable
{
public:
    // Constructs a table for a grid of the given dimensions.
    // All Q-values initialised to 0.0.
    QTable(int width, int height);

    // ---- Core Q-value access ------------------------------------------------

    // Returns Q(position, action). Returns 0.0 for any unseen state.
    double getValue(const Position& position, Action action) const;

    // Writes Q(position, action) = value.
    // Called by QLearningAgent after each Bellman update.
    void setValue(const Position& position, Action action, double value);

    // ---- Policy helpers -----------------------------------------------------

    // Returns the action with the highest Q-value at this position.
    // Ties broken by lowest action enum value (deterministic).
    // Used by the greedy (exploit) branch of ε-greedy policy.
    Action getBestAction(const Position& position) const;

    // Returns the maximum Q-value across all actions at this position.
    // This is the max_a'(Q(s',a')) term in the Bellman equation.
    double getMaxValue(const Position& position) const;

    // ---- Diagnostics --------------------------------------------------------

    // Resets all Q-values to 0.0. Useful for repeating training from scratch.
    void reset();

private:
    int width_;
    int height_;

    // Flat map: Position → array of 4 Q-values (one per Action).
    // unordered_map chosen for O(1) average lookup — the agent calls getValue
    // and setValue on every single step of every episode.
    //
    // Alternative considered: vector<vector<array<double,4>>> grid.
    // Rejected because it pre-allocates memory for every cell including
    // unreachable obstacle cells. unordered_map only stores visited states.
    std::unordered_map<Position, std::array<double, 4>, PositionHash> table_;

    // Number of actions — fixed at 4 (UP/DOWN/LEFT/RIGHT).
    // Named constant avoids magic number 4 scattered through the implementation.
    static constexpr int NUM_ACTIONS = 4;
};

#endif  // QTABLE_H
