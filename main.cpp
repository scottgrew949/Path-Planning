// main.cpp
// Robot Path Planning System — demonstration driver.
//
// Compile (g++ C++17, macOS / Linux):
//   g++ -std=c++17 -Wall -Wextra -O2 \
//       main.cpp \
//       core/Position.cpp core/Types.cpp \
//       environment/Cell.cpp environment/Environment.cpp \
//       planning/algorithms/AStar.cpp \
//       planning/algorithms/Dijkstra.cpp \
//       planning/algorithms/BFS.cpp \
//       utils/ProbabilityUtils.cpp \
//       visualization/Visualizer.cpp \
//       -o pathplanning

#include <iostream>
#include <vector>
#include <map>
#include <memory>
#include <chrono>
#include <string>

#include "core/Position.h"
#include "core/Types.h"
#include "environment/Environment.h"
#include "planning/IPathfinder.h"
#include "planning/algorithms/AStar.h"
#include "planning/algorithms/Dijkstra.h"
#include "planning/algorithms/BFS.h"
#include "planning/algorithms/BidirectionalAStar.h"
#include "planning/algorithms/ThetaStar.h"
#include "utils/ProbabilityUtils.h"
#include "visualization/Visualizer.h"
#include "rl/RLEnvironment.h"
#include "rl/QLearningAgent.h"

using namespace std;

// ---- Helper: time a single algorithm run ------------------------------------
static PathResult runTimed(IPathfinder&       algo,
                            const Environment& env,
                            const Position&    start,
                            const Position&    goal)
{
    auto t0 = chrono::high_resolution_clock::now();
    vector<Position> path = algo.findPath(env, start, goal);
    auto t1 = chrono::high_resolution_clock::now();

    double ms = chrono::duration<double, milli>(t1 - t0).count();

    PathResult result;
    result.algorithmName = algo.getName();
    result.path          = path;
    result.elapsedMs     = ms;
    result.nodesExplored = algo.getNodesExplored();

    if (path.size() >= 2)
    {
        result.pathCost = env.moveCost(path[0], path[0], path[1]);
        for(size_t i = 1; i < path.size()-1; ++i)
            result.pathCost += env.moveCost(path[i-1], path[i], path[i+1]);
    }

    return result;
}

// ---- Helper: build a hand-crafted 12x16 scenario ----------------------------
static Environment buildScenario()
{
    Environment env(41, 41);
    env.setStart(Position(0, 0));
    env.setGoal(Position(38, 40));
    env.generateLabyrinth(0.3);

    return env;
}

// =============================================================================
int main()
{
    // =========================================================================
    // Section 1: Environment setup
    // =========================================================================
    Visualizer::printSection("Environment Setup");

    Environment env = buildScenario();
    Visualizer::displayGrid(env);

    // =========================================================================
    // Section 2: Run all algorithms via IPathfinder interface
    // =========================================================================
    Visualizer::printSection("Pathfinding Algorithms");
    AStar astar;
    PathResult r = runTimed(astar, env, env.getStart(), env.getGoal());
    Visualizer::displayPath(env, r.path, r.algorithmName);

    vector<unique_ptr<IPathfinder>> algos;
    algos.push_back(make_unique<AStar>());
    algos.push_back(make_unique<Dijkstra>());
    algos.push_back(make_unique<BFS>());
    algos.push_back(make_unique<BidirectionalAStar>());
    algos.push_back(make_unique<ThetaStar>());

    vector<PathResult> results;
    for (auto& algo : algos)
    {
        env.reset();
        PathResult r = runTimed(*algo, env, env.getStart(), env.getGoal());
        results.push_back(r);
        Visualizer::displayPath(env, r.path, r.algorithmName);
        Visualizer::displayStats(r.algorithmName, r.path, r.elapsedMs, r.pathCost);
    }

    // =========================================================================
    // Section 3: Algorithm comparison table
    // =========================================================================
    Visualizer::printSection("Algorithm Comparison");

    Visualizer::displaySummaryTable(results);

    // =========================================================================
    // Section 4: Probability — sensor fusion demo
    // =========================================================================
    Visualizer::printSection("Sensor Uncertainty (Bayesian)");

    // demonstrate bayesUpdateSensor:
    double posterior = ProbabilityUtils::bayesUpdateSensor(0.2, 0.9, 0.1, true);
    cout << "Prior: 0.20 → Posterior after sensor hit: " << posterior << '\n';
    //       This is the core of occupancy grid updating in real autonomous vehicles.

    // demonstrate expectedValue with path costs:
    vector<double> costs = {results[0].pathCost, results[1].pathCost, results[2].pathCost, results[3].pathCost, results[4].pathCost};
    vector<double> probs = {0.2, 0.2, 0.2, 0.2, 0.2};
    double ev = ProbabilityUtils::expectedValue(costs, probs);
    cout << "Expected path cost: " << ev << '\n';

    // demonstrate entropy:
    vector<double> uniform = {0.2, 0.2, 0.2, 0.2, 0.2};
    cout << "Route entropy (5 equal options): "
        << ProbabilityUtils::entropy(uniform) << " bits\n";

    // =========================================================================
    // Section 5: Reinforcement Learning — tabular Q-learning
    // =========================================================================
    Visualizer::printSection("Reinforcement Learning (Tabular Q-Learning)");

    // Reset the environment to a clean state before RL training
    env.reset();

    // Wrap the grid in a gym-style interface
    RLEnvironment rlEnv(env);

    // Construct the agent with hyperparameters:
    //   learningRate  = 0.1   — small, stable updates
    //   discountFactor= 0.95  — values long-term reward (goal is far away)
    //   epsilonStart  = 1.0   — fully random at start, agent knows nothing
    //   epsilonMin    = 0.05  — always keep 5% exploration
    //   epsilonDecay  = 0.995 — slow decay, enough episodes to explore the maze
    QLearningAgent agent(rlEnv, 0.1, 0.95, 1.0, 0.05, 0.995);

    int numEpisodes        = 5000;
    int maxStepsPerEpisode = rlEnv.getWidth() * rlEnv.getHeight() * 4;

    cout << "Training for " << numEpisodes << " episodes...\n";
    vector<TrainingResult> trainingHistory = agent.train(numEpisodes, maxStepsPerEpisode);

    // Print every 500th episode as a learning curve sample
    cout << "\nEpisode | Goal Reached | Steps | Total Reward | Epsilon\n";
    cout << "--------|--------------|-------|--------------|--------\n";
    for (int episodeIndex = 0; episodeIndex < numEpisodes; ++episodeIndex)
    {
        if ((episodeIndex + 1) % 500 == 0)
        {
            const TrainingResult& snapshot = trainingHistory[episodeIndex];
            cout << snapshot.episodeNumber   << "\t| "
                 << (snapshot.goalReached ? "YES" : "NO") << "\t\t| "
                 << snapshot.stepsToGoal     << "\t| "
                 << snapshot.totalReward     << "\t\t| "
                 << snapshot.epsilonAtEnd    << "\n";
        }
    }

    // Extract and display the greedy path learned by the agent
    env.reset();
    vector<Position> learnedPath = agent.extractGreedyPath(maxStepsPerEpisode);

    if (learnedPath.empty())
    {
        cout << "\nAgent did not learn a complete path. Try more episodes.\n";
    }
    else
    {
        cout << "\nLearned path length: " << learnedPath.size() << " steps\n";
        Visualizer::displayPath(env, learnedPath, "Q-Learning");
    }

    return 0;
}
