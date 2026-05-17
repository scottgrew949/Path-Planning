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
//       planning/algorithms/BidirectionalAStar.cpp \
//       planning/algorithms/ThetaStar.cpp \
//       planning/algorithms/JPS.cpp \
//       planning/algorithms/DStarLite.cpp \
//       planning/algorithms/RRT.cpp \
//       rl/RLAgent.cpp rl/QLearningAgent.cpp rl/DynaQAgent.cpp \
//       rl/QTable.cpp rl/RLEnvironment.cpp \
//       utils/ProbabilityUtils.cpp \
//       visualization/Visualizer.cpp \
//       -o pathplanning

#include <iostream>
#include <fstream>
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
#include "planning/algorithms/JPS.h"
#include "planning/algorithms/DStarLite.h"
#include "planning/algorithms/RRT.h"
#include "utils/ProbabilityUtils.h"
#include "visualization/Visualizer.h"
#include "rl/RLEnvironment.h"
#include "rl/QLearningAgent.h"
#include "rl/DynaQAgent.h"

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

    for (size_t i = 0; i + 1 < path.size(); ++i)
    {
        const Position& prev = (i == 0) ? path[0] : path[i - 1];
        result.pathCost += env.moveCost(prev, path[i], path[i + 1]);
    }

    return result;
}

// ---- Helper: build a 151x41 labyrinth scenario ------------------------------
static Environment buildScenario()
{
    Environment env(201, 41);
    env.setStart(Position(0, 0));
    env.setGoal(Position(200, 40));
    env.generateLabyrinth(0.4);

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
    algos.push_back(make_unique<JPS>());
    algos.push_back(make_unique<DStarLite>());
    algos.push_back(make_unique<RRT>());

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
    vector<double> costs;
    for (const auto& res : results) costs.push_back(res.pathCost);
    vector<double> probs(costs.size(), 1.0 / costs.size());
    double ev = ProbabilityUtils::expectedValue(costs, probs);
    cout << "Expected path cost: " << ev << '\n';

    // demonstrate entropy:
    vector<double> uniform(costs.size(), 1.0 / costs.size());
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

    // Collect RL greedy path lengths for final summary table
    vector<pair<string,int>> rlSummary;

    // Shared hyperparameters for both agents:
    //   learningRate  = 0.1   — small, stable updates
    //   discountFactor= 0.95  — values long-term reward (goal is far away)
    //   epsilonStart  = 1.0   — fully random at start, agent knows nothing
    //   epsilonMin    = 0.05  — always keep 5% exploration
    //   epsilonDecay  = 0.995 — slow decay, enough episodes to explore the maze
    const double learningRate   = 0.1;
    const double discountFactor = 0.95;
    const double epsilonStart   = 1.0;
    const double epsilonMin     = 0.05;
    const double epsilonDecay   = 0.995;

    const int numEpisodes        = 10000;
    const int maxStepsPerEpisode = rlEnv.getWidth() * rlEnv.getHeight() * 4;

    // ---- Q-Learning ---------------------------------------------------------
    {
        QLearningAgent agent(rlEnv, learningRate, discountFactor,
                             epsilonStart, epsilonMin, epsilonDecay);

        cout << "Training for " << numEpisodes << " episodes...\n";
        cout << "\nEpisode | Goal Reached | Steps | Total Reward | Epsilon\n";
        cout << "--------|--------------|-------|--------------|--------\n";
        cout.flush();

        vector<TrainingResult> trainingHistory;
        trainingHistory.reserve(numEpisodes);
        for (int ep = 1; ep <= numEpisodes; ++ep)
        {
            TrainingResult r = agent.runEpisode(ep, maxStepsPerEpisode);
            trainingHistory.push_back(r);
            if (ep % 1000 == 0)
            {
                cout << r.episodeNumber   << "\t| "
                     << (r.goalReached ? "YES" : "NO") << "\t\t| "
                     << r.stepsToGoal     << "\t| "
                     << r.totalReward     << "\t\t| "
                     << r.epsilonAtEnd    << "\n";
                cout.flush();
            }
        }

        // Write full training history to CSV
        {
            ofstream csv("qlearning_training.csv");
            csv << "episode,total_reward,steps,goal_reached,epsilon\n";
            for (const TrainingResult& r : trainingHistory)
                csv << r.episodeNumber << ","
                    << r.totalReward   << ","
                    << r.stepsToGoal   << ","
                    << (r.goalReached ? 1 : 0) << ","
                    << r.epsilonAtEnd  << "\n";
            cout << "Training data written to qlearning_training.csv\n";
        }

        // Reset rlEnv so extractGreedyPath starts from the true start position
        rlEnv.reset();
        vector<Position> learnedPath = agent.extractGreedyPath(maxStepsPerEpisode);

        if (learnedPath.empty())
        {
            cout << "\nAgent did not learn a complete path. Try more episodes.\n";
            rlSummary.push_back({"Q-Learning", -1});
        }
        else
        {
            cout << "\nLearned path length: " << learnedPath.size() << " steps\n";
            rlSummary.push_back({"Q-Learning", (int)learnedPath.size()});
        }
    }

    // ---- Dyna-Q (model-based RL, n=10) --------------------------------------
    Visualizer::printSection("Dyna-Q (model-based RL, n=10)");

    // Reset rlEnv between agent runs so Dyna-Q starts from clean state
    rlEnv.reset();

    {
        const int planningSteps = 10;
        DynaQAgent agent(rlEnv, learningRate, discountFactor,
                         epsilonStart, epsilonMin, epsilonDecay,
                         planningSteps);

        cout << "Training for " << numEpisodes << " episodes...\n";
        cout << "\nEpisode | Goal Reached | Steps | Total Reward | Epsilon\n";
        cout << "--------|--------------|-------|--------------|--------\n";
        cout.flush();

        vector<TrainingResult> trainingHistory;
        trainingHistory.reserve(numEpisodes);
        for (int ep = 1; ep <= numEpisodes; ++ep)
        {
            TrainingResult r = agent.runEpisode(ep, maxStepsPerEpisode);
            trainingHistory.push_back(r);
            if (ep % 1000 == 0)
            {
                cout << r.episodeNumber   << "\t| "
                     << (r.goalReached ? "YES" : "NO") << "\t\t| "
                     << r.stepsToGoal     << "\t| "
                     << r.totalReward     << "\t\t| "
                     << r.epsilonAtEnd    << "\n";
                cout.flush();
            }
        }

        // Write full training history to CSV
        {
            ofstream csv("dynaq_training.csv");
            csv << "episode,total_reward,steps,goal_reached,epsilon\n";
            for (const TrainingResult& r : trainingHistory)
                csv << r.episodeNumber << ","
                    << r.totalReward   << ","
                    << r.stepsToGoal   << ","
                    << (r.goalReached ? 1 : 0) << ","
                    << r.epsilonAtEnd  << "\n";
            cout << "Training data written to dynaq_training.csv\n";
        }

        // Reset rlEnv so extractGreedyPath starts from the true start position
        rlEnv.reset();
        vector<Position> learnedPath = agent.extractGreedyPath(maxStepsPerEpisode);

        if (learnedPath.empty())
        {
            cout << "\nAgent did not learn a complete path. Try more episodes.\n";
            rlSummary.push_back({"Dyna-Q (n=10)", -1});
        }
        else
        {
            cout << "\nLearned path length: " << learnedPath.size() << " steps\n";
            rlSummary.push_back({"Dyna-Q (n=10)", (int)learnedPath.size()});
        }
    }

    // =========================================================================
    // Section 6: Final unified comparison
    // =========================================================================
    Visualizer::printSection("Full System Comparison");
    Visualizer::displayFinalSummary(results, rlSummary);

    return 0;
}
