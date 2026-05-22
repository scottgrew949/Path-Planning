// main.cpp
// Robot Path Planning System — demonstration driver.
//
// Compile (g++ C++17, macOS / Linux):
//   g++ -std=c++17 -Wall -Wextra -O2 \
//       main.cpp \
//       core/Position.cpp core/Types.cpp \
//       environment/Cell.cpp environment/Environment.cpp \
//       environment/DynamicEnvironment.cpp environment/SensorModel.cpp \
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
//       planning/CurriculumScheduler.cpp \
//       utils/ProbabilityUtils.cpp \
//       visualization/Visualizer.cpp \
//       -o pathplanning

#include <iostream>
#include <fstream>
#include <vector>
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
#include "planning/CurriculumScheduler.h"
#include "environment/DynamicEnvironment.h"
#include "environment/SensorModel.h"

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
    for (unique_ptr<IPathfinder>& algo : algos)
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
    for (const PathResult& res : results) costs.push_back(res.pathCost);
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

    CurriculumScheduler sched(numEpisodes);
    cout << "\nCurriculum Learning Schedule:\n";
    sched.printSchedule();
    cout << "\n";

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
            if (sched.isStageTransition(ep - 1)) {
                const CurriculumScheduler::Stage& stage = sched.getStageForEpisode(ep - 1);
                env.generateLabyrinth(stage.loopDensity, static_cast<unsigned>(stage.stageIndex + 1) * 37u);
                rlEnv.reset();
            }
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

        int bestQLSteps = -1;
        for (const TrainingResult& result : trainingHistory)
            if (result.goalReached && (bestQLSteps == -1 || result.stepsToGoal < bestQLSteps))
                bestQLSteps = result.stepsToGoal;

        if (bestQLSteps == -1)
        {
            cout << "\nAgent did not reach goal in any episode.\n";
            rlSummary.push_back({"Q-Learning", -1});
        }
        else
        {
            cout << "\nBest path length during training: " << bestQLSteps << " steps\n";
            rlSummary.push_back({"Q-Learning", bestQLSteps});
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
            if (sched.isStageTransition(ep - 1)) {
                const CurriculumScheduler::Stage& stage = sched.getStageForEpisode(ep - 1);
                env.generateLabyrinth(stage.loopDensity, static_cast<unsigned>(stage.stageIndex + 1) * 37u);
                rlEnv.reset();
            }
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

        int bestDQSteps = -1;
        for (const TrainingResult& result : trainingHistory)
            if (result.goalReached && (bestDQSteps == -1 || result.stepsToGoal < bestDQSteps))
                bestDQSteps = result.stepsToGoal;

        if (bestDQSteps == -1)
        {
            cout << "\nAgent did not reach goal in any episode.\n";
            rlSummary.push_back({"Dyna-Q (n=10)", -1});
        }
        else
        {
            cout << "\nBest path length during training: " << bestDQSteps << " steps\n";
            rlSummary.push_back({"Dyna-Q (n=10)", bestDQSteps});
        }
    }

    // =========================================================================
    // Section 6: Final unified comparison
    // =========================================================================
    Visualizer::printSection("Full System Comparison");
    Visualizer::displayFinalSummary(results, rlSummary);

    // =========================================================================
    // Section 7: Dynamic obstacles + partial observability (Phase 8)
    // =========================================================================
    Visualizer::printSection("Dynamic Obstacles + Partial Observability");

    // 21-wide x 11-tall grid — small enough for readable terminal output.
    DynamicEnvironment dynamicEnv(21, 11);
    dynamicEnv.setStart(Position(0,  0));
    dynamicEnv.setGoal( Position(20, 10));

    // Static L-shaped wall that forces the path through a constrained corridor.
    for (int y = 2; y <= 6; ++y)   dynamicEnv.setObstacle(Position(7, y));
    for (int x = 8; x <= 14; ++x)  dynamicEnv.setObstacle(Position(x, 6));

    // Dynamic obstacle 1 — horizontal patrol along row y=8 (below the static wall), columns 2..17.
    // Moves every 2 ticks (slower obstacle — like a slow-moving vehicle).
    // Row y=8 is chosen to avoid the static wall at x=7,y=2..6 and x=8..14,y=6.
    std::vector<Position> horizontalPatrol;
    for (int x =  2; x <= 17; ++x) horizontalPatrol.push_back(Position(x, 8));
    for (int x = 16; x >=  3; --x) horizontalPatrol.push_back(Position(x, 8));
    dynamicEnv.addDynamicObstacle(horizontalPatrol, 2);

    // Dynamic obstacle 2 — vertical patrol along column x=16, rows 1..8.
    // Moves every tick (faster obstacle — like a pedestrian).
    std::vector<Position> verticalPatrol;
    for (int y = 1; y <= 8; ++y) verticalPatrol.push_back(Position(16, y));
    for (int y = 7; y >= 2; --y) verticalPatrol.push_back(Position(16, y));
    dynamicEnv.addDynamicObstacle(verticalPatrol, 1);

    cout << "Environment: 21x11  Start: (0,0)  Goal: (20,10)\n";
    cout << "Dynamic obstacles: " << dynamicEnv.getDynamicObstacleCount() << "\n";
    cout << "  Obstacle 1 — horizontal patrol y=8, x=2..17, 1 step per 2 ticks\n";
    cout << "  Obstacle 2 — vertical patrol   x=16, y=1..8, 1 step per tick\n\n";

    // D* Lite for replanning.
    // Note: DStarLite::updateObstacle() is not yet implemented (stub) —
    // each replan calls findPath() from scratch. This is functionally correct;
    // true incremental replanning is a future optimisation.
    DStarLite dstarPhase8;

    // Initial plan — obstacles at their starting positions.
    std::vector<Position> dynamicPath = dstarPhase8.findPath(dynamicEnv,
                                                              dynamicEnv.getStart(),
                                                              dynamicEnv.getGoal());
    cout << "--- Tick 0 (initial) ---\n";
    Visualizer::displayPath(dynamicEnv, dynamicPath, "D* Lite");
    if (dynamicPath.empty())
        cout << "  [no path found]\n";
    else
        cout << "  Path length: " << dynamicPath.size() << " steps\n\n";

    // Simulate ticks and replan after each batch.
    const int ticksPerSnapshot = 5;
    const int totalTicks       = 15;

    for (int tick = 1; tick <= totalTicks; ++tick)
    {
        dynamicEnv.tick();

        if (tick % ticksPerSnapshot == 0)
        {
            dynamicEnv.reset();  // clear PATH/VISITED overlays only
            dynamicPath = dstarPhase8.findPath(dynamicEnv,
                                               dynamicEnv.getStart(),
                                               dynamicEnv.getGoal());
            cout << "--- Tick " << tick << " ---\n";
            Visualizer::displayPath(dynamicEnv, dynamicPath, "D* Lite");
            if (dynamicPath.empty())
                cout << "  [no path — obstacles blocked all routes]\n\n";
            else
                cout << "  Path length: " << dynamicPath.size() << " steps\n\n";
        }
    }

    // Partial observability — SensorModel at the agent's start position.
    // Range 5, no noise (perfect sensor within range).
    SensorModel sensor(5, 0.0, 0.0);
    std::vector<SensorModel::Observation> sensorReadings =
        sensor.observe(dynamicEnv.getStart(), dynamicEnv);

    int obstaclesDetected = 0;
    for (const SensorModel::Observation& reading : sensorReadings)
        if (reading.reportedAsObstacle) ++obstaclesDetected;

    cout << "Sensor observation from start (0,0):\n";
    cout << "  Range:              " << sensor.getSensorRange() << " (Manhattan)\n";
    cout << "  Cells scanned:      " << sensorReadings.size()  << "\n";
    cout << "  Obstacles detected: " << obstaclesDetected       << "\n";
    cout << "  Cells clear:        " << sensorReadings.size() - obstaclesDetected << "\n";
    cout << "  (cells beyond range are unknown — planner uses prior beliefs)\n\n";

    // Noisy sensor demo — 10% false positive, 5% false negative.
    SensorModel noisySensor(5, 0.1, 0.05);
    std::vector<SensorModel::Observation> noisyReadings =
        noisySensor.observe(dynamicEnv.getStart(), dynamicEnv);

    int noisyObstacles = 0;
    for (const SensorModel::Observation& reading : noisyReadings)
        if (reading.reportedAsObstacle) ++noisyObstacles;

    cout << "Noisy sensor (FP=10%, FN=5%) from start (0,0):\n";
    cout << "  Obstacles reported: " << noisyObstacles << "\n";
    cout << "  (perfect sensor reported " << obstaclesDetected
         << " — difference is sensor noise)\n";

    return 0;
}
