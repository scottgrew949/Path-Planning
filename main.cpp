// main.cpp
// Robot Path Planning System — interactive menu driver.
//
// Build (preferred): ./build.sh 2   →  ./pathplanning
//
// Manual compile (g++ C++17, macOS / Linux) — must match CPP_SOURCES in build.sh:
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
//       planning/algorithms/MCTS.cpp \
//       planning/algorithms/CBS.cpp \
//       rl/RLAgent.cpp rl/QLearningAgent.cpp rl/DynaQAgent.cpp \
//       rl/QTable.cpp rl/RLEnvironment.cpp rl/TDLambdaAgent.cpp \
//       planning/CurriculumScheduler.cpp \
//       planning/hybrid/HeuristicNetwork.cpp planning/hybrid/NeuralAStar.cpp \
//       utils/ProbabilityUtils.cpp visualization/Visualizer.cpp \
//       tests/SmokeTests.cpp \
//       -o pathplanning

#include <iostream>
#include <fstream>
#include <vector>
#include <memory>
#include <chrono>
#include <string>
#include <cstdlib>
#include <algorithm>

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
#include "planning/algorithms/MCTS.h"
#include "planning/algorithms/CBS.h"
#include "utils/ProbabilityUtils.h"
#include "visualization/Visualizer.h"
#include "rl/RLEnvironment.h"
#include "rl/QLearningAgent.h"
#include "rl/DynaQAgent.h"
#include "planning/CurriculumScheduler.h"
#include "environment/DynamicEnvironment.h"
#include "environment/SensorModel.h"
#include "rl/TDLambdaAgent.h"
#include "planning/hybrid/NeuralAStar.h"
#include "core/RunProfile.h"
#include "tests/SmokeTests.h"

using namespace std;

// =============================================================================
// Session settings (menu toggles RunProfile for all training sections)
// =============================================================================

static const char* NEURAL_WEIGHTS_PATH = "python/data/weights.bin";

static RunProfile gRunProfile = RunProfile::FULL;

static string runProfileLabel(RunProfile profile)
{
    return (profile == RunProfile::FULL) ? "FULL" : "QUICK";
}

static vector<int> buildStageLengthsVector(RunProfile profile)
{
    vector<int> lengths;
    lengths.reserve(curriculumStageCount());
    for (int stageIndex = 0; stageIndex < curriculumStageCount(); ++stageIndex)
        lengths.push_back(curriculumStageLengths(profile)[stageIndex]);
    return lengths;
}

static bool neuralWeightsFileExists()
{
    ifstream weightsFile(NEURAL_WEIGHTS_PATH);
    return weightsFile.good();
}

// =============================================================================
// Shared helpers
// =============================================================================

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

static Environment buildScenario()
{
    Environment env(201, 41);
    env.setStart(Position(0, 0));
    env.setGoal(Position(200, 40));
    env.generateLabyrinth(0.4);
    return env;
}

static void runPythonScript(const string& relPath)
{
    string command = "venv/bin/python3 python/" + relPath;
    cout << "\nRunning: " << command << "\n\n";
    std::system(command.c_str());
}

static void runBuildPythonModule()
{
    cout << "\nRunning: ./build.sh 3\n\n";
    std::system("./build.sh 3");
}

static void runSmokeTests()
{
    SmokeTestSummary summary = SmokeTests::runAll();
    if (summary.allPassed())
        cout << "All smoke tests passed.\n";
    else
        cout << "Some smoke tests failed.\n";
}

static Environment buildCompactScenario()
{
    Environment env(41, 41);
    env.setStart(Position(0, 0));
    env.setGoal(Position(40, 40));
    env.generateLabyrinth(0.35, 42);
    return env;
}


// =============================================================================
// Demo sections — each is self-contained
// =============================================================================

// Returns results so runAllCppDemos() can pass them to the summary.
static vector<PathResult> sectionAlgorithms(Environment& env)
{
    Visualizer::printSection("Classical Pathfinding Algorithms");
    cout << "All 9 algorithms run on the same 201x41 labyrinth from corner to corner.\n";
    cout << "They know the full map upfront — this is called 'complete information' planning.\n";
    cout << "Self-driving analog: offline route planning before the car moves.\n\n";
    cout << "What to notice: A* and Dijkstra find the shortest path but explore many cells.\n";
    cout << "JPS and Theta* skip redundant cells — same quality, less work.\n";
    cout << "RRT* and MCTS are probabilistic — they explore randomly, so paths vary each run.\n\n";
    cout << "  length  = number of steps in the returned path\n";
    cout << "  cost    = total movement cost (straight moves cost 1.0, turns add a small penalty)\n";
    cout << "  time    = wall-clock milliseconds the algorithm took to find the path\n\n";

    vector<unique_ptr<IPathfinder>> algos;
    algos.push_back(make_unique<AStar>());
    algos.push_back(make_unique<Dijkstra>());
    algos.push_back(make_unique<BFS>());
    algos.push_back(make_unique<BidirectionalAStar>());
    algos.push_back(make_unique<ThetaStar>());
    algos.push_back(make_unique<JPS>());
    algos.push_back(make_unique<DStarLite>());
    algos.push_back(make_unique<RRT>());
    algos.push_back(make_unique<MCTS>(2000, 1.414, 300));

    vector<PathResult> results;
    for (unique_ptr<IPathfinder>& algo : algos)
    {
        env.reset();
        PathResult r = runTimed(*algo, env, env.getStart(), env.getGoal());
        results.push_back(r);
        Visualizer::displayStats(r.algorithmName, r.path, r.elapsedMs, r.pathCost);
    }

    Visualizer::printSection("Algorithm Comparison");
    cout << "The key insight: nodes explored measures search intelligence, not just speed.\n";
    cout << "A* uses a heuristic (estimated distance to goal) to ignore dead-end directions early.\n";
    cout << "Dijkstra has no heuristic — it fans out in all directions equally, exploring more.\n";
    cout << "BFS treats all moves as equal cost — finds fewest steps but ignores turn penalties.\n";
    cout << "JPS prunes symmetric paths (why explore 10 cells in a corridor when 1 jump covers them?).\n";
    cout << "Theta* allows any-angle paths — cuts corners Dijkstra/A* can't because they're grid-locked.\n";
    cout << "RRT*/MCTS are stochastic — path quality varies, but they scale to continuous spaces.\n\n";
    cout << "  nodes explored = cells examined before finding the path — fewer = smarter algorithm\n\n";
    Visualizer::displaySummaryTable(results);

    return results;
}

static void sectionCBS()
{
    Visualizer::printSection("Multi-Agent Pathfinding (CBS)");
    cout << "3 robots navigate their own routes through a shared open grid without colliding.\n";
    cout << "CBS plans individually first, then detects conflicts and adds constraints until\n";
    cout << "all paths are collision-free. Agents have genuinely different distances to travel\n";
    cout << "so their path lengths should differ — longer routes take more steps.\n";
    cout << "Self-driving analog: intersection management for a fleet of autonomous vehicles.\n\n";
    cout << "  path length = steps each robot takes to reach its individual goal\n";
    cout << "  total cost  = sum of all path lengths — the fleet efficiency metric\n";
    cout << "  CT nodes    = conflict-resolution attempts made (grows exponentially with agent count)\n\n";

    // Purpose-built 30x20 grid with sparse random obstacles — gives CBS agents
    // multiple genuine route options so paths reflect actual distance differences.
    Environment cbsEnv(30, 20);
    cbsEnv.setStart(Position(0, 0));
    cbsEnv.setGoal(Position(29, 19));
    cbsEnv.generateRandom(0.15, 42);   // fixed seed: reproducible, low density for open routes

    std::vector<Agent> agents = {
        Agent(Position( 0,  0), Position(29, 19)),   // long diagonal
        Agent(Position( 0, 10), Position(29, 10)),   // straight across middle
        Agent(Position( 0, 19), Position(29,  0))    // diagonal the other way
    };

    CBS cbs;
    MultiAgentPaths agentPaths = cbs.findPaths(cbsEnv, agents);

    cout << "Agents: " << agents.size() << "\n";
    if (agentPaths.empty())
    {
        cout << "No solution found within search budget.\n";
    }
    else
    {
        int totalCost = 0;
        for (size_t agentIndex = 0; agentIndex < agentPaths.size(); ++agentIndex)
        {
            int pathLength = static_cast<int>(agentPaths[agentIndex].size());
            totalCost += pathLength;
            cout << "  Agent " << agentIndex << " path length: " << pathLength << " steps\n";
        }
        cout << "Total cost (sum of path lengths): " << totalCost << "\n";
        cout << "CT nodes expanded: " << cbs.getNodesExpanded() << "\n";
    }
}

// results may be empty when called standalone — fallback to example costs.
static void sectionBayesian(const vector<PathResult>& results)
{
    Visualizer::printSection("Sensor Uncertainty (Bayesian)");
    cout << "Classical planning assumes the map is perfectly known. Real robots don't have that luxury.\n";
    cout << "Bayes' rule is how a robot updates its belief about the world when a sensor fires.\n";
    cout << "If a LiDAR reports an obstacle (sensor hit), how confident should we be it's really there?\n";
    cout << "That depends on: how often was an obstacle there before (prior), how reliable is the sensor\n";
    cout << "(true positive rate), and how often does it false-alarm (false positive rate).\n";
    cout << "This math sits inside every occupancy grid in every autonomous vehicle on the road.\n\n";
    cout << "  prior       = probability of obstacle before the sensor fires (our initial belief)\n";
    cout << "  posterior   = updated probability after the sensor reports a hit\n";
    cout << "  expected cost = probability-weighted average of all possible path costs\n";
    cout << "  entropy     = information content of the route choice — 0 bits means one obvious answer\n\n";

    double posterior = ProbabilityUtils::bayesUpdateSensor(0.2, 0.9, 0.1, true);
    cout << "Prior: 0.20  →  Posterior after sensor hit: " << posterior << '\n';

    vector<double> costs;
    if (!results.empty())
        for (const PathResult& res : results) costs.push_back(res.pathCost);
    else
        costs = {120.0, 135.0, 200.0, 145.0, 160.0};

    vector<double> probs(costs.size(), 1.0 / static_cast<double>(costs.size()));
    double ev = ProbabilityUtils::expectedValue(costs, probs);
    cout << "Expected path cost: " << ev << '\n';

    vector<double> uniform = probs;
    cout << "Route entropy (" << costs.size() << " equal options): "
         << ProbabilityUtils::entropy(uniform) << " bits\n";
}

// Returns rl summary so runAllCppDemos() can pass it to displayFinalSummary.
static vector<pair<string,int>> sectionTabularRL(Environment& env, RunProfile profile)
{
    const vector<int> stageLengths = buildStageLengthsVector(profile);
    const int numEpisodes = totalCurriculumEpisodes(profile);
    const int logInterval   = trainingLogInterval(profile);

    Visualizer::printSection("Reinforcement Learning (Tabular Q-Learning)");
    cout << "Run profile: " << runProfileLabel(profile) << "\n";
    cout << "The agent gets NO map. It starts knowing nothing and learns purely from consequences.\n";
    cout << "Runs " << numEpisodes << " episodes. Snapshot every " << logInterval << ".\n";
    cout << "Every move returns a reward signal: +100 for reaching the goal, -1 per valid step,\n";
    cout << "-10 for hitting a wall. Over thousands of episodes the agent builds a Q-table:\n";
    cout << "a lookup table of 'at position X, action Y is worth Z expected future reward.'\n";
    cout << "Early episodes: the agent wanders randomly. Late episodes: it follows a learned path.\n";
    cout << "Self-driving analog: a robot that learns to navigate a building without being programmed.\n\n";
    cout << "What to watch: the curriculum scheduler replaces the maze mid-training with a harder one.\n";
    cout << "This causes visible regressions — the agent was learning one maze, then the floor moves.\n";
    cout << "Steps spike upward and Goal Reached can flip back to NO after the maze changes.\n";
    cout << "This is intentional: curriculum learning exposes the agent to progressively harder mazes\n";
    cout << "so the final policy generalises rather than memorising a single layout.\n";
    cout << "Epsilon decays to 0.05 early and stays there — exploration is mostly off by episode 1000.\n\n";
    cout << "  Episode      = one full run start-to-goal (or timeout at max steps)\n";
    cout << "  Goal Reached = did the agent find the goal this episode?\n";
    cout << "  Steps        = moves made — high early (wandering), lower as policy improves\n";
    cout << "  Total Reward = sum of all rewards — negative early, rises toward 0 as agent improves\n";
    cout << "  Epsilon      = exploration rate: 1.0 = pure random, 0.05 = 95% learned policy\n\n";

    env.reset();
    RLEnvironment rlEnv(env);
    vector<pair<string,int>> rlSummary;

    const double learningRate   = 0.1;
    const double discountFactor = 0.95;
    const double epsilonStart   = 1.0;
    const double epsilonMin     = 0.05;
    const double epsilonDecay        = 0.995;
    const int    maxStepsPerEpisode  = rlEnv.getWidth() * rlEnv.getHeight() * 4;

    CurriculumScheduler sched(stageLengths);
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

        vector<TrainingResult> history;
        history.reserve(numEpisodes);
        for (int episodeNumber = 1; episodeNumber <= numEpisodes; ++episodeNumber)
        {
            if (sched.isStageTransition(episodeNumber - 1)) {
                const CurriculumScheduler::Stage& stage = sched.getStageForEpisode(episodeNumber - 1);
                env.generateLabyrinth(stage.loopDensity,
                                      static_cast<unsigned>(stage.stageIndex + 1) * 37u);
                rlEnv.reset();
            }
            TrainingResult r = agent.runEpisode(episodeNumber, maxStepsPerEpisode);
            history.push_back(r);
            if (episodeNumber % logInterval == 0)
            {
                cout << r.episodeNumber << "\t| "
                     << (r.goalReached ? "YES" : "NO") << "\t\t| "
                     << r.stepsToGoal  << "\t| "
                     << r.totalReward  << "\t\t| "
                     << r.epsilonAtEnd << "\n";
                cout.flush();
            }
        }

        {
            ofstream csv("qlearning_training.csv");
            csv << "episode,total_reward,steps,goal_reached,epsilon\n";
            for (const TrainingResult& r : history)
                csv << r.episodeNumber << ","
                    << r.totalReward   << ","
                    << r.stepsToGoal   << ","
                    << (r.goalReached ? 1 : 0) << ","
                    << r.epsilonAtEnd  << "\n";
            cout << "Training data written to qlearning_training.csv\n";
        }

        auto bestIt = std::min_element(history.begin(), history.end(),
            [](const TrainingResult& a, const TrainingResult& b) {
                if (!a.goalReached) return false;
                if (!b.goalReached) return true;
                return a.stepsToGoal < b.stepsToGoal;
            });
        int best = (bestIt != history.end() && bestIt->goalReached) ? bestIt->stepsToGoal : -1;

        if (best == -1) cout << "\nAgent did not reach goal in any episode.\n";
        else            cout << "\nBest path length during training: " << best << " steps\n";
        rlSummary.push_back({"Q-Learning", best});
    }

    // ---- Dyna-Q (n=10) ------------------------------------------------------
    Visualizer::printSection("Dyna-Q (model-based RL, n=10)");
    cout << "Dyna-Q adds one idea to Q-Learning: after each real step, replay 10 past transitions\n";
    cout << "from memory. The agent 'imagines' experiences it has already had, squeezing 10 extra\n";
    cout << "Q-value updates out of every real environment interaction.\n";
    cout << "This is model-based RL: the agent builds an internal model of the world and plans in it.\n";
    cout << "The n=10 in the title means 10 imagined steps per real step — 11x the learning signal.\n";
    cout << "Watch: Dyna-Q should reach the goal in fewer episodes than Q-Learning above.\n\n";
    cout << "Same columns as Q-Learning above.\n\n";
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

        vector<TrainingResult> history;
        history.reserve(numEpisodes);
        for (int episodeNumber = 1; episodeNumber <= numEpisodes; ++episodeNumber)
        {
            if (sched.isStageTransition(episodeNumber - 1)) {
                const CurriculumScheduler::Stage& stage = sched.getStageForEpisode(episodeNumber - 1);
                env.generateLabyrinth(stage.loopDensity,
                                      static_cast<unsigned>(stage.stageIndex + 1) * 37u);
                rlEnv.reset();
            }
            TrainingResult r = agent.runEpisode(episodeNumber, maxStepsPerEpisode);
            history.push_back(r);
            if (episodeNumber % logInterval == 0)
            {
                cout << r.episodeNumber << "\t| "
                     << (r.goalReached ? "YES" : "NO") << "\t\t| "
                     << r.stepsToGoal  << "\t| "
                     << r.totalReward  << "\t\t| "
                     << r.epsilonAtEnd << "\n";
                cout.flush();
            }
        }

        {
            ofstream csv("dynaq_training.csv");
            csv << "episode,total_reward,steps,goal_reached,epsilon\n";
            for (const TrainingResult& r : history)
                csv << r.episodeNumber << ","
                    << r.totalReward   << ","
                    << r.stepsToGoal   << ","
                    << (r.goalReached ? 1 : 0) << ","
                    << r.epsilonAtEnd  << "\n";
            cout << "Training data written to dynaq_training.csv\n";
        }

        auto bestIt = std::min_element(history.begin(), history.end(),
            [](const TrainingResult& a, const TrainingResult& b) {
                if (!a.goalReached) return false;
                if (!b.goalReached) return true;
                return a.stepsToGoal < b.stepsToGoal;
            });
        int best = (bestIt != history.end() && bestIt->goalReached) ? bestIt->stepsToGoal : -1;

        if (best == -1) cout << "\nAgent did not reach goal in any episode.\n";
        else            cout << "\nBest path length during training: " << best << " steps\n";
        rlSummary.push_back({"Dyna-Q (n=10)", best});
    }

    return rlSummary;
}

static void sectionTDLambda(RunProfile profile)
{
    Visualizer::printSection("Reinforcement Learning (TD-λ Eligibility Traces)");
    cout << "Run profile: " << runProfileLabel(profile) << "\n";
    cout << "Standard Q-Learning updates only the last (state, action) when a reward arrives.\n";
    cout << "In a long maze, credit for reaching the goal takes thousands of episodes to propagate\n";
    cout << "back to the first few steps. TD-λ fixes this: it keeps a decaying trace of every\n";
    cout << "visited (state, action) and gives ALL of them credit in a single pass.\n";
    cout << "λ = 0 is identical to Q-Learning. λ = 1 spreads credit all the way back.\n";
    cout << "Self-driving analog: distributing blame for a crash across all steering decisions\n";
    cout << "that led to it, not just the final one.\n\n";
    cout << "What to watch: compare convergence speed vs Q-Learning (menu option 4).\n";
    cout << "With λ = 0.9, credit flows back ~10 steps per real step — faster goal propagation.\n\n";
    cout << "  λ (lambda) = trace decay: 0 = standard TD, 1 = Monte Carlo-like full credit\n";
    cout << "  Same episode/step/reward/epsilon columns as Q-Learning.\n\n";

    const vector<int> stageLengths = buildStageLengthsVector(profile);
    const int numEpisodes  = totalCurriculumEpisodes(profile);
    const int logInterval  = trainingLogInterval(profile);

    Environment env = buildScenario();
    env.reset();
    RLEnvironment rlEnv(env);

    const double learningRate   = 0.1;
    const double discountFactor = 0.95;
    const double epsilonStart   = 1.0;
    const double epsilonMin     = 0.05;
    const double epsilonDecay   = 0.995;
    const double lambda         = 0.9;
    const int    maxStepsPerEpisode = rlEnv.getWidth() * rlEnv.getHeight() * 4;

    CurriculumScheduler sched(stageLengths);
    TDLambdaAgent agent(rlEnv, learningRate, discountFactor,
                        epsilonStart, epsilonMin, epsilonDecay, lambda);

    cout << "Training for " << numEpisodes << " episodes (λ = " << lambda << ")...\n";
    cout << "\nEpisode | Goal Reached | Steps | Total Reward | Epsilon\n";
    cout << "--------|--------------|-------|--------------|--------\n";
    cout.flush();

    vector<TrainingResult> history;
    history.reserve(numEpisodes);
    for (int episodeNumber = 1; episodeNumber <= numEpisodes; ++episodeNumber)
    {
        if (sched.isStageTransition(episodeNumber - 1)) {
            const CurriculumScheduler::Stage& stage = sched.getStageForEpisode(episodeNumber - 1);
            env.generateLabyrinth(stage.loopDensity,
                                  static_cast<unsigned>(stage.stageIndex + 1) * 37u);
            rlEnv.reset();
        }
        TrainingResult r = agent.runEpisode(episodeNumber, maxStepsPerEpisode);
        history.push_back(r);
        if (episodeNumber % logInterval == 0)
        {
            cout << r.episodeNumber << "\t| "
                 << (r.goalReached ? "YES" : "NO") << "\t\t| "
                 << r.stepsToGoal  << "\t| "
                 << r.totalReward  << "\t\t| "
                 << r.epsilonAtEnd << "\n";
            cout.flush();
        }
    }

    auto bestIt = std::min_element(history.begin(), history.end(),
        [](const TrainingResult& a, const TrainingResult& b) {
            if (!a.goalReached) return false;
            if (!b.goalReached) return true;
            return a.stepsToGoal < b.stepsToGoal;
        });
    int best = (bestIt != history.end() && bestIt->goalReached) ? bestIt->stepsToGoal : -1;

    if (best == -1) cout << "\nAgent did not reach goal in any episode.\n";
    else            cout << "\nBest path length during training: " << best << " steps\n";
}

static void sectionNeuralAStar()
{
    Visualizer::printSection("Neural A* (learned heuristic)");
    cout << "Compares standard A* (Manhattan heuristic) with Neural A* using weights from training.\n";
    cout << "Weights file: " << NEURAL_WEIGHTS_PATH << "\n\n";

    if (!neuralWeightsFileExists())
    {
        cout << "Weights file not found.\n";
        cout << "From the main menu: Python Training → 1 Generate data, 2 Train heuristic.\n";
        cout << "Or run: ./build.sh 11 && ./build.sh 12\n";
        return;
    }

    Environment env = buildCompactScenario();
    const Position start = env.getStart();
    const Position goal  = env.getGoal();

    AStar astar;
    NeuralAStar neuralAstar(NEURAL_WEIGHTS_PATH);

    env.reset();
    PathResult astarResult = runTimed(astar, env, start, goal);

    env.reset();
    PathResult neuralResult = runTimed(neuralAstar, env, start, goal);

    cout << "Standard A*:\n";
    Visualizer::displayStats(astarResult.algorithmName, astarResult.path,
                             astarResult.elapsedMs, astarResult.pathCost);
    cout << "  nodes explored: " << astarResult.nodesExplored << "\n\n";

    cout << "Neural A*:\n";
    Visualizer::displayStats(neuralResult.algorithmName, neuralResult.path,
                             neuralResult.elapsedMs, neuralResult.pathCost);
    cout << "  nodes explored: " << neuralResult.nodesExplored << "\n";
    cout << "  heuristic calls: " << neuralAstar.getHeuristicCallCount() << "\n";
}

static void sectionDynamicObstacles(RunProfile profile)
{
    Visualizer::printSection("Dynamic Obstacles + Partial Observability");
    cout << "Run profile: " << runProfileLabel(profile) << "\n";
    cout << "Classical planning assumed a static map. Real environments change — cars move,\n";
    cout << "people walk, doors open. This demo combines two ideas from autonomous vehicles:\n\n";
    cout << "D* Lite replanning: when an obstacle moves, don't replan from scratch. D* Lite\n";
    cout << "repairs only the parts of the path that changed. Efficient for live replanning.\n\n";
    cout << "Partial observability: the robot can't see the whole map — only within sensor range.\n";
    cout << "Cells beyond range are unknown. Noisy sensors add false positives (sees phantom obstacles)\n";
    cout << "and false negatives (misses real ones). The Bayesian update from the previous section\n";
    cout << "is how you fuse noisy sensor readings into a reliable occupancy grid.\n\n";
    cout << "  Tick          = one simulation time step — obstacles move, robot may replan\n";
    cout << "  Path length   = current best path length after replanning around moved obstacles\n";
    cout << "  Range         = sensor detection radius in Manhattan distance (cells)\n";
    cout << "  Cells scanned = total cells the sensor inspected this observation\n";
    cout << "  Obstacles detected = cells the sensor reported as blocked (includes false positives)\n";
    cout << "  FP = false positive rate (sensor says blocked when it isn't)\n";
    cout << "  FN = false negative rate (sensor misses a real obstacle)\n\n";

    DynamicEnvironment dynamicEnv(21, 11);
    dynamicEnv.setStart(Position(0,  0));
    dynamicEnv.setGoal( Position(20, 10));

    for (int y = 2; y <= 6;  ++y) dynamicEnv.setObstacle(Position(7,  y));
    for (int x = 8; x <= 14; ++x) dynamicEnv.setObstacle(Position(x,  6));

    std::vector<Position> horizontalPatrol;
    for (int x =  2; x <= 17; ++x) horizontalPatrol.push_back(Position(x, 8));
    for (int x = 16; x >=  3; --x) horizontalPatrol.push_back(Position(x, 8));
    dynamicEnv.addDynamicObstacle(horizontalPatrol, 2);

    std::vector<Position> verticalPatrol;
    for (int y = 1; y <= 8; ++y) verticalPatrol.push_back(Position(16, y));
    for (int y = 7; y >= 2; --y) verticalPatrol.push_back(Position(16, y));
    dynamicEnv.addDynamicObstacle(verticalPatrol, 1);

    cout << "Environment: 21x11  Start: (0,0)  Goal: (20,10)\n";
    cout << "Dynamic obstacles: " << dynamicEnv.getDynamicObstacleCount() << "\n";
    cout << "  Obstacle 1 — horizontal patrol y = 8,  x = 2..17, 1 step per 2 ticks\n";
    cout << "  Obstacle 2 — vertical patrol   x = 16, y = 1..8,  1 step per tick\n\n";

    DStarLite dstar;
    std::vector<Position> dynamicPath = dstar.findPath(dynamicEnv,
                                                        dynamicEnv.getStart(),
                                                        dynamicEnv.getGoal());
    cout << "--- Tick 0 (initial) ---\n";
    if (dynamicPath.empty()) cout << "  [no path found]\n";
    else cout << "  Path length: " << dynamicPath.size() << " steps\n\n";

    const int ticksPerSnapshot = 5;
    const int totalTicks       = 15;
    for (int tick = 1; tick <= totalTicks; ++tick)
    {
        dynamicEnv.tick();
        if (tick % ticksPerSnapshot == 0)
        {
            dynamicEnv.reset();
            dynamicPath = dstar.findPath(dynamicEnv,
                                         dynamicEnv.getStart(),
                                         dynamicEnv.getGoal());
            cout << "--- Tick " << tick << " ---\n";
            if (dynamicPath.empty())
                cout << "  [no path — obstacles blocked all routes]\n\n";
            else
                cout << "  Path length: " << dynamicPath.size() << " steps\n\n";
        }
    }

    SensorModel sensor(5, 0.0, 0.0);
    std::vector<SensorModel::Observation> readings =
        sensor.observe(dynamicEnv.getStart(), dynamicEnv);

    int obstaclesDetected = static_cast<int>(
        std::count_if(readings.begin(), readings.end(),
                      [](const SensorModel::Observation& obs) { return obs.reportedAsObstacle; }));

    cout << "Sensor observation from start (0,0):\n";
    cout << "  Range:              " << sensor.getSensorRange() << " (Manhattan)\n";
    cout << "  Cells scanned:      " << readings.size()         << "\n";
    cout << "  Obstacles detected: " << obstaclesDetected        << "\n";
    cout << "  Cells clear:        " << readings.size() - obstaclesDetected << "\n";
    cout << "  (cells beyond range are unknown — planner uses prior beliefs)\n\n";

    SensorModel noisySensor(5, 0.1, 0.05);
    std::vector<SensorModel::Observation> noisyReadings =
        noisySensor.observe(dynamicEnv.getStart(), dynamicEnv);

    int noisyObstacles = static_cast<int>(
        std::count_if(noisyReadings.begin(), noisyReadings.end(),
                      [](const SensorModel::Observation& obs) { return obs.reportedAsObstacle; }));

    cout << "Noisy sensor (FP = 10%, FN = 5%) from start (0,0):\n";
    cout << "  Obstacles reported: " << noisyObstacles << "\n";
    cout << "  (perfect sensor reported " << obstaclesDetected
         << " — difference is sensor noise)\n";

    // ---- Tabular RL on the dynamic grid -------------------------------------
    Visualizer::printSection("Tabular RL on Dynamic Grid");
    cout << "Q-Learning trains while obstacles move between episodes.\n";
    cout << "After training, the greedy policy is evaluated once on the current layout.\n\n";

    const int dynamicRlEpisodes = (profile == RunProfile::FULL) ? 400 : 120;
    const int dynamicLogInterval = (profile == RunProfile::FULL) ? 100 : 40;

    dynamicEnv.reset();
    RLEnvironment rlEnvironment(dynamicEnv);
    QLearningAgent qLearningAgent(rlEnvironment, 0.1, 0.95, 1.0, 0.05, 0.99);
    const int maxStepsPerEpisode = rlEnvironment.getWidth() * rlEnvironment.getHeight() * 2;

    cout << "Training for " << dynamicRlEpisodes << " episodes on dynamic grid...\n";
    for (int episodeNumber = 1; episodeNumber <= dynamicRlEpisodes; ++episodeNumber)
    {
        qLearningAgent.runEpisode(episodeNumber, maxStepsPerEpisode);
        dynamicEnv.tick();
        if (episodeNumber % dynamicLogInterval == 0)
            cout << "  completed episode " << episodeNumber << "\n";
    }

    vector<Position> greedyPath = qLearningAgent.extractGreedyPath(maxStepsPerEpisode);
    if (greedyPath.empty())
        cout << "Greedy policy did not reach the goal within step limit.\n";
    else
        cout << "Greedy path length after training: " << greedyPath.size() << " steps\n";
}

// =============================================================================
// Run all C++ demos in the original sequence with final comparison
// =============================================================================

static void runAllCppDemos(RunProfile profile)
{
    Environment env = buildScenario();

    vector<PathResult> results = sectionAlgorithms(env);
    sectionCBS();
    sectionBayesian(results);
    vector<pair<string,int>> rlSummary = sectionTabularRL(env, profile);
    sectionDynamicObstacles(profile);
    sectionNeuralAStar();

    Visualizer::printSection("Full System Comparison");
    Visualizer::displayFinalSummary(results, rlSummary);
}

static void runGoldenPath()
{
    Visualizer::printSection("Golden Path (quick verification)");
    cout << "Runs a short slice of each major C++ area using QUICK profile.\n";
    cout << "Use this after code changes to confirm the system still hangs together.\n\n";

    runSmokeTests();

    Environment env = buildCompactScenario();
    cout << "\nClassical planners (A*, JPS, Dijkstra) on 41x41 labyrinth:\n\n";
    vector<unique_ptr<IPathfinder>> goldenAlgos;
    goldenAlgos.push_back(make_unique<AStar>());
    goldenAlgos.push_back(make_unique<JPS>());
    goldenAlgos.push_back(make_unique<Dijkstra>());
    vector<PathResult> results;
    for (unique_ptr<IPathfinder>& algo : goldenAlgos)
    {
        env.reset();
        PathResult result = runTimed(*algo, env, env.getStart(), env.getGoal());
        results.push_back(result);
        Visualizer::displayStats(result.algorithmName, result.path,
                                 result.elapsedMs, result.pathCost);
    }

    sectionBayesian(results);
    sectionTabularRL(env, RunProfile::QUICK);
    sectionDynamicObstacles(RunProfile::QUICK);
    sectionNeuralAStar();

    cout << "\nGolden path complete.\n";
}

// =============================================================================
// Menu
// =============================================================================

static void menuCppDemos()
{
    string input;
    while (true)
    {
        cout << "\n";
        Visualizer::printSection("C++ Demos");
        cout << "  Training profile: " << runProfileLabel(gRunProfile) << " (toggle from main menu)\n\n";
        cout << "  1. Classical Pathfinding (9 algorithms)\n";
        cout << "  2. Multi-Agent Pathfinding (CBS)\n";
        cout << "  3. Bayesian Sensor Fusion\n";
        cout << "  4. Tabular RL — Q-Learning + Dyna-Q\n";
        cout << "  5. Dynamic Obstacles + Sensors + RL\n";
        cout << "  6. TD-λ Eligibility Traces\n";
        cout << "  7. Neural A* (requires python/data/weights.bin)\n";
        cout << "  0. Back\n";
        cout << "\nChoice: ";
        if (!(cin >> input)) break;

        if      (input == "1") { auto env = buildScenario(); sectionAlgorithms(env); }
        else if (input == "2") { sectionCBS(); }
        else if (input == "3") { sectionBayesian({}); }
        else if (input == "4") { auto env = buildScenario(); sectionTabularRL(env, gRunProfile); }
        else if (input == "5") { sectionDynamicObstacles(gRunProfile); }
        else if (input == "6") { sectionTDLambda(gRunProfile); }
        else if (input == "7") { sectionNeuralAStar(); }
        else if (input == "0") { break; }
        else { cout << "Unknown option.\n"; }
    }
}

static void menuTraining()
{
    string input;
    while (true)
    {
        cout << "\n";
        Visualizer::printSection("Python — Training");
        cout << "  1.  Generate Heuristic Training Data\n";
        cout << "  2.  Train Heuristic Network (Neural A*)\n";
        cout << "  3.  Train DQN (Double + Dueling)\n";
        cout << "  4.  Train DQN + HER\n";
        cout << "  5.  Train PPO\n";
        cout << "  6.  Train Decision Transformer\n";
        cout << "  7.  Train Trajectory Diffuser\n";
        cout << "  8.  Train SAC (entropy-regularised, off-policy)\n";
        cout << "  9.  Train AlphaZero (MCTS + neural policy/value)\n";
        cout << "  10. Train World Models (neural Dyna-Q)\n";
        cout << "  11. Train LSTM PPO (recurrent, partial observability)\n";
        cout << "  0.  Back\n";
        cout << "\nChoice: ";
        if (!(cin >> input)) break;

        if      (input == "1")  { runPythonScript("generate_heuristic_data.py"); }
        else if (input == "2")  { runPythonScript("train_heuristic_net.py"); }
        else if (input == "3")  { runPythonScript("train_dqn.py"); }
        else if (input == "4")  { runPythonScript("train_dqn_her.py"); }
        else if (input == "5")  { runPythonScript("train_ppo.py"); }
        else if (input == "6")  { runPythonScript("train_decision_transformer.py"); }
        else if (input == "7")  { runPythonScript("train_diffuser.py"); }
        else if (input == "8")  { runPythonScript("train_sac.py"); }
        else if (input == "9")  { runPythonScript("train_alphazero.py"); }
        else if (input == "10") { runPythonScript("train_world_model.py"); }
        else if (input == "11") { runPythonScript("train_lstm_ppo.py"); }
        else if (input == "0")  { break; }
        else { cout << "Unknown option.\n"; }
    }
}

static void menuBenchmarks()
{
    string input;
    while (true)
    {
        cout << "\n";
        Visualizer::printSection("Python — Benchmarks & Visualisation");
        cout << "  1. Benchmark Neural A*\n";
        cout << "  2. Benchmark All Algorithms (statistical)\n";
        cout << "  3. Compare: Tabular Q-Learning vs DQN (same maze)\n";
        cout << "  4. Deep RL Comparison (DQN vs SAC vs PPO vs LSTM, 3 seeds)\n";
        cout << "  5. World Model Generalization Test\n";
        cout << "  0. Back\n";
        cout << "\nChoice: ";
        if (!(cin >> input)) break;

        if      (input == "1") { runPythonScript("benchmark_neural_astar.py"); }
        else if (input == "2") { runPythonScript("statistical_benchmark.py"); }
        else if (input == "3") { runPythonScript("benchmark_compare.py"); }
        else if (input == "4") { runPythonScript("benchmark_deep_rl.py"); }
        else if (input == "5") { runPythonScript("benchmark_world_model_gen.py"); }
        else if (input == "0") { break; }
        else { cout << "Unknown option.\n"; }
    }
}

static void toggleRunProfile()
{
    gRunProfile = (gRunProfile == RunProfile::FULL) ? RunProfile::QUICK : RunProfile::FULL;
    cout << "Training profile is now: " << runProfileLabel(gRunProfile) << "\n";
    cout << "  FULL  = " << totalCurriculumEpisodes(RunProfile::FULL)  << " tabular RL episodes\n";
    cout << "  QUICK = " << totalCurriculumEpisodes(RunProfile::QUICK) << " tabular RL episodes\n";
}

int main()
{
    string input;
    while (true)
    {
        cout << "\n";
        Visualizer::printSection("PathPlanning System");
        cout << "  Training profile: " << runProfileLabel(gRunProfile) << "\n\n";
        cout << "  1. C++ Demos\n";
        cout << "  2. Python — Training\n";
        cout << "  3. Python — Benchmarks & Visualisation\n";
        cout << "  S. Smoke tests — C++ (fast regression checks)\n";
        cout << "  P. Smoke tests — Python (network shapes, agent construction)\n";
        cout << "  G. Golden path (quick end-to-end C++ tour)\n";
        cout << "  A. Run all C++ demos (uses current profile)\n";
        cout << "  B. Build Python module (./build.sh 3)\n";
        cout << "  T. Toggle FULL / QUICK training profile\n";
        cout << "  0. Exit\n";
        cout << "\nChoice: ";
        if (!(cin >> input)) break;

        if      (input == "1")                 { menuCppDemos(); }
        else if (input == "2")                 { menuTraining(); }
        else if (input == "3")                 { menuBenchmarks(); }
        else if (input == "s" || input == "S") { runSmokeTests(); }
        else if (input == "p" || input == "P") { runPythonScript("tests/smoke_tests.py"); }
        else if (input == "g" || input == "G") { runGoldenPath(); }
        else if (input == "a" || input == "A") { runAllCppDemos(gRunProfile); }
        else if (input == "b" || input == "B") { runBuildPythonModule(); }
        else if (input == "t" || input == "T") { toggleRunProfile(); }
        else if (input == "0")                 { break; }
        else { cout << "Unknown option.\n"; }
    }

    return 0;
}
