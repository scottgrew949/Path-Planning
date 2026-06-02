// tests/SmokeTests.cpp
#include "SmokeTests.h"
#include "../core/Types.h"
#include "../core/Position.h"
#include "../environment/Environment.h"
#include "../planning/algorithms/AStar.h"
#include "../planning/algorithms/Dijkstra.h"
#include "../utils/ProbabilityUtils.h"
#include "../rl/RLEnvironment.h"
#include "../rl/QLearningAgent.h"
#include <cmath>
#include <iostream>
#include <string>

using namespace std;

namespace
{
    int gPassedCount = 0;
    int gFailedCount = 0;

    void check(bool condition, const string& testName)
    {
        if (condition)
        {
            ++gPassedCount;
            cout << "  [PASS] " << testName << '\n';
        }
        else
        {
            ++gFailedCount;
            cout << "  [FAIL] " << testName << '\n';
        }
    }

    void checkNear(double actual, double expected, double tolerance, const string& testName)
    {
        check(std::abs(actual - expected) <= tolerance, testName);
    }

    void testBayesianSensorUpdate()
    {
        double posterior = ProbabilityUtils::bayesUpdateSensor(0.2, 0.9, 0.1, true);
        checkNear(posterior, 0.692307, 0.001, "bayesUpdateSensor posterior after hit");
    }

    void testExpectedValueAndEntropy()
    {
        vector<double> costs = { 100.0, 200.0 };
        vector<double> probabilities = { 0.5, 0.5 };
        checkNear(ProbabilityUtils::expectedValue(costs, probabilities), 150.0, 0.001,
                  "expectedValue uniform two outcomes");
        checkNear(ProbabilityUtils::entropy(probabilities), 1.0, 0.001,
                  "entropy two equal outcomes");
    }

    void testEnvironmentObstacleLookup()
    {
        Environment environment(10, 10);
        environment.setStart(Position(0, 0));
        environment.setGoal(Position(9, 9));
        environment.setObstacle(Position(5, 5));

        check(environment.isObstacle(Position(5, 5)), "setObstacle marks cell");
        check(environment.isValid(Position(0, 1)), "start neighbour is walkable");
        check(!environment.isValid(Position(5, 5)), "obstacle cell is not valid");
    }

    void testAStarOnOpenGrid()
    {
        Environment environment(5, 5);
        environment.setStart(Position(0, 0));
        environment.setGoal(Position(4, 4));

        AStar astar;
        vector<Position> path = astar.findPath(environment, environment.getStart(), environment.getGoal());

        check(!path.empty(), "A* finds path on empty 5x5 grid");
        check(path.front() == environment.getStart(), "A* path starts at start");
        check(path.back() == environment.getGoal(), "A* path ends at goal");
    }

    void testDijkstraMatchesAStarCostOnOpenGrid()
    {
        Environment environment(8, 8);
        environment.setStart(Position(0, 0));
        environment.setGoal(Position(7, 7));

        AStar astar;
        Dijkstra dijkstra;
        vector<Position> astarPath = astar.findPath(environment, environment.getStart(), environment.getGoal());
        vector<Position> dijkstraPath = dijkstra.findPath(environment, environment.getStart(), environment.getGoal());

        check(!astarPath.empty() && !dijkstraPath.empty(),
              "A* and Dijkstra both find paths on open grid");

        double astarCost = 0.0;
        double dijkstraCost = 0.0;
        for (size_t stepIndex = 0; stepIndex + 1 < astarPath.size(); ++stepIndex)
        {
            const Position& previous = (stepIndex == 0) ? astarPath[0] : astarPath[stepIndex - 1];
            astarCost += environment.moveCost(previous, astarPath[stepIndex], astarPath[stepIndex + 1]);
        }
        for (size_t stepIndex = 0; stepIndex + 1 < dijkstraPath.size(); ++stepIndex)
        {
            const Position& previous = (stepIndex == 0) ? dijkstraPath[0] : dijkstraPath[stepIndex - 1];
            dijkstraCost += environment.moveCost(previous, dijkstraPath[stepIndex], dijkstraPath[stepIndex + 1]);
        }
        checkNear(astarCost, dijkstraCost, 0.01, "A* and Dijkstra equal cost on open grid");
    }

    void testQLearningSingleEpisode()
    {
        Environment environment(12, 12);
        environment.setStart(Position(0, 0));
        environment.setGoal(Position(11, 11));
        environment.generateLabyrinth(0.2, 99);

        RLEnvironment rlEnvironment(environment);
        QLearningAgent agent(rlEnvironment, 0.1, 0.95, 1.0, 0.05, 0.99);
        TrainingResult episodeResult = agent.runEpisode(1, 500);

        check(episodeResult.episodeNumber == 1, "QLearningAgent runEpisode returns episode number");
        check(episodeResult.stepsToGoal > 0, "QLearningAgent takes at least one step");
    }
}

SmokeTestSummary SmokeTests::runAll()
{
    gPassedCount = 0;
    gFailedCount = 0;

    cout << "\n============== Smoke Tests ==============\n";
    cout << "Fast checks for core math, grid, search, and one RL episode.\n\n";

    testBayesianSensorUpdate();
    testExpectedValueAndEntropy();
    testEnvironmentObstacleLookup();
    testAStarOnOpenGrid();
    testDijkstraMatchesAStarCostOnOpenGrid();
    testQLearningSingleEpisode();

    cout << "\nResults: " << gPassedCount << " passed, " << gFailedCount << " failed.\n";

    SmokeTestSummary summary;
    summary.passedCount = gPassedCount;
    summary.failedCount = gFailedCount;
    return summary;
}
