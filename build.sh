#!/bin/bash
# build.sh — all build commands for the PathPlanning project.
# Run from project root. Pass a target number or omit to build the full binary.
#
# Usage:
#   ./build.sh        — full C++ binary (all algorithms + tabular RL)
#   ./build.sh 1      — classical pathfinding only (no RL)
#   ./build.sh 2      — full C++ binary (default)
#   ./build.sh 3      — Python .so binding via setup.py
#   ./build.sh 4      — DQN deep RL training (Python/PyTorch)
#   ./build.sh 5      — tabular RL training curves (matplotlib)

TARGET=${1:-2}

case $TARGET in

  1)
    echo "==> Building classical pathfinding only..."
    set -e
    g++ -std=c++17 -Wall -Wextra -O2 \
        main.cpp \
        core/Position.cpp core/Types.cpp \
        environment/Cell.cpp environment/Environment.cpp \
        planning/algorithms/AStar.cpp \
        planning/algorithms/Dijkstra.cpp \
        planning/algorithms/BFS.cpp \
        planning/algorithms/BidirectionalAStar.cpp \
        planning/algorithms/ThetaStar.cpp \
        planning/algorithms/JPS.cpp \
        planning/algorithms/DStarLite.cpp \
        planning/algorithms/RRT.cpp \
        utils/ProbabilityUtils.cpp \
        visualization/Visualizer.cpp \
        -o pathplanning_classical
    echo "Build OK → ./pathplanning_classical"
    ;;

  2)
    echo "==> Building full binary (all algorithms + tabular RL)..."
    set -e
    g++ -std=c++17 -Wall -Wextra -O2 \
        main.cpp \
        core/Position.cpp core/Types.cpp \
        environment/Cell.cpp environment/Environment.cpp \
        planning/algorithms/AStar.cpp \
        planning/algorithms/Dijkstra.cpp \
        planning/algorithms/BFS.cpp \
        planning/algorithms/BidirectionalAStar.cpp \
        planning/algorithms/ThetaStar.cpp \
        planning/algorithms/JPS.cpp \
        planning/algorithms/DStarLite.cpp \
        planning/algorithms/RRT.cpp \
        rl/RLAgent.cpp rl/QLearningAgent.cpp rl/DynaQAgent.cpp \
        rl/QTable.cpp rl/RLEnvironment.cpp \
        utils/ProbabilityUtils.cpp \
        visualization/Visualizer.cpp \
        -o pathplanning
    echo "Build OK → ./pathplanning"
    ;;

  3)
    echo "==> Building Python .so binding (pybind11)..."
    # Requires: pip install pybind11
    set -e
    python setup.py build_ext --inplace
    python -c "import pathplanning; print('Binding OK')"
    ;;

  4)
    echo "==> Running DQN deep RL training (PyTorch)..."
    # Requires: ./build.sh 3 first, pip install torch
    set -e
    source venv/bin/activate 2>/dev/null || true
    python python/train_dqn.py
    ;;

  5)
    echo "==> Plotting tabular RL training curves..."
    # Requires: ./pathplanning run first (generates qlearning_training.csv, dynaq_training.csv)
    set -e
    source venv/bin/activate 2>/dev/null || true
    python training_curves.py
    ;;

  *)
    echo "Unknown target '$TARGET'. Valid: 1 2 3 4 5"
    exit 1
    ;;

esac
