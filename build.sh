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
#   ./build.sh 6      — Phase 6: BC training  (imitation learning)
#   ./build.sh 7      — Phase 6: DAgger training
#   ./build.sh 8      — Phase 6: benchmark all imitation policies
#   ./build.sh 9      — Phase 7: statistical benchmark (1000 seeds)
#   ./build.sh 10     — Phase 7: benchmark plots (bar charts + classical vs RL)

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
        planning/CurriculumScheduler.cpp \
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

  6)
    echo "==> Running Behavioral Cloning training..."
    # Requires: ./build.sh 3 first, pip install torch
    set -e
    source venv/bin/activate 2>/dev/null || true
    python python/train_bc.py
    ;;

  7)
    echo "==> Running DAgger training..."
    # Requires: ./build.sh 3 first, pip install torch
    set -e
    source venv/bin/activate 2>/dev/null || true
    python python/train_dagger.py
    ;;

  8)
    echo "==> Running imitation learning benchmark..."
    # Requires: ./build.sh 6 and ./build.sh 7 first (generates bc_model.pth, dagger_model.pth)
    set -e
    source venv/bin/activate 2>/dev/null || true
    python python/benchmark_imitation.py
    ;;

  9)
    echo "==> Running Phase 7 statistical benchmark (1000 seeds)..."
    # Requires: ./build.sh 3 first (pybind .so), bc_model.pth + dagger_model.pth optional
    set -e
    source venv/bin/activate 2>/dev/null || true
    python python/statistical_benchmark.py
    ;;

  10)
    echo "==> Plotting Phase 7 benchmark results..."
    # Requires: ./build.sh 9 first (generates benchmark_results.csv)
    set -e
    source venv/bin/activate 2>/dev/null || true
    python python/benchmark_plot.py
    ;;

  *)
    echo "Unknown target '$TARGET'. Valid: 1 2 3 4 5 6 7 8 9 10"
    exit 1
    ;;

esac
