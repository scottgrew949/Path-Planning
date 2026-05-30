#!/bin/bash
# build.sh — all build commands for the PathPlanning project.
# Run from project root. Pass a target number or omit to build the full binary.
#
# Usage:
#   ./build.sh        — full C++ binary (all algorithms + tabular RL)
#   ./build.sh 1      — full binary with debug symbols (-g, no -O2)
#   ./build.sh 2      — full C++ binary (default)
#   ./build.sh 3      — Python .so binding via setup.py
#   ./build.sh 4      — DQN deep RL training (Python/PyTorch)
#   ./build.sh 5      — tabular RL training curves (matplotlib)
#   ./build.sh 6      — Phase 6: BC training  (imitation learning)
#   ./build.sh 7      — Phase 6: DAgger training
#   ./build.sh 8      — Phase 6: benchmark all imitation policies
#   ./build.sh 9      — Phase 7: statistical benchmark (1000 seeds)
#   ./build.sh 10     — Phase 7: benchmark plots (bar charts + classical vs RL)
#   ./build.sh 11     — Phase 9: generate heuristic training data (backward Dijkstra, 500 mazes)
#   ./build.sh 12     — Phase 9: train heuristic network → python/data/weights.bin
#   ./build.sh 13     — Phase 9: Neural A* benchmark vs standard A* (100 test mazes)
#   ./build.sh 14     — Phase 9: heuristic quality visualisation (scatter plot + node count bar)
#   ./build.sh 15     — Phase 10: train Transformer PPO policy (full grid state)
#   ./build.sh 16     — Phase 10: train Decision Transformer (offline, A* demonstrations)
#   ./build.sh 17     — Phase 10: train Trajectory Diffuser (path generation via DDPM)
#   ./build.sh 18     — Phase 10: benchmark all policies (MLP PPO vs Transformer PPO vs DT vs Diffuser)

TARGET=${1:-2}

if [ "$TARGET" = "clean" ]; then
    echo "==> Cleaning build artifacts..."
    rm -rf build/
    rm -f pathplanning pathplanning_debug
    rm -f pathplanning*.so
    echo "Clean OK"
    exit 0
fi

case $TARGET in

  1)
    echo "==> Building full binary with debug symbols..."
    set -e
    g++ -std=c++17 -Wall -Wextra -g \
        main.cpp \
        core/Position.cpp core/Types.cpp \
        environment/Cell.cpp environment/Environment.cpp \
        environment/DynamicEnvironment.cpp environment/SensorModel.cpp \
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
        -o pathplanning_debug
    echo "Build OK → ./pathplanning_debug"
    ;;

  2)
    echo "==> Building full binary (all algorithms + tabular RL + dynamic env)..."
    set -e
    g++ -std=c++17 -Wall -Wextra -O2 \
        main.cpp \
        core/Position.cpp core/Types.cpp \
        environment/Cell.cpp environment/Environment.cpp \
        environment/DynamicEnvironment.cpp environment/SensorModel.cpp \
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
    python python/visualization/training_curves.py
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

  11)
    echo "==> Phase 9: Generating heuristic training data (500 mazes, backward Dijkstra)..."
    # Requires: ./build.sh 3 first (pybind11 .so)
    # Output:   python/data/heuristic_training.npy  (~450K rows, shape (N,5))
    set -e
    source venv/bin/activate 2>/dev/null || true
    python python/generate_heuristic_data.py
    ;;

  12)
    echo "==> Phase 9: Training heuristic network (60 epochs, exports weights.bin + weights.pt)..."
    # Requires: ./build.sh 11 first (generates heuristic_training.npy)
    # Output:   python/data/weights.bin  (C++ binary format)
    #           python/data/weights.pt   (PyTorch state dict, for visualisation)
    set -e
    source venv/bin/activate 2>/dev/null || true
    cd python && python train_heuristic_net.py && cd ..
    ;;

  13)
    echo "==> Phase 9: Neural A* vs standard A* benchmark (100 test mazes)..."
    # Requires: ./build.sh 3 + ./build.sh 12 (weights.bin must exist)
    # Output:   prints mean±std table: nodes_explored, path_length, time_ms
    set -e
    source venv/bin/activate 2>/dev/null || true
    python python/benchmark_neural_astar.py
    ;;

  14)
    echo "==> Phase 9: Heuristic quality visualisation..."
    # Requires: ./build.sh 12 (weights.bin + weights.pt must exist)
    # Output:   heuristic_quality.png  (scatter plot + node count comparison)
    set -e
    source venv/bin/activate 2>/dev/null || true
    cd python && python visualization/heuristic_quality.py && cd ..
    ;;

  15)
    echo "==> Phase 10: Training Transformer PPO policy..."
    # Requires: ./build.sh 3 (pybind11 .so)
    # Output:   python/data/transformer_policy.pt
    set -e
    source venv/bin/activate 2>/dev/null || true
    cd python && python train_transformer_policy.py && cd ..
    ;;

  16)
    echo "==> Phase 10: Training Decision Transformer (offline, A* demonstrations)..."
    # Requires: ./build.sh 3 (pybind11 .so)
    # Output:   python/data/decision_transformer.pt
    set -e
    source venv/bin/activate 2>/dev/null || true
    cd python && python train_decision_transformer.py && cd ..
    ;;

  17)
    echo "==> Phase 10: Training Trajectory Diffuser (DDPM on A* paths)..."
    # Requires: ./build.sh 3 (pybind11 .so)
    # Output:   python/data/trajectory_diffuser.pt
    set -e
    source venv/bin/activate 2>/dev/null || true
    cd python && python train_diffuser.py && cd ..
    ;;

  18)
    echo "==> Phase 10: Benchmarking all Phase 10 policies..."
    # Requires: ./build.sh 15, 16, and 17 all complete
    # Output:   prints comparison table to stdout
    set -e
    source venv/bin/activate 2>/dev/null || true
    cd python && python benchmark_phase10.py && cd ..
    ;;

  19)
    echo "==> Running live path planning demo (A* + dynamic obstacles)..."
    # Requires: ./build.sh 3 first (pybind11 .so)
    # Requires: pip install matplotlib numpy
    set -e
    source venv/bin/activate 2>/dev/null || true
    python python/live_demo.py
    ;;

  *)
    echo "Unknown target '$TARGET'. Valid: 1–19"
    exit 1
    ;;

esac
