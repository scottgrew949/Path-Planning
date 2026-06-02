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
#   ./build.sh 21     — Train SAC (entropy-regularised, off-policy, twin critics)
#   ./build.sh 22     — Train AlphaZero (MCTS + neural policy/value network)
#   ./build.sh 23     — Train World Models (neural Dyna-Q, imagined rollouts)
#   ./build.sh 24     — Train LSTM PPO (recurrent policy, partial observability)
#   ./build.sh 26     — Deep RL benchmark (DQN vs SAC vs PPO vs LSTM, curriculum, 3 seeds)
#   ./build.sh 27     — World Model generalization test (train on 1 maze, eval on 4)

TARGET=${1:-2}

# All .cpp files required by main.cpp (keep in sync with main.cpp compile comment)
CPP_SOURCES=(
    main.cpp
    core/Position.cpp
    core/Types.cpp
    environment/Cell.cpp
    environment/Environment.cpp
    environment/DynamicEnvironment.cpp
    environment/SensorModel.cpp
    planning/algorithms/AStar.cpp
    planning/algorithms/Dijkstra.cpp
    planning/algorithms/BFS.cpp
    planning/algorithms/BidirectionalAStar.cpp
    planning/algorithms/ThetaStar.cpp
    planning/algorithms/JPS.cpp
    planning/algorithms/DStarLite.cpp
    planning/algorithms/RRT.cpp
    planning/algorithms/MCTS.cpp
    planning/algorithms/CBS.cpp
    rl/RLAgent.cpp
    rl/QLearningAgent.cpp
    rl/DynaQAgent.cpp
    rl/TDLambdaAgent.cpp
    rl/QTable.cpp
    rl/RLEnvironment.cpp
    planning/CurriculumScheduler.cpp
    planning/hybrid/HeuristicNetwork.cpp
    planning/hybrid/NeuralAStar.cpp
    utils/ProbabilityUtils.cpp
    visualization/Visualizer.cpp
    tests/SmokeTests.cpp
)

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
    g++ -std=c++17 -Wall -Wextra -g "${CPP_SOURCES[@]}" -o pathplanning_debug
    echo "Build OK → ./pathplanning_debug"
    ;;

  2)
    echo "==> Building full binary (all algorithms + tabular RL + dynamic env)..."
    set -e
    g++ -std=c++17 -Wall -Wextra -O2 "${CPP_SOURCES[@]}" -o pathplanning
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
    echo "==> Target 5 removed (training_curves.py deleted)."
    echo "    Training CSVs are written by ./pathplanning menu option 4."
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
    echo "==> Target 14 removed (heuristic_quality.py deleted)."
    echo "    Use ./build.sh 13 to benchmark Neural A* vs standard A*."
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
    echo "==> DQN training with PER + HER (goal-conditioned, sparse reward)..."
    # Requires: ./build.sh 3 (pybind11 .so), pip install torch
    # PER: prioritized experience replay — high-TD-error transitions sampled more often
    # HER: hindsight experience replay — failed episodes relabelled as successes
    set -e
    source venv/bin/activate 2>/dev/null || true
    python python/train_dqn_her.py
    ;;

  20)
    echo "==> MVP: Unified benchmark — all algorithms, 100 mazes..."
    # Requires: ./build.sh 3 (pybind11 .so). Neural A* optional (./build.sh 12)
    # Output:   python/data/benchmark_all_results.csv + stdout table
    set -e
    source venv/bin/activate 2>/dev/null || true
    python python/benchmark_all.py
    ;;

  21)
    echo "==> Training SAC (Soft Actor-Critic, entropy-regularised off-policy)..."
    # Requires: ./build.sh 3 (pybind11 .so), pip install torch
    # Output:   stdout training log (episode/reward/critic loss/actor loss/alpha)
    set -e
    source venv/bin/activate 2>/dev/null || true
    python python/train_sac.py
    ;;

  22)
    echo "==> Training AlphaZero (MCTS + neural policy + value network)..."
    # Requires: ./build.sh 3 (pybind11 .so), pip install torch
    # Output:   stdout training log (episode/goal reached/example count)
    set -e
    source venv/bin/activate 2>/dev/null || true
    python python/train_alphazero.py
    ;;

  23)
    echo "==> Training World Models (dynamics + reward networks, imagined rollouts)..."
    # Requires: ./build.sh 3 (pybind11 .so), pip install torch
    # Output:   stdout training log + evaluation success rate
    set -e
    source venv/bin/activate 2>/dev/null || true
    python python/train_world_model.py
    ;;

  25)
    echo "==> Comparing Tabular Q-Learning vs DQN on the same maze..."
    # Requires: ./build.sh 3 (pybind11 .so), pip install torch
    # Output:   stdout comparison table (success rate, avg steps, param count)
    set -e
    source venv/bin/activate 2>/dev/null || true
    python python/benchmark_compare.py
    ;;

  24)
    echo "==> Training LSTM PPO (recurrent policy for partial observability)..."
    # Requires: ./build.sh 3 (pybind11 .so), pip install torch
    # Output:   stdout training log (episode/total reward/loss)
    set -e
    source venv/bin/activate 2>/dev/null || true
    python python/train_lstm_ppo.py
    ;;

  26)
    echo "==> Deep RL benchmark (DQN vs SAC vs PPO vs LSTM, curriculum, 3 seeds)..."
    set -e
    source venv/bin/activate 2>/dev/null || true
    python python/benchmark_deep_rl.py
    ;;

  27)
    echo "==> World Model generalization test (train on 1 maze, eval on 4)..."
    set -e
    source venv/bin/activate 2>/dev/null || true
    python python/benchmark_world_model_gen.py
    ;;

  28)
    echo "==> Python smoke tests (network shapes, agent construction, buffer ops)..."
    # Requires: pip install torch — no pybind11 .so needed
    # Runs in under 5 seconds. Use after any change to networks/ or agents/.
    set -e
    source venv/bin/activate 2>/dev/null || true
    python python/tests/smoke_tests.py
    ;;

  *)
    echo "Unknown target '$TARGET'. Valid: 1–28"
    exit 1
    ;;

esac
