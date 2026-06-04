#!/bin/bash
# build.sh — build and training commands for the PathPlanning project.
# Run from project root.
#
# Usage:
#   ./build.sh              — full C++ binary        → ./pathplanning
#   ./build.sh debug        — debug binary           → ./pathplanning_debug
#   ./build.sh py           — pybind11 .so (required before all Python targets)
#
#   ./build.sh dqn          — DQN + Double + Dueling + HER
#   ./build.sh sac          — Soft Actor-Critic (off-policy, twin critics)
#   ./build.sh ppo          — PPO + LSTM-PPO (recurrent, partial observability)
#   ./build.sh alphazero    — AlphaZero (MCTS + neural value/policy, self-play)
#   ./build.sh world        — World Model (dynamics net + imagined rollouts)
#   ./build.sh bc           — Behavioural Cloning    → bc_model.pth
#   ./build.sh dagger       — DAgger                 → dagger_model.pth
#   ./build.sh heuristic    — Generate data + train heuristic net → weights.bin
#   ./build.sh attention    — Transformer + Decision Transformer + Diffuser
#
#   ./build.sh bench        — all benchmarks (requires relevant models trained)
#   ./build.sh clean        — remove binaries and .so

TARGET=${1:-cpp}

CPP_SOURCES=(
    main.cpp
    core/Position.cpp
    core/Types.cpp
    environment/Cell.cpp
    environment/Environment.cpp
    environment/DynamicEnvironment.cpp
    environment/SensorModel.cpp
    environment/KalmanTracker.cpp
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
    planning/hierarchical/AbstractMap.cpp
    planning/hierarchical/HierarchicalPlanner.cpp
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

PY="source venv/bin/activate 2>/dev/null || true"

case $TARGET in

  cpp|"")
    echo "==> Building full C++ binary..."
    set -e
    g++ -std=c++17 -Wall -Wextra -O2 "${CPP_SOURCES[@]}" -o pathplanning
    echo "Build OK → ./pathplanning"
    ;;

  debug)
    echo "==> Building debug binary..."
    set -e
    g++ -std=c++17 -Wall -Wextra -g "${CPP_SOURCES[@]}" -o pathplanning_debug
    echo "Build OK → ./pathplanning_debug"
    ;;

  py)
    echo "==> Building pybind11 .so..."
    set -e
    python setup.py build_ext --inplace
    python -c "import pathplanning; print('Binding OK')"
    ;;

  dqn)
    echo "==> Training DQN (Double + Dueling + HER)..."
    set -e; eval "$PY"
    python python/train_dqn.py
    python python/train_dqn_her.py
    ;;

  sac)
    echo "==> Training SAC..."
    set -e; eval "$PY"
    python python/train_sac.py
    ;;

  ppo)
    echo "==> Training PPO + LSTM-PPO..."
    set -e; eval "$PY"
    python python/train_ppo.py
    python python/train_lstm_ppo.py
    ;;

  alphazero)
    echo "==> Training AlphaZero..."
    set -e; eval "$PY"
    python python/train_alphazero.py
    ;;

  world)
    echo "==> Training World Model..."
    set -e; eval "$PY"
    python python/train_world_model.py
    ;;

  bc)
    echo "==> Training Behavioural Cloning → bc_model.pth..."
    set -e; eval "$PY"
    python python/train_bc.py
    ;;

  dagger)
    echo "==> Training DAgger → dagger_model.pth..."
    set -e; eval "$PY"
    python python/train_dagger.py
    ;;

  heuristic)
    echo "==> Generating heuristic data + training network → weights.bin..."
    set -e; eval "$PY"
    python python/generate_heuristic_data.py
    cd python && python train_heuristic_net.py && cd ..
    ;;

  attention)
    echo "==> Training Transformer + Decision Transformer + Diffuser..."
    set -e; eval "$PY"
    cd python
    python train_transformer_policy.py
    python train_decision_transformer.py
    python train_diffuser.py
    cd ..
    ;;

  bench)
    echo "==> Running all benchmarks..."
    set -e; eval "$PY"
    python python/statistical_benchmark.py
    python python/benchmark_deep_rl.py
    python python/benchmark_imitation.py
    python python/benchmark_world_model_gen.py
    python python/tests/smoke_tests.py
    ;;

  clean)
    echo "==> Cleaning build artifacts..."
    rm -f pathplanning pathplanning_debug
    rm -f pathplanning*.so build/*.so
    echo "Clean OK"
    ;;

  *)
    echo "Unknown target '$TARGET'."
    echo "Valid: (none) | debug | py | dqn | sac | ppo | alphazero | world | bc | dagger | heuristic | attention | bench | clean"
    exit 1
    ;;

esac
