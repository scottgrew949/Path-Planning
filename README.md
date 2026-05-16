# # Robot Path Planning System
    2
    3 A ground-up C++ implementation of a robot navigation system bridging classi
      cal AI planning and modern reinforcement learning. Simulates the core softw
      are stack of an autonomous vehicle: perceiving an environment, planning a p
      ath through it, and learning from experience.
    4
    5 Built entirely without external AI/ML libraries. Every algorithm, data stru
      cture, and learning mechanism implemented from scratch in production-qualit
      y C++17.
    6
    7 ---
    8
    9 ## What It Demonstrates
   10
   11 - **Classical graph search** — A\*, Dijkstra, BFS, Bidirectional A\*, Theta
      \* (any-angle), JPS (jump point search), D\* Lite (incremental replanning),
       RRT (random tree)
   12 - **Tabular RL from scratch** — Q-learning, Bellman equation, epsilon-greed
      y exploration, Dyna-Q model-based planning
   13 - **Deep RL** — DQN, Double DQN, Dueling DQN via C++ environment + PyTorch
      bridge
   14 - **Clean systems architecture** — abstract interfaces, polymorphism, separ
      ation of concerns, zero external AI dependencies in the C++ core
   15 - **Real-world relevance** — every component maps directly to autonomous ve
      hicle technology
   16
   17 ### Self-Driving Car Analog
   18
   19 | Component | AV Equivalent |
   20 |---|---|
   21 | Environment (occupancy grid) | What LiDAR produces |
   22 | Classical algorithms | Global route planner (Google Maps layer) |
   23 | RL agent | Local motion policy (reacts to dynamic obstacles) |
   24 | DQN | Neural net replacing lookup table for large state spaces |
   25 | Pi + Camera | Real-time perception pipeline feeding live obstacle data |
   26
   27 ---
   28
   29 ## Build
   30
   31 All commands run from the project root.
   32
   33 ```bash
   34 ./build.sh        # full binary — all 8 algorithms + Q-Learning + Dyna-Q (d
      efault)
   35 ./build.sh 1      # classical pathfinding only (no RL, fastest compile)
   36 ./build.sh 2      # full binary (same as default)
   37 ./build.sh 3      # Python .so binding via pybind11 (required before DQN)
   38 ./build.sh 4      # DQN deep RL training (PyTorch)
   39 ./build.sh 5      # tabular RL training curves (matplotlib)
   40 ```
   41
   42 Run the full binary:
   43 ```bash
   44 ./pathplanning
   45 ```
   46
   47 ---
   48
   49 ## Algorithms
   50
   51 ### Classical Pathfinding
   52
   53 | Algorithm | Strategy | Optimal? | Notes |
   54 |---|---|---|---|
   55 | A\* | Best-first + heuristic | Yes | Manhattan distance heuristic |
   56 | Dijkstra | Uniform cost | Yes | No directional bias |
   57 | BFS | Hop count | Unweighted | Ignores edge costs |
   58 | Bidirectional A\* | Meets in the middle | Yes | ~3.5x fewer nodes than A\
      * |
   59 | Theta\* | Any-angle A\* | Near-optimal | Line-of-sight shortcuts via Bres
      enham |
   60 | JPS | Jump point search | Yes | Skips symmetric corridor paths |
   61 | D\* Lite | Backward A\* | Yes | Incremental replan on obstacle change |
   62 | RRT | Random tree | No | Probabilistic, continuous-space capable |
   63
   64 ### Tabular RL (C++)
   65
   66 | Agent | Method | Notes |
   67 |---|---|---|
   68 | Q-Learning | Temporal difference | 1 Bellman update per real step |
   69 | Dyna-Q | Model-based TD | n=10 imagined updates per real step, ~10x faste
      r convergence |
   70
   71 ### Deep RL (Python + PyTorch)
   72
   73 - DQN with replay buffer and target network
   74 - Double DQN (decoupled action selection / value estimation)
   75 - Dueling DQN (value stream + advantage stream)
   76 - PPO / Actor-Critic (in progress)
   77
   78 ---
   75 - Dueling DQN (value stream + advantage stream)
   76 - PPO / Actor-Critic (in progress)
   77
   78 ---
   79
   80 ## Architecture
   81
   82 ```
   83 core/               Position, Types, enums
   84 environment/        Occupancy grid (dual: vector<Cell> + bitset<10000>)
   85 planning/
   86   IPathfinder.h     Abstract interface — all 8 algorithms polymorphic
   87   algorithms/       AStar, Dijkstra, BFS, BidirectionalAStar, ThetaStar,
   88                     JPS, DStarLite, RRT
   89 rl/
   90   RLAgent           Abstract base (mirrors IPathfinder)
   91   QLearningAgent    Tabular Q-learning
   92   DynaQAgent        Model-based RL with world model
   93   RLEnvironment     Gym-style wrapper (reset / step)
   94   QTable            State-action value store
   95 utils/              ProbabilityUtils (Bayesian sensor fusion, entropy)
   96 visualization/      Grid and path rendering, summary table
   97 python/             DQN, Dueling DQN, Double DQN, Actor-Critic, PPO
   98 ```
   99
  100 ---
  101
  102 ## Output
  103
  104 Running `./pathplanning`:
  105
  106 1. **Environment** — 201×41 labyrinth grid rendered to terminal
  107 2. **Pathfinding** — all 8 algorithms run, path overlaid on grid, stats printed
  108 3. **Algorithm comparison table** — path length, cost, time, nodes explored
  109 4. **Bayesian sensor fusion** — occupancy grid update demo
  110 5. **Q-Learning training** — 10k episodes, convergence table, greedy path
  111 6. **Dyna-Q training** — 10k episodes, convergence table, greedy path
  112 7. **CSV output** — `qlearning_training.csv`, `dynaq_training.csv` for plotting
  113
  114 Plot training curves after running:
  115 ```bash
  116 ./build.sh 5
  117 ```
  118
  119 ---
