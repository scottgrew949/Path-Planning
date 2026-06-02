# Robot Path Planning System

A ground-up C++ implementation of a robot navigation system bridging classical AI planning and modern reinforcement learning. Simulates the core software stack of an autonomous vehicle: perceiving an environment, planning a path through it, and learning from experience.

Built entirely without external AI/ML libraries. Every algorithm, data structure, and learning mechanism implemented from scratch in production-quality C++17.

---

## What It Demonstrates

- **Classical graph search** — A\*, Dijkstra, BFS, Bidirectional A\*, Theta\* (any-angle), JPS (jump point search), D\* Lite (incremental replanning), RRT (random tree)
- **Tabular RL from scratch** — Q-learning, Bellman equation, epsilon-greedy exploration, Dyna-Q model-based planning
- **Deep RL** — DQN, Double DQN, Dueling DQN via C++ environment + PyTorch bridge
- **Clean systems architecture** — abstract interfaces, polymorphism, separation of concerns, zero external AI dependencies in the C++ core
- **Real-world relevance** — every component maps directly to autonomous vehicle technology

### Self-Driving Car Analog

| Component | AV Equivalent |
|---|---|
| Environment (occupancy grid) | What LiDAR produces |
| Classical algorithms | Global route planner (Google Maps layer) |
| RL agent | Local motion policy (reacts to dynamic obstacles) |
| DQN | Neural net replacing lookup table for large state spaces |
| Pi + Camera | Real-time perception pipeline feeding live obstacle data |

---

## Build

All commands run from the project root.

```bash
./build.sh        # full binary — planners + Q-Learning + Dyna-Q + TD(λ) (default)
./build.sh 1      # full binary with debug symbols (-g)
./build.sh 2      # full binary with -O2 (same sources as default)
./build.sh 3      # Python .so binding via pybind11 (required before DQN)
./build.sh 4      # DQN deep RL training (PyTorch)
./build.sh 5      # tabular RL training curves (matplotlib)
```

Run the interactive driver (one binary, all C++ and Python entry points):
```bash
./build.sh 2
./pathplanning
```

### Main menu

| Key | Action |
|-----|--------|
| `1` | C++ demos (classical, CBS, Bayes, tabular RL, dynamic+RL, TD-λ, Neural A*) |
| `2` | Python training scripts (heuristic net, DQN, PPO, …) |
| `3` | Python benchmarks and plots |
| `S` | Smoke tests — fast regression checks (~seconds) |
| `G` | Golden path — quick end-to-end C++ tour (uses QUICK profile for RL) |
| `A` | Run all C++ demos in sequence |
| `B` | Build Python module (`./build.sh 3`) |
| `T` | Toggle **FULL** / **QUICK** training (15k vs 1k curriculum episodes) |
| `0` | Exit |

**FULL** vs **QUICK** applies to tabular RL sections (menu 4, 5, 6, and option A). Classical planning is unchanged.

**Neural A*** (C++ menu 7) needs `python/data/weights.bin` from Python Training → generate data → train heuristic (or `./build.sh 11`–`12`). The same weights are used by Python benchmarks via `./build.sh 3`.

---

## Algorithms

### Classical Pathfinding

| Algorithm | Strategy | Optimal? | Notes |
|---|---|---|---|
| A\* | Best-first + heuristic | Yes | Manhattan distance heuristic |
| Dijkstra | Uniform cost | Yes | No directional bias |
| BFS | Hop count | Unweighted | Ignores edge costs |
| Bidirectional A\* | Meets in the middle | Yes | ~3.5x fewer nodes than A\* |
| Theta\* | Any-angle A\* | Near-optimal | Line-of-sight shortcuts via Bresenham |
| JPS | Jump point search | Yes | Skips symmetric corridor paths |
| D\* Lite | Backward A\* | Yes | Incremental replan on obstacle change |
| RRT | Random tree | No | Probabilistic, continuous-space capable |

### Tabular RL (C++)

| Agent | Method | Notes |
|---|---|---|
| Q-Learning | Temporal difference | 1 Bellman update per real step |
| Dyna-Q | Model-based TD | n=10 imagined updates per real step, ~10x faster convergence |

### Deep RL (Python + PyTorch)

- DQN with replay buffer and target network
- Double DQN (decoupled action selection / value estimation)
- Dueling DQN (value stream + advantage stream)
- PPO / Actor-Critic (in progress)

---

## Architecture

```
core/               Position, Types, enums
environment/        Occupancy grid (dual: vector<Cell> + bitset<10000>)
planning/
  IPathfinder.h     Abstract interface — all 8 algorithms polymorphic
  algorithms/       AStar, Dijkstra, BFS, BidirectionalAStar, ThetaStar,
                    JPS, DStarLite, RRT
rl/
  RLAgent           Abstract base (mirrors IPathfinder)
  QLearningAgent    Tabular Q-learning
  DynaQAgent        Model-based RL with world model
  RLEnvironment     Gym-style wrapper (reset / step)
  QTable            State-action value store
utils/              ProbabilityUtils (Bayesian sensor fusion, entropy)
visualization/      Grid and path rendering, summary table
python/             DQN, Dueling DQN, Double DQN, Actor-Critic, PPO
```

---

## Output

Running `./pathplanning`:

1. **Environment** — 201×41 labyrinth grid rendered to terminal
2. **Pathfinding** — all 8 algorithms run, path overlaid on grid, stats printed
3. **Algorithm comparison table** — path length, cost, time, nodes explored
4. **Bayesian sensor fusion** — occupancy grid update demo
5. **Q-Learning training** — 10k episodes, convergence table, greedy path
6. **Dyna-Q training** — 10k episodes, convergence table, greedy path
7. **CSV output** — `qlearning_training.csv`, `dynaq_training.csv` for plotting

Plot training curves after running:
```bash
./build.sh 5
```

---

## Hardware Extension (Planned)

Raspberry Pi + overhead camera feeds live obstacle detection into `env.setObstacle()`. RL agent navigates a physical grid mirroring the real world in real time.

---

## Roadmap

- [x] Phase 1 — Classical planning (A\*, Dijkstra, BFS, Bidirectional A\*, Theta\*)
- [x] Phase 2 — Additional algorithms (JPS, D\* Lite, RRT)
- [x] Phase 3 — Tabular RL (Q-Learning, Dyna-Q)
- [x] Phase 4 — Deep RL (DQN, Double DQN, Dueling DQN)
- [ ] Phase 5 — Policy gradient (PPO / Actor-Critic)
- [ ] Phase 6 — Imitation learning (behavioural cloning, DAgger)
- [ ] Phase 7 — Curriculum learning + statistical benchmarking
- [ ] Phase 8 — Dynamic obstacles + partial observability
- [ ] Phase 9 — Neural A\* (learned heuristic)
- [ ] Phase 10 — Attention-based policy (Transformer, Decision Transformer)
- [ ] Phase 11 — Hardware integration (Raspberry Pi + camera)
- [ ] Phase 12 — OpenStreetMap integration
