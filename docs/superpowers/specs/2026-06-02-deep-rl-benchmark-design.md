# Deep RL Benchmark + World Model Generalization — Design Spec
**Date:** 2026-06-02

## Problem

The project has 9+ RL algorithms but no way to compare whether they actually work or which is better. Training scripts are isolated — each trains on a single fixed maze with no curriculum. World Models trains on one maze and is never tested on new ones.

## Goals

1. Fair cross-algorithm comparison: DQN, SAC, PPO, LSTM PPO — same maze, same episode budget, same eval protocol, 3 seeds, mean±std reported.
2. Curriculum for deep RL agents matching the C++ tabular curriculum philosophy: easy → hard mazes.
3. World Models generalization test: does the trained model transfer to held-out mazes?

## Files

### New
- `python/benchmark_deep_rl.py` — unified benchmark runner (goal 1 + 2)
- `python/benchmark_world_model_gen.py` — world model generalization test (goal 3)

### Unchanged
All existing training scripts, networks, agents — benchmark imports them, does not modify them.

---

## benchmark_deep_rl.py

### Curriculum

| Stage | Obstacle density | Episodes | Maze seed |
|-------|-----------------|----------|-----------|
| 1     | 0.10            | 500      | 11        |
| 2     | 0.20            | 500      | 22        |
| 3     | 0.30            | 1000     | 33        |
| 4     | 0.35            | 1000     | 44        |

Total: 3000 episodes per algorithm per seed.
Maze swaps between stages — weights are kept (not reset). Intentional regression, same as tabular curriculum.

### Evaluation

- 100 greedy episodes (epsilon=0, no sampling noise)
- Held-out maze: density=0.35, seed=999 — never seen during training
- Same eval maze for all algorithms and all seeds

### Algorithms

| Algorithm | Network/Agent | Key difference |
|-----------|--------------|----------------|
| DQN       | DQNNetwork   | Off-policy, epsilon-greedy, Double+Dueling |
| SAC       | SACAgent     | Off-policy, entropy-regularized, twin critics |
| PPO       | PPOAgent + ActorCriticNetwork | On-policy, clipped surrogate |
| LSTM PPO  | LSTMActorCriticNetwork | Recurrent, partial observability |

### Per-algorithm adapter interface

Each adapter exposes:
- `train_one_stage(env, episodes)` — train for N episodes on current env
- `swap_maze(env)` — update env reference, keep weights
- `greedy_policy(state_tensor) -> int` — used during evaluation

### Multi-seed protocol

Run 3 seeds. Each seed:
1. Re-create all 4 agents from scratch (fresh weights)
2. Run full curriculum (4 stages, 3000 episodes total)
3. Evaluate on held-out maze (100 episodes)
4. Record: success_rate, avg_steps_on_success, first_goal_episode, train_time_seconds

Report mean ± std across 3 seeds for each metric.

### Output

```
============================================================
Deep RL Benchmark — 3000 eps, 4 curriculum stages, 3 seeds
Eval: 100 greedy eps, held-out maze (density=0.35, seed=999)
============================================================
Algorithm   Success%       Avg Steps      First Goal     Time(s)
------------------------------------------------------------
DQN         78.3 ± 4.1    142.1 ± 8.3    ep 312 ± 24    48 ± 3
SAC         82.0 ± 3.2    138.4 ± 6.1    ep 287 ± 18    61 ± 4
PPO         71.0 ± 5.5    159.2 ± 11.2   ep 401 ± 31    44 ± 2
LSTM PPO    85.3 ± 2.8    131.7 ± 5.9    ep 251 ± 15    73 ± 5
============================================================
```

Results also saved to `python/data/benchmark_deep_rl_results.csv`.

### CONCEPT blocks to include

- Why curriculum matters for deep RL (same as tabular: prevents policy memorising one layout)
- Why held-out eval maze is critical (training maze success rate is misleading)
- Why 3 seeds (single run variance is too high to draw conclusions)
- What first_goal_episode measures (sample efficiency — how fast does each algorithm start learning?)

---

## benchmark_world_model_gen.py

### Protocol

1. Train world model on maze A (density=0.3, seed=1) — full 3-phase pipeline
2. Evaluate policy on 4 mazes, 50 greedy episodes each:
   - Training maze (seed=1) — baseline
   - Maze B (seed=2)
   - Maze C (seed=3)
   - Maze D (seed=4)
3. Report success rate per maze + drop from training maze

### Output

```
================================================
World Model Generalization Test
Train maze: density=0.3, seed=1
================================================
Maze        Seed    Success%    vs Train
------------------------------------------------
Train       1       72%         baseline
Held-out B  2       31%         -41%
Held-out C  3       28%         -44%
Held-out D  4       35%         -37%
================================================
Interpretation: large drop = model memorised maze structure.
Small drop = policy learned general movement principles.
```

### CONCEPT block

- Explain what the drop reveals: DynamicsNetwork predicts position from (state, action). If drop is large, the policy relied on maze-specific dynamics that don't transfer. This is the fundamental limitation of model-based RL without domain randomization.
- Self-driving analog: training simulator built from one road layout; testing on a different city.

---

## Build / menu wiring

- `./build.sh 26` → `benchmark_deep_rl.py`
- `./build.sh 27` → `benchmark_world_model_gen.py`
- Menu: Benchmarks → option 4 (Deep RL comparison), option 5 (World Model generalization)
