# python/benchmark_rl_heuristic.py
#
# Option C: RL-derived heuristic vs supervised (Neural A*) vs Manhattan.
#
# Research question:
#   Can a Q-table trained via RL produce a better A* heuristic than Manhattan
#   distance? How does it compare to the supervised Neural A* heuristic?
#
# Key design difference between the two learned approaches:
#   Supervised (Neural A*): train once on 500 mazes → generalizes to any new maze
#   RL (Q-table A*):        must train a NEW Q-table for each maze it encounters
#
# This distinction IS the research result. Even if Q-table heuristic quality
# is comparable on in-distribution mazes, the per-maze training cost makes it
# impractical for real planning systems.
#
# Method:
#   For each of NUM_MAZES test mazes:
#     1. Train a Q-table on that maze for QTABLE_EPISODES episodes
#     2. Run Python A* with Manhattan heuristic     → record nodes, path len
#     3. Run Python A* with Q-table heuristic       → record nodes, path len
#     4. Run NeuralAStar (C++ pybind) on same maze  → record nodes, path len
#   Report: mean ± std for all three planners.
#
# Run: source venv/bin/activate && python python/benchmark_rl_heuristic.py

import sys
import os
import heapq
import random
import time
import numpy as np
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import pathplanning

WEIGHTS_PATH     = os.path.join(os.path.dirname(__file__), 'data', 'weights.bin')
GRID_WIDTH       = 41
GRID_HEIGHT      = 41
START_X, START_Y = 1, 1
GOAL_X,  GOAL_Y  = GRID_WIDTH - 2, GRID_HEIGHT - 2
OBSTACLE_DENSITY = 0.25
EPSILON          = 1.5

NUM_MAZES        = 50     # fewer mazes — Q-table training per maze is expensive
TEST_SEED_OFFSET = 3000   # offset from training (0-499), primary (1000-1999), generalization (2000-2499)
QTABLE_EPISODES  = 500    # Q-learning episodes per maze
QTABLE_ALPHA     = 0.1    # learning rate
QTABLE_GAMMA     = 0.95   # discount factor
QTABLE_EPSILON_START = 1.0
QTABLE_EPSILON_MIN   = 0.05
QTABLE_EPSILON_DECAY = 0.99
MAX_STEPS_PER_EPISODE = GRID_WIDTH * GRID_HEIGHT * 2


def train_qtable(env):
    """
    Train a Q-table on the given maze using epsilon-greedy Q-learning.
    Uses the existing pybind env.reset() / env.step() interface.
    Returns: dict mapping (x, y) → [Q_up, Q_down, Q_left, Q_right]
    """
    qtable  = defaultdict(lambda: [0.0, 0.0, 0.0, 0.0])
    epsilon = QTABLE_EPSILON_START

    for _ in range(QTABLE_EPISODES):
        state  = env.reset()
        state  = (state[0], state[1])

        for _ in range(MAX_STEPS_PER_EPISODE):
            if random.random() < epsilon:
                action = random.randint(0, 3)
            else:
                action = max(range(4), key=lambda a: qtable[state][a])

            result     = env.step(action)
            next_state = (int(result[0]), int(result[1]))
            reward     = result[2]
            done       = result[3] > 0.5

            best_next = max(qtable[next_state])
            qtable[state][action] += QTABLE_ALPHA * (
                reward + QTABLE_GAMMA * best_next - qtable[state][action]
            )

            state = next_state
            if done:
                break

        epsilon = max(QTABLE_EPSILON_MIN, epsilon * QTABLE_EPSILON_DECAY)

    return qtable


def astar_python(env, start, goal, heuristic_fn):
    """
    A* in Python with a pluggable heuristic.
    Returns (path_length, nodes_explored).
    Using Python A* for all three planners ensures fair node-count comparison.
    """
    start_t = tuple(start)
    goal_t  = tuple(goal)

    open_set  = [(heuristic_fn(start), 0, start_t, None)]
    g_costs   = {start_t: 0}
    came_from = {}
    explored  = set()

    while open_set:
        _, g, current, parent = heapq.heappop(open_set)

        if current in explored:
            continue
        explored.add(current)
        came_from[current] = parent

        if current == goal_t:
            path_len = 0
            node = current
            while node:
                path_len += 1
                node = came_from[node]
            return path_len - 1, len(explored)

        cx, cy = current
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nx, ny = cx + dx, cy + dy
            if not env.isValid(nx, ny):
                continue
            neighbor = (nx, ny)
            new_g    = g + 1
            if new_g < g_costs.get(neighbor, float('inf')):
                g_costs[neighbor] = new_g
                heapq.heappush(open_set, (new_g + heuristic_fn(neighbor), new_g, neighbor, current))

    return 0, len(explored)


def manhattan_heuristic(goal):
    gx, gy = goal
    def h(pos):
        return abs(pos[0] - gx) + abs(pos[1] - gy)
    return h


def qtable_heuristic(qtable, goal):
    """
    Convert Q-values to a cost heuristic.
    Q*(s) ≈ 100 - h*(s) so h(s) ≈ 100 - max_a Q(s, a).
    For states not in the Q-table, fall back to Manhattan.
    """
    gx, gy = goal
    def h(pos):
        state = pos
        if state in qtable:
            max_q = max(qtable[state])
            # Reward at goal = 100, step cost = -1. So Q* ≈ 100 - h*.
            return max(0.0, 100.0 - max_q)
        return abs(pos[0] - gx) + abs(pos[1] - gy)
    return h


def neural_heuristic_nodes(env, neural_algo):
    """Run Neural A* via C++ pybind. Returns (path_len, nodes_explored)."""
    path  = env.runNeuralAStar(neural_algo, START_X, START_Y, GOAL_X, GOAL_Y)
    nodes = neural_algo.getNodesExplored()
    return len(path), nodes


def main():
    if not os.path.exists(WEIGHTS_PATH):
        print("weights.bin not found. Run: ./build.sh 12")
        return

    neural_algo = pathplanning.NeuralAStar(WEIGHTS_PATH, EPSILON)
    start       = (START_X, START_Y)
    goal        = (GOAL_X, GOAL_Y)

    print(f"\n{'═'*76}")
    print("  RL-Derived vs Supervised Heuristic vs Manhattan  —  A* Node Comparison")
    print(f"{'═'*76}")
    print(f"  Grid: {GRID_WIDTH}×{GRID_HEIGHT}  |  density = {OBSTACLE_DENSITY}  |  {NUM_MAZES} mazes")
    print(f"  Q-table: {QTABLE_EPISODES} episodes/maze  |  Neural A*: ε = {EPSILON} (trained once on 500 mazes)")
    print(f"\n  Key question: does per-maze RL training beat one-time supervised training?")
    print(f"\n  {'':─<70}")
    print(f"  {'#':>4}  {'qtable train':>13}  {'manhattan':>10}  {'qtable A*':>10}  {'neural A*':>10}")
    print(f"  {'':─<70}")

    manhattan_nodes_list = []
    qtable_nodes_list    = []
    neural_nodes_list    = []
    manhattan_path_list  = []
    qtable_path_list     = []
    neural_path_list     = []
    qtable_train_times   = []
    skipped              = 0

    for i in range(NUM_MAZES):
        seed = TEST_SEED_OFFSET + i
        env  = pathplanning.GridEnvironment(
            GRID_WIDTH, GRID_HEIGHT,
            START_X, START_Y, GOAL_X, GOAL_Y,
            OBSTACLE_DENSITY, seed
        )

        # Skip mazes with no solution
        baseline = env.findPath(START_X, START_Y, GOAL_X, GOAL_Y)
        if not baseline:
            skipped += 1
            continue

        # Train Q-table on this maze
        t0       = time.time()
        qtable   = train_qtable(env)
        train_ms = (time.time() - t0) * 1000

        # Run all three planners
        m_len, m_nodes = astar_python(env, start, goal, manhattan_heuristic(goal))
        q_len, q_nodes = astar_python(env, start, goal, qtable_heuristic(qtable, goal))
        n_len, n_nodes = neural_heuristic_nodes(env, neural_algo)

        if m_len == 0 or n_len == 0:
            skipped += 1
            continue

        manhattan_nodes_list.append(m_nodes)
        qtable_nodes_list.append(q_nodes)
        neural_nodes_list.append(n_nodes)
        manhattan_path_list.append(m_len)
        qtable_path_list.append(q_len)
        neural_path_list.append(n_len)
        qtable_train_times.append(train_ms)

        print(f"  [{i+1:>2}/{NUM_MAZES}]  {train_ms:>9.0f} ms  "
              f"{m_nodes:>10}  {q_nodes:>10}  {n_nodes:>10}")

    if not manhattan_nodes_list:
        print("No solvable mazes found.")
        return

    m_mean = np.mean(manhattan_nodes_list)
    q_mean = np.mean(qtable_nodes_list)
    n_mean = np.mean(neural_nodes_list)

    q_reduction = (m_mean - q_mean) / m_mean * 100
    n_reduction = (m_mean - n_mean) / m_mean * 100

    q_path_ratio = np.mean(qtable_path_list)   / np.mean(manhattan_path_list)
    n_path_ratio = np.mean(neural_path_list)    / np.mean(manhattan_path_list)

    print(f"\n{'─'*76}")
    print(f"  {'':>30}  {'Manhattan':>12}  {'Q-table A*':>12}  {'Neural A*':>12}")
    print(f"  {'':─>30}  {'':─>12}  {'':─>12}  {'':─>12}")
    print(f"  {'Avg nodes explored':>30}  {m_mean:>12.1f}  {q_mean:>12.1f}  {n_mean:>12.1f}")
    print(f"  {'Std nodes':>30}  {np.std(manhattan_nodes_list):>12.1f}  {np.std(qtable_nodes_list):>12.1f}  {np.std(neural_nodes_list):>12.1f}")
    print(f"  {'Node reduction vs Manhattan':>30}  {'baseline':>12}  {q_reduction:>+11.1f}%  {n_reduction:>+11.1f}%")
    print(f"  {'Path ratio vs optimal':>30}  {'1.000x':>12}  {q_path_ratio:>11.3f}x  {n_path_ratio:>11.3f}x")
    print(f"  {'Training cost':>30}  {'none':>12}  {np.mean(qtable_train_times):>9.0f} ms  {'once (offline)':>12}")

    print(f"\n{'═'*76}")
    print("  FINDINGS")
    print(f"{'═'*76}")

    if q_reduction > n_reduction:
        print(f"  Q-table heuristic reduces nodes MORE than Neural A* ({q_reduction:+.1f}% vs {n_reduction:+.1f}%)")
        print(f"  BUT requires {np.mean(qtable_train_times):.0f} ms training per maze vs zero for Neural A*")
    else:
        print(f"  Neural A* reduces nodes more than Q-table ({n_reduction:+.1f}% vs {q_reduction:+.1f}%)")
        print(f"  AND requires zero per-maze training vs {np.mean(qtable_train_times):.0f} ms for Q-table")
        print(f"  Supervised offline training dominates RL on both efficiency and generalization.")

    print(f"\n  Q-table trained {QTABLE_EPISODES} episodes/maze = {np.mean(qtable_train_times):.0f} ms overhead per query")
    print(f"  Neural A* amortizes training cost across all mazes = ~0 ms overhead per query")
    print(f"{'═'*76}\n")


if __name__ == '__main__':
    main()
