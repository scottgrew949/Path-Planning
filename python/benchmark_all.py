# python/benchmark_all.py
#
# PURPOSE: Single-command benchmark — all algorithms, same 100 test mazes.
#
# CONCEPT — Why benchmark together?
#   Running each algorithm on an IDENTICAL maze set removes sampling noise.
#   Any difference in node count or path length is purely algorithmic,
#   not a lucky/unlucky maze draw. This is controlled comparison.
#
# Run: source venv/bin/activate && python python/benchmark_all.py

import sys
import os
import time
import csv
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import pathplanning

WEIGHTS_PATH     = os.path.join(os.path.dirname(__file__), 'data', 'weights.bin')
RESULTS_CSV_PATH = os.path.join(os.path.dirname(__file__), 'data', 'benchmark_all_results.csv')

GRID_WIDTH       = 41
GRID_HEIGHT      = 41
START_X, START_Y = 1, 1
GOAL_X,  GOAL_Y  = 39, 39
OBSTACLE_DENSITY = 0.25
NUM_MAZES        = 100
SEED_OFFSET      = 5000
NEURAL_EPSILON   = 1.5


def run_all_algorithms(env, neural_algo):
    """
    Run every registered algorithm on the given environment.
    Returns a dict: algo_name -> {'path_len': int, 'nodes': int, 'ms': float}
    path_len == 0 means the algorithm failed to find a path.
    """
    algorithms = [
        ('A*',            lambda: env.findPath(START_X, START_Y, GOAL_X, GOAL_Y)),
        ('Dijkstra',      lambda: env.findPathDijkstra(START_X, START_Y, GOAL_X, GOAL_Y)),
        ('BFS',           lambda: env.findPathBFS(START_X, START_Y, GOAL_X, GOAL_Y)),
        ('BiDir A*',      lambda: env.findPathBidirAStar(START_X, START_Y, GOAL_X, GOAL_Y)),
        ('Theta*',        lambda: env.findPathThetaStar(START_X, START_Y, GOAL_X, GOAL_Y)),
        ('JPS',           lambda: env.findPathJPS(START_X, START_Y, GOAL_X, GOAL_Y)),
    ]

    results = {}
    for algo_name, run_fn in algorithms:
        t0     = time.perf_counter()
        path   = run_fn()
        elapsed_ms = (time.perf_counter() - t0) * 1000
        results[algo_name] = {
            'path_len': len(path),
            'nodes':    env.getNodesExplored(),
            'ms':       elapsed_ms,
        }

    if neural_algo is not None:
        t0   = time.perf_counter()
        path = env.runNeuralAStar(neural_algo, START_X, START_Y, GOAL_X, GOAL_Y)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        results['Neural A*'] = {
            'path_len': len(path),
            'nodes':    neural_algo.getNodesExplored(),
            'ms':       elapsed_ms,
        }

    return results


def main():
    neural_algo = None
    if os.path.exists(WEIGHTS_PATH):
        neural_algo = pathplanning.NeuralAStar(WEIGHTS_PATH, NEURAL_EPSILON)
        print(f"Neural A* loaded (ε={NEURAL_EPSILON})")
    else:
        print(f"weights.bin not found — Neural A* excluded. Run: ./build.sh 12")

    algo_names = ['A*', 'Dijkstra', 'BFS', 'BiDir A*', 'Theta*', 'JPS']
    if neural_algo:
        algo_names.append('Neural A*')

    collected = {name: {'path_len': [], 'nodes': [], 'ms': []} for name in algo_names}
    skipped = 0

    print(f"\n  Grid {GRID_WIDTH}×{GRID_HEIGHT}  |  density={OBSTACLE_DENSITY}  |  {NUM_MAZES} mazes  |  seeds {SEED_OFFSET}–{SEED_OFFSET+NUM_MAZES-1}")
    print(f"  {'maze':>5}  " + "  ".join(f"{'nodes':>7}" for _ in algo_names))

    for maze_index in range(NUM_MAZES):
        seed = SEED_OFFSET + maze_index
        env  = pathplanning.GridEnvironment(
            GRID_WIDTH, GRID_HEIGHT,
            START_X, START_Y, GOAL_X, GOAL_Y,
            OBSTACLE_DENSITY, seed
        )

        # Skip mazes with no solution (use A* as the reference check)
        baseline = env.findPath(START_X, START_Y, GOAL_X, GOAL_Y)
        if not baseline:
            skipped += 1
            continue

        maze_results = run_all_algorithms(env, neural_algo)

        for name in algo_names:
            result = maze_results.get(name)
            if result and result['path_len'] > 0:
                collected[name]['path_len'].append(result['path_len'])
                collected[name]['nodes'].append(result['nodes'])
                collected[name]['ms'].append(result['ms'])

        if (maze_index + 1) % 20 == 0:
            print(f"  [{maze_index+1:>3}/{NUM_MAZES}]  " +
                  "  ".join(
                      f"{np.mean(collected[n]['nodes']):>7.0f}" if collected[n]['nodes'] else f"{'N/A':>7}"
                      for n in algo_names
                  ))

    solved = NUM_MAZES - skipped
    print(f"\n  Solved {solved}/{NUM_MAZES} mazes  ({skipped} skipped — no path exists)\n")

    col_width = 14
    header_line = f"  {'Algorithm':<14}" + "".join(
        f"{'nodes (mean±std)':>{col_width+2}}  {'path (mean±std)':>{col_width}}  {'ms (mean)':>10}"
        for _ in ['once']
    )

    sep = "═" * 76
    print(f"  {sep}")
    print(f"  {'Algorithm':<14}  {'Nodes (mean±std)':>18}  {'Path len (mean±std)':>22}  {'Time ms':>10}")
    print(f"  {'─'*14}  {'─'*18}  {'─'*22}  {'─'*10}")

    astar_nodes_mean = np.mean(collected['A*']['nodes']) if collected['A*']['nodes'] else 1.0

    csv_rows = []
    for name in algo_names:
        node_vals = collected[name]['nodes']
        path_vals = collected[name]['path_len']
        ms_vals   = collected[name]['ms']

        if not node_vals:
            print(f"  {name:<14}  {'N/A (all failed)':>54}")
            continue

        node_mean  = np.mean(node_vals)
        node_std   = np.std(node_vals)
        path_mean  = np.mean(path_vals)
        path_std   = np.std(path_vals)
        ms_mean    = np.mean(ms_vals)
        success_pct = len(node_vals) / solved * 100

        reduction = (astar_nodes_mean - node_mean) / astar_nodes_mean * 100

        print(f"  {name:<14}  {node_mean:>8.0f} ±{node_std:>6.0f}  "
              f"{path_mean:>10.1f} ±{path_std:>6.1f}  {ms_mean:>9.3f}  "
              f"[{reduction:>+.1f}%]  success={success_pct:.0f}%")

        csv_rows.append({
            'algorithm':     name,
            'solved_mazes':  len(node_vals),
            'nodes_mean':    f"{node_mean:.1f}",
            'nodes_std':     f"{node_std:.1f}",
            'path_len_mean': f"{path_mean:.1f}",
            'path_len_std':  f"{path_std:.1f}",
            'time_ms_mean':  f"{ms_mean:.4f}",
            'node_reduction_vs_astar_pct': f"{reduction:.1f}",
        })

    print(f"  {sep}")
    print(f"\n  [reduction %] = node count vs A* baseline. Negative = more nodes than A*.\n")

    os.makedirs(os.path.dirname(RESULTS_CSV_PATH), exist_ok=True)
    with open(RESULTS_CSV_PATH, 'w', newline='') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=csv_rows[0].keys())
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"  Results saved → {RESULTS_CSV_PATH}")


if __name__ == '__main__':
    main()
