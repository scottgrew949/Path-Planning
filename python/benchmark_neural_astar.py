# python/benchmark_neural_astar.py
#
# Research question: Does a learned heuristic beat Manhattan distance for A*?
# Under what conditions? Where does it break down?
#
# Experiment: sweep maze density and epsilon on 1000 test mazes each.
# Report nodes explored, path quality, and time per condition.
#
# Run: source venv/bin/activate && python python/benchmark_neural_astar.py

import sys
import os
import time
import numpy as np
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import pathplanning

WEIGHTS_PATH   = os.path.join(os.path.dirname(__file__), 'data', 'weights.bin')
GRID_WIDTH     = 41
GRID_HEIGHT    = 41
START_X, START_Y = 1, 1
GOAL_X,  GOAL_Y  = GRID_WIDTH - 2, GRID_HEIGHT - 2
NUM_MAZES      = 1000
TEST_SEED_OFFSET = 1000  # no overlap with training seeds 0–499

# Conditions to sweep
DENSITIES = [0.10, 0.20, 0.30, 0.40]
EPSILONS  = [1.0, 1.5, 2.0]


def run_astar(env):
    path  = env.findPath(START_X, START_Y, GOAL_X, GOAL_Y)
    nodes = env.getNodesExplored()
    return len(path), nodes


def run_neural_astar(env, neural_algo):
    path        = env.runNeuralAStar(neural_algo, START_X, START_Y, GOAL_X, GOAL_Y)
    nodes       = neural_algo.getNodesExplored()
    return len(path), nodes


def benchmark_condition(density, epsilon, neural_algo):
    """Run both planners on NUM_MAZES mazes at given density and epsilon."""
    astar_paths, astar_nodes     = [], []
    neural_paths, neural_nodes   = [], []
    skipped = 0

    for i in range(NUM_MAZES):
        seed = TEST_SEED_OFFSET + i
        env  = pathplanning.GridEnvironment(
            GRID_WIDTH, GRID_HEIGHT,
            START_X, START_Y, GOAL_X, GOAL_Y,
            density, seed
        )

        a_path, a_nodes = run_astar(env)
        if a_path == 0:   # no path exists in this maze
            skipped += 1
            continue

        n_path, n_nodes = run_neural_astar(env, neural_algo)
        if n_path == 0:   # Neural A* failed — skip maze to avoid polluting stats
            skipped += 1
            continue

        astar_paths.append(a_path)
        astar_nodes.append(a_nodes)
        neural_paths.append(n_path)
        neural_nodes.append(n_nodes)

    return {
        "astar_path_mean":   np.mean(astar_paths),
        "astar_path_std":    np.std(astar_paths),
        "astar_nodes_mean":  np.mean(astar_nodes),
        "astar_nodes_std":   np.std(astar_nodes),
        "neural_path_mean":  np.mean(neural_paths),
        "neural_path_std":   np.std(neural_paths),
        "neural_nodes_mean": np.mean(neural_nodes),
        "neural_nodes_std":  np.std(neural_nodes),
        "node_reduction_pct": (np.mean(astar_nodes) - np.mean(neural_nodes)) / np.mean(astar_nodes) * 100,
        "path_ratio":         np.mean(neural_paths) / np.mean(astar_paths),
        "solved":             len(astar_paths),
        "skipped":            skipped,
    }


def print_density_table(results_by_density, epsilon):
    """Table: how does Neural A* performance vary with maze difficulty?"""
    if epsilon == 1.0:
        note = "NOTE: at ε = 1.0, the learned heuristic is constrained to be admissible\n  (never overestimates true cost), so it explores the same nodes as standard\n  A* — no reduction. Raising ε trades path optimality for search speed."
    else:
        note = f"paths guaranteed ≤ {epsilon}× optimal cost"
    print(f"\n{'─'*72}")
    print(f"  Neural A* (ε = {epsilon}) vs standard A* (Manhattan heuristic)")
    print(f"  ε = epsilon = heuristic weight:  f(n) = g(n) + ε × h_neural(n)")
    print(f"  {note}")
    print(f"{'─'*72}")
    print(f"  {'density':>9}  {'A* nodes (mean±std)':>22}  {'Neural nodes (mean±std)':>25}  {'nodes saved':>12}  {'path overhead':>14}")
    print(f"  {'':─>9}  {'':─>22}  {'':─>25}  {'':─>12}  {'':─>14}")

    for density, r in sorted(results_by_density.items()):
        overhead = (r['path_ratio'] - 1) * 100
        print(f"  {density:>9.2f}  "
              f"{r['astar_nodes_mean']:>10.1f} ± {r['astar_nodes_std']:>6.1f}  "
              f"{r['neural_nodes_mean']:>12.1f} ± {r['neural_nodes_std']:>6.1f}  "
              f"{r['node_reduction_pct']:>+11.1f}%  "
              f"{overhead:>+12.1f}%")
    print()


def print_epsilon_table(results_by_epsilon, density):
    """Table: how does epsilon affect the nodes vs path quality tradeoff?"""
    print(f"\n{'─'*72}")
    print(f"  ε (heuristic weight) tradeoff  |  density={density}  |  {NUM_MAZES} mazes")
    print(f"  ε controls aggressiveness: higher ε = fewer nodes explored but longer paths")
    print(f"  f(n) = g(n) + ε × h_neural(n)   where g = cost so far, h = learned estimate")
    print(f"{'─'*72}")
    print(f"  {'ε':>5}  {'neural nodes':>14}  {'nodes saved':>12}  {'path overhead':>14}  {'quality':>22}")
    print(f"  {'':─>5}  {'':─>14}  {'':─>12}  {'':─>14}  {'':─>22}")

    for epsilon, r in sorted(results_by_epsilon.items()):
        overhead = (r['path_ratio'] - 1) * 100
        if overhead < 0.1:
            verdict = "identical to optimal"
        elif overhead < 3.0:
            verdict = "near-optimal (< 3%)"
        elif overhead < 7.0:
            verdict = "acceptable (< 7%)"
        else:
            verdict = "degraded"
        print(f"  {epsilon:>5.1f}  "
              f"{r['neural_nodes_mean']:>10.1f} ± {r['neural_nodes_std']:>4.0f}  "
              f"{r['node_reduction_pct']:>+11.1f}%  "
              f"{overhead:>+12.1f}%  "
              f"{verdict:>22}")
    print()


def print_summary(all_results):
    """Find best and worst conditions. State the core finding."""
    best_reduction = max(all_results.values(), key=lambda r: r["node_reduction_pct"])
    worst_quality  = max(all_results.values(), key=lambda r: r["path_ratio"])
    best_key       = max(all_results, key=lambda k: all_results[k]["node_reduction_pct"])
    worst_key      = max(all_results, key=lambda k: all_results[k]["path_ratio"])

    print(f"\n{'═'*70}")
    print("  FINDINGS")
    print(f"{'═'*70}")
    print(f"  Best node reduction:  {best_reduction['node_reduction_pct']:+.1f}%  "
          f"(density={best_key[0]}, ε={best_key[1]})")
    print(f"  Worst path quality:   {worst_quality['path_ratio']:.3f}x optimal  "
          f"(density={worst_key[0]}, ε={worst_key[1]})")
    print()

    # Find breakeven density (where reduction goes negative)
    for density in sorted(DENSITIES):
        r = all_results.get((density, 1.5))
        if r and r["node_reduction_pct"] < 0:
            print(f"  Neural A* becomes WORSE than Manhattan at density={density}")
            break
    else:
        print(f"  Neural A* beats Manhattan across all tested densities at ε=1.5")

    print(f"{'═'*70}\n")


def main():
    if not os.path.exists(WEIGHTS_PATH):
        print("weights.bin not found. Run: ./build.sh 12")
        return

    neural_algos = {eps: pathplanning.NeuralAStar(WEIGHTS_PATH, eps) for eps in EPSILONS}

    print(f"\n{'═'*72}")
    print("  Neural A* Benchmark  —  Learned Heuristic vs Manhattan Distance")
    print(f"{'═'*72}")
    print(f"  Grid:       {GRID_WIDTH}×{GRID_HEIGHT}")
    print(f"  Mazes:      {NUM_MAZES} per condition (seeds {TEST_SEED_OFFSET}–{TEST_SEED_OFFSET+NUM_MAZES-1}, never seen during training)")
    print(f"  Densities:  {DENSITIES}  (fraction of cells that are walls)")
    print(f"  Epsilons:   {EPSILONS}  (heuristic weight — higher = faster but less optimal)")
    print(f"  Conditions: {len(DENSITIES)} × {len(EPSILONS)} = {len(DENSITIES)*len(EPSILONS)} total")
    print(f"\n  Running...")
    print(f"  {'':─<55}")
    print(f"  {'#':>4}  {'density':>9}  {'ε (weight)':>12}  {'time':>6}  {'solvable':>10}")
    print(f"  {'':─<55}")

    all_results = {}
    total = len(DENSITIES) * len(EPSILONS)
    done  = 0

    for density in DENSITIES:
        for epsilon in EPSILONS:
            done += 1
            print(f"  [{done:>2}/{total}]  density = {density:.2f}   ε = {epsilon:.1f}  ", end="", flush=True)
            start = time.time()
            r = benchmark_condition(density, epsilon, neural_algos[epsilon])
            elapsed = time.time() - start
            all_results[(density, epsilon)] = r
            print(f"  {elapsed:>4.1f}s    {r['solved']}/{NUM_MAZES}")

    print(f"\n{'═'*72}")
    print("  RESULTS")
    print(f"  Nodes explored = how many cells A* had to examine before finding the path.")
    print(f"  Fewer nodes = faster search = better heuristic.")
    print(f"  Path overhead = how much longer Neural A*'s path is vs the optimal A* path.")
    print(f"{'═'*72}")

    # Table 1: node reduction vs maze density at each epsilon
    for epsilon in EPSILONS:
        results_by_density = {d: all_results[(d, epsilon)] for d in DENSITIES}
        print_density_table(results_by_density, epsilon)

    # Table 2: epsilon tradeoff at medium density
    results_by_epsilon = {eps: all_results[(0.20, eps)] for eps in EPSILONS}
    print_epsilon_table(results_by_epsilon, density=0.20)

    # Summary: core finding
    print_summary(all_results)


if __name__ == '__main__':
    main()
